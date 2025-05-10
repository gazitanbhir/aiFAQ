import logging
import json
import os
from functools import lru_cache
from enum import Enum, auto

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
# Allow overriding paths via .env, default to assets/data/ relative to main.py
PERSONA_FILE_PATH = os.getenv("PERSONA_FILE_PATH", os.path.join(BASE_DIR, "assets/data/persona.txt"))
FAQ_FILE_PATH = os.getenv("FAQ_FILE_PATH", os.path.join(BASE_DIR, "assets/data/faq.json"))

MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 5))
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Aura")
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"

# --- Logging Setup ---
log_level = logging.DEBUG if DEV_MODE else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Essential Configuration Check ---
if not GOOGLE_AI_API_KEY:
    logger.error("FATAL: GOOGLE_AI_API_KEY environment variable not set.")
    raise ValueError("Missing Google AI API Key configuration.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FAQ Chatbot API - AuraCoreTech",
    description="API endpoint and frontend for Aura, the AuraCoreTech intent-aware AI assistant.",
    version="1.4.0", # Incremented for intent handling
)

# --- CORS Middleware ---
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    # Add your deployed frontend URL(s) here
    "https://auracoretech.com/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- User Intent Definition ---
class UserIntent(Enum):
    GENERAL_INQUIRY = auto()
    SERVICE_EXPLORATION = auto()
    PRICING_QUOTE = auto()
    TECHNICAL_SUPPORT = auto()
    CONSULTATION_BOOKING = auto()
    PROJECT_ONBOARDING_FOLLOWUP = auto()
    RANDOM_BROWSING = auto()
    UNKNOWN = auto()

# --- Ensure data files exist (or create defaults) ---
def ensure_file_exists(filepath, default_content=""):
    """Checks if a file exists, creates it with default content if not."""
    if not os.path.exists(filepath):
        try:
            dir_name = os.path.dirname(filepath)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"Created directory: {dir_name}")

            with open(filepath, "w", encoding="utf-8") as f:
                if isinstance(default_content, (dict, list)):
                    json.dump(default_content, f, indent=2)
                else:
                    f.write(str(default_content))
            logger.warning(f"Created default file: {filepath}. Please ensure content is correct.")
        except IOError as e:
            logger.error(f"Failed to create default file at {filepath}: {e}")
        except Exception as e:
             logger.error(f"An unexpected error occurred creating default file {filepath}: {e}")

# Default content (used if files are missing)
default_persona_content = """
Fallback Persona: Basic Assistant
Role: Answer questions based on provided FAQ data.
Tone: Neutral and informative.
Instructions: Stick strictly to the FAQ data. If the answer isn't there, state that clearly. Keep answers very short.
"""
default_faq_content = {
  "title": "Fallback FAQ",
  "instructions": "Answer based *only* on the Q&A pairs listed.",
  "faq_data": {
    "General": [
      ["What is this service?", "This is a fallback FAQ. The main data file was not found or is invalid."],
      ["Is support available?", "Please check the system configuration and ensure faq.json exists and contains support contact details."]
    ]
  }
}

ensure_file_exists(PERSONA_FILE_PATH, default_persona_content)
ensure_file_exists(FAQ_FILE_PATH, default_faq_content)


# --- Gemini AI Configuration ---
generation_config = {
    "temperature": 0.3, # Lower temperature for factual, guided responses
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 250, # Enforce conciseness from persona
    "response_mime_type": "text/plain",
}

safety_settings = {
    "HATE": "BLOCK_MEDIUM_AND_ABOVE",
    "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
    "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE",
}

# --- AI Model Initialization ---
model = None
try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    logger.info(f"Gemini AI Model '{model.model_name}' configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {e}", exc_info=True)
    # App will continue but health check will fail and responses will error

# --- Data Loading Functions ---
@lru_cache(maxsize=1)
def load_persona_instructions(filename: str = PERSONA_FILE_PATH) -> str:
    """Loads the persona instructions text file."""
    logger.debug(f"Attempting to load persona from: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as file:
            instructions = file.read().strip()
            if not instructions:
                logger.warning(f"Persona file '{filename}' is empty. Using default prompt.")
                return str(default_persona_content)
            logger.info(f"Persona instructions loaded successfully from '{filename}'.")
            return instructions
    except FileNotFoundError:
        logger.error(f"CRITICAL: Persona file not found at {filename}. Using default.")
        ensure_file_exists(filename, default_persona_content) # Attempt recreation
        return str(default_persona_content)
    except Exception as e:
        logger.error(f"Error loading persona from {filename}: {e}", exc_info=True)
        return str(default_persona_content)

@lru_cache(maxsize=1)
def load_faq_data(filename: str = FAQ_FILE_PATH) -> str:
    """Loads and formats FAQ data from the JSON structure into a string for the prompt."""
    logger.debug(f"Attempting to load FAQ data from: {filename}")
    faq_context_lines = []
    raw_faq_data = {} # Store raw data for potential direct lookups if needed later

    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            raw_faq_data = data # Store loaded data

            faq_data_section = data.get("faq_data", {})
            if not faq_data_section:
                 logger.warning(f"No 'faq_data' object found or empty in {filename}.")
                 return "(No FAQs loaded - data structure missing or empty)", raw_faq_data

            # Iterate through sections and their QA pairs
            for section_title, qa_list in faq_data_section.items():
                 faq_context_lines.append(f"\n## Section: {section_title}\n")
                 if isinstance(qa_list, list): # Top-level section like "Video Editing"
                     for i, qa_pair in enumerate(qa_list, 1):
                         if isinstance(qa_pair, list) and len(qa_pair) == 2:
                             q, a = map(str, qa_pair)
                             faq_context_lines.append(f"{i}. Q: {q}\n   A: {a}")
                         else:
                             logger.warning(f"Skipping malformed QA pair in section '{section_title}': {qa_pair}")
                 elif isinstance(qa_list, dict): # Nested sections like "Security" subsections
                     for sub_section_title, sub_qa_list in qa_list.items():
                          if isinstance(sub_qa_list, list):
                               faq_context_lines.append(f"### Subsection: {sub_section_title}\n")
                               for i, qa_pair in enumerate(sub_qa_list, 1):
                                   if isinstance(qa_pair, list) and len(qa_pair) == 2:
                                       q, a = map(str, qa_pair)
                                       faq_context_lines.append(f"{i}. Q: {q}\n   A: {a}")
                                   else:
                                       logger.warning(f"Skipping malformed QA pair in subsection '{section_title}/{sub_section_title}': {qa_pair}")
                          else:
                               logger.warning(f"Skipping malformed subsection content in '{section_title}': {sub_section_title}")
                 else:
                     logger.warning(f"Skipping malformed section content for '{section_title}'. Expected list or dict.")


            if not faq_context_lines:
                 logger.warning(f"FAQ data loaded from '{filename}' but resulted in no formatted content.")
                 return "(No valid FAQs found in loaded data)", raw_faq_data

            logger.info(f"FAQ data loaded and formatted successfully from '{filename}'.")
            # Return both formatted string and raw data dict
            return "\n".join(faq_context_lines), raw_faq_data

    except FileNotFoundError:
        logger.error(f"CRITICAL: FAQ file not found: {filename}. Chatbot will lack context.")
        ensure_file_exists(filename, default_faq_content) # Attempt recreation
        return "(FAQ file missing - using fallback)", {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in FAQ file: {filename}. Error: {e}. FAQ context unavailable.")
        return "(Error loading FAQs: Invalid JSON)", {}
    except Exception as e:
        logger.error(f"Error reading or processing FAQ file '{filename}': {e}", exc_info=True)
        return "(Error loading FAQs)", {}


# --- Conversation History (Simple In-Memory Store) ---
conversation_history = []

# --- Intent Detection Function ---
@lru_cache(maxsize=128) # Cache recent intent detections
def detect_intent(user_message: str) -> UserIntent:
    """Uses the LLM to classify the user message into a predefined intent."""
    if not model:
        logger.error("Intent detection model not available.")
        return UserIntent.UNKNOWN

    # Ensure all enum members are included except UNKNOWN
    intent_list = "\n".join([f"- {intent.name}" for intent in UserIntent if intent != UserIntent.UNKNOWN])

    prompt = f"""
Analyze the following user message and classify its primary intent based ONLY on the list provided below. Choose the single best fit.

User Message: "{user_message}"

Possible Intents:
{intent_list}

Output ONLY the single intent name (e.g., GENERAL_INQUIRY, TECHNICAL_SUPPORT). If none fit well or the message is nonsensical, output UNKNOWN.
Intent: """

    try:
        # Use a potentially faster config for classification
        intent_gen_config = {"temperature": 0.1, "max_output_tokens": 50, "top_p": 1.0, "top_k": 1}
        response = model.generate_content(
            prompt,
            generation_config=intent_gen_config,
            safety_settings=safety_settings # Apply safety here too
        )

        # Handle potential blocks or empty responses in intent detection
        if not response.candidates or not hasattr(response.candidates[0], 'content') or not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
             logger.warning(f"Intent detection failed or blocked for message: '{user_message[:50]}...' - Reason: {response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback, 'block_reason') else 'Unknown'}. Defaulting to UNKNOWN.")
             return UserIntent.UNKNOWN

        detected_intent_str = response.text.strip().upper()

        # Map string back to Enum member safely
        for intent_enum in UserIntent:
            if intent_enum.name == detected_intent_str:
                logger.info(f"Detected intent for '{user_message[:50]}...': {intent_enum.name}")
                return intent_enum

        logger.warning(f"Intent detection returned unexpected value: '{detected_intent_str}'. Defaulting to UNKNOWN.")
        return UserIntent.UNKNOWN

    except Exception as e:
        logger.error(f"Error during intent detection API call: {e}", exc_info=True)
        return UserIntent.UNKNOWN

# --- Core Chat Logic ---
def generate_bot_response(question: str, persona_instructions: str, faq_context: str, intent_hint: str = "") -> str:
    """Generates a response using AI, guided by persona, FAQs, and intent hint."""
    global conversation_history
    global model

    if not model:
        logger.error("AI model is not initialized. Cannot generate response.")
        return "[System Error: AI Model not available. Please check configuration.]"

    prompt_lines = [
        "START OF ASSISTANT INSTRUCTIONS",
        f"Your name is {ASSISTANT_NAME}.",
        persona_instructions, # Persona includes detailed intent handling, creativity, conciseness rules
        f"Hint based on detected user intent: {intent_hint}" if intent_hint else "User intent was not specifically determined, rely on general instructions.",
        "Strictly adhere to all instructions, especially regarding response length, factual basis (FAQ only), and intent-specific handling described in the persona.",
        "END OF ASSISTANT INSTRUCTIONS",
        "\n",
        "--- Company FAQ Information ---",
        faq_context, # Formatted Q&A string or error message
        "--- End of Company FAQ Information ---",
        "\n",
        "--- Conversation History (Oldest to Newest) ---",
    ]

    # Add limited history (newest turns first in prompt is sometimes better)
    relevant_history = conversation_history[-(MAX_HISTORY_LENGTH):]
    history_pairs = []
    for q_hist, a_hist in relevant_history:
         history_pairs.append(f"User: {q_hist}")
         history_pairs.append(f"{ASSISTANT_NAME}: {a_hist}")

    if not history_pairs:
         prompt_lines.append("(No previous turns in this conversation)")
    else:
        prompt_lines.extend(history_pairs)

    prompt_lines.append("--- End of Conversation History ---")
    prompt_lines.append("\n--- Current User Query ---")
    prompt_lines.append(f"User: {question}")
    prompt_lines.append("--- End of User Query ---")
    prompt_lines.append(f"\n{ASSISTANT_NAME}:") # Prompt the model for its response

    full_prompt = "\n".join(filter(None, prompt_lines))
    # logger.debug(f"Prompt sent to AI:\n-------\n{full_prompt}\n-------") # Uncomment for deep debugging

    try:
        response = model.generate_content(full_prompt) # Uses the main generation_config
        answer = ""

        if not response.candidates:
            block_reason = "Unknown"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                block_reason = response.prompt_feedback.block_reason.name
            logger.warning(f"AI response blocked or empty. Reason: {block_reason}")
            answer = f"(I'm unable to provide a response due to content safety filters ({block_reason}). Please rephrase your query based on our FAQ topics.)"
        elif hasattr(response, 'parts') and response.parts:
             answer = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'): # Fallback
             answer = response.text

        if not answer or not answer.strip():
             logger.warning("AI response generated but contained no meaningful text.")
             # Check if it was blocked at the candidate level
             finish_reason = "Unknown"
             safety_ratings = []
             if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                finish_reason = response.candidates[0].finish_reason.name
             if response.candidates and hasattr(response.candidates[0], 'safety_ratings'):
                 safety_ratings = [f"{r.category.name}: {r.probability.name}" for r in response.candidates[0].safety_ratings]

             if finish_reason == "SAFETY":
                 answer = f"(My response was blocked due to safety concerns [{', '.join(safety_ratings)}]. Please ask about topics covered in our FAQs.)"
             else:
                 answer = "(I could not generate a specific answer for that query based on the provided FAQs. Could you please ask differently or check if the topic is covered?)"


        # Update history ONLY if the response wasn't a system error/block message (heuristic)
        if not answer.startswith("[System Error") and not "(I'm unable to provide a response" in answer and not "(My response was blocked" in answer :
            conversation_history.append((question, answer.strip()))
            # Prune history
            while len(conversation_history) > MAX_HISTORY_LENGTH:
                 conversation_history.pop(0)
        else:
            logger.warning("Skipping history update due to system error or blocked response.")


        return answer.strip()

    except ValueError as ve: # E.g., Invalid API key
        logger.error(f"ValueError during AI API call: {str(ve)}", exc_info=True)
        return "[System Error: Configuration issue with the AI service. Please check logs.]"
    except Exception as e:
        logger.error(f"Error during AI API call or processing: {str(e)}", exc_info=True)
        return "[System Error: I encountered a problem generating a response. Please try again shortly.]"


# --- Pydantic Models for API Validation ---
class ChatRequest(BaseModel):
    message: str
    reset_context: bool = False

class ChatResponse(BaseModel):
    response: str
    context_cleared: bool = False
    intent_detected: str | None = None # Add detected intent to response


# --- API Endpoints ---

# Load data once on startup and inject dependencies
@lru_cache() # Cache the combined result
def get_common_dependencies():
    persona = load_persona_instructions()
    faq_text, faq_raw = load_faq_data()
    # Log loaded state once after first call
    logger.info("Persona and FAQ data loaded/cached.")
    return {"persona": persona, "faq_text": faq_text, "faq_raw": faq_raw}

@app.post("/api/chat",
          response_model=ChatResponse,
          summary="Process Chat Message with Aura (Intent-Aware)",
          tags=["Chatbot"])
async def api_chat(
    request_data: ChatRequest,
    common_data: dict = Depends(get_common_dependencies) # Inject cached data
):
    """
    Receives user message, detects intent, uses Aura's persona and FAQ data
    to generate a context-aware response. Optionally resets conversation history.
    """
    global conversation_history
    context_cleared = False
    persona = common_data["persona"]
    faq_text = common_data["faq_text"]
    # faq_raw = common_data["faq_raw"] # Raw data available if needed for direct lookups

    if request_data.reset_context:
        conversation_history = []
        logger.info("Conversation history reset via API request.")
        context_cleared = True
        # Return a standard greeting after reset
        return ChatResponse(
            response=f"Hello! I'm {ASSISTANT_NAME}, your guide to AuraCoreTech based on our FAQs. The context has been reset. How can I assist you?",
            context_cleared=True,
            intent_detected="CONTEXT_RESET"
        )

    if not request_data.message or not request_data.message.strip():
        logger.warning("Received empty message in API request.")
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    user_message = request_data.message.strip()
    logger.info(f"Received message: '{user_message}'")

    # --- Intent Detection Step ---
    intent = detect_intent(user_message)
    intent_name = intent.name # Get string name for logging/response

    # --- Intent-Based Handling ---
    # The persona now dictates how the LLM should handle these intents based on FAQ content.
    # We just provide a hint to the main generator.
    intent_specific_instructions = ""
    if intent != UserIntent.UNKNOWN:
        intent_specific_instructions = f"The user's query seems related to {intent_name}. Handle according to the specific instructions for this intent in the persona, using ONLY the FAQ data for information or contact details."
    else:
        intent_specific_instructions = "The user's intent is unclear. Try to answer based on the general FAQ context, or if it seems completely unrelated, state that the query is outside the scope of the available FAQ information as per persona instructions."

    logger.info(f"Handling intent '{intent_name}' using FAQ context guided by persona.")

    # --- Generate Response using LLM ---
    # No direct responses needed here as the persona guides the LLM based on intent
    answer = generate_bot_response(
        question=user_message,
        persona_instructions=persona,
        faq_context=faq_text,
        intent_hint=intent_specific_instructions
    )

    logger.info(f"Generated response (Intent: {intent_name}): '{answer[:100]}...'")
    return ChatResponse(response=answer, context_cleared=context_cleared, intent_detected=intent_name)


# Health Check Endpoint
@app.get("/health",
         status_code=200,
         summary="Health Check",
         tags=["System"],
         response_description="Returns the operational status of the API.")
async def health_check():
    """Checks AI model status, data file accessibility, and basic operation."""
    global model
    persona_ok = False
    faq_ok = False
    model_ok = False
    model_name = "Not Initialized"
    details = {}

    # Check data file loading (uses cache, forces reload check indirectly)
    try:
        # Re-call loading functions to check current accessibility
        load_persona_instructions.cache_clear()
        load_faq_data.cache_clear()
        get_common_dependencies.cache_clear() # Clear combined cache too
        deps = get_common_dependencies()
        persona_ok = bool(deps["persona"] and not deps["persona"].startswith("Fallback Persona"))
        faq_ok = bool(deps["faq_text"] and not deps["faq_text"].startswith("(FAQ file missing") and not deps["faq_text"].startswith("(Error loading FAQs"))
        details["persona_loaded_ok"] = persona_ok
        details["faq_loaded_ok"] = faq_ok
    except Exception as e:
        logger.warning(f"Health check: Data loading issue - {e}", exc_info=DEV_MODE)
        details["data_loading_error"] = str(e)


    # Check AI model status
    if model is not None and hasattr(model, 'model_name'):
         model_ok = True
         model_name = model.model_name
         # Optional: Try a minimal test call? (Can add latency/cost)
         # try:
         #    test_response = model.generate_content("Test", generation_config={"max_output_tokens": 5})
         #    model_ok = bool(test_response.text)
         # except Exception as test_e:
         #    model_ok = False
         #    details["model_test_error"] = str(test_e)
    elif GOOGLE_AI_API_KEY and not model:
         model_ok = False
         model_name = "Initialization Failed (Check Logs)"
    elif not GOOGLE_AI_API_KEY:
         model_ok = False
         model_name = "Configuration Error (API Key Missing)"

    status = "ok" if persona_ok and faq_ok and model_ok else "error"
    status_code = 200 if status == "ok" else 503 # Service Unavailable

    # Manually construct response to set status code correctly via FastAPI
    response_body = {
        "status": status,
        "details": {
             "model_status": "configured" if model_ok else "error",
             "model_name": model_name,
             "persona_file_status": "accessible" if persona_ok else "error_or_default",
             "faq_data_status": "accessible_and_parsed" if faq_ok else "error_or_missing",
             "current_history_length": len(conversation_history),
             **details # Add specific error details if any
        }
    }
    # FastAPI sets status code based on exceptions or explicit status_code in decorator/response object
    # To ensure 503 is returned when status is 'error', we might need to raise HTTPException
    # However, returning the detailed JSON with status 'error' and 200 OK might be acceptable for some health checks.
    # For stricter checks, uncomment the raise:
    # if status == 'error':
    #     raise HTTPException(status_code=503, detail=response_body)

    return response_body


# --- Serve Frontend Static Files ---
if os.path.exists(FRONTEND_DIR) and os.path.isdir(FRONTEND_DIR):
    try:
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
        logger.info(f"Serving static files from directory: {FRONTEND_DIR}")
    except Exception as e:
        logger.error(f"Failed to mount static files directory '{FRONTEND_DIR}': {e}")
        # Add fallback even if mounting fails
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def read_root_mount_error():
            return """
            <html><head><title>AuraCoreTech API Error</title></head><body>
            <h1>AuraCoreTech FAQ Chatbot API</h1>
            <p>API is running, but there was an error serving the frontend.</p>
            <p>Check API documentation at <a href="/docs">/docs</a>.</p></body></html>
            """
else:
    logger.warning(f"Frontend directory '{FRONTEND_DIR}' not found. Static file serving disabled.")
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def read_root_fallback():
        return """
        <html><head><title>AuraCoreTech API</title></head><body>
        <h1>AuraCoreTech FAQ Chatbot API</h1>
        <p>API is running, but the frontend directory was not found.</p>
        <p>Check API documentation at <a href="/docs">/docs</a>.</p></body></html>
        """


# --- Main Execution Guard ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting AuraCoreTech FAQ Chatbot Backend API server... (DEV_MODE={DEV_MODE})")
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" # Listen on all interfaces

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=DEV_MODE, # Enable reload only if DEV_MODE is true
        log_level=log_level.lower() # Pass log level to uvicorn
    )