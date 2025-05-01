import logging
import json
import os
from functools import lru_cache

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
PERSONA_FILE_PATH = os.getenv("PERSONA_FILE_PATH", "assets/data/persona.txt")
FAQ_FILE_PATH = os.getenv("FAQ_FILE_PATH", "assets/data/faq.json")
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 3)) # Keep history short for FAQ focus
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Assistant")
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Essential Configuration Check ---
if not GOOGLE_AI_API_KEY:
    logger.error("FATAL: GOOGLE_AI_API_KEY environment variable not set.")
    raise ValueError("Missing Google AI API Key configuration.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FAQ Chatbot API",
    description="API endpoint for an FAQ-focused AI assistant.",
    version="1.1.0",
)

# --- CORS Middleware ---
# Adjust origins for your specific frontend deployment
origins = [
    "http://localhost",
    "http://localhost:8080", # Common local dev port for frontend
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "http://0.0.0.0:8080", # <--- ADD THIS LINE

    # Add your deployed frontend URL(s) here
    "https://auracoretech.com/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Now includes the origin the browser is reporting
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

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
                f.write(default_content)
            logger.warning(f"Created default file: {filepath}. Please customize it.")
        except IOError as e:
            logger.error(f"Failed to create default file at {filepath}: {e}")
            # Decide if this is critical enough to stop the app
            # raise IOError(f"Could not create essential file: {filepath}") from e

default_persona = PERSONA_FILE_PATH
default_faq = FAQ_FILE_PATH

ensure_file_exists(PERSONA_FILE_PATH, default_persona)
ensure_file_exists(FAQ_FILE_PATH, default_faq)

# --- Gemini AI Configuration ---
generation_config = {
    "temperature": 0.3, # Lower temperature for more factual, less creative responses suitable for FAQs
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 200, # Adjust as needed based on typical answer length
    "response_mime_type": "text/plain",
}

safety_settings = {
    "HATE": "BLOCK_MEDIUM_AND_ABOVE",
    "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
    "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE",
}

try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Or "gemini-pro"
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    logger.info(f"Gemini AI Model '{model.model_name}' configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {e}", exc_info=True)
    raise RuntimeError(f"Gemini AI configuration failed: {e}")


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
                return default_persona # Fallback to a minimal default
            logger.info(f"Persona instructions loaded successfully from '{filename}'.")
            return instructions
    except FileNotFoundError:
        logger.error(f"CRITICAL: Persona file not found at {filename}. Using default.")
        return default_persona
    except Exception as e:
        logger.error(f"Error loading persona from {filename}: {e}", exc_info=True)
        return default_persona # Fallback on error

@lru_cache(maxsize=1)
def load_faq_data(filename: str = FAQ_FILE_PATH) -> str:
    """Loads and formats FAQ data from JSON file into a string for the prompt."""
    logger.debug(f"Attempting to load FAQ data from: {filename}")
    faq_string = ""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            faqs = data.get("faqs", [])
            if not faqs:
                 logger.warning(f"No 'faqs' array found or array is empty in {filename}.")
                 return "(No FAQs loaded)"

            faq_list_str = []
            for i, item in enumerate(faqs, 1):
                q = item.get("question", f"FAQ {i} - Missing Question")
                a = item.get("answer", f"FAQ {i} - Missing Answer")
                faq_list_str.append(f"{i}. Q: {q}\n   A: {a}") # Numbered list for clarity
            faq_string = "\n\n".join(faq_list_str) # Separate FAQs clearly
            logger.info(f"FAQ data loaded and formatted successfully from '{filename}'. Found {len(faqs)} FAQs.")

    except FileNotFoundError:
        logger.warning(f"FAQ file not found: {filename}. Chatbot will lack FAQ context.")
        faq_string = "(FAQ file missing)"
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in FAQ file: {filename}. FAQ context unavailable.")
        faq_string = "(Error loading FAQs: Invalid JSON)"
    except Exception as e:
        logger.error(f"Error reading FAQ file '{filename}': {e}", exc_info=True)
        faq_string = "(Error loading FAQs)"

    # Return even if empty/error, so the prompt structure is maintained
    return faq_string if faq_string else "(No FAQs loaded)"


# --- Conversation History (Simple In-Memory Store - Per Instance) ---
# Warning: Global state. Not suitable for multi-user production without session management.
conversation_history = []

# --- Core Chat Logic ---
def generate_bot_response(question: str, persona_instructions: str, faq_context: str) -> str:
    """Generates a response using AI, focusing on persona and provided FAQs."""
    global conversation_history

    # Build the prompt with clear sections
    prompt_lines = [
        "START OF ASSISTANT INSTRUCTIONS",
        persona_instructions,
        "END OF ASSISTANT INSTRUCTIONS",
        "\n",
        "--- Company FAQ Information ---",
        faq_context, # This contains the formatted Q&A list or an error message
        "--- End of Company FAQ Information ---",
        "\n",
        "--- Conversation History (Oldest to Newest) ---",
    ]

    # Add limited history
    history_to_include = conversation_history[-(MAX_HISTORY_LENGTH * 2):] # Get last N Q&A pairs
    if not history_to_include:
         prompt_lines.append("(No history for this session)")
    else:
        for q_hist, a_hist in history_to_include:
            prompt_lines.append(f"User: {q_hist}")
            prompt_lines.append(f"{ASSISTANT_NAME}: {a_hist}")

    prompt_lines.append("--- End of Conversation History ---")
    prompt_lines.append("\n--- Current User Query ---")
    prompt_lines.append(f"User: {question}")
    prompt_lines.append("--- End of User Query ---")
    prompt_lines.append(f"\n{ASSISTANT_NAME}:") # Prompt the model for its response

    full_prompt = "\n".join(prompt_lines)
    # logger.debug(f"Prompt sent to AI:\n-------\n{full_prompt}\n-------") # Uncomment for deep debugging

    try:
        response = model.generate_content(full_prompt)

        answer = ""
        # Safely extract text, checking for potential blocks
        if not response.candidates:
             logger.warning("AI response has no candidates. Check prompt feedback.")
             # Check for blocking reasons if available
             block_reason = "Unknown"
             try:
                 block_reason = response.prompt_feedback.block_reason.name
             except AttributeError:
                 pass # No feedback available
             answer = f"(My response was blocked due to: {block_reason}. Please rephrase your question or ask about allowed topics.)"
        elif response.parts:
             answer = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'):
             answer = response.text # Fallback

        # Handle cases where generation succeeds but text is empty
        if not answer:
             logger.warning("AI response was generated but contained no text.")
             answer = "(I was unable to generate a text response for that. Please try again.)"

        # Update history list
        conversation_history.append((question, answer))
        # Prune history
        while len(conversation_history) > MAX_HISTORY_LENGTH:
             conversation_history.pop(0) # Remove oldest Q&A pair

        return answer.strip()

    except Exception as e:
        logger.error(f"Error during AI API call or processing: {str(e)}", exc_info=True)
        # Provide a generic system error message
        return "[System Error: An issue occurred while generating the response. Please try again later.]"


# --- Pydantic Models for API Validation ---
class ChatRequest(BaseModel):
    message: str
    reset_context: bool = False

class ChatResponse(BaseModel):
    response: str
    context_cleared: bool = False


# --- API Endpoint ---
@app.post("/api/chat",
          response_model=ChatResponse,
          summary="Process FAQ Chat Message",
          tags=["Chatbot"])
async def api_chat(
    request_data: ChatRequest,
    persona: str = Depends(load_persona_instructions),
    faq_data: str = Depends(load_faq_data) # Inject loaded FAQ data
):
    """
    Receives user message, uses persona and FAQ data to generate response.
    """
    global conversation_history
    context_cleared = False

    if request_data.reset_context:
        conversation_history = []
        logger.info("Conversation history reset via API request.")
        context_cleared = True
        return ChatResponse(
            response="Chat context has been reset.",
            context_cleared=True
        )

    if not request_data.message or not request_data.message.strip():
        logger.warning("Received empty message in API request.")
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    # Generate response using the core logic, passing loaded persona and FAQs
    answer = generate_bot_response(request_data.message, persona, faq_data)
    return ChatResponse(response=answer, context_cleared=context_cleared)


# --- Health Check Endpoint ---
@app.get("/health",
         status_code=200,
         summary="Health Check",
         tags=["System"],
         response_description="Returns the operational status of the API.")
async def health_check():
    """Basic health check including data file accessibility."""
    persona_ok = False
    faq_ok = False
    try:
        # Use the loading functions to check accessibility (uses cache)
        load_persona_instructions()
        persona_ok = True
    except Exception as e:
        logger.warning(f"Health check: Persona file issue - {e}")
    try:
        load_faq_data()
        faq_ok = True
    except Exception as e:
        logger.warning(f"Health check: FAQ file issue - {e}")

    return {
        "status": "ok",
        "persona_file_accessible": persona_ok,
        "faq_data_accessible": faq_ok,
        "current_history_length": len(conversation_history)
        }

# --- Main Execution Guard ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FAQ Chatbot Backend API server...")
    # Use PORT environment variable provided by Render/hosting, default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Use 0.0.0.0 host to be accessible externally
    # Set reload=True only if DEV_MODE env var is set to "true"
    reload_flag = os.environ.get("DEV_MODE", "false").lower() == "true"
    logger.info(f"Running Uvicorn: host=0.0.0.0, port={port}, reload={reload_flag}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload_flag)