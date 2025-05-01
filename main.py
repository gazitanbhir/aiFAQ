# --- START OF FILE main.py ---

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
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Aura") # Match persona name
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
    title="FAQ Chatbot API - AuraCoreTech",
    description="API endpoint for Aura, the AuraCoreTech FAQ-focused AI assistant.",
    version="1.2.0", # Incremented version
)

# --- CORS Middleware ---
# Adjust origins for your specific frontend deployment
origins = [
    "http://localhost",
    "http://localhost:8080", # Common local dev port for frontend
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "http://0.0.0.0:8080", # Included as requested, though browser usually sends specific host

    # Add your deployed frontend URL(s) here
    # e.g., "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
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

            # Special handling if default content is a file path itself
            if default_content == filepath and not os.path.exists(default_content):
                 logger.warning(f"Default content for {filepath} points to itself and doesn't exist. Creating empty file.")
                 content_to_write = "" # Create empty instead of writing the path
            else:
                 content_to_write = default_content # Assume it's actual content

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content_to_write)
            logger.warning(f"Created default file: {filepath}. Please ensure content is correct.")
        except IOError as e:
            logger.error(f"Failed to create default file at {filepath}: {e}")
            # Decide if this is critical enough to stop the app
            # raise IOError(f"Could not create essential file: {filepath}") from e

# Default content strings (placeholders if files are missing)
default_persona_content = """
Fallback Persona: Basic Assistant
Role: Answer questions based on provided FAQ data.
Tone: Neutral and informative.
Instructions: Stick strictly to the FAQ data. If the answer isn't there, state that clearly.
"""
default_faq_content = """
{
  "title": "Fallback FAQ",
  "instructions": "No specific instructions loaded.",
  "faq_data": {
    "General": [
      ["What is this service?", "This is a fallback FAQ. The main data file was not found or is invalid."],
      ["Is support available?", "Please check the system configuration."]
    ]
  }
}
"""

# Use default *content* if files are missing, not the file path itself
ensure_file_exists(PERSONA_FILE_PATH, default_persona_content)
ensure_file_exists(FAQ_FILE_PATH, default_faq_content)


# --- Gemini AI Configuration ---
generation_config = {
    "temperature": 0.3, # Lower temperature for factual FAQ responses
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 300, # Increased slightly for potentially longer FAQ answers
    "response_mime_type": "text/plain",
}

# Safety settings remain the same
safety_settings = {
    "HATE": "BLOCK_MEDIUM_AND_ABOVE",
    "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
    "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE",
}

try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Efficient model choice
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
                return default_persona_content # Fallback to default content
            logger.info(f"Persona instructions loaded successfully from '{filename}'.")
            return instructions
    except FileNotFoundError:
        logger.error(f"CRITICAL: Persona file not found at {filename}. Using default.")
        # Attempt to create it if ensure_file failed earlier, maybe permissions changed
        ensure_file_exists(filename, default_persona_content)
        return default_persona_content # Return default content
    except Exception as e:
        logger.error(f"Error loading persona from {filename}: {e}", exc_info=True)
        return default_persona_content # Fallback on error

@lru_cache(maxsize=1)
def load_faq_data(filename: str = FAQ_FILE_PATH) -> str:
    """Loads and formats FAQ data from the NEW JSON structure into a string."""
    logger.debug(f"Attempting to load FAQ data from: {filename}")
    faq_context_lines = []
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

            # Optional: Include the main instructions from the JSON if needed
            # json_instructions = data.get("instructions", "")
            # if json_instructions:
            #     faq_context_lines.append(f"FAQ Instructions: {json_instructions}\n")

            faq_data = data.get("faq_data", {}) # Expecting the 'faq_data' object
            if not faq_data:
                 logger.warning(f"No 'faq_data' object found or object is empty in {filename}.")
                 return "(No FAQs loaded - data structure missing or empty)"

            # Iterate through sections and their QA pairs
            for section_title, qa_list in faq_data.items():
                if isinstance(qa_list, list): # Handle top-level sections
                    faq_context_lines.append(f"\n## Section: {section_title}\n")
                    for i, qa_pair in enumerate(qa_list, 1):
                        if isinstance(qa_pair, list) and len(qa_pair) == 2:
                            q, a = qa_pair
                            faq_context_lines.append(f"{i}. Q: {q}\n   A: {a}")
                        else:
                             logger.warning(f"Skipping malformed QA pair in section '{section_title}': {qa_pair}")
                elif isinstance(qa_list, dict): # Handle nested subsections (like Security & Compliance)
                    faq_context_lines.append(f"\n## Section: {section_title}\n")
                    for sub_section_title, sub_qa_list in qa_list.items():
                         if isinstance(sub_qa_list, list):
                              faq_context_lines.append(f"### Subsection: {sub_section_title}\n")
                              for i, qa_pair in enumerate(sub_qa_list, 1):
                                   if isinstance(qa_pair, list) and len(qa_pair) == 2:
                                       q, a = qa_pair
                                       faq_context_lines.append(f"{i}. Q: {q}\n   A: {a}")
                                   else:
                                       logger.warning(f"Skipping malformed QA pair in subsection '{section_title}/{sub_section_title}': {qa_pair}")
                         else:
                              logger.warning(f"Skipping malformed subsection content in '{section_title}': {sub_section_title}")

            if not faq_context_lines:
                 logger.warning(f"FAQ data loaded from '{filename}' but resulted in no formatted content.")
                 return "(No valid FAQs found in loaded data)"

            logger.info(f"FAQ data loaded and formatted successfully from '{filename}'.")
            return "\n".join(faq_context_lines)

    except FileNotFoundError:
        logger.error(f"CRITICAL: FAQ file not found: {filename}. Chatbot will lack context.")
         # Attempt to create it if ensure_file failed earlier
        ensure_file_exists(filename, default_faq_content)
        return "(FAQ file missing - using fallback)"
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in FAQ file: {filename}. Error: {e}. FAQ context unavailable.")
        return "(Error loading FAQs: Invalid JSON)"
    except Exception as e:
        logger.error(f"Error reading or processing FAQ file '{filename}': {e}", exc_info=True)
        return "(Error loading FAQs)"


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
        f"Your name is {ASSISTANT_NAME}.", # Explicitly state name if not in persona text
        persona_instructions,
        "You MUST answer questions based *only* on the information provided in the 'Company FAQ Information' section below.",
        "If the answer is not found in the FAQ information, clearly state that you don't have information on that topic based on the provided context and suggest contacting AuraCoreTech directly.",
        "Do not invent information or use external knowledge.",
        "END OF ASSISTANT INSTRUCTIONS",
        "\n",
        "--- Company FAQ Information ---",
        faq_context, # This now contains the correctly formatted Q&A list or an error message
        "--- End of Company FAQ Information ---",
        "\n",
        "--- Conversation History (Oldest to Newest) ---",
    ]

    # Add limited history
    # Create pairs of (user_message, bot_response) for history context
    history_pairs = []
    temp_history = conversation_history[:] # Work with a copy
    while len(temp_history) > MAX_HISTORY_LENGTH:
        temp_history.pop(0)

    for q_hist, a_hist in temp_history:
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

    full_prompt = "\n".join(prompt_lines)
    # logger.debug(f"Prompt sent to AI:\n-------\n{full_prompt}\n-------") # Uncomment for deep debugging if needed

    try:
        # Use the model variable directly
        response = model.generate_content(full_prompt)

        answer = ""
        # Safely extract text, checking for potential blocks
        if not response.candidates:
             logger.warning("AI response has no candidates. Check safety settings or prompt complexity.")
             # Check for blocking reasons if available
             block_reason = "Unknown"
             try:
                 # Accessing safety feedback correctly
                 if response.prompt_feedback.block_reason:
                      block_reason = response.prompt_feedback.block_reason.name
             except AttributeError:
                 pass # No feedback available or structure differs
             answer = f"(I'm unable to provide a response based on the current safety settings ({block_reason}). Please rephrase your question or ask about topics covered in our FAQ.)"
        elif response.parts:
             answer = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'):
             answer = response.text # Fallback if parts structure isn't used

        # Handle cases where generation succeeds but text is empty
        if not answer.strip(): # Check if the stripped answer is empty
             logger.warning("AI response was generated but contained no meaningful text.")
             answer = "(I could not generate a specific answer for that query based on the provided FAQs. Could you please ask differently?)"

        # Update history list (only store valid Q&A)
        conversation_history.append((question, answer))
        # Prune history (already handled by slicing when building prompt, but good practice here too)
        while len(conversation_history) > MAX_HISTORY_LENGTH:
             conversation_history.pop(0) # Remove oldest Q&A pair

        return answer.strip()

    except Exception as e:
        logger.error(f"Error during AI API call or processing: {str(e)}", exc_info=True)
        # Provide a generic system error message
        return "[System Error: I encountered a problem trying to generate a response. Please try again in a moment.]"


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
          summary="Process FAQ Chat Message with Aura",
          tags=["Chatbot"])
async def api_chat(
    request_data: ChatRequest,
    persona: str = Depends(load_persona_instructions), # Inject loaded persona
    faq_data: str = Depends(load_faq_data) # Inject loaded & formatted FAQ data
):
    """
    Receives user message, uses Aura's persona and AuraCoreTech FAQ data
    to generate a context-aware response.
    """
    global conversation_history
    context_cleared = False

    if request_data.reset_context:
        conversation_history = []
        logger.info("Conversation history reset via API request.")
        context_cleared = True
        # Return immediately after reset
        return ChatResponse(
            response=f"Hello! I'm {ASSISTANT_NAME}. The chat context has been reset. How can I help you based on our FAQs?",
            context_cleared=True
        )

    if not request_data.message or not request_data.message.strip():
        logger.warning("Received empty message in API request.")
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    # Generate response using the core logic, passing loaded persona and FAQs
    logger.info(f"Received message: '{request_data.message}'")
    answer = generate_bot_response(request_data.message, persona, faq_data)
    logger.info(f"Generated response: '{answer[:100]}...'") # Log start of response
    return ChatResponse(response=answer, context_cleared=context_cleared)


# --- Health Check Endpoint ---
@app.get("/health",
         status_code=200,
         summary="Health Check",
         tags=["System"],
         response_description="Returns the operational status of the API.")
async def health_check():
    """Basic health check including data file accessibility and AI model status."""
    persona_ok = False
    faq_ok = False
    model_ok = False
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

    # Simple check if the model object exists
    if 'model' in globals() and model is not None:
         model_ok = True
         model_name = model.model_name
    else:
         model_name = "Not Initialized"


    return {
        "status": "ok",
        "model_status": "configured" if model_ok else "error",
        "model_name": model_name,
        "persona_file_accessible": persona_ok,
        "faq_data_accessible": faq_ok,
        "current_history_length": len(conversation_history)
        }

# --- Main Execution Guard ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AuraCoreTech FAQ Chatbot Backend API server...")
    # Use PORT environment variable provided by Render/hosting, default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Use 0.0.0.0 host to be accessible externally
    # Set reload=True only if DEV_MODE env var is set to "true"
    reload_flag = os.environ.get("DEV_MODE", "false").lower() == "true"
    logger.info(f"Running Uvicorn: host=0.0.0.0, port={port}, reload={reload_flag}")
    # Ensure the app object is passed correctly as a string "module:app_object"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload_flag)

# --- END OF FILE main.py ---