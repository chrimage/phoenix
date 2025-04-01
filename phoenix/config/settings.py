# phoenix/config/settings.py - Configuration for the Phoenix Project
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in the project root
# Assumes the script is run from the project root or the .env file is discoverable
project_root = Path(__file__).parent.parent.parent # Adjust if necessary
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- API Keys & User Info ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # In a real app, might raise an error or log a critical warning
    print("Warning: GOOGLE_API_KEY not found in environment variables or .env file.")
    # raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

USER_NAME = os.getenv("USER_NAME", "the user") # Load user name, default if not set

# --- Model Configuration ---
EMBEDDING_MODEL = "models/text-embedding-004" # Or your preferred Gemini embedding model
SUMMARY_MODEL_NAME = "gemini-2.0-flash" # Model for generating summaries
CHAT_MODEL = "gemini-2.5-pro-exp-03-25" # Updated default model per user request
CHAT_TEMPERATURE = 0.7 # Define temperature

# --- Summary & Insight Configuration ---
MIN_TOKENS_FOR_SUMMARY = 1000 # Minimum conversation length before generating a summary
TARGET_CONV_SUMMARY_TOKENS = 250 # Target token length for conversation summaries
TARGET_INSIGHT_NOTE_TOKENS = 50  # Target token length for each insight note
MAX_INSIGHT_NOTES_PER_CONV = 15  # Maximum number of insight notes to extract per conversation

# --- Chunking Configuration ---
TARGET_CHUNK_TOKENS = 500 # Aim for chunks around this size
SENTENCE_OVERLAP = 2 # Add this many sentences of overlap between chunks

# --- Database Configuration ---
# Default paths relative to the project root
DATA_DIR = project_root / "data"
CHROMA_DIR = str(DATA_DIR / "chroma_memory") # ChromaDB needs a string path
DB_FILE_NAME = str(DATA_DIR / "phoenix_memory.db") # SQLite needs a string path

# --- Retrieval Configuration ---
TOP_K_CHUNKS = 5 # Number of relevant chunks to retrieve
TOP_K_INSIGHTS = 10 # Number of relevant insight notes to retrieve
CONTEXT_TOKEN_LIMIT = 7000 # Max tokens for context (retrieved chunks + history)

# --- spaCy Configuration ---
SPACY_MODEL = "en_core_web_sm"

# --- Validation (Optional but recommended) ---
# You could add checks here to ensure critical paths exist or models are valid formats

print("⚙️ Phoenix Configuration Loaded ⚙️")
print(f" - User Name: {USER_NAME}")
print(f" - Embedding Model: {EMBEDDING_MODEL}")
print(f" - Chat Model: {CHAT_MODEL}")
print(f" - Chroma Path: {CHROMA_DIR}")
print(f" - SQLite Path: {DB_FILE_NAME}")
