# phoenix/config/settings.py - Centralized Configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

# --- User Settings ---
USER_NAME = os.getenv("USER_NAME", "the user") # Load user name, default if not set

# --- Model Configuration ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-pro-latest") # Default chat model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004") # Embedding model
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemini-1.5-flash-latest") # Model for summaries/insights

# --- Chat Configuration ---
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.7)) # LLM temperature for chat

# --- Retrieval Configuration ---
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", 5)) # Number of relevant chunks to retrieve
TOP_K_INSIGHTS = int(os.getenv("TOP_K_INSIGHTS", 10)) # Number of relevant insight notes to retrieve
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 7000)) # Max tokens for combined context

# --- Indexing Configuration ---
# Chunking
TARGET_CHUNK_TOKENS = int(os.getenv("TARGET_CHUNK_TOKENS", 500)) # Target size for text chunks
SENTENCE_OVERLAP = int(os.getenv("SENTENCE_OVERLAP", 2)) # Sentence overlap between chunks
# Summarization & Insights
MIN_TOKENS_FOR_SUMMARY = int(os.getenv("MIN_TOKENS_FOR_SUMMARY", 1000)) # Min conversation tokens to trigger summary/insights
TARGET_CONV_SUMMARY_TOKENS = int(os.getenv("TARGET_CONV_SUMMARY_TOKENS", 250)) # Target token length for summaries
TARGET_INSIGHT_NOTE_TOKENS = int(os.getenv("TARGET_INSIGHT_NOTE_TOKENS", 50)) # Target token length for each insight
MAX_INSIGHT_NOTES_PER_CONV = int(os.getenv("MAX_INSIGHT_NOTES_PER_CONV", 15)) # Max insights per conversation

# --- Database Configuration ---
# Ensure data is stored within the 'data' directory
DATA_DIR = "data"
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(DATA_DIR, "chroma_memory")) # Directory for ChromaDB persistent storage
DB_FILE_NAME = os.getenv("DB_FILE_NAME", os.path.join(DATA_DIR, "phoenix_memory.db")) # Filename for SQLite metadata DB
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "phoenix_memory") # Name for the ChromaDB collection

# --- spaCy Model ---
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# --- Print Configuration on Load (Optional) ---
# print("--- Phoenix Configuration Loaded ---")
# print(f"User Name: {USER_NAME}")
# print(f"Chat Model: {CHAT_MODEL}")
# print(f"Embedding Model: {EMBEDDING_MODEL}")
# print(f"Summary Model: {SUMMARY_MODEL}")
# print(f"Chroma Dir: {CHROMA_DIR}")
# print(f"SQLite DB: {DB_FILE_NAME}")
# print("---------------------------------")
