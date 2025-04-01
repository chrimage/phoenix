#!/usr/bin/env python3
# phoenix/cli.py - Command Line Interface Entry Point
import typer
from typing_extensions import Annotated
from typing import Optional
import os
import shutil # For recreate_db option

# --- Import Project Modules ---
# It's generally better to import specific classes/functions
# to avoid polluting the namespace and make dependencies clearer.
try:
    from phoenix.config import settings
    from phoenix.storage.sqlite_store import SqliteStore
    from phoenix.storage.chroma_store import ChromaStore
    from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter
    from phoenix.indexing.parsers import OpenAIParser
    from phoenix.indexing.chunker import chunk_conversation
    from phoenix.indexing.summarizer import generate_conversation_summary
    from phoenix.indexing.insight_generator import generate_insight_notes
    from phoenix.indexing.indexer_service import IndexerService
    from phoenix.chat.context_builder import assemble_chat_context
    from phoenix.chat.response_generator import generate_chat_response
    from phoenix.chat.chat_service import ChatService
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project root directory or the 'phoenix' package is installed correctly.")
    # Optionally, add instructions on setting PYTHONPATH if needed
    # print("You might need to set PYTHONPATH=. (run from project root)")
    exit(1)

# --- Typer App Initialization ---
app = typer.Typer(
    name="phoenix",
    help="🐦 Phoenix Memory: Index and chat with your conversation history.",
    add_completion=True,
)

# --- Helper Functions (Component Instantiation) ---
# These functions centralize the creation of service dependencies

def get_llm_adapter() -> GoogleGenAIAdapter:
    """Instantiates and returns the LLM adapter."""
    return GoogleGenAIAdapter()

def get_sqlite_store() -> SqliteStore:
    """Instantiates and returns the SQLite store."""
    return SqliteStore(db_path=settings.DB_FILE_NAME)

def get_chroma_store() -> ChromaStore:
    """Instantiates and returns the ChromaDB store."""
    return ChromaStore(
        chroma_dir=settings.CHROMA_DIR,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        api_key=settings.GOOGLE_API_KEY
    )

def get_indexer_service() -> IndexerService:
    """Instantiates and returns the IndexerService with its dependencies."""
    return IndexerService(
        parser=OpenAIParser(),
        chunker_func=chunk_conversation,
        summarizer_func=generate_conversation_summary,
        insight_generator_func=generate_insight_notes,
        sqlite_store=get_sqlite_store(),
        chroma_store=get_chroma_store(),
        llm_adapter=get_llm_adapter()
    )

def get_chat_service() -> ChatService:
    """Instantiates and returns the ChatService with its dependencies."""
    return ChatService(
        chroma_store=get_chroma_store(),
        sqlite_store=get_sqlite_store(),
        llm_adapter=get_llm_adapter(),
        context_builder_func=assemble_chat_context,
        response_generator_func=generate_chat_response,
        insight_generator_func=generate_insight_notes # Pass insight generator
    )


# --- CLI Commands ---

@app.command()
def index(
    input_file: Annotated[str, typer.Option("--input", "-i", help="Path to the input OpenAI conversations JSON file (e.g., 'conversations.json').", rich_help_panel="Indexing Options")],
    max_convos: Annotated[Optional[int], typer.Option("--max", "-m", help="Maximum number of conversations to process (default: process all).", rich_help_panel="Indexing Options")] = None,
    recreate_db: Annotated[bool, typer.Option("--recreate", help="Delete existing ChromaDB directory and SQLite database before starting.", rich_help_panel="Database Options")] = False,
    chroma_dir_override: Annotated[Optional[str], typer.Option("--chroma-dir", help="Override the ChromaDB directory path from settings.", rich_help_panel="Database Options")] = None,
    db_path_override: Annotated[Optional[str], typer.Option("--db-path", help="Override the SQLite database file path from settings.", rich_help_panel="Database Options")] = None,
):
    """
    🔥 Process and index conversation data into the memory stores.
    """
    print("--- Starting Phoenix Indexing ---")

    # Handle database recreation
    chroma_path = chroma_dir_override or settings.CHROMA_DIR
    sqlite_path = db_path_override or settings.DB_FILE_NAME

    if recreate_db:
        if os.path.exists(chroma_path):
            print(f"⚠️ Deleting existing ChromaDB directory: {chroma_path}")
            try:
                shutil.rmtree(chroma_path)
            except OSError as e:
                print(f"Error deleting directory {chroma_path}: {e}")
                # Decide if this is fatal or just a warning
        if os.path.exists(sqlite_path):
            print(f"⚠️ Deleting existing SQLite database: {sqlite_path}")
            try:
                os.remove(sqlite_path)
            except OSError as e:
                print(f"Error deleting file {sqlite_path}: {e}")

    # Override settings if paths were provided via CLI
    # This is a bit simplistic; a more robust config system might be better
    if chroma_dir_override: settings.CHROMA_DIR = chroma_dir_override
    if db_path_override: settings.DB_FILE_NAME = db_path_override

    try:
        # Instantiate service (which will instantiate dependencies)
        indexer_service = get_indexer_service()
        # Run the indexing process
        indexer_service.process_input_file(input_path=input_file, max_convos=max_convos)
    except Exception as e:
        print(f"\n--- Indexing Failed ---")
        print(f"An error occurred during the indexing process: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

    print("\n--- Phoenix Indexing Finished ---")


@app.command()
def chat(
    chroma_dir_override: Annotated[Optional[str], typer.Option("--chroma-dir", help="Override the ChromaDB directory path from settings.", rich_help_panel="Database Options")] = None,
    db_path_override: Annotated[Optional[str], typer.Option("--db-path", help="Override the SQLite database file path from settings.", rich_help_panel="Database Options")] = None,
):
    """
    💬 Start an interactive chat session using the indexed memory.
    """
    print("--- Starting Phoenix Chat ---")

    # Override settings if paths were provided via CLI
    if chroma_dir_override: settings.CHROMA_DIR = chroma_dir_override
    if db_path_override: settings.DB_FILE_NAME = db_path_override

    try:
        # Instantiate service
        chat_service = get_chat_service()
        # Run the interactive chat loop
        chat_service.run_interactive_chat()
    except ImportError as e:
         # Catch import errors specifically if components failed to load
         print(f"\n--- Chat Failed to Start ---")
         print(f"Import Error: {e}")
         print("Ensure all dependencies are installed and the application structure is correct.")
         raise typer.Exit(code=1)
    except Exception as e:
        print(f"\n--- Chat Failed ---")
        print(f"An error occurred during the chat session: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

    print("\n--- Phoenix Chat Finished ---")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Check if spaCy model is available before running commands that need it
    # (Chunker loads it globally, so error happens early if missing)
    try:
        # Attempt a minimal import that triggers spaCy load in chunker
        from phoenix.indexing.chunker import nlp as spacy_nlp_check
        if not spacy_nlp_check: # Should not happen if load didn't raise error
             raise ImportError("spaCy model check failed.")
    except ImportError as e:
        # Error message printed by chunker.py, just exit here.
        print("Exiting due to missing spaCy model.")
        exit(1)
    except Exception as e:
        print(f"Unexpected error during spaCy model check: {e}")
        exit(1)

    # Run the Typer app
    app()
