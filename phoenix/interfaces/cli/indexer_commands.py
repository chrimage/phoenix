# phoenix/interfaces/cli/indexer_commands.py - CLI logic for the 'index' command
import argparse
import os
import shutil
import sys

# Ensure project root is discoverable for imports, assuming standard structure
project_root_cli = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root_cli not in sys.path:
    sys.path.insert(0, project_root_cli)

# Import necessary components after path adjustment
try:
    from phoenix.config import settings
    from phoenix.adapters.parsers.openai_parser import OpenAIParser
    from phoenix.adapters.llm.gemini_client import GeminiClient
    from phoenix.adapters.vector_db.chroma_client import ChromaDBClient
    from phoenix.adapters.metadata_db.sqlite_client import SQLiteClient
    from phoenix.services.indexing_service import IndexingService
except ImportError as e:
    print(f"❌ Error importing modules in indexer_commands.py: {e}")
    print("Ensure the script is run correctly relative to the project structure.")
    sys.exit(1)

def configure_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the index command to the argparse parser."""
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input OpenAI conversations JSON file (e.g., 'path/to/conversations.json')."
    )
    parser.add_argument(
        "--chroma-dir",
        default=settings.CHROMA_DIR,
        help=f"Path to the ChromaDB directory (default configured: {settings.CHROMA_DIR})."
    )
    parser.add_argument(
        "--db-path",
        default=settings.DB_FILE_NAME,
        help=f"Path to the SQLite database file (default configured: {settings.DB_FILE_NAME})."
    )
    parser.add_argument(
        "--max-convos",
        type=int,
        default=None,
        help="Maximum number of conversations to process (default: process all)."
    )
    parser.add_argument(
        "--recreate-db",
        action="store_true",
        help="Delete the existing ChromaDB directory and SQLite database before starting, if they exist."
    )
    # Add other potential arguments here (e.g., --log-level)

def run_index_command(args: argparse.Namespace):
    """Executes the indexing process based on parsed arguments."""
    print("--- Running Phoenix Indexer ---")
    print(f"Input file: {args.input}")
    print(f"ChromaDB directory: {args.chroma_dir}")
    print(f"SQLite DB path: {args.db_path}")
    if args.max_convos:
        print(f"Max conversations: {args.max_convos}")
    if args.recreate_db:
        print("Recreate DB flag set: Existing databases will be deleted.")

    # Handle database recreation if requested
    if args.recreate_db:
        if os.path.exists(args.chroma_dir):
            print(f"⚠️ Deleting existing ChromaDB directory: {args.chroma_dir}")
            try:
                shutil.rmtree(args.chroma_dir)
                print("   Deleted.")
            except OSError as e:
                print(f"❌ Error deleting ChromaDB directory: {e}")
                # Decide whether to proceed or exit
                # sys.exit(1)
        if os.path.exists(args.db_path):
            print(f"⚠️ Deleting existing SQLite database: {args.db_path}")
            try:
                os.remove(args.db_path)
                print("   Deleted.")
            except OSError as e:
                print(f"❌ Error deleting SQLite database: {e}")
                # Decide whether to proceed or exit
                # sys.exit(1)

    # Instantiate adapters and services
    # Error handling for adapter initialization is mostly within the adapters themselves
    try:
        print("\nInitializing components...")
        parser = OpenAIParser()
        llm_client = GeminiClient() # Requires GOOGLE_API_KEY in env
        # Pass potentially overridden paths from args to clients
        chroma_client = ChromaDBClient(chroma_dir=args.chroma_dir)
        sqlite_client = SQLiteClient(db_path=args.db_path)

        indexing_service = IndexingService(
            parser=parser,
            llm_client=llm_client,
            chroma_client=chroma_client,
            sqlite_client=sqlite_client
        )
        print("Components initialized successfully.")

    except ValueError as ve: # Catch config errors like missing API key
        print(f"\n❌ Configuration Error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to initialize components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run the indexing process
    try:
        indexing_service.run_indexing(
            input_path=args.input,
            max_convos=args.max_convos
        )
        print("\n--- Indexing Command Finished ---")
    except Exception as e:
        # Catch errors during the indexing run itself
        print(f"\n❌ An error occurred during the indexing run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Note: The main execution logic is handled by interfaces/cli/main.py
# This script only defines the configuration and execution function for the 'index' command.
