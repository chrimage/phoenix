# phoenix/interfaces/cli/chat_commands.py - CLI logic for the 'chat' command
import argparse
import os
import sys

# Ensure project root is discoverable for imports
project_root_cli = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root_cli not in sys.path:
    sys.path.insert(0, project_root_cli)

# Import necessary components after path adjustment
try:
    from phoenix.config import settings
    from phoenix.adapters.llm.gemini_client import GeminiClient
    from phoenix.adapters.vector_db.chroma_client import ChromaDBClient
    from phoenix.adapters.metadata_db.sqlite_client import SQLiteClient
    from phoenix.services.context_assembly import ContextAssembler
    from phoenix.services.chat_service import ChatService
except ImportError as e:
    print(f"❌ Error importing modules in chat_commands.py: {e}")
    print("Ensure the script is run correctly relative to the project structure.")
    sys.exit(1)

def configure_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the chat command to the argparse parser."""
    # Add arguments if needed, e.g., overriding model, temperature, db paths
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
        "--chat-model",
        default=settings.CHAT_MODEL,
        help=f"Chat model to use (default configured: {settings.CHAT_MODEL})."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings.CHAT_TEMPERATURE,
        help=f"Chat generation temperature (default configured: {settings.CHAT_TEMPERATURE})."
    )
    # Add other potential arguments like --top-k-chunks, --top-k-insights

def run_chat_command(args: argparse.Namespace):
    """Executes the interactive chat loop based on parsed arguments."""
    print("--- Starting Phoenix Chat Interface ---")
    print(f"Using ChromaDB at: {args.chroma_dir}")
    print(f"Using SQLite DB at: {args.db_path}")
    print(f"Using Chat Model: {args.chat_model}") # Note: Model override isn't fully plumbed through yet
    print(f"Using Temperature: {args.temperature}") # Note: Temp override isn't fully plumbed through yet

    # Instantiate adapters and services
    try:
        print("\nInitializing components...")
        # TODO: Plumb through args overrides for model/temp/paths if implemented
        llm_client = GeminiClient() # Uses settings internally for default model
        chroma_client = ChromaDBClient(chroma_dir=args.chroma_dir)
        sqlite_client = SQLiteClient(db_path=args.db_path)
        context_assembler = ContextAssembler()

        chat_service = ChatService(
            llm_client=llm_client,
            chroma_client=chroma_client,
            sqlite_client=sqlite_client,
            context_assembler=context_assembler
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

    # Start the interactive loop
    try:
        chat_service.start_new_chat() # Initialize the session

        print("\n--- 🐦 Phoenix Chat Interface ---")
        print("Enter your message. Type 'exit' or 'quit' to end.")
        print("Type 'clear' to reset conversation history (starts a new session).")
        print("----------------------------------")

        while True:
            try:
                user_input = input(f"{settings.USER_NAME}: ") # Use configured user name
                if user_input.lower() in ["exit", "quit"]:
                    chat_service.end_chat()
                    break
                if user_input.lower() == "clear":
                    chat_service.end_chat()
                    print("\n[Chat history cleared, starting new session...]")
                    chat_service.start_new_chat()
                    print("----------------------------------")
                    continue
                if not user_input.strip():
                    continue

                # Process the message using the chat service
                ai_response = chat_service.process_user_message(user_input)

                # Print the response
                print(f"\nPhoenix: {ai_response}")
                print("-" * 20) # Separator

            except KeyboardInterrupt:
                print("\nExiting...")
                chat_service.end_chat()
                break
            except EOFError: # Handle Ctrl+D
                 print("\nExiting...")
                 chat_service.end_chat()
                 break

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the chat session: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Chat Session Ended ---")

# Note: The main execution logic is handled by interfaces/cli/main.py
# This script only defines the configuration and execution function for the 'chat' command.
