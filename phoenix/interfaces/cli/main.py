# phoenix/interfaces/cli/main.py - Main CLI entry point
import argparse
import os
import sys

# Ensure the project root is in the Python path
# This allows importing 'phoenix' modules from the CLI script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import command functions after adjusting path
try:
    from phoenix.interfaces.cli import indexer_commands
    from phoenix.interfaces.cli import chat_commands
    print("CLI command modules imported successfully.")
except ImportError as e:
    print(f"❌ Error importing CLI command modules: {e}")
    print("Please ensure the script is run from the project root or the phoenix package is installed correctly.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred during import: {e}")
    sys.exit(1)


def main():
    """Parses command-line arguments and dispatches to the appropriate command function."""
    parser = argparse.ArgumentParser(
        description="🐦 Phoenix: Your AI assistant with persistent memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Indexer Command ---
    parser_index = subparsers.add_parser(
        "index",
        help="Index conversation data into Phoenix memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add arguments specific to the indexer command
    indexer_commands.configure_parser(parser_index)
    parser_index.set_defaults(func=indexer_commands.run_index_command)

    # --- Chat Command ---
    parser_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session with Phoenix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add arguments specific to the chat command
    chat_commands.configure_parser(parser_chat)
    parser_chat.set_defaults(func=chat_commands.run_chat_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute the selected command function
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            print(f"\n❌ An error occurred while executing command '{args.command}':")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # This should not happen if subparsers are required
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
