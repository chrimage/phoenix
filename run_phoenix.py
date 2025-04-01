#!/usr/bin/env python3
# run_phoenix.py - Main executable script for the Phoenix project

import sys
import os

# Ensure the 'phoenix' package directory is in the Python path
# This allows running the script from the project root
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import the main function from the CLI interface
    from phoenix.interfaces.cli.main import main
except ImportError as e:
    print(f"❌ Error importing Phoenix CLI: {e}")
    print("Please ensure you are running this script from the project root directory")
    print("and that the 'phoenix' directory structure is correct.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred during import: {e}")
    sys.exit(1)


if __name__ == "__main__":
    # Execute the main CLI function
    main()
