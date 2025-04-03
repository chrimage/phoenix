# Phoenix Memory Project

## Overview

Phoenix Memory is a Python-based system designed to index and interact with conversation histories, specifically targeting OpenAI's `conversations.json` export format. It leverages Retrieval-Augmented Generation (RAG) to provide a chat interface that can recall relevant information from past conversations.

The project uses a unified command-line interface (CLI) powered by Typer, located in `phoenix/cli.py`, which provides commands for indexing conversation data and chatting with the indexed memory.

## Features

*   Parses OpenAI `conversations.json` format.
*   Chunks conversations into manageable, overlapping segments using spaCy for sentence boundary detection.
*   Generates concise summaries for long conversations using a Google Gemini model.
*   Extracts key insights (about the user, AI, or interaction) from conversations.
*   Creates vector embeddings for text chunks, summaries, and insights using Google's `text-embedding-004` model.
*   Stores embeddings and associated text/metadata in a persistent ChromaDB database for efficient semantic search.
*   Stores conversation-level metadata (title, timestamps, etc.) in an SQLite database.
*   Provides a RAG-based chat interface via the `chat` command that:
    *   Retrieves relevant context (chunks, summaries, insights) from ChromaDB based on the user's query.
    *   Constructs a prompt including the retrieved context for a Google Gemini chat model.
    *   Generates conversational responses informed by past interactions.
    *   Saves the current chat turn back into memory for future recall.
    *   Generates and saves insights for the current turn in real-time.
*   Offers an `index` command to process and store conversation data.
*   Configuration managed via environment variables (using `.env` file) and `phoenix/config/settings.py`.

## Setup

Follow these steps to set up your development environment.

1.  **Prerequisites**:
    *   Git
    *   Python 3.11 (The `setup_dev.sh` script includes a check for Fedora; ensure you have Python 3.11 installed on your system).

2.  **Clone the Repository**:
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd phoenix-memory          # Replace phoenix-memory with your directory name
    ```

3.  **Run Setup Script**:
    *   The easiest way to set up is using the provided script. It will:
        *   Create a Python virtual environment (`venv`).
        *   Install required dependencies from `requirements.txt`.
        *   Download the necessary spaCy language model (`en_core_web_sm`).
        *   Create a `.env` file template if one doesn't exist.
    ```bash
    bash setup_dev.sh
    ```

4.  **Configure API Key**:
    *   The setup script creates a `.env` file if it's missing. Edit this file:
    ```dotenv
    # .env
    GOOGLE_API_KEY=your_google_api_key_here
    ```
    *   Replace `your_google_api_key_here` with your actual Google Generative AI API key.
    *   You can obtain a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

5.  **Activate Virtual Environment**:
    *   Before running any project commands, activate the virtual environment created by the script:
    ```bash
    source venv/bin/activate
    ```
    *   You'll need to do this every time you open a new terminal session to work on the project.

**(Alternative Manual Setup)**

If you prefer not to use the script:

```bash
# 1. Ensure Python 3.11 is available
# 2. Create virtual environment
python3.11 -m venv venv
# 3. Activate it
source venv/bin/activate
# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# 5. Download spaCy model
python -m spacy download en_core_web_sm
# 6. Create and populate .env file manually (see step 4 above)
```

## Usage

All commands are run using the `phoenix` CLI entry point after activating the virtual environment (`source venv/bin/activate`).

1.  **Prepare Conversation Data**:
    *   Ensure you have your OpenAI `conversations.json` file ready.

2.  **Run the Indexer (`index` command)**:
    *   Process and store your conversation data.
    ```bash
    python -m phoenix.cli index --input /path/to/your/conversations.json
    ```
    *   **Common Options**:
        *   `--input` / `-i` (Required): Path to the `conversations.json` file.
        *   `--max` / `-m`: Maximum number of conversations to process (e.g., `--max 10`). Defaults to processing all.
        *   `--recreate`: Delete existing ChromaDB and SQLite data before indexing. **Use this if re-indexing from scratch.**
        *   `--chroma-dir`: Override the ChromaDB storage directory (defaults to `data/chroma_memory`).
        *   `--db-path`: Override the SQLite database file path (defaults to `data/phoenix_memory.db`).
    *   **Example with Recreate**:
        ```bash
        python -m phoenix.cli index --input conversations.json --recreate
        ```

3.  **Run the Chat Interface (`chat` command)**:
    *   Start an interactive chat session using the indexed memory.
    ```bash
    python -m phoenix.cli chat
    ```
    *   **Common Options**:
        *   `--chroma-dir`: Specify the ChromaDB directory (must match the one used for indexing).
        *   `--db-path`: Specify the SQLite database path (must match the one used for indexing).
    *   Interact with the chat agent. Type `exit` or `quit` to end the session.

## Configuration

Configuration is primarily managed through environment variables, which are loaded from a `.env` file in the project root by `phoenix/config/settings.py`.

*   **`GOOGLE_API_KEY`**: (Required) Your Google Generative AI API key.
*   **`USER_NAME`**: Your name for personalization (defaults to "the user").
*   **`CHAT_MODEL`**: Gemini model for chat responses (defaults to `gemini-1.5-pro-latest`).
*   **`EMBEDDING_MODEL`**: Embedding model (defaults to `models/text-embedding-004`).
*   **`SUMMARY_MODEL`**: Gemini model for summaries/insights (defaults to `gemini-1.5-flash-latest`).
*   **`CHROMA_DIR`**: Path to ChromaDB storage (defaults to `data/chroma_memory`).
*   **`DB_FILE_NAME`**: Path to SQLite database (defaults to `data/phoenix_memory.db`).
*   **`CHROMA_COLLECTION_NAME`**: Name of the ChromaDB collection (defaults to `phoenix_memory`).
*   **`SPACY_MODEL`**: spaCy language model to use (defaults to `en_core_web_sm`).
*   Other settings related to chunking, retrieval limits, etc., can be found in `phoenix/config/settings.py`.

**Note**: Command-line arguments like `--chroma-dir` and `--db-path` for the `index` and `chat` commands will override the values loaded from the environment or `.env` file.
