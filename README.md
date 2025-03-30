# Phoenix Memory Project

## Overview

Phoenix Memory is a Python-based system designed to index and interact with conversation histories, specifically targeting OpenAI's `conversations.json` export format. It leverages Retrieval-Augmented Generation (RAG) to provide a chat interface that can recall relevant information from past conversations.

The project consists of two main components:

1.  **`phoenix_indexer.py`**: Processes, chunks, summarizes, generates insights, embeds, and stores conversation data into a vector database (ChromaDB) and a metadata database (SQLite).
2.  **`phoenix_chat.py`**: Provides an interactive command-line interface (CLI) to chat with an AI model (Google Gemini) that has access to the indexed conversation memory.

## Features

*   Parses OpenAI `conversations.json` format.
*   Chunks conversations into manageable, overlapping segments using spaCy for sentence boundary detection.
*   Generates concise summaries for long conversations using a Google Gemini model.
*   Extracts key insights (about the user, AI, or interaction) from conversations.
*   Creates vector embeddings for text chunks, summaries, and insights using Google's `text-embedding-004` model.
*   Stores embeddings and associated text/metadata in a persistent ChromaDB database for efficient semantic search.
*   Stores conversation-level metadata (title, timestamps, etc.) in an SQLite database.
*   Provides a RAG-based chat interface (`phoenix_chat.py`) that:
    *   Retrieves relevant context (chunks, summaries, insights) from ChromaDB based on the user's query.
    *   Constructs a prompt including the retrieved context for a Google Gemini chat model.
    *   Generates conversational responses informed by past interactions.
    *   Saves the current chat turn back into memory for future recall.
    *   Generates and saves insights for the current turn in real-time.

## How it Works

1.  **Indexing (`phoenix_indexer.py`)**:
    *   Reads the input `conversations.json` file.
    *   Parses each conversation, extracting messages and metadata.
    *   For sufficiently long conversations, it generates a summary and extracts key insights using a specified Gemini model (`gemini-2.0-flash` by default).
    *   Splits the conversation text into overlapping chunks based on sentence boundaries (using spaCy).
    *   Uses the configured Google embedding model (`models/text-embedding-004`) via ChromaDB's integration to generate vector embeddings for each text chunk, summary, and insight.
    *   Stores the text, embeddings, and metadata (like conversation ID, timestamps, document type) in a ChromaDB collection (`phoenix_memory`).
    *   Stores high-level conversation metadata in an SQLite database (`phoenix_memory.db`).

2.  **Chat (`phoenix_chat.py`)**:
    *   Initializes connections to the ChromaDB and SQLite databases.
    *   Initializes the Google Gemini chat model (`gemini-2.5-pro-exp-03-25` by default).
    *   When the user enters a query:
        *   It queries the ChromaDB collection for text chunks and insights semantically similar to the user's input.
        *   It assembles a context string from the retrieved results, prioritizing summaries and insights while respecting token limits.
        *   It sends the assembled context and the user's query to the Gemini chat model.
        *   It displays the AI's response.
        *   It saves the user query and AI response as a new chunk in ChromaDB.
        *   It generates and saves insights related to this specific turn into ChromaDB.

## Setup

1.  **Prerequisites**:
    *   Python 3.8 or higher
    *   `pip` (Python package installer)

2.  **Clone the Repository (if applicable)**:
    ```bash
    # git clone <repository-url>
    # cd phoenix-memory
    ```
    *(Assuming the code is in a directory named `phoenix-memory`)*

3.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download spaCy Model**:
    ```bash
    python -m spacy download en_core_web_sm
    ```

6.  **Configure API Key**:
    *   Create a file named `.env` in the project's root directory.
    *   Add your Google Generative AI API key to the `.env` file:
        ```
        GOOGLE_API_KEY='YOUR_API_KEY_HERE'
        ```
    *   You can obtain a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

1.  **Prepare Conversation Data**:
    *   Ensure you have your OpenAI `conversations.json` file.

2.  **Run the Indexer**:
    *   Execute the indexer script, providing the path to your conversations file.
    ```bash
    python phoenix_indexer.py --input /path/to/your/conversations.json
    ```
    *   **Options**:
        *   `--chroma-dir`: Specify a directory for ChromaDB data (defaults to `chroma_memory`).
        *   `--db-path`: Specify a path for the SQLite database (defaults to `phoenix_memory.db`).
        *   `--max-convos`: Limit the number of conversations to process (e.g., `--max-convos 10`).
        *   `--recreate-db`: **Important:** Use this flag to delete existing ChromaDB and SQLite data before indexing. This is necessary if you want to re-index from scratch or after code changes affecting data structure (like removing tags).
            ```bash
            python phoenix_indexer.py --input /path/to/your/conversations.json --recreate-db
            ```

3.  **Run the Chat Interface**:
    *   Once indexing is complete, start the chat interface:
    ```bash
    python phoenix_chat.py
    ```
    *   **Options**:
        *   `--chroma-dir`: Specify the ChromaDB directory (must match the one used for indexing).
        *   `--db-path`: Specify the SQLite database path (must match the one used for indexing).
    *   Interact with the chat agent. Type `exit` or `quit` to end the session. Type `clear` to reset the current chat history (without affecting the long-term memory).

## Configuration

Key configuration options are located near the top of `phoenix_indexer.py` and `phoenix_chat.py`:

*   `EMBEDDING_MODEL`: The embedding model used (e.g., `models/text-embedding-004`, ensure consistency between scripts).
*   `SUMMARY_MODEL_NAME`: The Gemini model used for summaries and initial insights in the indexer (e.g., `gemini-2.0-flash`).
*   `CHAT_MODEL`: The Gemini model used for the chat interface (e.g., `gemini-2.5-pro-exp-03-25`).
*   `CHROMA_DIR`, `DB_FILE_NAME`: Database paths.
*   `TOP_K_CHUNKS`, `TOP_K_INSIGHTS`: Number of items to retrieve during chat.
*   `CONTEXT_TOKEN_LIMIT`: Maximum tokens allocated for retrieved context in the chat prompt.
