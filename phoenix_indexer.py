# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "google-generativeai", # Reverted to original package for ChromaDB compatibility
#   "spacy",
#   "chromadb",
#   "python-dotenv",
#   "tqdm",
# Removed pip dependency as it didn't solve the spacy download issue within uv run
# ]
# ///
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# phoenix_indexer.py - The Firebird's Forge: Indexing and Embedding Memories

import argparse
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai # Reverted import
import spacy
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tqdm import tqdm
# Import types module for error handling if needed
from google.generativeai import types # Reverted types import

# --- Configuration & Constants ---

# Load environment variables (.env file)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

# Reverted: Configure the original library
genai.configure(api_key=GOOGLE_API_KEY)
# client = genai.Client(api_key=GOOGLE_API_KEY) # Removed client initialization

# Model Configuration
EMBEDDING_MODEL = "models/text-embedding-004" # Or your preferred Gemini embedding model
SUMMARY_MODEL_NAME = "gemini-2.0-flash" # Model for generating summaries

# Summary Configuration
MIN_TOKENS_FOR_SUMMARY = 1000 # Minimum conversation length before generating a summary
TARGET_CONV_SUMMARY_TOKENS = 250 # Target token length for conversation summaries
TARGET_INSIGHT_NOTE_TOKENS = 50  # Target token length for each insight note
MAX_INSIGHT_NOTES_PER_CONV = 15  # Maximum number of insight notes to extract per conversation

# Chunking Configuration
# Aim for chunks around this size. Actual size will vary based on sentence boundaries.
# REDUCED from 1000 to 500 to provide a larger safety margin against the 36KB payload limit.
TARGET_CHUNK_TOKENS = 500
# Add this many sentences of overlap between chunks
SENTENCE_OVERLAP = 2

# Database Configuration
CHROMA_DIR = "chroma_memory"
DB_FILE_NAME = "phoenix_memory.db"

# Load spaCy model.
# NOTE: The 'spacy' dependency above installs the library,
# but you still need to download the specific language model separately.
# Run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    # Reverted again: Instruct user to download manually. Automatic download is unreliable in uv run.
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run the following command in your terminal (outside of this script execution):")
    print("  python -m spacy download en_core_web_sm")
    exit(1)

print("🔥 Phoenix Indexer Initialized 🔥")
print(f" - Embedding Model: {EMBEDDING_MODEL}")
print(f" - Target Chunk Tokens: ~{TARGET_CHUNK_TOKENS} (allows 20% overshoot)")
print(f" - Sentence Overlap: {SENTENCE_OVERLAP}")
print(f" - Chroma Database: {CHROMA_DIR}")

# --- Database Setup ---

def setup_database(db_path: str):
    """Initializes the SQLite database and tables."""
    print(f"Setting up SQLite database at: {db_path}")
    
    # Check if we need to create a new database
    is_new_db = not os.path.exists(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Conversations table: Stores metadata about each conversation
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        conv_id TEXT PRIMARY KEY,
        title TEXT,
        create_time REAL, -- Unix timestamp
        model_slug TEXT,
        metadata TEXT -- Store other details as JSON
    );
    """)
    
    conn.commit()
    
    if is_new_db:
        print("SQLite database schema initialized.")
    else:
        print("Connected to existing SQLite database.")
    
    return conn

def setup_chroma_client(chroma_dir: str):
    """Initialize the ChromaDB client and return it."""
    print(f"Setting up ChromaDB client at: {chroma_dir}")
    
    # Initialize the embedding function with Google Generative AI
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name=EMBEDDING_MODEL
    )
    
    # Create a persistent client that will store data on disk
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Create or get the collection
    collection = client.get_or_create_collection(
        name="phoenix_memory",
        embedding_function=google_ef,
        metadata={"description": "Phoenix memory storage for RAG"}
    )
    
    print(f"ChromaDB collection 'phoenix_memory' ready.")
    return client, collection

# --- Conversation Parsing (Adapted from original) ---

class OpenAIParser:
    """Parser for OpenAI conversation exports (conversations.json)."""

    def parse_conversation_data(self, data: Dict) -> Dict | None:
        """Parses a single conversation entry from the conversations.json list."""
        if not isinstance(data, dict) or not data.get('mapping'):
            print(f"Skipping invalid conversation data: {str(data)[:100]}...")
            return None # Skip items that aren't valid conversation dicts

        conv_id = data.get('id', f"gen_{uuid.uuid4()}") # Generate ID if missing
        title = data.get('title', 'Untitled Conversation')
        create_time = data.get('create_time') # Keep as original timestamp (float)
        model_slug = data.get('current_model_slug') or data.get('model_slug') or "unknown"

        # Extract messages in chronological order
        messages = self._extract_thread(data.get('mapping', {}), data.get('current_node'))

        if not messages:
            print(f"Warning: No messages extracted for conversation '{title}' (ID: {conv_id}). Skipping.")
            return None

        # Basic metadata
        metadata = {
            "original_id": data.get("id"),
            "update_time": data.get("update_time"),
        }

        return {
            "conv_id": conv_id,
            "title": title,
            "create_time": float(create_time) if create_time else time.time(),
            "model_slug": model_slug,
            "messages": messages, # List of {"msg_id": str, "role": str, "content": str, "timestamp": float}
            "metadata": json.dumps(metadata)
        }

    def _extract_thread(self, mapping: Dict, current_node_id: str | None) -> list:
        """Extracts the main message thread leading to the current node."""
        if not mapping or not current_node_id or current_node_id not in mapping:
            # Try finding a root if current_node is invalid
            roots = [nid for nid, node in mapping.items() if not node.get('parent')]
            if not roots: return []
            # If multiple roots, maybe pick the one with the latest timestamp? For now, pick first.
            current_node_id = roots[0]
            # We might need a more robust way to find the *actual* current node if `current_node` is bad

        thread = []
        node_id = current_node_id

        while node_id:
            node = mapping.get(node_id)
            if not node: break # Should not happen in valid data

            message_data = node.get('message')
            if message_data and message_data.get("content"): # Only include nodes with actual message content
                role = message_data.get("author", {}).get("role", "system")
                if role not in ["user", "assistant", "system"]:
                    role = "system" # Normalize unknown roles

                content_data = message_data.get("content", {})
                content_type = content_data.get("content_type", "text")
                text_content = ""

                if content_type == "text":
                    parts = content_data.get("parts", [])
                    if parts and isinstance(parts[0], str):
                        text_content = parts[0]
                elif content_type == "code": # Handle code blocks
                     text_content = f"```\n{content_data.get('text', '')}\n```"
                # Add handling for other types like multimodal later if needed
                elif content_type in ["tether_Browse_display", "multimodal_text"]:
                     # Try to extract text - might need refinement based on actual data
                     parts = content_data.get("parts", [])
                     if isinstance(parts, list) and parts:
                         if isinstance(parts[0], str):
                              text_content = parts[0]
                         elif isinstance(parts[0], dict): # Simple handling for potential dict parts
                              text_content = json.dumps(parts[0])
                     elif isinstance(content_data.get("text"), str):
                          text_content = content_data["text"]
                     else:
                          text_content = f"[{content_type} content]"
                else:
                     text_content = f"[{content_type} content omitted]"


                if text_content.strip(): # Only add if there's non-whitespace content
                    msg_time = message_data.get('create_time')
                    thread.append({
                        "msg_id": message_data.get("id", f"msg_{uuid.uuid4()}"),
                        "role": role,
                        "content": text_content,
                        "timestamp": float(msg_time) if msg_time else time.time() # Fallback timestamp
                    })

            node_id = node.get('parent') # Move up the chain

        return thread[::-1] # Reverse to get chronological order

# --- Summary Generation ---

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string using whitespace splitting."""
    return len(text.split())

def generate_summary(prompt: str, model, max_output_tokens: int) -> str:
    """Generate a summary using the provided generative model.
    
    Args:
        prompt: The prompt text to send to the model
        model: The generative model to use
        max_output_tokens: Maximum tokens allowed in the response
        
    Returns:
        The generated summary text or empty string on error
    """
    try:
        # Set up generation config
        # Reverted: Use original GenerationConfig
        generation_config = types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.2,  # Lower temperature for factual summaries
        )
        
        # Generate the summary using the passed model object
        # The 'model' object is passed into this function now
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract and return the text
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

# --- Insight Notes Generation ---

def generate_insight_notes(conversation_text: str, model, max_output_tokens: int) -> List[Tuple[str, List[str]]]:
    """Generate insight notes with tags from a conversation.
    
    Args:
        conversation_text: The full text of the conversation
        model: The generative model to use
        max_output_tokens: Maximum tokens allowed in the response
        
    Returns:
        A list of tuples (insight_text, [tag1, tag2, ...]) or empty list on error
    """
    # Revised prompt to extract insights AND key facts
    prompt = f"""Analyze the following conversation.

Extract important, relatively stable insights about:
1. Recurring user interests, preferences, or demonstrated knowledge/skill levels. Avoid temporary goals unless they represent a significant shift or decision.
2. Key capabilities or limitations demonstrated by the AI assistant during the interaction.
3. Core topics discussed repeatedly, established facts, significant decisions made, or explicit action items agreed upon.

ALSO, extract any specific key facts mentioned, such as:
- User's name (if stated)
- Specific project names, tools, or locations mentioned
- Explicitly stated goals or preferences not covered by general insights

For each insight OR fact:
- Make it a single, concise, standalone statement.
- Start the line with "Insight:" for general insights.
- Start the line with "Fact:" for specific factual details.
- Ensure it's likely to remain relevant beyond the immediate context of this conversation.
- Avoid speculating on short-term intentions. Focus on observed patterns or explicit statements.

Example Format:
Insight: User prefers concise code examples.
Fact: User mentioned their name is Chris.
Insight: AI demonstrated image generation capabilities.
Fact: Project discussed is named 'LyricVideoMaker'.

Provide up to {MAX_INSIGHT_NOTES_PER_CONV} insights/facts, prioritizing the most significant and stable ones.

Conversation:
---
{conversation_text}
---

Insights and Facts:""" # Changed the final label

    try:
        # Set up generation config
        # Reverted: Use original GenerationConfig
        generation_config = types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.2,  # Lower temperature for factual insights
        )
        
        # Generate the insights using the passed model object
        # The 'model' object is passed into this function now
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Parse the response
        response_text = response.text.strip()
        insights = []
        
        # Split by double newlines to separate insights
        insight_blocks = [block.strip() for block in response_text.split("\n\n") if block.strip()]
        
        for block in insight_blocks:
            # Check if block follows our expected format
            insight_line = None
            tags_line = None
            
            lines = block.split("\n")
            # Check if the block starts with "Insight:" or "Fact:"
            if lines and (lines[0].startswith("Insight:") or lines[0].startswith("Fact:")):
                # Extract the text after the prefix
                insight_or_fact_text = lines[0].split(":", 1)[1].strip()
                if insight_or_fact_text:
                    # Add the insight/fact (tags are no longer parsed)
                    insights.append((insight_or_fact_text, [])) # Append with empty list for tags
        
        return insights
    except Exception as e:
        print(f"Error generating insight notes: {e}")
        return []

# --- Text Chunking ---

def chunk_conversation(conversation: Dict) -> list[Dict]:
    """Splits a conversation into overlapping chunks based on sentences."""
    chunks = []
    all_sentences = [] # Store tuples of (sentence_text, msg_id, timestamp)

    # Use spaCy to split messages into sentences, preserving metadata
    for msg in conversation["messages"]:
        if msg["content"] and isinstance(msg["content"], str):
            doc = nlp(msg["content"])
            for sent in doc.sents:
                # Store sentence text along with original message ID and timestamp
                all_sentences.append((sent.text, msg["msg_id"], msg["timestamp"]))

    if not all_sentences:
        return [] # No text content to chunk

    current_chunk_sentences = []
    current_chunk_tokens = 0
    current_message_ids = set()
    start_time = all_sentences[0][2] # Timestamp of the first sentence

    for i, (sent_text, msg_id, timestamp) in enumerate(all_sentences):
        # Rough token estimation for the sentence
        sent_tokens = len(sent_text.split()) # Simple whitespace split for speed

        # If adding this sentence exceeds target size (and chunk not empty), finalize previous chunk
        if current_chunk_sentences and (current_chunk_tokens + sent_tokens > TARGET_CHUNK_TOKENS * 1.2): # Allow 20% overshoot
            chunk_id = f"chunk_{conversation['conv_id']}_{uuid.uuid4()}"
            chunk_text = " ".join(s[0] for s in current_chunk_sentences)
            end_time = current_chunk_sentences[-1][2] # Timestamp of last sentence in chunk

            chunks.append({
                "chunk_id": chunk_id,
                "conv_id": conversation["conv_id"],
                "chunk_text": chunk_text.strip(),
                "start_time": start_time,
                "end_time": end_time,
                "message_ids": json.dumps(sorted(list(current_message_ids))),
                "token_count": current_chunk_tokens # Store the estimated tokens
            })

            # Start new chunk, adding overlap
            # Overlap goes back SENTENCE_OVERLAP sentences, but ensure indices are valid
            overlap_start_index = max(0, len(current_chunk_sentences) - SENTENCE_OVERLAP)
            current_chunk_sentences = current_chunk_sentences[overlap_start_index:]
            current_chunk_tokens = sum(len(s[0].split()) for s in current_chunk_sentences)
            current_message_ids = set(s[1] for s in current_chunk_sentences)
            start_time = current_chunk_sentences[0][2] if current_chunk_sentences else timestamp # Update start time

        # Add current sentence to the chunk
        current_chunk_sentences.append((sent_text, msg_id, timestamp))
        current_chunk_tokens += sent_tokens
        current_message_ids.add(msg_id)
        # Ensure start_time reflects the actual earliest time in the current chunk buffer
        if len(current_chunk_sentences) == 1:
             start_time = timestamp


    # Add the last remaining chunk
    if current_chunk_sentences:
        chunk_id = f"chunk_{conversation['conv_id']}_{uuid.uuid4()}"
        chunk_text = " ".join(s[0] for s in current_chunk_sentences)
        end_time = current_chunk_sentences[-1][2] # Timestamp of last sentence

        chunks.append({
            "chunk_id": chunk_id,
            "conv_id": conversation["conv_id"],
            "chunk_text": chunk_text.strip(),
            "start_time": start_time,
            "end_time": end_time,
            "message_ids": json.dumps(sorted(list(current_message_ids))),
            "token_count": current_chunk_tokens
        })

    return chunks

# --- Indexing Logic ---

def index_conversations(input_path: str, chroma_dir: str, db_path: str, max_convos: int | None = None):
    """Main function to parse, chunk, embed, and store conversations."""
    if not Path(input_path).exists():
        print(f"Error: Input file not found at {input_path}")
        return

    parser = OpenAIParser()
    
    # Reverted: Initialize the generative model for summaries and insights
    print(f"Initializing generative model: {SUMMARY_MODEL_NAME}")
    summary_model = genai.GenerativeModel(SUMMARY_MODEL_NAME)
    
    # Set up SQLite database for conversation metadata
    try:
        conn = setup_database(db_path)
        print(f"SQLite database ready for storing conversation metadata.")
    except Exception as e:
        print(f"ERROR: Failed to set up SQLite database: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print(f"Setting up ChromaDB client for indexing...")
    # Set up ChromaDB client and collection
    try:
        client, collection = setup_chroma_client(chroma_dir)
        print(f"ChromaDB setup successful. Collection name: 'phoenix_memory'")
    except Exception as e:
        print(f"ERROR: Failed to set up ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return
    
    processed_conv_count = 0
    processed_chunk_count = 0
    processed_insight_count = 0
    failed_chunk_count = 0
    failed_conv_count = 0
    conv_metadata = {}  # Store conversation metadata by conv_id

    try:
        print(f"Loading conversations from {input_path}...")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            print(f"Successfully loaded JSON data from {input_path}")
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parsing error in {input_path}: {e}")
            print(f"Check that the file contains valid JSON data")
            return
        except Exception as e:
            print(f"ERROR: Failed to read input file: {e}")
            return

        if not isinstance(all_data, list):
            print(f"ERROR: Expected a list of conversations in {input_path}, found {type(all_data)}")
            return

        print(f"Found {len(all_data)} potential conversation entries.")

        # Apply limit if specified
        data_to_process = all_data[:max_convos] if max_convos else all_data
        
        # First, parse all conversations to extract create_time
        print("Parsing all conversations to prepare for chronological processing...")
        parsed_conversations = []
        parser = OpenAIParser()
        
        for conv_idx, conv_data in enumerate(tqdm(data_to_process, desc="Parsing Conversations")):
            try:
                parsed_conv = parser.parse_conversation_data(conv_data)
                if parsed_conv:  # Skip invalid/empty conversations
                    parsed_conversations.append(parsed_conv)
            except Exception as e:
                print(f"ERROR parsing conversation {conv_idx}: {e}")
                failed_conv_count += 1
        
        # Sort conversations by create_time (oldest first)
        parsed_conversations.sort(key=lambda x: x["create_time"])
        print(f"Sorted {len(parsed_conversations)} valid conversations chronologically (oldest first)")
        
        # Process each conversation in chronological order
        print(f"Processing {len(parsed_conversations)} conversations in chronological order...")
        for conv_idx, parsed_conv in enumerate(tqdm(parsed_conversations, desc="🔥 Indexing Conversations")):
            # Convert timestamp to readable date for display
            create_date = datetime.fromtimestamp(parsed_conv["create_time"]).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[Conv {conv_idx+1}/{len(parsed_conversations)}] Processing conversation from {create_date}")
            print(f"  - Conversation ID: {parsed_conv['conv_id']}, Title: {parsed_conv['title']}")
            print(f"  - Found {len(parsed_conv['messages'])} messages")

            # Store conversation metadata in both memory (for chunk metadata) and SQLite
            conv_id = parsed_conv["conv_id"]
            
            # Store in memory for use with chunks
            conv_metadata[conv_id] = {
                "title": parsed_conv["title"],
                "create_time": parsed_conv["create_time"],
                "model_slug": parsed_conv["model_slug"],
                "metadata": parsed_conv["metadata"]
            }
            
            # Store in SQLite database
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO conversations (conv_id, title, create_time, model_slug, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        conv_id,
                        parsed_conv["title"],
                        parsed_conv["create_time"],
                        parsed_conv["model_slug"],
                        parsed_conv["metadata"]
                    )
                )
                conn.commit()
                print(f"  - Saved conversation metadata to SQLite database")
            except sqlite3.Error as e:
                print(f"ERROR storing conversation metadata in SQLite: {e}")
                conn.rollback()
                failed_conv_count += 1
                # Continue with chunks even if SQLite storage fails

            # --- Generate Conversation Summary ---
            conv_summary_text = None
            
            # Combine all messages into a single text for summarization
            full_conv_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in parsed_conv['messages']])
            conv_tokens = estimate_tokens(full_conv_text)
            
            if conv_tokens > MIN_TOKENS_FOR_SUMMARY:
                print(f"  - Conversation {parsed_conv['conv_id']} exceeds token threshold ({conv_tokens} > {MIN_TOKENS_FOR_SUMMARY}). Generating summary...")
                summary_prompt = f"Summarize the key topics, decisions, and outcomes of the following conversation concisely (target ~{TARGET_CONV_SUMMARY_TOKENS} tokens):\n\n---\n{full_conv_text}\n---\n\nSummary:"
                try:
                    # Reverted: Pass the summary_model object
                    conv_summary_text = generate_summary(summary_prompt, summary_model, int(TARGET_CONV_SUMMARY_TOKENS * 1.2))
                    if conv_summary_text:
                        print(f"  - Generated conversation summary ({estimate_tokens(conv_summary_text)} tokens)")
                    else:
                        print("  - Summary generation failed or returned empty.")
                except Exception as e:
                    print(f"  - ERROR generating conversation summary: {e}")
            else:
                print(f"  - Conversation below token threshold ({conv_tokens} <= {MIN_TOKENS_FOR_SUMMARY}). Skipping summary.")

            # --- Generate Insight Notes ---
            insight_notes = []
            if conv_tokens > MIN_TOKENS_FOR_SUMMARY:
                print(f"  - Generating insight notes for conversation {parsed_conv['conv_id']}...")
                try:
                    # Call the LLM to generate insights with tags
                    max_output_tokens = int(TARGET_INSIGHT_NOTE_TOKENS * MAX_INSIGHT_NOTES_PER_CONV * 2)  # Allow buffer for format
                    # Reverted: Pass the summary_model object
                    insight_notes = generate_insight_notes(full_conv_text, summary_model, max_output_tokens)
                    
                    if insight_notes:
                        print(f"  - Generated {len(insight_notes)} insights/facts") # Indent this line
                    else: # Correct indentation for else
                        print("  - Insight/fact generation failed or returned empty.")
                except Exception as e: # Correct indentation for except
                    print(f"  - ERROR generating insights/facts: {e}")
                    insight_notes = [] # Keep this assignment within except
            else: # Correct indentation for outer else
                print(f"  - Conversation below token threshold. Skipping insight note generation.")
            
            # --- Chunk Conversation ---
            print(f"  - Chunking conversation {conv_id}...")
            try:
                chunks = chunk_conversation(parsed_conv)
                if not chunks:
                    print(f"  - No chunks generated for conversation {conv_id}. Skipping.")
                    continue
                print(f"  - Generated {len(chunks)} chunks")
            except Exception as e:
                print(f"ERROR chunking conversation {conv_id}: {e}")
                continue

            # --- Process Chunks (ChromaDB will handle embedding) ---
            chunk_texts = []
            chunk_ids = []
            chunk_metadatas = []
            
            print(f"  - Preparing chunks for ChromaDB insertion...")
            for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"  Processing Chunks for Conv {processed_conv_count+1}", leave=False)):
                try:
                    # Prepare data for batch insertion
                    chunk_texts.append(chunk["chunk_text"])
                    chunk_ids.append(chunk["chunk_id"])
                    
                    # Convert message_ids from JSON string to list for metadata
                    message_ids = json.loads(chunk["message_ids"])
                    
                    # Create metadata dictionary (will be stored with each chunk)
                    metadata = {
                        "conv_id": chunk["conv_id"],
                        "title": conv_metadata[conv_id]["title"],
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                        "token_count": chunk["token_count"],
                        "message_count": len(message_ids),
                        "model_slug": conv_metadata[conv_id]["model_slug"]
                    }
                    chunk_metadatas.append(metadata)
                except Exception as e:
                    print(f"ERROR preparing chunk {chunk_idx} of conversation {conv_id}: {e}")
                    # We'll continue with other chunks if one fails
            
            if not chunk_texts:
                print(f"  - No valid chunks to add for conversation {conv_id}. Skipping.")
                continue
                
            # Prepare summary for ChromaDB if one was generated
            summary_texts_to_add = []
            summary_ids_to_add = []
            summary_metadatas_to_add = []
            
            if conv_summary_text:
                summary_id = f"summary_{parsed_conv['conv_id']}"
                summary_token_count = estimate_tokens(conv_summary_text)
                summary_metadata = {
                    "doc_type": "conversation_summary",
                    "original_conv_id": parsed_conv['conv_id'],
                    "summary_token_count": summary_token_count,
                    "title": parsed_conv["title"],
                    "create_time": parsed_conv["create_time"],
                    "model_slug": parsed_conv["model_slug"]
                }
                summary_texts_to_add.append(conv_summary_text)
                summary_ids_to_add.append(summary_id)
                summary_metadatas_to_add.append(summary_metadata)
            
            # Prepare insight notes for ChromaDB
            insight_texts_to_add = []
            insight_ids_to_add = []
            insight_metadatas_to_add = []
            
            for i, (insight_text, tags) in enumerate(insight_notes):
                insight_id = f"insight_{parsed_conv['conv_id']}_{uuid.uuid4()}"
                insight_token_count = estimate_tokens(insight_text)
                
                # Create metadata for this insight
                insight_metadata = {
                    "doc_type": "insight_note",
                    # "tags": json.dumps(tags),  # Tags removed
                    "original_conv_id": parsed_conv['conv_id'],
                    "insight_token_count": insight_token_count,
                    "title": parsed_conv["title"],
                    "create_time": parsed_conv["create_time"],
                    "model_slug": parsed_conv["model_slug"]
                }
                
                insight_texts_to_add.append(insight_text)
                insight_ids_to_add.append(insight_id)
                insight_metadatas_to_add.append(insight_metadata)
            
            # Add chunks in batch to ChromaDB
            print(f"  - Adding {len(chunk_texts)} chunks to ChromaDB...")
            try:
                collection.add(
                    documents=chunk_texts,
                    ids=chunk_ids,
                    metadatas=chunk_metadatas
                )
                print(f"  - Successfully added chunks to ChromaDB")
                processed_chunk_count += len(chunks)
            except Exception as e:
                print(f"ERROR adding chunks for conversation {conv_id} to ChromaDB: {e}")
                print(f"  - First few characters of a sample chunk text: '{chunk_texts[0][:100]}...'")
                print(f"  - Sample chunk ID: {chunk_ids[0]}")
                print(f"  - Sample metadata: {chunk_metadatas[0]}")
                failed_chunk_count += len(chunks)
                continue # Skip adding summary/insights if chunks failed
                
            # Add summary to ChromaDB if one was generated
            if summary_texts_to_add:
                print(f"  - Adding conversation summary to ChromaDB...")
                try:
                    collection.add(
                        documents=summary_texts_to_add,
                        ids=summary_ids_to_add,
                        metadatas=summary_metadatas_to_add
                    )
                    print("  - Successfully added summary to ChromaDB")
                except Exception as e:
                    print(f"  - ERROR adding summary for conversation {conv_id} to ChromaDB: {e}")
            
            # Add insight notes/facts to ChromaDB if any were generated
            if insight_texts_to_add:
                print(f"  - Adding {len(insight_texts_to_add)} insights/facts to ChromaDB...")
                try:
                    collection.add(
                        documents=insight_texts_to_add,
                        ids=insight_ids_to_add,
                        metadatas=insight_metadatas_to_add
                    )
                    print(f"  - Successfully added insights/facts to ChromaDB")
                    processed_insight_count += len(insight_texts_to_add)
                    
                    # Print a preview of insights/facts
                    print("\n=== INSIGHTS/FACTS PREVIEW ===")
                    preview_count = min(3, len(insight_texts_to_add))
                    for i in range(preview_count):
                        # We don't store the prefix, so just print the text
                        print(f"- {insight_texts_to_add[i]}")
                        print("---")
                    print("============================\n")
                except Exception as e:
                    print(f"  - ERROR adding insights/facts for conversation {conv_id} to ChromaDB: {e}")

            processed_conv_count += 1
            print(f"  - Successfully processed conversation {conv_id} ({processed_conv_count}/{len(data_to_process)})")

    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_path}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from {input_path}: {e}")
        print("Check that the file contains valid JSON data")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during indexing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the SQLite connection
        if conn:
            conn.close()
            print("SQLite connection closed.")

    print("\n--- 🔥 Indexing Complete 🔥 ---")
    print(f"Successfully processed {processed_conv_count} conversations.")
    print(f"Created and embedded {processed_chunk_count} chunks.")
    print(f"Created and embedded {processed_insight_count} insights/facts.")
    if failed_chunk_count > 0:
        print(f"⚠️ Failed to process {failed_chunk_count} chunks.")
    print(f"Memory stored in: {chroma_dir}")
    print("------------------------------")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix Indexer: Process and embed conversation data.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input OpenAI conversations JSON file (e.g., 'path/to/conversations.json')."
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_DIR,
        help=f"Path to the ChromaDB directory (default: {CHROMA_DIR})."
    )
    parser.add_argument(
        "--db-path",
        default=DB_FILE_NAME,
        help=f"Path to the SQLite database file (default: {DB_FILE_NAME})."
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

    args = parser.parse_args()

    # Handle database recreation
    if args.recreate_db:
        if os.path.exists(args.chroma_dir):
            print(f"⚠️ Deleting existing ChromaDB directory: {args.chroma_dir}")
            import shutil
            shutil.rmtree(args.chroma_dir)
        if os.path.exists(args.db_path):
            print(f"⚠️ Deleting existing SQLite database: {args.db_path}")
            os.remove(args.db_path)

    # Run the indexing process
    index_conversations(args.input, args.chroma_dir, args.db_path, args.max_convos)
