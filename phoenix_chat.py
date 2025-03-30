# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "google-generativeai", # Reverted to original package for ChromaDB compatibility
#   "chromadb",
#   "python-dotenv",
#   "tenacity",
# ]
# ///
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# phoenix_chat.py - The Firebird's Dialogue: Chatting with Memory

import argparse
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime

import google.generativeai as genai # Reverted import
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tenacity import (retry, stop_after_attempt, wait_exponential,
                    retry_if_exception_type)
# Import types module for error handling
from google.generativeai import types # Reverted types import

# --- Configuration & Constants ---

# Load environment variables (.env file)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USER_NAME = os.getenv("USER_NAME", "the user") # Load user name, default if not set
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

# Reverted: Configure the original library
genai.configure(api_key=GOOGLE_API_KEY)
# client = genai.Client(api_key=GOOGLE_API_KEY) # Removed client initialization

# Model Configuration
CHAT_MODEL = "gemini-2.5-pro-exp-03-25" # Updated default model per user request
EMBEDDING_MODEL = "models/text-embedding-004" # Should match indexer
CHAT_TEMPERATURE = 0.7 # Define temperature

# Retrieval Configuration
TOP_K_CHUNKS = 5 # Number of relevant chunks to retrieve
TOP_K_INSIGHTS = 10 # Number of relevant insight notes to retrieve
CONTEXT_TOKEN_LIMIT = 7000 # Max tokens for context (retrieved chunks + history) leaving room for response

# Database Configuration
CHROMA_DIR = "chroma_memory" # Should match indexer
DB_FILE_NAME = "phoenix_memory.db" # Should match indexer

print("🐦 Phoenix Chat Initialized 🐦")
print(f" - User Name: {USER_NAME}") # Display configured user name
print(f" - Chat Model: {CHAT_MODEL}")
print(f" - Embedding Model: {EMBEDDING_MODEL}")
print(f" - Max Context Tokens: ~{CONTEXT_TOKEN_LIMIT}")
print(f" - Chunks to Retrieve: {TOP_K_CHUNKS}")
print(f" - Insights to Retrieve: {TOP_K_INSIGHTS}")
print(f" - Insight Notes Integration: Active")

# --- Database Setup ---

def setup_sqlite_database(db_path: str):
    """Connects to the SQLite database for conversation metadata."""
    print(f"Setting up SQLite database at: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Error: SQLite database file not found at {db_path}.")
        print("Please run the phoenix_indexer.py script first.")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        print(f"Connected to SQLite database: {db_path}")
        return conn
    except Exception as e:
        print(f"Error connecting to SQLite database: {e}")
        return None

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
    
    # Get the collection (should already be created by the indexer)
    try:
        collection = client.get_collection(
            name="phoenix_memory",
            embedding_function=google_ef
        )
        print(f"Connected to ChromaDB collection 'phoenix_memory'")
    except Exception as e:
        print(f"Error connecting to ChromaDB collection: {e}")
        print("Creating a new collection...")
        collection = client.create_collection(
            name="phoenix_memory",
            embedding_function=google_ef,
            metadata={"description": "Phoenix memory storage for RAG"}
        )
        print(f"Created new ChromaDB collection 'phoenix_memory'")
    
    return client, collection

def search_relevant_chunks(collection, query_text: str, top_k: int) -> list[dict]:
    """Searches for chunks similar to the query text using ChromaDB."""
    if not query_text or not query_text.strip():
        return []

    try:
        # ChromaDB will handle the embedding internally
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where={"doc_type": {"$ne": "insight_note"}}  # Exclude insight notes - we'll handle them separately
        )
        
        # Format the results to match the expected structure
        formatted_chunks = []
        
        if results["ids"] and results["ids"][0]:  # Check if we got any results
            for i, doc_id in enumerate(results["ids"][0]):
                # Create a chunk_info dict with the same structure as before
                chunk_info = {
                    "chunk_id": doc_id,
                    "chunk_text": results["documents"][0][i],
                    "similarity": 1 - (results["distances"][0][i] if results["distances"][0][i] <= 1 else results["distances"][0][i]/100)
                }
                
                # Add metadata fields
                metadata = results["metadatas"][0][i]
                for key, value in metadata.items():
                    chunk_info[key] = value
                
                formatted_chunks.append(chunk_info)
                
            # Ensure results are sorted by similarity (highest first)
            formatted_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        return formatted_chunks

    except Exception as e:
        print(f"Error searching for chunks: {e}")
        return []

def search_relevant_insights(collection, query_text: str, top_k: int) -> list[dict]:
    """Searches for insight notes similar to the query text using ChromaDB."""
    if not query_text or not query_text.strip():
        return []

    try:
        # ChromaDB will handle the embedding internally
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where={"doc_type": "insight_note"}  # Only get insight notes
        )
        
        # Format the results
        formatted_insights = []
        
        if results["ids"] and results["ids"][0]:  # Check if we got any results
            for i, doc_id in enumerate(results["ids"][0]):
                # Create an insight_info dict
                insight_info = {
                    "insight_id": doc_id,
                    "insight_text": results["documents"][0][i],
                    "similarity": 1 - (results["distances"][0][i] if results["distances"][0][i] <= 1 else results["distances"][0][i]/100)
                }
                
                # Add metadata fields
                metadata = results["metadatas"][0][i]
                for key, value in metadata.items():
                    # Skip the 'tags' key if it exists, otherwise add the key/value
                    if key != 'tags':
                        insight_info[key] = value
                
                formatted_insights.append(insight_info)
                
            # Ensure results are sorted by similarity (highest first)
            formatted_insights.sort(key=lambda x: x['similarity'], reverse=True)
        
        return formatted_insights

    except Exception as e:
        print(f"Error searching for insight notes: {e}")
        return []

# --- Context Assembly ---

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string using whitespace splitting."""
    return len(text.split())

def assemble_chat_context(retrieved_chunks: list[dict], retrieved_insights: list[dict], max_tokens: int) -> str:
    """Formats retrieved chunks and insights into a context string within token limits."""
    total_tokens = 0
    context_parts = []
    max_tokens_per_section = max_tokens // 2  # Split token budget between chunks and insights
    
    # Process insight notes first
    if retrieved_insights:
        insights_context = "--- USER MODEL AND CONVERSATION INSIGHTS ---\n\n"
        insights_tokens = estimate_tokens(insights_context)
        
        # Process insights directly without grouping by tags
        processed_insight_ids = set() # Keep track of processed insights to avoid duplicates if any
        for insight in retrieved_insights:
            if insight['insight_id'] in processed_insight_ids:
                continue

            insight_text = insight['insight_text']
            insight_token_count = estimate_tokens(insight_text)
            
            if insights_tokens + insight_token_count <= max_tokens_per_section:
                # Add insight without tags
                insights_context += f"• {insight_text}\n\n"
                insights_tokens += insight_token_count
                processed_insight_ids.add(insight['insight_id'])
            else:
                break # Stop adding insights if token limit is reached
        
        if insights_tokens > estimate_tokens("--- USER MODEL AND CONVERSATION INSIGHTS ---\n\n"): # Check if any insights were actually added
            context_parts.append(insights_context)
            total_tokens += insights_tokens
    
    # Process regular chunks and summaries
    if retrieved_chunks:
        chunks_context = "--- RELEVANT PREVIOUS CONVERSATION DETAILS ---\n\n"
        chunks_tokens = estimate_tokens(chunks_context)
        
        # Sort chunks - prioritize summaries first, then by similarity
        summaries = []
        regular_chunks = []
        
        for chunk in retrieved_chunks:
            # Check if this is a summary or regular chunk
            if chunk.get('doc_type') == 'conversation_summary':
                summaries.append(chunk)
            else:
                regular_chunks.append(chunk)
        
        # Process summaries first
        for summary in summaries:
            chunk_text = summary['chunk_text']
            chunk_tokens = summary.get('summary_token_count') or estimate_tokens(chunk_text)
            
            # Check if adding this summary exceeds the limit
            if chunks_tokens + chunk_tokens <= max_tokens_per_section:
                chunks_context += f"[Summary of Conversation {summary.get('original_conv_id', 'unknown')[:8]} - {summary.get('title', 'No title')}]\n{chunk_text}\n\n"
                chunks_tokens += chunk_tokens
        
        # Process regular chunks
        for chunk in regular_chunks:
            chunk_text = chunk['chunk_text']
            chunk_tokens = chunk.get('token_count') or estimate_tokens(chunk_text)

            # Check if adding this chunk exceeds the limit
            if chunks_tokens + chunk_tokens <= max_tokens_per_section:
                chunks_context += f"----\n[From Conversation {chunk['conv_id'][:8]} around {datetime.fromtimestamp(chunk['start_time']).strftime('%Y-%m-%d')}]\n{chunk_text}\n\n"
                chunks_tokens += chunk_tokens
            else:
                # Try adding a truncated version if there's significant space left
                if max_tokens_per_section - chunks_tokens > 50:  # Need at least 50 tokens space
                    allowed_chars = int((max_tokens_per_section - chunks_tokens) * 3.5)  # Estimate chars from tokens
                    truncated_text = chunk_text[:allowed_chars] + "..."
                    chunks_context += f"----\n[From Conversation {chunk['conv_id'][:8]} around {datetime.fromtimestamp(chunk['start_time']).strftime('%Y-%m-%d')}]\n{truncated_text}\n\n"
                    chunks_tokens += estimate_tokens(truncated_text)  # Recalculate token count
                break  # Stop adding chunks
        
        if chunks_tokens > 0:
            context_parts.append(chunks_context)
            total_tokens += chunks_tokens
    
    # Combine all parts into a single context string
    full_context = "\n".join(context_parts)
    
    return full_context.strip() if total_tokens > 0 else "No relevant context found."

def synthesize_user_model(retrieved_insights: list[dict], model) -> str:
    """Generate a synthesized user model from insight notes."""
    if not retrieved_insights:
        return ""
    
    # Use all retrieved insights for user model synthesis (no tag filtering)
    if not retrieved_insights:
        return ""
    
    user_insights = retrieved_insights # Use all insights
    
    # Create prompt for synthesis
    prompt = "Based on the following insights about the user, create a brief summary of the user's profile including interests, preferences, knowledge levels, and goals.\n\nInsights:\n"
    
    for insight in user_insights:
        prompt += f"- {insight['insight_text']}\n"
    
    prompt += "\nSynthesized User Profile:"
    
    # Call the LLM for synthesis
    try:
        # Reverted: Use original GenerationConfig and model.generate_content
        generation_config = types.GenerationConfig(
            max_output_tokens=300,
            temperature=0.2
        )
        # The 'model' object is passed into this function now
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text.strip()
    except (types.ResponseBlockedError, types.StopCandidateError, types.InvalidArgumentError) as e:
        print(f"Gemini API Error during user model synthesis: {e}")
        return f"User profile synthesis unavailable due to API error: {type(e).__name__}"
    except Exception as e:
        print(f"Unexpected error generating user model synthesis: {e}")
        return "User profile synthesis unavailable due to unexpected error."

# --- Main Chat Logic ---

def save_conversation_metadata(conn: sqlite3.Connection, conv_id: str, title: str, timestamp: float):
    """Saves or updates conversation metadata in SQLite database."""
    if not conn:
        return False
        
    try:
        cursor = conn.cursor()
        
        # Basic metadata
        metadata = {
            "source": "phoenix_chat.py",
            "create_time": timestamp
        }
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO conversations (conv_id, title, create_time, model_slug, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conv_id,
                title,
                timestamp,
                CHAT_MODEL,
                json.dumps(metadata)
            )
        )
        conn.commit()
        print(f"[Conversation metadata saved to SQLite database]")
        return True
    except sqlite3.Error as e:
        print(f"Error saving conversation metadata: {e}")
        if conn:
            conn.rollback()
        return False

def save_conversation_turn(collection, conn: sqlite3.Connection, conv_id: str, title: str, user_message: str, ai_response: str, timestamp: float):
    """Saves a conversation turn (user + AI message) as a chunk in ChromaDB."""
    try:
        # Create a unique ID for this chunk
        chunk_id = f"chunk_{conv_id}_{uuid.uuid4()}"
        
        # Combine user message and AI response
        chunk_text = f"User: {user_message}\n\nAI: {ai_response}"
        
        # Generate message IDs for user and AI messages
        user_msg_id = f"msg_{uuid.uuid4()}"
        ai_msg_id = f"msg_{uuid.uuid4()}"
        
        # Estimate token count (simple approximation)
        token_count = len(chunk_text.split())
        
        # Create metadata for this conversation turn
        metadata = {
            "conv_id": conv_id,
            "start_time": timestamp,
            "end_time": timestamp,  # Same timestamp for start/end since it's a single turn
            "token_count": token_count,
            "message_count": 2,  # User + AI
            "model_slug": CHAT_MODEL,
            "title": title
        }
        
        print("[Saving conversation to memory...]")
        
        # Add to ChromaDB collection (embeddings generated automatically)
        collection.add(
            documents=[chunk_text],
            ids=[chunk_id],
            metadatas=[metadata]
        )
        
        # Also update metadata in SQLite if connection is available
        if conn:
            save_conversation_metadata(conn, conv_id, title, timestamp)
        
        print(f"[Conversation turn saved as chunk {chunk_id[:8]}]")
        return True
    except Exception as e:
        print(f"Error saving conversation turn: {e}")
        return False

def save_insight_notes(collection, conv_id: str, chat_history: list, timestamp: float, model, title: str):
    """Generate and save insight notes from the entire conversation."""
    try:
        # Format the chat history into a conversation text
        conversation_text = ""
        for msg in chat_history:
            role = "User" if msg.role == "user" else "AI"
            conversation_text += f"{role}: {msg.parts[0].text}\n\n"
        
        # Skip if conversation is too short
        if len(conversation_text.split()) < 20:  # Arbitrary threshold
            return False
        
        print("[Generating insight notes for current conversation...]")
        
        # Prompt designed to extract insights AND key facts from the entire conversation
        prompt = f"""Analyze the following complete conversation.

Extract 3-5 important insights about:
1. The user (interests, goals, preferences, knowledge levels)
2. The AI assistant (capabilities demonstrated, limitations, style)
3. The interaction (topics discussed, decisions made, action items)

ALSO, extract any specific key facts mentioned, such as:
- User's name (if stated)
- Specific project names, tools, or locations mentioned
- Explicitly stated goals or preferences not covered by general insights

For each insight OR fact:
- Make it a single, concise, standalone statement.
- Start the line with "Insight:" for general insights.
- Start the line with "Fact:" for specific factual details.

Example Format:
Insight: User prefers concise code examples.
Fact: User mentioned their name is Chris.
Insight: AI demonstrated image generation capabilities.
Fact: Project discussed is named 'LyricVideoMaker'.

Complete Conversation:
---
{conversation_text}
---

Insights and Facts:""" # Changed the final label

        # Generate the insights
        # Reverted: Use original GenerationConfig and model.generate_content
        generation_config = types.GenerationConfig(
            max_output_tokens=300,
            temperature=0.2
        )
        
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
            
        if not insights:
            print("  - No insights or facts generated")
            return False
            
        print(f"  - Generated {len(insights)} insights/facts")
            
        # Prepare insight notes/facts for ChromaDB
        insight_texts_to_add = []
        insight_ids_to_add = []
        insight_metadatas_to_add = []
        
        for i, (insight_text, tags) in enumerate(insights):
            insight_id = f"insight_{conv_id}_{uuid.uuid4()}"
            insight_token_count = estimate_tokens(insight_text)
            
            # Create metadata for this insight
            insight_metadata = {
                "doc_type": "insight_note",
                # "tags": json.dumps(tags),  # Tags removed
                "original_conv_id": conv_id,
                "insight_token_count": insight_token_count,
                "title": title,
                "create_time": timestamp,
                "model_slug": CHAT_MODEL
            }
            
            insight_texts_to_add.append(insight_text)
            insight_ids_to_add.append(insight_id)
            insight_metadatas_to_add.append(insight_metadata)
        
        # Add insight notes to ChromaDB
        print(f"  - Adding {len(insight_texts_to_add)} insights/facts to ChromaDB...")
        
        collection.add(
            documents=insight_texts_to_add,
            ids=insight_ids_to_add,
            metadatas=insight_metadatas_to_add
        )
        
        print(f"  - Successfully added insights/facts to ChromaDB")
        
        # Print a preview (distinguish between Insight and Fact if possible, though they are stored the same way)
        print("\n=== INSIGHTS/FACTS PREVIEW ===")
        preview_count = min(3, len(insight_texts_to_add))
        for i in range(preview_count):
            # We don't store the prefix, so just print the text
            print(f"- {insight_texts_to_add[i]}") 
            print("---")
        print("============================\n")

        return True
    except (types.ResponseBlockedError, types.StopCandidateError, types.InvalidArgumentError) as e:
        print(f"Gemini API Error during insight generation: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error generating insight notes: {e}")
        return False

def run_chat_interface(chroma_dir: str, db_path: str):
    """Runs the main interactive chat loop."""
    # Set up SQLite database for conversation metadata
    conn = setup_sqlite_database(db_path)
    
    # Set up ChromaDB client and collection
    client, collection = setup_chroma_client(chroma_dir)
    
    # Reverted: Initialize the Generative Model for chat
    chat_model = genai.GenerativeModel(CHAT_MODEL)
    # Define generation config (without system instruction)
    chat_config = types.GenerationConfig(
        temperature=CHAT_TEMPERATURE
    )

    # Start a persistent chat session
    # History will be managed internally by the chat_session object
    chat_session = chat_model.start_chat(
        history=[]
    )
    # print("Chat session created.") # Removed extra print

    # Generate a unique conversation ID
    conv_id = str(uuid.uuid4())
    
    print("\n--- 🐦 Phoenix Chat Interface ---")
    print("Enter your message. Type 'exit' or 'quit' to end.")
    print("Type 'clear' to reset conversation history.")
    print(f"[New conversation started with ID: {conv_id[:8]}]")
    print("----------------------------------")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                # Generate insights at the end of the session
                if chat_session.history:
                    print("[Generating insight notes for the entire conversation...]")
                    save_insight_notes(
                        collection,
                        conv_id,
                        chat_session.history,
                        time.time(),
                        chat_model,
                        title
                    )
                break
            if user_input.lower() == "clear":
                # Reset the chat session
                chat_session = chat_model.start_chat(
                    history=[]
                )
                print("\n[Chat history cleared]\n")
                continue
            if not user_input.strip():
                continue

            start_time = time.time()
            title = f"Chat Session {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M')}"

            print("[Thinking... finding relevant memories...]")

            # 1a. Search for insights specifically about the user (if name is known)
            user_specific_context = ""
            if USER_NAME != "the user":
                print(f"[Searching for insights about user: {USER_NAME}]")
                user_insights = search_relevant_insights(collection, USER_NAME, 5) # Get top 5 insights about the user
                if user_insights:
                    print(f"[Found {len(user_insights)} insights specifically about {USER_NAME}]")
                    user_specific_context = f"--- KEY INFORMATION ABOUT THE USER ({USER_NAME}) ---\n"
                    for insight in user_insights:
                        user_specific_context += f"- {insight['insight_text']}\n"
                    user_specific_context += "---\n\n"
                else:
                    print(f"[No specific insights found for user: {USER_NAME}]")

            # 1b. Search for relevant chunks based on the current user message
            retrieved_chunks = search_relevant_chunks(collection, user_input, TOP_K_CHUNKS)
            
            if retrieved_chunks:
                print(f"[Found {len(retrieved_chunks)} relevant memory chunks]")
            else:
                print("[No highly relevant memories found]")
                
            # 2. Search for insight notes
            retrieved_insights = search_relevant_insights(collection, user_input, TOP_K_INSIGHTS)
            if retrieved_insights:
                print(f"[Found {len(retrieved_insights)} relevant insight notes]")
            else:
                print("[No relevant insight notes found]")

            # 3. Assemble context from retrieved chunks and insights based on the *current message*
            message_context_str = assemble_chat_context(retrieved_chunks, retrieved_insights, CONTEXT_TOKEN_LIMIT)

            # 4. Build the final prompt with clear sections and instructions
            prompt_parts = []
            if user_specific_context:
                prompt_parts.append(f"--- Background Information about the User ({USER_NAME}) ---\n{user_specific_context.replace('--- KEY INFORMATION ABOUT THE USER ('+USER_NAME+') ---', '').replace('---', '').strip()}") # Add user context if available

            if message_context_str != "No relevant context found.":
                 prompt_parts.append(f"--- Relevant Conversation History & Insights ---\n{message_context_str.strip()}") # Add message context if available

            # Add the core instruction and the user's query
            prompt_parts.append(f"--- Current User Query ---\nPlease respond directly to the following query from the user, using the background information and conversation history above for context if relevant:\n\nUser: {user_input}")

            final_prompt = "\n\n".join(prompt_parts) # Join sections with double newlines

            # 5. Generate response using the chat session
            print("[Generating response...]")
            # The chat_session object now manages history internally.
            # We send the combined context + user query as the prompt.
            try:
                time.sleep(1.1) # Keep basic rate limit delay
                # Reverted: Pass generation_config to send_message
                response = chat_session.send_message(final_prompt, generation_config=chat_config)
                ai_response = response.text
            except (types.ResponseBlockedError, types.StopCandidateError, types.InvalidArgumentError) as e:
                print(f"Gemini API Error during chat response generation: {e}")
                ai_response = f"[Error generating response due to API issue: {type(e).__name__}]"
            except Exception as e:
                print(f"Unexpected error generating chat response: {e}")
                ai_response = "[Unexpected error generating response]" # Placeholder error message

            end_time = time.time()

            # 5. Print response
            # History is managed by chat_session, no need to manually append here
            print(f"\nPhoenix: {ai_response}")
            print(f"[Response generated in {end_time - start_time:.2f}s]")

            # 6. Save this conversation turn to ChromaDB and SQLite
            # Note: We still save the turn for long-term memory/search,
            # even though chat_session handles short-term history for the API call.
            save_result = save_conversation_turn(
                collection, 
                conn,
                conv_id,
                title,
                user_input,
                ai_response,
                start_time
            )
            
            # Insight notes generation moved to end of session

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()

    print("Chat session ended. Goodbye!")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix Chat: Interact with an AI agent with memory.")
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
    # Add other arguments if needed (e.g., --model, --top-k)

    args = parser.parse_args()

    run_chat_interface(args.chroma_dir, args.db_path)
