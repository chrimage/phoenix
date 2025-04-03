# phoenix/indexing/indexer_service.py - Orchestrates the Indexing Pipeline
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from tqdm import tqdm
import traceback

# Project Modules
from phoenix.config import settings
from phoenix.core.models import ParsedConversation, Chunk, Summary, InsightNote, Message, ConversationMetadata
from phoenix.core.utils import estimate_tokens
from phoenix.storage.sqlite_store import SqliteStore
from phoenix.storage.chroma_store import ChromaStore
from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter
from phoenix.indexing.parsers import OpenAIParser
from phoenix.indexing.chunker import chunk_conversation
from phoenix.indexing.summarizer import generate_conversation_summary
from phoenix.indexing.insight_generator import generate_insight_notes

class IndexerService:
    """
    Service class to manage the conversation indexing pipeline.
    """
    def __init__(self,
                 parser: OpenAIParser,
                 chunker_func, # Pass the function itself
                 summarizer_func, # Pass the function
                 insight_generator_func, # Pass the function
                 sqlite_store: SqliteStore,
                 chroma_store: ChromaStore,
                 llm_adapter: GoogleGenAIAdapter):
        """
        Initializes the IndexerService.

        Args:
            parser: An instance of a conversation parser (e.g., OpenAIParser).
            chunker_func: The function used for chunking (e.g., chunk_conversation).
            summarizer_func: The function for generating summaries.
            insight_generator_func: The function for generating insights.
            sqlite_store: An instance of SqliteStore.
            chroma_store: An instance of ChromaStore.
            llm_adapter: An instance of GoogleGenAIAdapter.
        """
        self.parser = parser
        self.chunk_conversation = chunker_func
        self.generate_summary = summarizer_func
        self.generate_insights = insight_generator_func
        self.sqlite_store = sqlite_store
        self.chroma_store = chroma_store
        self.llm_adapter = llm_adapter
        print("IndexerService initialized.")

    def process_input_file(self, input_path: str, max_convos: Optional[int] = None):
        """
        Processes a conversation export file, indexes the data, and stores it.

        Args:
            input_path: Path to the input conversation file (e.g., JSON).
            max_convos: Maximum number of conversations to process. Defaults to None (process all).
        """
        if not Path(input_path).exists():
            print(f"Error: Input file not found at {input_path}")
            return

        processed_conv_count = 0
        processed_chunk_count = 0
        processed_summary_count = 0
        processed_insight_count = 0
        failed_conv_count = 0
        failed_add_count = 0 # Count failures during ChromaDB add operations

        try:
            # Ensure database connections are established
            # Using context managers ensures they are closed properly
            with self.sqlite_store, self.chroma_store:
                print(f"Loading conversations from {input_path}...")
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        all_data = json.load(f)
                    print(f"Successfully loaded JSON data from {input_path}")
                except json.JSONDecodeError as e:
                    print(f"ERROR: JSON parsing error in {input_path}: {e}")
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

                # --- 1. Parse and Sort Conversations ---
                print("Parsing all conversations to prepare for chronological processing...")
                parsed_conversations: List[ParsedConversation] = []
                for conv_idx, conv_data in enumerate(tqdm(data_to_process, desc="Parsing Conversations")):
                    try:
                        parsed_conv = self.parser.parse_conversation_data(conv_data)
                        if parsed_conv:
                            parsed_conversations.append(parsed_conv)
                    except Exception as e:
                        print(f"ERROR parsing conversation entry {conv_idx}: {e}")
                        failed_conv_count += 1

                # Sort conversations by create_time (oldest first)
                parsed_conversations.sort(key=lambda x: x.metadata.create_time)
                print(f"Sorted {len(parsed_conversations)} valid conversations chronologically.")
                total_to_process = len(parsed_conversations)

                # --- 2. Process Each Conversation ---
                print(f"Processing {total_to_process} conversations...")
                for conv_idx, parsed_conv in enumerate(tqdm(parsed_conversations, desc="🔥 Indexing Conversations")):
                    conv_id = parsed_conv.metadata.conv_id
                    create_date = datetime.fromtimestamp(parsed_conv.metadata.create_time).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\n[Conv {conv_idx+1}/{total_to_process}] Processing '{parsed_conv.metadata.title}' from {create_date} (ID: {conv_id[:8]})")

                    documents_to_add = [] # Collect chunks, summary, insights for this convo

                    try:
                        # --- 2a. Save Metadata ---
                        if not self.sqlite_store.save_conversation_metadata(parsed_conv.metadata):
                            print(f"Warning: Failed to save metadata to SQLite for {conv_id}. Continuing...")
                            # Decide if this is critical enough to skip the conversation

                        # --- 2b. Generate Summary ---
                        summary = self.generate_summary(parsed_conv, self.llm_adapter)
                        if summary:
                            documents_to_add.append(summary)
                            processed_summary_count += 1

                        # --- 2c. Generate Insights ---
                        insights = self.generate_insights(parsed_conv, self.llm_adapter)
                        if insights:
                            documents_to_add.extend(insights)
                            processed_insight_count += len(insights)

                        # --- 2d. Chunk Conversation ---
                        print(f"  - Chunking conversation {conv_id[:8]}...")
                        chunks = self.chunk_conversation(parsed_conv)
                        if chunks:
                            print(f"  - Generated {len(chunks)} chunks")
                            # Metadata is updated within chroma_store.add_documents now
                            # for chunk in chunks:
                            #    chunk.conv_meta_ref = parsed_conv.metadata # REMOVED
                            documents_to_add.extend(chunks)
                            processed_chunk_count += len(chunks)
                        else:
                            print(f"  - No chunks generated for conversation {conv_id[:8]}.")


                        # --- 2e. Add Documents to ChromaDB ---
                        if documents_to_add:
                            print(f"  - Adding {len(documents_to_add)} documents (chunks/summary/insights) to ChromaDB...")
                            try:
                                self.chroma_store.add_documents(documents_to_add)
                                print(f"  - Successfully added documents for {conv_id[:8]}.")
                            except Exception as e_add:
                                print(f"ERROR adding documents for conversation {conv_id[:8]} to ChromaDB: {e_add}")
                                failed_add_count += len(documents_to_add)
                                # Continue to next conversation even if adding fails for this one
                        else:
                            print(f"  - No documents (chunks/summary/insights) to add for {conv_id[:8]}.")

                        processed_conv_count += 1

                    except Exception as e_conv:
                        print(f"\nERROR processing conversation {conv_id}: {e_conv}")
                        traceback.print_exc()
                        failed_conv_count += 1
                        # Continue to the next conversation

        except FileNotFoundError:
            print(f"ERROR: Input file not found at {input_path}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Could not decode JSON from {input_path}: {e}")
        except ConnectionError as e:
             print(f"ERROR: Database connection failed: {e}")
        except Exception as e:
            print(f"\nFATAL ERROR: An unexpected error occurred during indexing: {e}")
            traceback.print_exc()
        finally:
            # Connections are handled by context managers now
            print("\n--- 🔥 Indexing Complete 🔥 ---")
            print(f"Successfully processed {processed_conv_count} conversations.")
            print(f" - Generated and stored {processed_chunk_count} chunks.")
            print(f" - Generated and stored {processed_summary_count} summaries.")
            print(f" - Generated and stored {processed_insight_count} insights/facts.")
            if failed_conv_count > 0:
                print(f"⚠️ Failed to fully process {failed_conv_count} conversations.")
            if failed_add_count > 0:
                 print(f"⚠️ Failed to add {failed_add_count} documents to ChromaDB.")
            print(f"Memory stored in: {settings.CHROMA_DIR} (ChromaDB) and {settings.DB_FILE_NAME} (SQLite)")
            print("------------------------------")
