# phoenix/services/indexing_service.py - Service for indexing conversations
import json
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from tqdm import tqdm

# Import core components, models, and settings
from phoenix.config import settings
from phoenix.core.models import Conversation, Chunk, InsightNote, ConversationSummary
from phoenix.core import text_processing

# Import adapters
from phoenix.adapters.parsers.openai_parser import OpenAIParser
from phoenix.adapters.llm.gemini_client import GeminiClient
from phoenix.adapters.vector_db.chroma_client import ChromaDBClient
from phoenix.adapters.metadata_db.sqlite_client import SQLiteClient

class IndexingService:
    """Orchestrates the conversation indexing process."""

    def __init__(
        self,
        parser: OpenAIParser,
        llm_client: GeminiClient,
        chroma_client: ChromaDBClient,
        sqlite_client: SQLiteClient
    ):
        """
        Initializes the IndexingService with necessary adapters.

        Args:
            parser: An instance of OpenAIParser.
            llm_client: An instance of GeminiClient.
            chroma_client: An instance of ChromaDBClient.
            sqlite_client: An instance of SQLiteClient.
        """
        self.parser = parser
        self.llm_client = llm_client
        self.chroma_client = chroma_client
        self.sqlite_client = sqlite_client
        print("🚀 Indexing Service Initialized 🚀")

    def run_indexing(self, input_path: str, max_convos: Optional[int] = None):
        """
        Executes the full indexing pipeline.

        Args:
            input_path: Path to the input conversations JSON file.
            max_convos: Maximum number of conversations to process (optional).
        """
        print(f"🔥 Starting indexing process for: {input_path}")
        processed_conv_count = 0
        processed_chunk_count = 0
        processed_summary_count = 0
        processed_insight_count = 0
        failed_conv_count = 0
        failed_chunk_adds = 0
        failed_summary_adds = 0
        failed_insight_adds = 0

        try:
            # 1. Load and Parse Data
            print(f"Loading conversations from {input_path}...")
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                print(f"Successfully loaded JSON data.")
            except FileNotFoundError:
                print(f"❌ ERROR: Input file not found at {input_path}")
                return
            except json.JSONDecodeError as e:
                print(f"❌ ERROR: JSON parsing error in {input_path}: {e}")
                return
            except Exception as e:
                print(f"❌ ERROR: Failed to read input file: {e}")
                return

            if not isinstance(all_data, list):
                print(f"❌ ERROR: Expected a list of conversations in {input_path}, found {type(all_data)}")
                return

            print(f"Found {len(all_data)} potential conversation entries.")
            data_to_process = all_data[:max_convos] if max_convos else all_data

            # 2. Parse all conversations first
            print("Parsing all conversations...")
            parsed_conversations: List[Conversation] = []
            for conv_idx, conv_data in enumerate(tqdm(data_to_process, desc="Parsing Conversations")):
                try:
                    parsed_conv = self.parser.parse_conversation_data(conv_data)
                    if parsed_conv:
                        parsed_conversations.append(parsed_conv)
                    else:
                         print(f"⚠️ Warning: Skipping invalid conversation data at index {conv_idx}.")
                         failed_conv_count += 1
                except Exception as e:
                    print(f"❌ ERROR parsing conversation data at index {conv_idx}: {e}")
                    failed_conv_count += 1

            if not parsed_conversations:
                print("❌ No valid conversations found to process. Exiting.")
                return

            # 3. Sort by create_time (oldest first)
            parsed_conversations.sort(key=lambda x: x.create_time)
            print(f"Sorted {len(parsed_conversations)} valid conversations chronologically.")

            # 4. Process each conversation
            print(f"Processing {len(parsed_conversations)} conversations...")
            for conv_idx, conversation in enumerate(tqdm(parsed_conversations, desc="🔥 Indexing Conversations")):
                conv_id = conversation.conv_id
                create_date = datetime.fromtimestamp(conversation.create_time).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[Conv {conv_idx+1}/{len(parsed_conversations)}] Processing '{conversation.title}' from {create_date} (ID: {conv_id[:8]})")
                print(f"  - Found {len(conversation.messages)} messages.")

                # 4a. Save Metadata to SQLite
                try:
                    if not self.sqlite_client.save_conversation_metadata(conversation):
                         print(f"⚠️ Warning: Failed to save metadata for conversation {conv_id} to SQLite.")
                         # Continue processing even if metadata save fails
                except Exception as e:
                    print(f"❌ ERROR saving metadata for {conv_id} to SQLite: {e}")

                # Combine messages for summary/insight generation
                full_conv_text = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation.messages])
                conv_tokens = text_processing.estimate_tokens(full_conv_text)

                # 4b. Generate Summary (if applicable)
                conv_summary: Optional[ConversationSummary] = None
                if conv_tokens > settings.MIN_TOKENS_FOR_SUMMARY:
                    print(f"  - Conversation exceeds token threshold ({conv_tokens} > {settings.MIN_TOKENS_FOR_SUMMARY}). Generating summary...")
                    summary_prompt = f"Summarize the key topics, decisions, and outcomes of the following conversation concisely (target ~{settings.TARGET_CONV_SUMMARY_TOKENS} tokens):\n\n---\n{full_conv_text}\n---\n\nSummary:"
                    try:
                        summary_text = self.llm_client.generate_text(
                            prompt=summary_prompt,
                            model_name=settings.SUMMARY_MODEL_NAME,
                            max_output_tokens=int(settings.TARGET_CONV_SUMMARY_TOKENS * 1.2),
                            temperature=0.2
                        )
                        if summary_text and not summary_text.startswith("[Error"):
                            summary_token_count = text_processing.estimate_tokens(summary_text)
                            print(f"  - Generated conversation summary ({summary_token_count} tokens)")
                            conv_summary = ConversationSummary(
                                summary_id=f"summary_{conv_id}",
                                original_conv_id=conv_id,
                                summary_text=summary_text,
                                summary_token_count=summary_token_count,
                                title=conversation.title,
                                create_time=conversation.create_time,
                                model_slug=conversation.model_slug
                            )
                        else:
                            print(f"  - Summary generation failed or returned empty/error: {summary_text}")
                    except Exception as e:
                        print(f"❌ ERROR generating conversation summary for {conv_id}: {e}")
                else:
                    print(f"  - Conversation below token threshold ({conv_tokens}). Skipping summary.")

                # 4c. Generate Insight Notes (if applicable)
                insight_notes: List[InsightNote] = []
                if conv_tokens > settings.MIN_TOKENS_FOR_SUMMARY: # Use same threshold for insights
                    print(f"  - Generating insight notes for conversation {conv_id}...")
                    # Revised prompt from original script
                    insight_prompt = f"""Analyze the following conversation.

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

Provide up to {settings.MAX_INSIGHT_NOTES_PER_CONV} insights/facts, prioritizing the most significant and stable ones.

Conversation:
---
{full_conv_text}
---

Insights and Facts:"""
                    try:
                        max_output_tokens = int(settings.TARGET_INSIGHT_NOTE_TOKENS * settings.MAX_INSIGHT_NOTES_PER_CONV * 2) # Allow buffer
                        generated_insights_text = self.llm_client.generate_text(
                            prompt=insight_prompt,
                            model_name=settings.SUMMARY_MODEL_NAME, # Use summary model for insights too
                            max_output_tokens=max_output_tokens,
                            temperature=0.2
                        )

                        if generated_insights_text and not generated_insights_text.startswith("[Error"):
                            # Parse the generated text (simple line splitting, check prefix)
                            parsed_texts = []
                            lines = generated_insights_text.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line.startswith("Insight:") or line.startswith("Fact:"):
                                     insight_text = line.split(":", 1)[1].strip()
                                     if insight_text:
                                         parsed_texts.append(insight_text)

                            if parsed_texts:
                                print(f"  - Generated and parsed {len(parsed_texts)} insights/facts.")
                                for text in parsed_texts:
                                    insight_token_count = text_processing.estimate_tokens(text)
                                    insight_notes.append(InsightNote(
                                        insight_id=f"insight_{conv_id}_{uuid.uuid4()}",
                                        original_conv_id=conv_id,
                                        insight_text=text,
                                        insight_token_count=insight_token_count,
                                        title=conversation.title,
                                        create_time=conversation.create_time,
                                        model_slug=conversation.model_slug
                                    ))
                            else:
                                print("  - Insight generation returned text, but no valid Insight:/Fact: lines found.")
                        else:
                            print(f"  - Insight generation failed or returned empty/error: {generated_insights_text}")
                    except Exception as e:
                        print(f"❌ ERROR generating insights for {conv_id}: {e}")
                else:
                     print(f"  - Conversation below token threshold. Skipping insight note generation.")

                # 4d. Chunk Conversation
                print(f"  - Chunking conversation {conv_id}...")
                try:
                    # Use the text_processing module function
                    chunks: List[Chunk] = text_processing.chunk_conversation(conversation)
                    if not chunks:
                        print(f"  - No chunks generated for conversation {conv_id}. Skipping chunk processing.")
                        # Still try to add summary/insights if they exist
                    else:
                        print(f"  - Generated {len(chunks)} chunks.")
                except Exception as e:
                    print(f"❌ ERROR chunking conversation {conv_id}: {e}")
                    failed_conv_count += 1
                    continue # Skip to next conversation if chunking fails critically

                # 4e. Add Chunks to ChromaDB
                if chunks:
                    try:
                        self.chroma_client.add_chunks(chunks)
                        processed_chunk_count += len(chunks)
                    except Exception as e:
                        print(f"❌ ERROR adding chunks for conversation {conv_id} to ChromaDB: {e}")
                        failed_chunk_adds += len(chunks)
                        # Continue to add summary/insights even if chunks fail

                # 4f. Add Summary to ChromaDB
                if conv_summary:
                    try:
                        self.chroma_client.add_summaries([conv_summary])
                        processed_summary_count += 1
                    except Exception as e:
                        print(f"❌ ERROR adding summary for conversation {conv_id} to ChromaDB: {e}")
                        failed_summary_adds += 1

                # 4g. Add Insights to ChromaDB
                if insight_notes:
                    try:
                        self.chroma_client.add_insight_notes(insight_notes)
                        processed_insight_count += len(insight_notes)
                        # Optional: Print preview
                        print("    Insights/Facts Preview:")
                        for note in insight_notes[:3]: print(f"      - {note.insight_text[:80]}...")
                    except Exception as e:
                        print(f"❌ ERROR adding insights for conversation {conv_id} to ChromaDB: {e}")
                        failed_insight_adds += len(insight_notes)

                processed_conv_count += 1
                print(f"  ✅ Successfully processed conversation {conv_id} ({processed_conv_count}/{len(parsed_conversations)})")

        except Exception as e:
            print(f"\n❌ An unexpected error occurred during the indexing process: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close SQLite connection if it's managed by the service (or handled in client's __del__)
            # self.sqlite_client.close() # Assuming SQLiteClient handles its own closure
            print("\n--- 🔥 Indexing Complete 🔥 ---")
            print(f"Successfully processed {processed_conv_count} conversations.")
            print(f"Added {processed_chunk_count} chunks to vector store.")
            print(f"Added {processed_summary_count} summaries to vector store.")
            print(f"Added {processed_insight_count} insights/facts to vector store.")
            if failed_conv_count > 0 or failed_chunk_adds > 0 or failed_summary_adds > 0 or failed_insight_adds > 0:
                print("--- Issues Encountered ---")
                if failed_conv_count > 0: print(f"⚠️ Failed to parse/process {failed_conv_count} conversations.")
                if failed_chunk_adds > 0: print(f"⚠️ Failed to add {failed_chunk_adds} chunks to ChromaDB.")
                if failed_summary_adds > 0: print(f"⚠️ Failed to add {failed_summary_adds} summaries to ChromaDB.")
                if failed_insight_adds > 0: print(f"⚠️ Failed to add {failed_insight_adds} insights to ChromaDB.")
            print("------------------------------")

# Example Usage (if run directly, though typically instantiated and run via CLI)
if __name__ == "__main__":
    print("Testing IndexingService setup...")
    # This requires setting up dummy adapters or mocking them for a real test
    # For now, just demonstrate instantiation possibility

    # Create dummy instances (replace with actual or mocked instances)
    parser = OpenAIParser()
    llm = GeminiClient() # Assumes GOOGLE_API_KEY is set in environment
    chroma = ChromaDBClient() # Assumes data dir exists or is created
    sqlite = SQLiteClient() # Assumes data dir exists or is created

    service = IndexingService(parser, llm, chroma, sqlite)
    print("IndexingService instantiated.")

    # To run indexing, you would call:
    # service.run_indexing("path/to/your/conversations.json")
    print("\nTo run indexing, execute the main CLI script.")
