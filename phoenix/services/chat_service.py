# phoenix/services/chat_service.py - Service for handling chat interactions
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import core components, models, and settings
from phoenix.config import settings
from phoenix.core.models import Conversation, Message, Chunk, InsightNote # Import necessary models
from phoenix.core import text_processing # For token estimation if needed

# Import adapters and other services
from phoenix.adapters.llm.gemini_client import GeminiClient, genai # Import genai for ChatSession type hint
from phoenix.adapters.vector_db.chroma_client import ChromaDBClient
from phoenix.adapters.metadata_db.sqlite_client import SQLiteClient
from phoenix.services.context_assembly import ContextAssembler

class ChatService:
    """Orchestrates the chat interaction process."""

    def __init__(
        self,
        llm_client: GeminiClient,
        chroma_client: ChromaDBClient,
        sqlite_client: SQLiteClient,
        context_assembler: ContextAssembler
    ):
        """
        Initializes the ChatService with necessary adapters and services.

        Args:
            llm_client: An instance of GeminiClient.
            chroma_client: An instance of ChromaDBClient.
            sqlite_client: An instance of SQLiteClient.
            context_assembler: An instance of ContextAssembler.
        """
        self.llm_client = llm_client
        self.chroma_client = chroma_client
        self.sqlite_client = sqlite_client
        self.context_assembler = context_assembler
        self._current_chat_session: Optional[genai.ChatSession] = None
        self._current_conv_id: Optional[str] = None
        self._current_conv_title: Optional[str] = None
        print("💬 Chat Service Initialized 💬")

    def start_new_chat(self, initial_history: Optional[List[Dict[str, Any]]] = None):
        """Starts a new chat session, generating a new conversation ID."""
        self._current_conv_id = str(uuid.uuid4())
        start_time = time.time()
        self._current_conv_title = f"Chat Session {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M')}"
        print(f"🚀 Starting new chat session (ID: {self._current_conv_id[:8]})")
        # Start LLM chat session
        self._current_chat_session = self.llm_client.start_chat_session(history=initial_history)
        # Save initial metadata (optional, could also save on first message)
        initial_convo = Conversation(
            conv_id=self._current_conv_id,
            title=self._current_conv_title,
            create_time=start_time,
            model_slug=settings.CHAT_MODEL,
            metadata={"source": "phoenix_chat_service"}
        )
        self.sqlite_client.save_conversation_metadata(initial_convo)

    def process_user_message(self, user_input: str) -> str:
        """
        Processes a single user message, retrieves context, generates a response,
        and saves the interaction.

        Args:
            user_input: The text message from the user.

        Returns:
            The AI's response text.
        """
        if not self._current_chat_session or not self._current_conv_id or not self._current_conv_title:
            # Start a new chat if one isn't active
            print("⚠️ No active chat session found. Starting a new one.")
            self.start_new_chat()
            # Need to ensure _current_chat_session is not None after start_new_chat
            if not self._current_chat_session:
                 return "[Error: Failed to initialize chat session]"


        start_time = time.time()
        print("[Thinking... finding relevant memories...]")

        # 1. Retrieve relevant context from ChromaDB
        try:
            # Query for chunks/summaries and insights separately
            retrieved_chunks = self.chroma_client.query_chunks_and_summaries(
                query_text=user_input,
                top_k=settings.TOP_K_CHUNKS
            )
            retrieved_insights = self.chroma_client.query_insights(
                query_text=user_input,
                top_k=settings.TOP_K_INSIGHTS
            )
            print(f"[Retrieved {len(retrieved_chunks)} chunks/summaries and {len(retrieved_insights)} insights]")
        except Exception as e:
            print(f"❌ Error retrieving context from ChromaDB: {e}")
            retrieved_chunks = []
            retrieved_insights = []

        # 2. Assemble context string
        context_str = self.context_assembler.assemble_chat_context(
            retrieved_chunks,
            retrieved_insights,
            settings.CONTEXT_TOKEN_LIMIT
        )

        # 3. Build the final prompt
        # Include user-specific info if USER_NAME is set (could be enhanced)
        user_specific_context = ""
        if settings.USER_NAME != "the user":
            # Simple placeholder - could involve a separate insight query for the user
            user_specific_context = f"--- Background Information about the User ({settings.USER_NAME}) ---\n(Note: Treat this user as {settings.USER_NAME})\n---\n\n"

        prompt_parts = []
        if user_specific_context:
            prompt_parts.append(user_specific_context)
        if context_str != "No relevant context found.":
            prompt_parts.append(f"--- Relevant Conversation History & Insights ---\n{context_str}\n---\n")

        prompt_parts.append(f"--- Current User Query ---\nPlease respond directly to the following query from {settings.USER_NAME}:\n\nUser: {user_input}")
        final_prompt = "\n".join(prompt_parts) # Use single newline between sections for final prompt

        # 4. Generate AI response via LLM Client
        print("[Generating response...]")
        try:
            ai_response = self.llm_client.send_chat_message(
                chat_session=self._current_chat_session,
                prompt=final_prompt,
                temperature=settings.CHAT_TEMPERATURE
            )
        except Exception as e:
            print(f"❌ Error generating AI response: {e}")
            ai_response = "[Error generating response]"

        end_time = time.time()
        print(f"[Response generated in {end_time - start_time:.2f}s]")

        # 5. Save conversation turn (user message + AI response)
        # Create a simple Chunk representation for saving this turn
        turn_chunk_id = f"chunk_{self._current_conv_id}_{uuid.uuid4()}"
        turn_text = f"User: {user_input}\n\nAI: {ai_response}"
        turn_token_count = text_processing.estimate_tokens(turn_text)

        turn_chunk = Chunk(
            chunk_id=turn_chunk_id,
            conv_id=self._current_conv_id,
            chunk_text=turn_text,
            start_time=start_time, # Use start time of processing this turn
            end_time=end_time,     # Use end time of processing this turn
            message_ids=[f"msg_{uuid.uuid4()}", f"msg_{uuid.uuid4()}"], # Generate dummy IDs for saving
            token_count=turn_token_count,
            title=self._current_conv_title,
            model_slug=settings.CHAT_MODEL
        )

        try:
            print("[Saving conversation turn to memory...]")
            self.chroma_client.add_chunks([turn_chunk])
            # Update conversation metadata in SQLite (e.g., last updated time, though not implemented here)
            # Could re-save the Conversation object with an updated timestamp if needed
            # self.sqlite_client.save_conversation_metadata(...)
            print(f"[Conversation turn saved as chunk {turn_chunk_id[:8]}]")
        except Exception as e:
            print(f"❌ Error saving conversation turn chunk: {e}")

        # Note: Insight generation for the *current* conversation is complex to do mid-stream.
        # It's often better done periodically or at the end of a session.
        # The `save_insight_notes` logic from the original chat script could be adapted
        # into a separate method here or in the IndexingService, run on demand.

        return ai_response

    def end_chat(self):
        """Cleans up the current chat session."""
        # Optional: Trigger end-of-session insight generation here if desired
        print(f"Ending chat session (ID: {self._current_conv_id[:8] if self._current_conv_id else 'N/A'})")
        self._current_chat_session = None
        self._current_conv_id = None
        self._current_conv_title = None

# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("Testing ChatService setup...")
    # Requires setting up dummy/mocked adapters

    # Create dummy instances (replace with actual or mocked instances)
    try:
        llm = GeminiClient() # Assumes GOOGLE_API_KEY is set
        chroma = ChromaDBClient() # Assumes data dir exists
        sqlite = SQLiteClient() # Assumes data dir exists
        assembler = ContextAssembler()

        service = ChatService(llm, chroma, sqlite, assembler)
        print("ChatService instantiated.")

        # Example interaction (would run in a loop in the CLI)
        service.start_new_chat()
        response1 = service.process_user_message("What is the capital of France?")
        print(f"\nPhoenix: {response1}")

        response2 = service.process_user_message("What did I just ask you about?")
        print(f"\nPhoenix: {response2}")

        service.end_chat()

    except ValueError as ve:
         print(f"Configuration Error: {ve}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")
        import traceback
        traceback.print_exc()
