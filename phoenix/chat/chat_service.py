# phoenix/chat/chat_service.py - Orchestrates the Interactive Chat Session
import time
import uuid
from datetime import datetime
import traceback
from typing import Optional, Any, List # Any for chat_session, List for type hint

# Project Modules
from phoenix.config import settings
from phoenix.core.models import Message, ConversationMetadata, InsightNote, ParsedConversation # InsightNote needed for saving
from phoenix.storage.sqlite_store import SqliteStore
from phoenix.storage.chroma_store import ChromaStore
from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter
from phoenix.chat.context_builder import assemble_chat_context, synthesize_user_model
from phoenix.chat.response_generator import generate_chat_response
# Import insight generator for end-of-session processing
from phoenix.indexing.insight_generator import generate_insight_notes
from phoenix.core.utils import estimate_tokens # For insight generation check

class ChatService:
    """
    Service class to manage the interactive chat session.
    """
    def __init__(self,
                 chroma_store: ChromaStore,
                 sqlite_store: SqliteStore,
                 llm_adapter: GoogleGenAIAdapter,
                 context_builder_func, # assemble_chat_context
                 # user_model_synthesizer_func, # synthesize_user_model - might integrate differently
                 response_generator_func, # generate_chat_response
                 insight_generator_func): # generate_insight_notes
        """
        Initializes the ChatService.

        Args:
            chroma_store: Instance of ChromaStore.
            sqlite_store: Instance of SqliteStore.
            llm_adapter: Instance of GoogleGenAIAdapter.
            context_builder_func: Function to assemble chat context.
            response_generator_func: Function to generate chat responses.
            insight_generator_func: Function to generate insights.
        """
        self.chroma_store = chroma_store
        self.sqlite_store = sqlite_store
        self.llm_adapter = llm_adapter
        self.assemble_context = context_builder_func
        # self.synthesize_user_model = user_model_synthesizer_func
        self.generate_response = response_generator_func
        self.generate_insights = insight_generator_func
        self.chat_session: Optional[Any] = None # Will hold the google.generativeai.ChatSession
        self.current_conv_id: Optional[str] = None
        self.current_conv_metadata: Optional[ConversationMetadata] = None
        print("ChatService initialized.")

    def _start_new_session(self):
        """Starts a new chat session and generates a conversation ID."""
        self.current_conv_id = f"conv_{uuid.uuid4()}"
        start_time = time.time()
        title = f"Chat Session {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M')}"
        self.current_conv_metadata = ConversationMetadata(
            conv_id=self.current_conv_id,
            title=title,
            create_time=start_time,
            model_slug=settings.CHAT_MODEL,
            metadata={"source": "phoenix_chat_service"}
        )
        # Start the LLM chat session
        self.chat_session = self.llm_adapter.start_chat(model_name=settings.CHAT_MODEL)
        print(f"\n[New conversation started - ID: {self.current_conv_id[:8]}]")
        # Save initial metadata
        self.sqlite_store.save_conversation_metadata(self.current_conv_metadata)

    def _save_final_insights(self):
        """Generates and saves insights at the end of a conversation."""
        if not self.chat_session or not self.chat_session.history or not self.current_conv_metadata:
            print("[No history found, skipping final insight generation.]")
            return

        print("\n[Generating final insight notes for the conversation...]")

        # Reconstruct ParsedConversation structure needed by generate_insight_notes
        # Note: This is slightly awkward; might refactor insight generator later
        # to accept history directly if possible.
        messages = []
        for msg in self.chat_session.history:
            # Need to handle potential multi-part messages if they occur
            content = msg.parts[0].text if msg.parts else "[Empty Message]"
            messages.append(Message(role=msg.role, content=content)) # Timestamps might be off here

        # Check token count before calling LLM
        full_conv_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        conv_tokens = estimate_tokens(full_conv_text)
        if conv_tokens <= settings.MIN_TOKENS_FOR_SUMMARY:
             print(f"[Conversation too short ({conv_tokens} tokens), skipping final insights.]")
             return

        parsed_conv_for_insights = ParsedConversation(
            metadata=self.current_conv_metadata,
            messages=messages
        )

        # Generate insights
        insight_notes: List[InsightNote] = self.generate_insights(
            parsed_conv=parsed_conv_for_insights,
            llm_adapter=self.llm_adapter
        )

        # Save insights to ChromaDB
        if insight_notes:
            try:
                # Attach conv metadata reference needed by ChromaStore's add_documents
                for note in insight_notes:
                     note.conv_meta_ref = self.current_conv_metadata # Attach ref
                self.chroma_store.add_documents(insight_notes)
                print(f"[Successfully saved {len(insight_notes)} final insights/facts to ChromaDB.]")
            except Exception as e:
                print(f"[Error saving final insights to ChromaDB: {e}]")
        else:
            print("[No final insights generated or saved.]")


    def run_interactive_chat(self):
        """Runs the main interactive chat loop."""
        try:
            # Ensure database connections are ready
            with self.sqlite_store, self.chroma_store:
                self._start_new_session() # Start the first session

                print("\n--- 🐦 Phoenix Chat Interface ---")
                print("Enter your message. Type 'exit' or 'quit' to end.")
                print("Type 'clear' to start a new conversation.")
                print("----------------------------------")

                while True:
                    try:
                        user_input = input(f"{settings.USER_NAME}: ")
                        if user_input.lower() in ["exit", "quit"]:
                            self._save_final_insights() # Generate insights before exiting
                            break
                        if user_input.lower() == "clear":
                            self._save_final_insights() # Generate insights for the ended session
                            self._start_new_session() # Start a fresh one
                            continue
                        if not user_input.strip():
                            continue

                        start_turn_time = time.time()
                        user_message = Message(role="user", content=user_input, timestamp=start_turn_time)

                        print("[Thinking... finding relevant memories...]")

                        # 1. Search for relevant context (chunks and insights)
                        retrieved_chunks = self.chroma_store.search_relevant_chunks(user_input)
                        retrieved_insights = self.chroma_store.search_relevant_insights(user_input)

                        if retrieved_chunks: print(f"[Found {len(retrieved_chunks)} relevant memory chunks]")
                        if retrieved_insights: print(f"[Found {len(retrieved_insights)} relevant insight notes]")

                        # 2. Assemble context string
                        context_str = self.assemble_context(retrieved_chunks, retrieved_insights)

                        # 3. Build the final prompt
                        # (Optional: Add user-specific context synthesis here if desired)
                        # user_model_summary = self.synthesize_user_model(retrieved_insights, self.llm_adapter)
                        prompt_parts = []
                        # if user_model_summary:
                        #     prompt_parts.append(f"--- User Profile Summary ---\n{user_model_summary}")

                        if context_str != "No relevant context found.":
                            prompt_parts.append(f"--- Relevant Context ---\n{context_str}")

                        prompt_parts.append(f"--- Current Query ---\nPlease respond directly to the following query from '{settings.USER_NAME}', using the context above if relevant:\n\n{settings.USER_NAME}: {user_input}")
                        final_prompt = "\n\n".join(prompt_parts)

                        # 4. Generate response
                        ai_response_text = self.generate_response(
                            chat_session=self.chat_session,
                            prompt=final_prompt,
                            llm_adapter=self.llm_adapter
                        )
                        end_turn_time = time.time()
                        ai_message = Message(role="model", content=ai_response_text, timestamp=end_turn_time) # 'model' role used by Gemini API history

                        # 5. Print response
                        print(f"\nPhoenix: {ai_response_text}")
                        print(f"[Response generated in {end_turn_time - start_turn_time:.2f}s]")

                        # 6. Save conversation turn
                        if self.current_conv_metadata:
                            # Save to ChromaDB
                            save_ok_chroma = self.chroma_store.save_conversation_turn_as_chunk(
                                self.current_conv_metadata, user_message, ai_message
                            )
                            # Update metadata timestamp in SQLite (optional, could just use create_time)
                            # self.current_conv_metadata.metadata["last_update_time"] = end_turn_time
                            # save_ok_sqlite = self.sqlite_store.save_conversation_metadata(self.current_conv_metadata)
                            # if not save_ok_chroma or not save_ok_sqlite:
                            #      print("[Warning: Failed to save conversation turn completely.]")
                        else:
                             print("[Warning: No current conversation metadata found, cannot save turn.]")


                    except KeyboardInterrupt:
                        print("\nExiting...")
                        self._save_final_insights()
                        break
                    except ConnectionError as e:
                         print(f"\nDatabase Connection Error: {e}. Please restart the application.")
                         break # Exit loop on connection error
                    except Exception as e:
                        print(f"\nAn unexpected error occurred in the chat loop: {e}")
                        traceback.print_exc()
                        # Decide whether to continue or break on other errors
                        # time.sleep(1) # Avoid rapid error loops

        except ConnectionError as e:
             print(f"Failed to connect to databases: {e}")
        except Exception as e:
            print(f"Failed to initialize ChatService components: {e}")
            traceback.print_exc()

        print("\nChat session ended. Goodbye!")


# Example Usage (Requires setting up real or mock components)
if __name__ == "__main__":
    print("Setting up components for ChatService test...")
    # This requires either real credentials and data or extensive mocking

    # Use mocks from previous tests for structure demonstration
    from phoenix.storage.sqlite_store import SqliteStore as MockSqliteStore # Reuse mock if simple
    from phoenix.storage.chroma_store import ChromaStore as MockChromaStore # Reuse mock
    from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter as MockLLMAdapter # Reuse mock
    from phoenix.chat.context_builder import assemble_chat_context as mock_context_builder
    from phoenix.chat.response_generator import generate_chat_response as mock_response_generator
    from phoenix.indexing.insight_generator import generate_insight_notes as mock_insight_generator

    print("NOTE: ChatService test uses basic mocks. Input 'hello', then 'exit'.")

    try:
        # Instantiate with mocks
        # Ensure mocks have necessary methods if reusing simple ones
        mock_chroma = MockChromaStore()
        mock_sqlite = MockSqliteStore()
        mock_llm = MockLLMAdapter()

        # Add dummy search methods to mocks if not present
        if not hasattr(mock_chroma, 'search_relevant_chunks'):
             mock_chroma.search_relevant_chunks = lambda q: []
        if not hasattr(mock_chroma, 'search_relevant_insights'):
             mock_chroma.search_relevant_insights = lambda q: []
        if not hasattr(mock_chroma, 'save_conversation_turn_as_chunk'):
             mock_chroma.save_conversation_turn_as_chunk = lambda *a: True

        chat_service = ChatService(
            chroma_store=mock_chroma,
            sqlite_store=mock_sqlite,
            llm_adapter=mock_llm,
            context_builder_func=mock_context_builder,
            response_generator_func=mock_response_generator,
            insight_generator_func=mock_insight_generator
        )

        # Run the chat (will require manual input: type 'hello', then 'exit')
        # chat_service.run_interactive_chat()

        print("\nChatService test finished (manual input required).")

    except Exception as e:
        print(f"\nError during ChatService test setup or execution: {e}")
        traceback.print_exc()
