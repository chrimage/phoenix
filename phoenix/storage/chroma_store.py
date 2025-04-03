# phoenix/storage/chroma_store.py - ChromaDB Vector Store Interactions
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Tuple, Union
import uuid
import time
import json

from phoenix.config import settings
from phoenix.core.models import Chunk, InsightNote, Summary, ConversationMetadata, Message
from phoenix.core.utils import estimate_tokens # Needed for saving turns

class ChromaStore:
    """Handles interactions with the ChromaDB vector store."""

    def __init__(self,
                 chroma_dir: str = settings.CHROMA_DIR,
                 collection_name: str = settings.CHROMA_COLLECTION_NAME,
                 embedding_model: str = settings.EMBEDDING_MODEL,
                 api_key: str = settings.GOOGLE_API_KEY):
        """
        Initializes the ChromaStore.

        Args:
            chroma_dir: Path to the ChromaDB persistent storage directory.
            collection_name: Name of the collection within ChromaDB.
            embedding_model: Name of the embedding model to use (from Google GenAI).
            api_key: Google API Key for the embedding function.
        """
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_function = None
        print(f"Initializing ChromaStore: Dir='{self.chroma_dir}', Collection='{self.collection_name}'")

    def connect(self):
        """Connects to ChromaDB and gets/creates the collection."""
        if self.collection:
            return # Already connected

        print(f"Setting up ChromaDB client at: {self.chroma_dir}")
        try:
            # Initialize the embedding function
            self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=self.api_key,
                model_name=self.embedding_model
            )

            # Create a persistent client
            self.client = chromadb.PersistentClient(path=self.chroma_dir)

            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Phoenix memory storage for RAG"}
            )
            print(f"ChromaDB collection '{self.collection_name}' ready.")

        except Exception as e:
            print(f"Error connecting to or setting up ChromaDB: {e}")
            self.client = None
            self.collection = None
            self.embedding_function = None
            raise # Re-raise the exception

    def _ensure_connected(self):
        """Ensures the client is connected before performing operations."""
        if not self.collection:
            print("ChromaDB not connected. Attempting to connect...")
            self.connect()
        if not self.collection:
             raise ConnectionError("Failed to connect to ChromaDB.")

    def add_documents(self, documents: List[Union[Chunk, Summary, InsightNote]], conv_meta: Optional[ConversationMetadata] = None):
        """
        Adds a batch of documents (Chunks, Summaries, or InsightNotes) to ChromaDB.
        Embeddings are generated automatically by ChromaDB based on the collection's function.
        Optionally accepts ConversationMetadata to update Chunk metadata.

        Args:
            documents: A list of Chunk, Summary, or InsightNote objects.
        """
        self._ensure_connected()

        if not documents:
            print("No documents provided to add.")
            return

        texts_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        print(f"Preparing {len(documents)} documents for ChromaDB insertion...")
        for doc in documents:
            try:
                # Ensure metadata is updated before adding
                if hasattr(doc, 'update_metadata') and callable(doc.update_metadata):
                    if isinstance(doc, Chunk):
                        if conv_meta: # Use the passed conv_meta for Chunks
                            doc.update_metadata(conv_meta)
                        else:
                            print(f"Warning: Adding Chunk {doc.doc_id} without ConversationMetadata for metadata update.")
                    else: # For Summary, InsightNote
                        doc.update_metadata()

                texts_to_add.append(doc.text)
                ids_to_add.append(doc.doc_id)
                metadatas_to_add.append(doc.metadata)
            except Exception as e:
                print(f"Error preparing document {getattr(doc, 'doc_id', 'UNKNOWN')} for ChromaDB: {e}")
                # Optionally skip this document or handle error differently

        if not texts_to_add:
            print("No valid documents prepared for addition.")
            return

        print(f"Adding {len(texts_to_add)} documents to ChromaDB collection '{self.collection_name}'...")
        try:
            self.collection.add(
                documents=texts_to_add,
                ids=ids_to_add,
                metadatas=metadatas_to_add
            )
            print(f"Successfully added {len(texts_to_add)} documents to ChromaDB.")
        except Exception as e:
            # Provide more context on failure
            print(f"ERROR adding documents to ChromaDB: {e}")
            print(f"  - Number of documents attempted: {len(texts_to_add)}")
            if texts_to_add:
                print(f"  - First document ID: {ids_to_add[0]}")
                print(f"  - First document metadata: {metadatas_to_add[0]}")
                print(f"  - First document text snippet: '{texts_to_add[0][:100]}...'")
            # Consider logging the failed batch for debugging
            raise # Re-raise to indicate failure

    def search_relevant_chunks(self, query_text: str, top_k: int = settings.TOP_K_CHUNKS) -> List[Dict]:
        """Searches for chunks similar to the query text."""
        self._ensure_connected()

        if not query_text or not query_text.strip():
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where={"doc_type": {"$ne": "insight_note"}} # Exclude insights
            )

            formatted_chunks = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 1.0
                    # Normalize distance to similarity (0-1). Assumes distance is cosine distance (0-2) or L2.
                    # Simple normalization: 1 - distance for cosine (0-1 range expected after embedding func),
                    # or 1 - (distance / max_possible_distance) for others.
                    # ChromaDB distances might vary. Let's use a simple 1 - distance approach for now.
                    similarity = 1.0 - (distance if distance is not None else 1.0)

                    chunk_info = {
                        "chunk_id": doc_id,
                        "chunk_text": results["documents"][0][i] if results["documents"] and results["documents"][0] else "",
                        "similarity": max(0.0, min(1.0, similarity)) # Clamp similarity between 0 and 1
                    }
                    # Add all metadata fields
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    chunk_info.update(metadata)
                    formatted_chunks.append(chunk_info)

                # Sort by similarity (highest first)
                formatted_chunks.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)

            return formatted_chunks

        except Exception as e:
            print(f"Error searching for chunks: {e}")
            return []

    def search_relevant_insights(self, query_text: str, top_k: int = settings.TOP_K_INSIGHTS) -> List[Dict]:
        """Searches for insight notes similar to the query text."""
        self._ensure_connected()

        if not query_text or not query_text.strip():
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where={"doc_type": "insight_note"} # Only insights
            )

            formatted_insights = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 1.0
                    similarity = 1.0 - (distance if distance is not None else 1.0)

                    insight_info = {
                        "insight_id": doc_id,
                        "insight_text": results["documents"][0][i] if results["documents"] and results["documents"][0] else "",
                        "similarity": max(0.0, min(1.0, similarity))
                    }
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    insight_info.update(metadata)
                    formatted_insights.append(insight_info)

                formatted_insights.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)

            return formatted_insights

        except Exception as e:
            print(f"Error searching for insight notes: {e}")
            return []

    def save_conversation_turn_as_chunk(self, conv_meta: ConversationMetadata, user_message: Message, ai_response: Message):
        """
        Saves a single conversation turn (user + AI) as a chunk in ChromaDB.
        This is typically used by the chat interface for ongoing memory.

        Args:
            conv_meta: Metadata of the conversation this turn belongs to.
            user_message: The user's Message object.
            ai_response: The AI's Message object.

        Returns:
            True if successful, False otherwise.
        """
        self._ensure_connected()

        try:
            chunk_id = f"chunk_{conv_meta.conv_id}_{uuid.uuid4()}"
            chunk_text = f"User: {user_message.content}\n\nAI: {ai_response.content}"
            token_count = estimate_tokens(chunk_text)
            timestamp = ai_response.timestamp # Use AI response time as the turn time

            chunk = Chunk(
                doc_id=chunk_id,
                text=chunk_text,
                conv_id=conv_meta.conv_id,
                start_time=user_message.timestamp, # Start time is user message time
                end_time=timestamp, # End time is AI message time
                token_count=token_count,
                message_ids=[user_message.msg_id, ai_response.msg_id]
            )
            # Pass conv_meta directly to add_documents for metadata update
            # chunk.conv_meta_ref = conv_meta # REMOVED

            print(f"[Saving conversation turn to ChromaDB as chunk {chunk_id[:8]}]")
            self.add_documents([chunk], conv_meta=conv_meta) # Pass conv_meta here
            return True

        except Exception as e:
            print(f"Error saving conversation turn as chunk: {e}")
            return False

    def delete_collection(self):
        """Deletes the entire ChromaDB collection. Use with caution!"""
        if not self.client:
            print("Client not initialized, cannot delete collection.")
            return

        print(f"WARNING: Attempting to delete ChromaDB collection: '{self.collection_name}'")
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None # Reset collection object
            print(f"Collection '{self.collection_name}' deleted successfully.")
        except Exception as e:
            # Catch potential exception if collection doesn't exist
            print(f"Error deleting collection '{self.collection_name}': {e}")
            print("It might not have existed.")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # ChromaDB PersistentClient doesn't have an explicit close method
        print("ChromaStore context exited.")
        pass

# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing ChromaStore...")
    # Use a test directory
    test_chroma_dir = "test_chroma_memory"
    test_collection_name = "test_phoenix_collection"

    # Clean up previous test run
    import shutil
    if os.path.exists(test_chroma_dir):
        print(f"Removing previous test directory: {test_chroma_dir}")
        shutil.rmtree(test_chroma_dir)

    try:
        # Use API Key from environment for testing
        if not settings.GOOGLE_API_KEY:
             raise ValueError("GOOGLE_API_KEY needed for testing ChromaStore")

        with ChromaStore(chroma_dir=test_chroma_dir, collection_name=test_collection_name) as store:
            print("Store connected via context manager.")

            # Create dummy data
            conv_meta = ConversationMetadata(conv_id="test_conv_chroma", title="Chroma Test")
            chunk1 = Chunk(doc_id="c1", text="This is the first test chunk.", conv_id="test_conv_chroma", start_time=time.time()-10, end_time=time.time()-5, token_count=6, message_ids=["m1"])
            chunk2 = Chunk(doc_id="c2", text="Another chunk for testing similarity.", conv_id="test_conv_chroma", start_time=time.time()-5, end_time=time.time(), token_count=6, message_ids=["m2"])
            insight1 = InsightNote(doc_id="i1", text="User likes testing.", original_conv_id="test_conv_chroma", title="Chroma Test")
            summary1 = Summary(doc_id="s1", text="This conversation was about testing ChromaDB.", original_conv_id="test_conv_chroma", title="Chroma Test")

            # Add documents
            print("\nAdding documents...")
            store.add_documents([chunk1, chunk2, insight1, summary1])

            # Wait briefly for indexing
            print("Waiting for indexing...")
            time.sleep(5) # Allow time for ChromaDB background processing

            # Search for chunks
            print("\nSearching for chunks related to 'similarity'...")
            results_chunks = store.search_relevant_chunks("similarity", top_k=1)
            print(f"Chunk search results: {results_chunks}")
            if results_chunks:
                assert results_chunks[0]['chunk_id'] == 'c2'
                print("Chunk search verification PASSED")
            else:
                 print("Chunk search verification FAILED (No results)")


            # Search for insights
            print("\nSearching for insights related to 'user preferences'...")
            results_insights = store.search_relevant_insights("user preferences", top_k=1)
            print(f"Insight search results: {results_insights}")
            if results_insights:
                 assert results_insights[0]['insight_id'] == 'i1'
                 print("Insight search verification PASSED")
            else:
                 print("Insight search verification FAILED (No results)")

            # Test saving a turn
            print("\nSaving a conversation turn...")
            user_msg = Message(role="user", content="Tell me about Chroma")
            ai_msg = Message(role="assistant", content="Chroma is a vector database.")
            save_turn_ok = store.save_conversation_turn_as_chunk(conv_meta, user_msg, ai_msg)
            print(f"Save turn successful: {save_turn_ok}")
            assert save_turn_ok

            print("\nWaiting for indexing...")
            time.sleep(5)

            print("\nSearching for chunks related to 'vector database'...")
            results_turn = store.search_relevant_chunks("vector database", top_k=1)
            print(f"Turn search results: {results_turn}")
            if results_turn:
                 assert "User: Tell me about Chroma" in results_turn[0]['chunk_text']
                 print("Turn search verification PASSED")
            else:
                 print("Turn search verification FAILED (No results)")


    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the test directory
        if os.path.exists(test_chroma_dir):
            # shutil.rmtree(test_chroma_dir)
            print(f"Test ChromaDB directory '{test_chroma_dir}' left for inspection.")
        else:
            print(f"Test ChromaDB directory '{test_chroma_dir}' not found for cleanup.")

    print("\nChromaStore testing finished.")
