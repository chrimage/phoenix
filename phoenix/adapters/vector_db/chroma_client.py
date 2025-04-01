# phoenix/adapters/vector_db/chroma_client.py - Adapter for ChromaDB interactions
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple

from phoenix.config import settings
# Import core models to potentially return structured data from queries
from phoenix.core.models import Chunk, InsightNote, ConversationSummary

# Define the collection name as a constant
COLLECTION_NAME = "phoenix_memory"

class ChromaDBClient:
    """Wraps ChromaDB interactions for storing and retrieving embeddings."""

    def __init__(self, chroma_dir: str = settings.CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        """
        Initializes the ChromaDB client, embedding function, and collection.

        Args:
            chroma_dir: Path to the directory for persistent storage.
            collection_name: Name of the collection to use.
        """
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_function = None

        self._initialize_client()
        self._initialize_embedding_function()
        self._initialize_collection()

    def _initialize_client(self):
        """Initializes the persistent ChromaDB client."""
        try:
            print(f"Initializing ChromaDB client at: {self.chroma_dir}")
            self._client = chromadb.PersistentClient(path=self.chroma_dir)
            print("ChromaDB client initialized.")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB client: {e}")
            raise # Propagate the error

    def _initialize_embedding_function(self):
        """Initializes the Google Generative AI embedding function."""
        if not settings.GOOGLE_API_KEY:
            raise ValueError("ChromaDBClient requires GOOGLE_API_KEY for embedding function.")
        try:
            print(f"Initializing Google Generative AI Embedding Function (Model: {settings.EMBEDDING_MODEL})")
            self._embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=settings.GOOGLE_API_KEY,
                model_name=settings.EMBEDDING_MODEL
            )
            print("Embedding function initialized.")
        except Exception as e:
            print(f"❌ Error initializing embedding function: {e}")
            raise

    def _initialize_collection(self):
        """Gets or creates the ChromaDB collection."""
        if not self._client or not self._embedding_function:
            raise RuntimeError("Client and embedding function must be initialized before the collection.")
        try:
            print(f"Getting or creating ChromaDB collection: '{self.collection_name}'")
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={"description": "Phoenix memory storage for RAG"} # Optional metadata
            )
            print(f"ChromaDB collection '{self.collection_name}' ready.")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB collection: {e}")
            raise

    def add_documents(self, documents: List[str], ids: List[str], metadatas: List[Dict[str, Any]]):
        """
        Adds multiple documents to the collection. ChromaDB handles embedding.

        Args:
            documents: A list of text documents to add.
            ids: A list of unique IDs corresponding to the documents.
            metadatas: A list of metadata dictionaries corresponding to the documents.
        """
        if not self._collection:
            raise RuntimeError("Collection is not initialized.")
        if not (len(documents) == len(ids) == len(metadatas)):
            raise ValueError("Length of documents, ids, and metadatas must be the same.")
        if not documents:
            print("⚠️ Warning: No documents provided to add.")
            return

        try:
            print(f"Adding {len(documents)} documents to collection '{self.collection_name}'...")
            # ChromaDB automatically handles embedding via the collection's embedding function
            self._collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Successfully added {len(documents)} documents.")
        except Exception as e:
            print(f"❌ Error adding documents to ChromaDB: {e}")
            # Consider logging details of the failed batch for debugging
            # print(f"   - First ID: {ids[0] if ids else 'N/A'}")
            # print(f"   - First Metadata: {metadatas[0] if metadatas else 'N/A'}")
            # print(f"   - First Document Snippet: {documents[0][:100] if documents else 'N/A'}...")
            raise # Re-raise the error for higher-level handling

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Optional[List[List[Any]]]]:
        """
        Queries the collection for documents similar to the query texts.

        Args:
            query_texts: A list of query strings.
            n_results: The number of results to return for each query.
            where_filter: Optional dictionary for metadata filtering (e.g., {"doc_type": "insight_note"}).
            include: List of fields to include in the results.

        Returns:
            A dictionary containing the query results (ids, documents, metadatas, distances).
            Returns None if the query fails.
        """
        if not self._collection:
            raise RuntimeError("Collection is not initialized.")
        if not query_texts:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]} # Return empty structure

        try:
            print(f"Querying collection '{self.collection_name}' with {len(query_texts)} text(s)...")
            query_args = {
                "query_texts": query_texts,
                "n_results": n_results,
                "include": include
            }
            if where_filter:
                print(f"Applying where filter: {where_filter}")
                query_args["where"] = where_filter
            else:
                print("No where filter applied.")

            results = self._collection.query(**query_args)
            print(f"Query successful. Found results for {len(results.get('ids', []))} queries.")
            return results
        except Exception as e:
            print(f"❌ Error querying ChromaDB: {e}")
            # Return an empty-like structure consistent with ChromaDB's output on error/no results
            return {"ids": None, "documents": None, "metadatas": None, "distances": None}

    # --- Convenience Methods for Specific Document Types ---

    def add_chunks(self, chunks: List[Chunk]):
        """Adds Chunk objects to the collection."""
        docs = [c.chunk_text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [{
            "conv_id": c.conv_id,
            "title": c.title,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "token_count": c.token_count,
            "message_count": len(c.message_ids), # Calculate here
            "model_slug": c.model_slug,
            "doc_type": "conversation_chunk" # Add type identifier
        } for c in chunks]
        self.add_documents(docs, ids, metadatas)

    def add_insight_notes(self, insights: List[InsightNote]):
        """Adds InsightNote objects to the collection."""
        docs = [i.insight_text for i in insights]
        ids = [i.insight_id for i in insights]
        metadatas = [{
            "original_conv_id": i.original_conv_id,
            "insight_token_count": i.insight_token_count,
            "title": i.title,
            "create_time": i.create_time,
            "model_slug": i.model_slug,
            "doc_type": i.doc_type # Should be "insight_note"
        } for i in insights]
        self.add_documents(docs, ids, metadatas)

    def add_summaries(self, summaries: List[ConversationSummary]):
        """Adds ConversationSummary objects to the collection."""
        docs = [s.summary_text for s in summaries]
        ids = [s.summary_id for s in summaries]
        metadatas = [{
            "original_conv_id": s.original_conv_id,
            "summary_token_count": s.summary_token_count,
            "title": s.title,
            "create_time": s.create_time,
            "model_slug": s.model_slug,
            "doc_type": s.doc_type # Should be "conversation_summary"
        } for s in summaries]
        self.add_documents(docs, ids, metadatas)

    def query_chunks_and_summaries(self, query_text: str, top_k: int = settings.TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        """Queries for relevant chunks and summaries using database-level filtering."""
        where_filter = {"doc_type": {"$in": ["conversation_chunk", "conversation_summary"]}}
        results = self.query(
            query_texts=[query_text],
            n_results=top_k, # Request only the number needed
            where_filter=where_filter
        )
        # Formatting now happens directly on the filtered results from the DB
        return self._format_query_results(results)

    def query_insights(self, query_text: str, top_k: int = settings.TOP_K_INSIGHTS) -> List[Dict[str, Any]]:
        """Queries specifically for relevant insight notes using database-level filtering."""
        where_filter = {"doc_type": "insight_note"}
        results = self.query(
            query_texts=[query_text],
            n_results=top_k, # Request only the number needed
            where_filter=where_filter
        )
        # Formatting now happens directly on the filtered results from the DB
        return self._format_query_results(results)

    def _format_query_results(self, results: Dict[str, Optional[List[List[Any]]]]) -> List[Dict[str, Any]]:
        """Helper to format raw ChromaDB query results into a list of dictionaries."""
        formatted_list = []
        if not results or results.get("ids") is None or not results["ids"][0]:
            return formatted_list # Return empty list if no results or error

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else [None] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [None] * len(ids)

        for i, doc_id in enumerate(ids):
            # Calculate similarity (assuming distance is Euclidean, lower is better)
            # Normalize distance to similarity (0 to 1). Handle potential None distance.
            similarity = None
            if distances[i] is not None:
                 # Simple normalization assuming distance is roughly 0-2 for normalized embeddings
                 similarity = max(0.0, 1.0 - (distances[i] / 2.0))
                 # Alternative: 1 / (1 + distances[i])

            item = {
                "id": doc_id,
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
                "similarity": similarity # Add calculated similarity
            }
            formatted_list.append(item)

        # Sort by similarity (highest first) if similarity was calculated
        if formatted_list and formatted_list[0].get("similarity") is not None:
            formatted_list.sort(key=lambda x: x['similarity'], reverse=True)
        # If similarity couldn't be calculated, sort by distance (lowest first)
        elif formatted_list and formatted_list[0].get("distance") is not None:
             formatted_list.sort(key=lambda x: x['distance'])


        return formatted_list

# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("Testing ChromaDBClient...")
    try:
        # Ensure the data directory exists for the test
        import os
        os.makedirs(settings.DATA_DIR, exist_ok=True)

        client = ChromaDBClient()

        # Test adding data (using dummy Chunk objects)
        print("\n--- Testing Add Chunks ---")
        dummy_chunks = [
            Chunk(chunk_id="test_chunk_1", conv_id="conv_abc", chunk_text="This is the first test chunk.", start_time=1678886400.0, end_time=1678886405.0, message_ids=["msg1"], token_count=7, title="Test Convo 1"),
            Chunk(chunk_id="test_chunk_2", conv_id="conv_abc", chunk_text="Another chunk follows the first one.", start_time=1678886406.0, end_time=1678886410.0, message_ids=["msg2"], token_count=7, title="Test Convo 1"),
            Chunk(chunk_id="test_chunk_3", conv_id="conv_xyz", chunk_text="A chunk from a different conversation.", start_time=1678887000.0, end_time=1678887005.0, message_ids=["msg_x"], token_count=7, title="Test Convo 2"),
        ]
        client.add_chunks(dummy_chunks)

        # Test querying
        print("\n--- Testing Query ---")
        query = "information about the first chunk"
        results = client.query_chunks_and_summaries(query, top_k=2)

        print(f"Query: '{query}'")
        if results:
            print("Results:")
            for i, res in enumerate(results):
                print(f"  {i+1}. ID: {res['id']}")
                print(f"     Similarity: {res['similarity']:.4f}" if res['similarity'] is not None else "N/A")
                print(f"     Text: {res['text'][:80]}...")
                print(f"     Metadata: {res['metadata']}")
        else:
            print("No results found or query failed.")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")
        import traceback
        traceback.print_exc()
