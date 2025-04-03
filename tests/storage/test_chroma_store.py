# tests/storage/test_chroma_store.py

import pytest
from unittest.mock import MagicMock, patch, call

# Adjust import path as necessary
from phoenix.storage.chroma_store import ChromaStore
from phoenix.core.models import Chunk # If needed

# --- Fixtures ---

@pytest.fixture
def mock_chromadb_client(mocker):
    """Provides a mocked chromadb client and collection."""
    mock_collection = mocker.MagicMock()
    mock_client = mocker.MagicMock()
    # Configure client mock to return the mock collection
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client, mock_collection

@pytest.fixture
@patch('chromadb.PersistentClient') # Patch the client used in ChromaStore.__init__
def chroma_store(mock_persistent_client, mock_chromadb_client, mocker):
    """Provides an instance of ChromaStore with mocked chromadb client."""
    mock_client_instance, mock_collection_instance = mock_chromadb_client
    # Make the PersistentClient patch return our mock client instance
    mock_persistent_client.return_value = mock_client_instance

    # Mock the embedding function if it's called during init or methods
    # Adjust the path 'google.generativeai.embed_content' if necessary
    with patch('google.generativeai.embed_content') as mock_embed_func:
        mock_embed_func.return_value = {'embedding': [0.1, 0.2]} # Example embedding

        store = ChromaStore(
            chroma_dir="dummy_test_dir",
            collection_name="test_collection",
            embedding_model="test_model",
            api_key="test_api_key" # Provide necessary args
        )
        # Store mocks for later access in tests if needed
        store._client = mock_client_instance
        store._collection = mock_collection_instance
        store._mock_embed_func = mock_embed_func # Store the mock embedding function
        yield store

# --- Test Cases ---

def test_chroma_store_initialization(chroma_store, mock_chromadb_client):
    """Tests if the ChromaStore initializes correctly and gets/creates collection."""
    mock_client, mock_collection = mock_chromadb_client
    mock_client.get_or_create_collection.assert_called_once_with(
        name="test_collection",
        embedding_function=chroma_store.embedding_function # Check if embedding func is passed
    )
    assert chroma_store._collection == mock_collection

def test_add_chunks(chroma_store, mock_chromadb_client):
    """Tests adding chunks to the Chroma collection."""
    _, mock_collection = mock_chromadb_client
    chunks_to_add = [
        {'id': 'c1', 'text': 'text 1', 'embedding': [0.1], 'metadata': {'conv_id': 'conv1'}},
        {'id': 'c2', 'text': 'text 2', 'embedding': [0.2], 'metadata': {'conv_id': 'conv1'}},
    ]

    chroma_store.add_chunks(chunks_to_add)

    # Check if collection.add was called with the correct arguments
    mock_collection.add.assert_called_once_with(
        ids=['c1', 'c2'],
        embeddings=[[0.1], [0.2]],
        documents=['text 1', 'text 2'],
        metadatas=[{'conv_id': 'conv1'}, {'conv_id': 'conv1'}]
    )

def test_add_chunks_empty(chroma_store, mock_chromadb_client):
    """Tests adding an empty list of chunks."""
    _, mock_collection = mock_chromadb_client
    chroma_store.add_chunks([])
    mock_collection.add.assert_not_called()

def test_query(chroma_store, mock_chromadb_client):
    """Tests querying the Chroma collection."""
    mock_client, mock_collection = mock_chromadb_client
    query_text = "search for this"
    n_results = 5
    mock_query_results = {
        'ids': [['id1', 'id2']],
        'documents': [['doc text 1', 'doc text 2']],
        'metadatas': [[{'conv_id': 'c1'}, {'conv_id': 'c2'}]],
        'distances': [[0.1, 0.2]]
    }
    mock_collection.query.return_value = mock_query_results

    # Mock the embedding function call during query
    chroma_store._mock_embed_func.return_value = {'embedding': [0.5, 0.6]}

    results = chroma_store.query(query_text, n_results=n_results)

    # Check that the embedding function was called for the query text
    chroma_store._mock_embed_func.assert_called_once_with(
        model=chroma_store.embedding_model_name,
        content=query_text,
        task_type="retrieval_query" # Or appropriate task type
    )

    # Check that collection.query was called correctly
    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.5, 0.6]], # Check the mocked query embedding
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'] # Check included fields
    )

    # Check that the results are processed correctly (depends on implementation)
    assert results == ['doc text 1', 'doc text 2'] # Example: returns only documents

def test_delete_collection(chroma_store, mock_chromadb_client):
    """Tests deleting the collection."""
    mock_client, _ = mock_chromadb_client
    collection_name = chroma_store.collection_name

    chroma_store.delete_collection()

    mock_client.delete_collection.assert_called_once_with(name=collection_name)


# Add tests for error handling (e.g., connection errors, embedding errors, query errors)
