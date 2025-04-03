# tests/indexing/test_indexer_service.py

import pytest
import json
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# Adjust import paths as necessary
from phoenix.indexing.indexer_service import IndexerService
from phoenix.core.models import Conversation # Assuming Conversation model is used

# --- Fixtures ---

@pytest.fixture
def mock_indexer_dependencies(mocker):
    """Provides mocked dependencies for IndexerService."""
    return {
        "parser": mocker.MagicMock(),
        "chunker_func": mocker.MagicMock(),
        "summarizer_func": mocker.MagicMock(),
        "insight_generator_func": mocker.MagicMock(),
        "sqlite_store": mocker.MagicMock(),
        "chroma_store": mocker.MagicMock(),
        "llm_adapter": mocker.MagicMock(),
    }

@pytest.fixture
def indexer_service(mock_indexer_dependencies):
    """Provides an instance of IndexerService with mocked dependencies."""
    return IndexerService(**mock_indexer_dependencies)

@pytest.fixture
def sample_convo_data():
    """Provides sample raw conversation data (like from JSON)."""
    # This should resemble the structure in your input JSON file
    return [
        {
            "title": "Test Convo 1",
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Hello"]}, "create_time": 1678886400}},
                "msg2": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Hi"]}, "create_time": 1678886401}},
            },
            "create_time": 1678886400,
            "update_time": 1678886401,
            # Add other relevant fields from your JSON structure
        },
        # Add more conversations if needed
    ]

@pytest.fixture
def sample_parsed_conversation(sample_convo_data):
    """Provides a sample parsed Conversation object."""
    # Create a Conversation object based on sample_convo_data
    # This depends on your parser's output format
    # For simplicity, let's assume the parser returns a Conversation object
    # (You might need to mock the parser's return value more accurately)
    from phoenix.core.models import Message # Import here or globally
    from datetime import datetime
    return Conversation(
        id="convo1", # Example ID
        title=sample_convo_data[0]['title'],
        messages=[
            Message(id="msg1", role="user", content="Hello", timestamp=datetime.fromtimestamp(1678886400)),
            Message(id="msg2", role="assistant", content="Hi", timestamp=datetime.fromtimestamp(1678886401)),
        ],
        create_time=datetime.fromtimestamp(sample_convo_data[0]['create_time']),
        update_time=datetime.fromtimestamp(sample_convo_data[0]['update_time']),
        # Add metadata etc. if parsed
    )


# --- Test Cases ---

def test_indexer_service_initialization(indexer_service, mock_indexer_dependencies):
    """Tests if the IndexerService initializes correctly."""
    assert isinstance(indexer_service, IndexerService)
    assert indexer_service.parser == mock_indexer_dependencies["parser"]
    assert indexer_service.sqlite_store == mock_indexer_dependencies["sqlite_store"]
    # ... check other attributes

@patch('builtins.open', new_callable=MagicMock)
@patch('json.load')
def test_process_input_file(mock_json_load, mock_open, indexer_service, mock_indexer_dependencies, sample_convo_data, sample_parsed_conversation):
    """Tests the processing of an input JSON file."""
    input_path = "dummy/path/conversations.json"
    mock_json_load.return_value = sample_convo_data

    # Mock the parser to return our sample parsed conversation
    mock_indexer_dependencies["parser"].parse_conversation.return_value = sample_parsed_conversation

    # Mock other processing functions
    mock_summary = "Generated summary"
    mock_insights = ["Insight A", "Insight B"]
    mock_chunks_data = [{'id': 'chunk1', 'text': 'Chunk text', 'embedding': [0.1], 'metadata': {}}]
    mock_indexer_dependencies["summarizer_func"].return_value = mock_summary
    mock_indexer_dependencies["insight_generator_func"].return_value = mock_insights
    mock_indexer_dependencies["chunker_func"].return_value = mock_chunks_data
    mock_indexer_dependencies["llm_adapter"].generate_embedding.return_value = [0.1] # Mock embedding generation

    # Mock DB add methods
    mock_indexer_dependencies["sqlite_store"].add_conversation.return_value = "convo1_db_id" # Example return

    # Execute the method
    indexer_service.process_input_file(input_path)

    # Assertions
    mock_open.assert_called_once_with(Path(input_path), 'r', encoding='utf-8')
    mock_json_load.assert_called_once()
    mock_indexer_dependencies["parser"].parse_conversation.assert_called_once_with(sample_convo_data[0])
    mock_indexer_dependencies["summarizer_func"].assert_called_once_with(sample_parsed_conversation, mock_indexer_dependencies["llm_adapter"])
    mock_indexer_dependencies["insight_generator_func"].assert_called_once_with(sample_parsed_conversation, mock_indexer_dependencies["llm_adapter"])

    # Check that the conversation (with summary/insights) was added to SQLite
    mock_indexer_dependencies["sqlite_store"].add_conversation.assert_called_once()
    call_args, _ = mock_indexer_dependencies["sqlite_store"].add_conversation.call_args
    saved_convo = call_args[0]
    assert saved_convo.summary == mock_summary
    assert saved_convo.metadata['insights'] == mock_insights # Check where insights are stored

    # Check chunking and embedding
    mock_indexer_dependencies["chunker_func"].assert_called_once_with(sample_parsed_conversation)
    mock_indexer_dependencies["llm_adapter"].generate_embedding.assert_called_once_with(mock_chunks_data[0]['text'])

    # Check that chunks were added to ChromaDB
    mock_indexer_dependencies["chroma_store"].add_chunks.assert_called_once_with([mock_chunks_data[0]]) # Check data format

def test_process_input_file_max_convos(mock_indexer_dependencies, indexer_service, sample_convo_data):
    """Tests the max_convos limit."""
    # Similar setup as above, but provide more data and set max_convos
    input_path = "dummy/path/conversations.json"
    multiple_convos_data = sample_convo_data * 3 # Simulate 3 conversations

    with patch('builtins.open', new_callable=MagicMock()), \
         patch('json.load', return_value=multiple_convos_data):

        # Mock parser to return distinct objects if necessary
        mock_indexer_dependencies["parser"].parse_conversation.side_effect = [
            MagicMock(id="c1"), MagicMock(id="c2"), MagicMock(id="c3")
        ]
        # Mock other funcs simply
        mock_indexer_dependencies["summarizer_func"].return_value = "summary"
        mock_indexer_dependencies["insight_generator_func"].return_value = []
        mock_indexer_dependencies["chunker_func"].return_value = [] # Simplify: assume no chunks for this test focus

        indexer_service.process_input_file(input_path, max_convos=2)

        # Assert parser was called only twice
        assert mock_indexer_dependencies["parser"].parse_conversation.call_count == 2
        # Assert DB add was called only twice
        assert mock_indexer_dependencies["sqlite_store"].add_conversation.call_count == 2


# Add tests for error handling (file not found, JSON decode error, DB errors, etc.)
