# tests/chat/test_chat_service.py

import pytest
from unittest.mock import MagicMock, patch

# Adjust import path as necessary
from phoenix.chat.chat_service import ChatService
from phoenix.core.models import Conversation, Message, Interaction

# --- Fixtures ---

@pytest.fixture
def mock_dependencies(mocker):
    """Provides mocked dependencies for ChatService."""
    return {
        "chroma_store": mocker.MagicMock(),
        "sqlite_store": mocker.MagicMock(),
        "llm_adapter": mocker.MagicMock(),
        "context_builder_func": mocker.MagicMock(),
        "response_generator_func": mocker.MagicMock(),
        "insight_generator_func": mocker.MagicMock(),
    }

@pytest.fixture
def chat_service(mock_dependencies):
    """Provides an instance of ChatService with mocked dependencies."""
    return ChatService(**mock_dependencies)

# --- Test Cases ---

def test_chat_service_initialization(chat_service, mock_dependencies):
    """Tests if the ChatService initializes correctly."""
    assert isinstance(chat_service, ChatService)
    assert chat_service.chroma_store == mock_dependencies["chroma_store"]
    assert chat_service.sqlite_store == mock_dependencies["sqlite_store"]
    assert chat_service.llm_adapter == mock_dependencies["llm_adapter"]
    # ... check other attributes

@pytest.mark.skip(reason="Requires more detailed mocking/setup")
def test_process_user_message(chat_service, mock_dependencies):
    """Tests the core logic of processing a user message."""
    user_input = "Tell me about project X."
    mock_context = "Relevant context from vector store."
    mock_response = "AI response about project X."
    mock_insights = ["Insight 1", "Insight 2"]
    mock_interaction_id = 123

    # Configure mock return values
    mock_dependencies["chroma_store"].query.return_value = ["doc1", "doc2"] # Simplified
    mock_dependencies["context_builder_func"].return_value = mock_context
    mock_dependencies["response_generator_func"].return_value = mock_response
    mock_dependencies["insight_generator_func"].return_value = mock_insights
    mock_dependencies["sqlite_store"].add_interaction.return_value = mock_interaction_id

    ai_response, insights = chat_service.process_user_message(user_input)

    assert ai_response == mock_response
    assert insights == mock_insights

    # Verify calls
    mock_dependencies["chroma_store"].query.assert_called_once_with(user_input, n_results=chat_service.n_results)
    mock_dependencies["context_builder_func"].assert_called_once_with(user_input, ["doc1", "doc2"])
    mock_dependencies["response_generator_func"].assert_called_once_with(user_input, mock_context, chat_service.llm_adapter)
    mock_dependencies["insight_generator_func"].assert_called_once_with(user_input, mock_response, mock_context, chat_service.llm_adapter)
    mock_dependencies["sqlite_store"].add_interaction.assert_called_once()
    # Add more detailed checks on the Interaction object saved

@pytest.mark.skip(reason="Requires mocking input() and testing loop logic")
def test_run_interactive_chat(chat_service, mocker):
    """Tests the interactive chat loop (simplified)."""
    # Mock input() to simulate user interaction
    mock_input = mocker.patch('builtins.input')
    mock_input.side_effect = ["hello", "quit"] # Simulate user typing 'hello' then 'quit'

    # Mock process_user_message to avoid testing its full logic here
    mocker.patch.object(chat_service, 'process_user_message', return_value=("AI says hello", ["Insight"]))

    # Mock print to capture output (optional)
    mock_print = mocker.patch('builtins.print')

    chat_service.run_interactive_chat()

    # Assertions
    assert mock_input.call_count == 2 # Called for "hello" and "quit"
    chat_service.process_user_message.assert_called_once_with("hello")
    # Check print calls if needed (e.g., ensure response and insights were printed)
    # Example: mock_print.assert_any_call("\n🤖 Assistant:", "AI says hello")

# Add tests for error handling, edge cases, different configurations etc.
