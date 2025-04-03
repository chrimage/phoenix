# tests/chat/test_response_generator.py

import pytest
from unittest.mock import MagicMock

# Adjust import path as necessary
from phoenix.chat.response_generator import generate_chat_response

# --- Fixtures ---

@pytest.fixture
def mock_llm_adapter(mocker):
    """Provides a mocked LLM adapter."""
    adapter = mocker.MagicMock()
    # Configure mock behavior as needed for tests
    adapter.generate_text.return_value = "This is a mocked LLM response."
    return adapter

# --- Test Cases ---

def test_generate_chat_response_basic(mock_llm_adapter):
    """Tests basic response generation."""
    user_query = "Hello there."
    context = "Some relevant context."
    expected_prompt_fragment = f"User Query: {user_query}" # Check key parts of the prompt
    expected_prompt_fragment_context = f"Context:\n{context}"
    expected_response = "This is a mocked LLM response."

    response = generate_chat_response(user_query, context, mock_llm_adapter)

    assert response == expected_response
    # Check that the LLM adapter was called with a prompt containing the query and context
    mock_llm_adapter.generate_text.assert_called_once()
    call_args, _ = mock_llm_adapter.generate_text.call_args
    generated_prompt = call_args[0]
    assert expected_prompt_fragment in generated_prompt
    assert expected_prompt_fragment_context in generated_prompt

def test_generate_chat_response_empty_context(mock_llm_adapter):
    """Tests response generation when context is empty."""
    user_query = "Any updates?"
    context = "" # Empty context
    expected_prompt_fragment = f"User Query: {user_query}"
    expected_response = "This is a mocked LLM response."

    response = generate_chat_response(user_query, context, mock_llm_adapter)

    assert response == expected_response
    mock_llm_adapter.generate_text.assert_called_once()
    call_args, _ = mock_llm_adapter.generate_text.call_args
    generated_prompt = call_args[0]
    assert expected_prompt_fragment in generated_prompt
    # Ensure the context section is handled gracefully (e.g., not present or marked empty)
    assert "Context:\n" in generated_prompt # Check based on actual prompt format

# Add more tests for different query/context combinations, error handling from LLM, etc.
