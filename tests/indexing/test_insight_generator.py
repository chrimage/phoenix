# tests/indexing/test_insight_generator.py

import pytest
from unittest.mock import MagicMock

# Adjust import path as necessary
from phoenix.indexing.insight_generator import generate_insight_notes
from phoenix.core.models import Conversation, Message # If needed for context

# --- Fixtures ---

@pytest.fixture
def mock_llm_adapter(mocker):
    """Provides a mocked LLM adapter."""
    adapter = mocker.MagicMock()
    # Example: Assume insights are returned as a list of strings in the text
    adapter.generate_text.return_value = "Insight 1\n- Insight 2\n* Insight 3"
    return adapter

@pytest.fixture
def sample_conversation_for_insights():
    """Provides a sample Conversation object for insight generation context."""
    # You might want a more specific conversation for testing insights
    from datetime import datetime
    now = datetime.now()
    msg1 = Message(id="m1", role="user", content="Discuss project goals.", timestamp=now)
    msg2 = Message(id="m2", role="assistant", content="Goal A, Goal B.", timestamp=now)
    return Conversation(id="c1", title="Goals", messages=[msg1, msg2], create_time=now, update_time=now)

# --- Test Cases ---

def test_generate_insight_notes_basic(mock_llm_adapter, sample_conversation_for_insights):
    """Tests basic insight generation."""
    expected_insights = ["Insight 1", "- Insight 2", "* Insight 3"] # Based on mock return value parsing

    # Test with conversation object
    insights = generate_insight_notes(sample_conversation_for_insights, mock_llm_adapter)

    assert insights == expected_insights
    mock_llm_adapter.generate_text.assert_called_once()
    # Add checks here to verify the prompt sent to the LLM included conversation content

def test_generate_insight_notes_from_text(mock_llm_adapter):
    """Tests insight generation from raw text inputs (if applicable)."""
    # This test assumes the function can also take raw strings,
    # adjust if it only takes Conversation objects.
    user_query = "User query text."
    ai_response = "AI response text."
    context = "Some context."
    expected_insights = ["Insight 1", "- Insight 2", "* Insight 3"]

    insights = generate_insight_notes(user_query, ai_response, context, mock_llm_adapter)

    assert insights == expected_insights
    mock_llm_adapter.generate_text.assert_called_once()
    # Add checks here to verify the prompt sent to the LLM included query, response, context

def test_generate_insight_notes_empty_response(mock_llm_adapter):
    """Tests insight generation when the LLM returns an empty response."""
    mock_llm_adapter.generate_text.return_value = "" # Simulate empty response
    user_query = "Query"
    ai_response = "Response"
    context = "Context"

    insights = generate_insight_notes(user_query, ai_response, context, mock_llm_adapter)

    assert insights == [] # Expect empty list if LLM gives nothing

def test_generate_insight_notes_no_insights(mock_llm_adapter):
    """Tests insight generation when the LLM response contains no clear insights."""
    mock_llm_adapter.generate_text.return_value = "Just some normal text, no bullet points."
    user_query = "Query"
    ai_response = "Response"
    context = "Context"

    insights = generate_insight_notes(user_query, ai_response, context, mock_llm_adapter)

    # The expected result depends on how the function parses insights.
    # If it strictly looks for lines starting with markers, it might be empty.
    # If it returns the whole text as one insight, adjust the assertion.
    assert insights == ["Just some normal text, no bullet points."] # Example: returns non-empty lines

# Add tests for error handling, different LLM responses, complex parsing scenarios.
