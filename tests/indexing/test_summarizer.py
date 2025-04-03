# tests/indexing/test_summarizer.py

import pytest
from unittest.mock import MagicMock
from datetime import datetime

# Adjust import path as necessary
from phoenix.indexing.summarizer import generate_conversation_summary
from phoenix.core.models import Conversation, Message

# --- Fixtures ---

@pytest.fixture
def mock_llm_adapter(mocker):
    """Provides a mocked LLM adapter."""
    adapter = mocker.MagicMock()
    adapter.generate_text.return_value = "This is a mocked summary of the conversation."
    return adapter

@pytest.fixture
def sample_conversation_for_summary():
    """Provides a sample Conversation object for summarization."""
    now = datetime.now()
    msg1 = Message(id="m1", role="user", content="Can we discuss the project plan?", timestamp=now)
    msg2 = Message(id="m2", role="assistant", content="Sure. Phase 1 involves X, Phase 2 involves Y.", timestamp=now)
    msg3 = Message(id="m3", role="user", content="What about the deadlines?", timestamp=now)
    msg4 = Message(id="m4", role="assistant", content="Phase 1 deadline is Q1, Phase 2 is Q2.", timestamp=now)
    return Conversation(
        id="convo_plan",
        title="Project Plan Discussion",
        messages=[msg1, msg2, msg3, msg4],
        create_time=now,
        update_time=now
    )

@pytest.fixture
def short_conversation():
    """Provides a very short conversation."""
    now = datetime.now()
    msg1 = Message(id="m_short", role="user", content="Hi", timestamp=now)
    return Conversation(id="c_short", title="Greeting", messages=[msg1], create_time=now, update_time=now)

# --- Test Cases ---

def test_generate_summary_basic(mock_llm_adapter, sample_conversation_for_summary):
    """Tests basic summary generation for a typical conversation."""
    expected_summary = "This is a mocked summary of the conversation."

    summary = generate_conversation_summary(sample_conversation_for_summary, mock_llm_adapter)

    assert summary == expected_summary
    mock_llm_adapter.generate_text.assert_called_once()
    # Check that the prompt contained the conversation content
    call_args, _ = mock_llm_adapter.generate_text.call_args
    prompt = call_args[0]
    assert "Can we discuss the project plan?" in prompt
    assert "Phase 1 deadline is Q1, Phase 2 is Q2." in prompt

def test_generate_summary_short_conversation(mock_llm_adapter, short_conversation):
    """Tests summary generation for a very short conversation."""
    # The LLM might still generate a summary, or the function might have logic
    # to handle short conversations differently (e.g., return title or first message).
    # This test assumes the LLM is still called.
    expected_summary = "This is a mocked summary of the conversation."

    summary = generate_conversation_summary(short_conversation, mock_llm_adapter)

    assert summary == expected_summary
    mock_llm_adapter.generate_text.assert_called_once()
    call_args, _ = mock_llm_adapter.generate_text.call_args
    prompt = call_args[0]
    assert "Hi" in prompt

def test_generate_summary_empty_conversation(mock_llm_adapter):
    """Tests summary generation for an empty conversation."""
    now = datetime.now()
    empty_convo = Conversation(id="c_empty", title="Empty", messages=[], create_time=now, update_time=now)

    # Expecting the function to handle this gracefully, likely returning None or empty string
    # without calling the LLM.
    summary = generate_conversation_summary(empty_convo, mock_llm_adapter)

    assert summary is None or summary == "" # Adjust based on actual behavior
    mock_llm_adapter.generate_text.assert_not_called()

def test_generate_summary_llm_error(mock_llm_adapter, sample_conversation_for_summary):
    """Tests handling of an error during LLM call."""
    mock_llm_adapter.generate_text.side_effect = Exception("LLM API Error")

    # Expect the function to either raise the exception or return a default value (e.g., None)
    with pytest.raises(Exception, match="LLM API Error"):
         generate_conversation_summary(sample_conversation_for_summary, mock_llm_adapter)
    # OR:
    # summary = generate_conversation_summary(sample_conversation_for_summary, mock_llm_adapter)
    # assert summary is None # Or some other indicator of failure

# Add tests for conversations with different structures, lengths, content types.
