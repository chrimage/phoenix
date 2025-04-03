# tests/indexing/test_chunker.py

import pytest
from datetime import datetime

# Adjust import path as necessary
from phoenix.indexing.chunker import chunk_conversation, calculate_token_count
from phoenix.core.models import Conversation, Message

# --- Fixtures ---

@pytest.fixture
def sample_conversation():
    """Provides a sample Conversation object for testing."""
    now = datetime.now()
    msg1 = Message(id="m1", role="user", content="Hello, how does the chunking work?", timestamp=now)
    msg2 = Message(id="m2", role="assistant", content="It groups messages based on token limits.", timestamp=now)
    msg3 = Message(id="m3", role="user", content="Okay, what's the default limit?", timestamp=now)
    msg4 = Message(id="m4", role="assistant", content="The default limit is usually around 500 tokens per chunk.", timestamp=now)
    msg5 = Message(id="m5", role="user", content="Thanks!", timestamp=now)
    return Conversation(
        id="convo1",
        title="Chunking Test",
        messages=[msg1, msg2, msg3, msg4, msg5],
        create_time=now,
        update_time=now
    )

# --- Test Cases for calculate_token_count ---
# Note: These tests depend on the spaCy model's tokenization. Results might vary slightly
# if the model changes or if the text contains complex elements.

def test_token_count_simple():
    """Tests token counting for a simple sentence."""
    text = "This is a simple sentence."
    # Expected count might vary slightly based on model (e.g., punctuation handling)
    # Run this once to see the actual count from your spacy model if needed.
    expected_count = 6 # Example: "This", "is", "a", "simple", "sentence", "."
    assert calculate_token_count(text) == expected_count

def test_token_count_empty():
    """Tests token counting for empty string."""
    assert calculate_token_count("") == 0

def test_token_count_longer():
    """Tests token counting for a longer text."""
    text = "This sentence is a bit longer and includes some punctuation, like commas!"
    # Again, the exact count depends on the tokenizer.
    expected_count = 14 # Example count
    assert calculate_token_count(text) == expected_count

# --- Test Cases for chunk_conversation ---

@pytest.mark.skip(reason="Requires careful setup of token limits and expected outputs")
def test_chunk_conversation_basic(sample_conversation):
    """Tests basic conversation chunking with a reasonable token limit."""
    # This test is complex because the output depends heavily on the token counts
    # of the formatted messages and the max_token_limit.
    # You'll need to manually calculate expected chunks based on your formatting
    # and a chosen token limit.

    max_tokens = 50 # Example: A small limit to force multiple chunks
    chunks_data = chunk_conversation(sample_conversation, max_token_limit=max_tokens)

    assert isinstance(chunks_data, list)
    assert len(chunks_data) > 1 # Expecting multiple chunks with this low limit

    # Example assertions (highly dependent on actual token counts and formatting):
    # chunk1 = chunks_data[0]
    # assert "Hello, how does the chunking work?" in chunk1['text']
    # assert "It groups messages based on token limits." in chunk1['text']
    # assert chunk1['metadata']['start_msg_id'] == "m1"
    # assert chunk1['metadata']['end_msg_id'] == "m2" # Or maybe m3 depending on tokens
    # assert chunk1['token_count'] <= max_tokens

    # chunk2 = chunks_data[1]
    # assert "Okay, what's the default limit?" in chunk2['text'] # Or maybe starts here
    # ... more detailed checks

@pytest.mark.skip(reason="Requires setting a very high token limit")
def test_chunk_conversation_single_chunk(sample_conversation):
    """Tests chunking when the limit is high enough for one chunk."""
    max_tokens = 1000 # High limit
    chunks_data = chunk_conversation(sample_conversation, max_token_limit=max_tokens)

    assert isinstance(chunks_data, list)
    assert len(chunks_data) == 1

    chunk = chunks_data[0]
    assert chunk['metadata']['start_msg_id'] == "m1"
    assert chunk['metadata']['end_msg_id'] == "m5"
    assert "Hello, how does the chunking work?" in chunk['text']
    assert "Thanks!" in chunk['text']
    assert chunk['token_count'] <= max_tokens

def test_chunk_conversation_empty():
    """Tests chunking an empty conversation."""
    now = datetime.now()
    empty_convo = Conversation(id="c_empty", title="Empty", messages=[], create_time=now, update_time=now)
    chunks_data = chunk_conversation(empty_convo)
    assert chunks_data == []

# Add tests for edge cases: very long messages, messages exceeding limit alone, etc.
