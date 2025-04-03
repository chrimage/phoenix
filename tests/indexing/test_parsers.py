# tests/indexing/test_parsers.py

import pytest
from datetime import datetime
import json

# Adjust import path as necessary
from phoenix.indexing.parsers import OpenAIParser
from phoenix.core.models import Conversation, Message

# --- Fixtures ---

@pytest.fixture
def openai_parser():
    """Provides an instance of the OpenAIParser."""
    return OpenAIParser()

@pytest.fixture
def sample_openai_convo_data():
    """Provides sample raw conversation data in OpenAI JSON format."""
    # This structure should match the actual JSON format you expect
    return {
        "title": "Sample OpenAI Chat",
        "create_time": 1678886400.123,
        "update_time": 1678886460.456,
        "mapping": {
            "msg1": {
                "id": "msg1",
                "message": {
                    "id": "msg1_inner",
                    "author": {"role": "user"},
                    "create_time": 1678886405.1,
                    "content": {"content_type": "text", "parts": ["Hello Assistant"]},
                    "status": "finished_successfully",
                    # ... other fields
                },
                "parent": "root",
                # ... other fields
            },
            "msg2": {
                "id": "msg2",
                "message": {
                    "id": "msg2_inner",
                    "author": {"role": "assistant"},
                    "create_time": 1678886410.2,
                    "content": {"content_type": "text", "parts": ["Hello User! How can I help?"]},
                    "status": "finished_successfully",
                    # ... other fields
                },
                "parent": "msg1",
                # ... other fields
            },
            "msg3": {
                 "id": "msg3",
                 "message": None, # Example of a node without a message
                 "parent": "msg2",
            }
        },
        "moderation_results": [],
        "current_node": "msg2", # Or msg3 if it exists
        # ... other top-level fields
    }

# --- Test Cases ---

def test_parse_conversation_basic(openai_parser, sample_openai_convo_data):
    """Tests parsing a basic valid OpenAI conversation structure."""
    conversation = openai_parser.parse_conversation(sample_openai_convo_data)

    assert isinstance(conversation, Conversation)
    assert conversation.title == "Sample OpenAI Chat"
    # Timestamps might need tolerance or careful comparison
    assert conversation.create_time == datetime.fromtimestamp(1678886400.123)
    assert conversation.update_time == datetime.fromtimestamp(1678886460.456)
    assert len(conversation.messages) == 2 # msg3 has no message content

    # Check message 1
    msg1 = conversation.messages[0]
    assert msg1.id == "msg1" # Uses the mapping key as ID
    assert msg1.role == "user"
    assert msg1.content == "Hello Assistant"
    assert msg1.timestamp == datetime.fromtimestamp(1678886405.1)
    assert msg1.metadata.get("openai_message_id") == "msg1_inner" # Check if metadata is added

    # Check message 2
    msg2 = conversation.messages[1]
    assert msg2.id == "msg2"
    assert msg2.role == "assistant"
    assert msg2.content == "Hello User! How can I help?"
    assert msg2.timestamp == datetime.fromtimestamp(1678886410.2)
    assert msg2.metadata.get("openai_message_id") == "msg2_inner"

def test_parse_conversation_missing_fields(openai_parser):
    """Tests parsing with potentially missing optional fields."""
    # Create data missing optional fields like title, times, etc.
    minimal_data = {
        "mapping": {
            "m1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Test"]}}}
        },
        "current_node": "m1"
    }
    conversation = openai_parser.parse_conversation(minimal_data)
    assert isinstance(conversation, Conversation)
    assert conversation.title is None or "Untitled" in conversation.title # Check default title
    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == "Test"
    # Check default timestamps if applicable

def test_parse_conversation_empty_mapping(openai_parser):
    """Tests parsing when the mapping is empty."""
    empty_data = {"mapping": {}, "current_node": None}
    conversation = openai_parser.parse_conversation(empty_data)
    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 0

def test_parse_conversation_invalid_structure(openai_parser):
    """Tests parsing with invalid or unexpected data structure."""
    invalid_data = {"foo": "bar"} # Completely wrong structure
    with pytest.raises(Exception): # Expect an error (KeyError, TypeError, etc.)
        openai_parser.parse_conversation(invalid_data)

    invalid_data_2 = {"mapping": {"msg1": {"message": None}}} # Missing essential message fields
    with pytest.raises(Exception):
        openai_parser.parse_conversation(invalid_data_2)


# Add more tests for edge cases: different content types, error messages, complex parent structures etc.
