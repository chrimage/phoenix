# tests/core/test_models.py

import pytest
from datetime import datetime

# Adjust import path as necessary
from phoenix.core.models import Message, Conversation, Chunk, Interaction

# --- Test Cases for Message ---

def test_message_creation():
    """Tests basic Message object creation."""
    now = datetime.now()
    msg = Message(
        id="msg1",
        role="user",
        content="Hello world",
        timestamp=now,
        metadata={"source": "test"}
    )
    assert msg.id == "msg1"
    assert msg.role == "user"
    assert msg.content == "Hello world"
    assert msg.timestamp == now
    assert msg.metadata == {"source": "test"}

def test_message_default_metadata():
    """Tests Message creation with default metadata."""
    now = datetime.now()
    msg = Message(
        id="msg2",
        role="assistant",
        content="Hi there",
        timestamp=now
    )
    assert msg.id == "msg2"
    assert msg.role == "assistant"
    assert msg.metadata == {} # Check default value

# Add more tests for Message validation if using Pydantic validators

# --- Test Cases for Conversation ---

def test_conversation_creation():
    """Tests basic Conversation object creation."""
    now = datetime.now()
    msg1 = Message(id="m1", role="user", content="Q", timestamp=now)
    msg2 = Message(id="m2", role="assistant", content="A", timestamp=now)
    convo = Conversation(
        id="convo1",
        title="Test Conversation",
        messages=[msg1, msg2],
        metadata={"tags": ["test", "basic"]},
        summary="A simple test conversation.",
        create_time=now,
        update_time=now
    )
    assert convo.id == "convo1"
    assert convo.title == "Test Conversation"
    assert len(convo.messages) == 2
    assert convo.messages[0] == msg1
    assert convo.metadata == {"tags": ["test", "basic"]}
    assert convo.summary == "A simple test conversation."
    assert convo.create_time == now

def test_conversation_empty_messages():
    """Tests Conversation creation with no messages."""
    now = datetime.now()
    convo = Conversation(
        id="convo2",
        title="Empty Convo",
        messages=[],
        create_time=now,
        update_time=now
    )
    assert convo.id == "convo2"
    assert len(convo.messages) == 0

# Add more tests for Conversation validation, methods if any

# --- Test Cases for Chunk ---

def test_chunk_creation():
    """Tests basic Chunk object creation."""
    now = datetime.now()
    chunk = Chunk(
        id="chunk1",
        conversation_id="convo1",
        text="This is a text chunk.",
        embedding=[0.1, 0.2, 0.3],
        metadata={"start_msg_id": "m1", "end_msg_id": "m2"},
        timestamp=now
    )
    assert chunk.id == "chunk1"
    assert chunk.conversation_id == "convo1"
    assert chunk.text == "This is a text chunk."
    assert chunk.embedding == [0.1, 0.2, 0.3]
    assert chunk.metadata == {"start_msg_id": "m1", "end_msg_id": "m2"}
    assert chunk.timestamp == now

# Add more tests for Chunk validation

# --- Test Cases for Interaction ---

def test_interaction_creation():
    """Tests basic Interaction object creation."""
    now = datetime.now()
    interaction = Interaction(
        id=1, # Assuming integer ID from DB
        query="User query text",
        response="AI response text",
        context="Retrieved context used",
        insights=["Insight 1", "Insight 2"],
        timestamp=now
    )
    assert interaction.id == 1
    assert interaction.query == "User query text"
    assert interaction.response == "AI response text"
    assert interaction.context == "Retrieved context used"
    assert interaction.insights == ["Insight 1", "Insight 2"]
    assert interaction.timestamp == now

# Add more tests for Interaction validation
