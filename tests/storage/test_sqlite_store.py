# tests/storage/test_sqlite_store.py

import pytest
import sqlite3
from datetime import datetime
import json
from pathlib import Path

# Adjust import path as necessary
from phoenix.storage.sqlite_store import SqliteStore
from phoenix.core.models import Conversation, Interaction

# --- Fixtures ---

@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a path to a temporary SQLite database file."""
    return tmp_path / "test_phoenix.db"

@pytest.fixture
def sqlite_store(temp_db_path):
    """Provides an instance of SqliteStore using a temporary database."""
    store = SqliteStore(db_path=str(temp_db_path))
    yield store
    # Optional: Clean up connection if needed, though usually handled by object lifecycle
    # store.conn.close() # Be careful if conn might already be closed

# --- Test Cases ---

def test_sqlite_store_initialization(sqlite_store, temp_db_path):
    """Tests if the SqliteStore initializes and creates the database file and tables."""
    assert isinstance(sqlite_store, SqliteStore)
    assert Path(temp_db_path).exists()

    # Check if tables were created
    try:
        cursor = sqlite_store.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations';")
        assert cursor.fetchone() is not None, "Conversations table not created"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions';")
        assert cursor.fetchone() is not None, "Interactions table not created"
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()

def test_add_and_get_conversation(sqlite_store):
    """Tests adding a conversation and retrieving it."""
    now = datetime.now()
    convo = Conversation(
        id="test_convo_1",
        title="My Test Conversation",
        messages=[], # Keep messages simple for this test
        metadata={"tags": ["sqlite", "test"], "insights": ["Insight X"]},
        summary="A conversation added during testing.",
        create_time=now,
        update_time=now
    )

    # Add conversation
    db_id = sqlite_store.add_conversation(convo)
    assert db_id == "test_convo_1" # Assuming add_conversation returns the ID

    # Get conversation
    retrieved_convo = sqlite_store.get_conversation(db_id)

    assert retrieved_convo is not None
    assert retrieved_convo.id == convo.id
    assert retrieved_convo.title == convo.title
    assert retrieved_convo.summary == convo.summary
    # Compare timestamps carefully (potential precision differences)
    assert abs((retrieved_convo.create_time - convo.create_time).total_seconds()) < 0.001
    assert abs((retrieved_convo.update_time - convo.update_time).total_seconds()) < 0.001
    # Metadata is stored as JSON, compare the loaded dict
    assert retrieved_convo.metadata == convo.metadata

def test_get_conversation_not_found(sqlite_store):
    """Tests getting a non-existent conversation."""
    retrieved_convo = sqlite_store.get_conversation("non_existent_id")
    assert retrieved_convo is None

def test_add_and_get_interaction(sqlite_store):
    """Tests adding an interaction and retrieving it."""
    now = datetime.now()
    interaction = Interaction(
        query="What is SQLite?",
        response="It's a relational database.",
        context="Context used for response.",
        insights=["SQLite is file-based", "It's widely used"],
        timestamp=now
    )

    # Add interaction
    interaction_id = sqlite_store.add_interaction(interaction)
    assert isinstance(interaction_id, int) # Expecting an auto-incremented ID

    # Get interaction (assuming a method exists, otherwise query directly for test)
    # Let's assume direct query for this example if get_interaction doesn't exist
    cursor = sqlite_store.conn.cursor()
    cursor.execute("SELECT id, query, response, context, insights, timestamp FROM interactions WHERE id = ?", (interaction_id,))
    row = cursor.fetchone()
    cursor.close()

    assert row is not None
    retrieved_id, retrieved_query, retrieved_response, retrieved_context, retrieved_insights_json, retrieved_timestamp_str = row

    assert retrieved_id == interaction_id
    assert retrieved_query == interaction.query
    assert retrieved_response == interaction.response
    assert retrieved_context == interaction.context
    assert json.loads(retrieved_insights_json) == interaction.insights
    retrieved_timestamp = datetime.fromisoformat(retrieved_timestamp_str)
    assert abs((retrieved_timestamp - interaction.timestamp).total_seconds()) < 0.001


def test_update_conversation_summary(sqlite_store):
    """Tests updating the summary of an existing conversation."""
    now = datetime.now()
    convo = Conversation(id="update_test", title="Initial Title", messages=[], create_time=now, update_time=now)
    sqlite_store.add_conversation(convo)

    new_summary = "This is the updated summary."
    new_update_time = datetime.now()
    sqlite_store.update_conversation_summary("update_test", new_summary, new_update_time)

    retrieved_convo = sqlite_store.get_conversation("update_test")
    assert retrieved_convo is not None
    assert retrieved_convo.summary == new_summary
    assert abs((retrieved_convo.update_time - new_update_time).total_seconds()) < 0.001

# Add tests for error handling (e.g., adding duplicate conversation ID, DB connection errors)
# Add tests for other methods like get_all_conversation_ids, update_metadata, etc.
