# phoenix/storage/sqlite_store.py - SQLite Database Interactions
import sqlite3
import os
import json
import time
from typing import Optional

from phoenix.config import settings
from phoenix.core.models import ConversationMetadata

class SqliteStore:
    """Handles interactions with the SQLite database for metadata."""

    def __init__(self, db_path: str = settings.DB_FILE_NAME):
        """
        Initializes the SqliteStore.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        print(f"Initializing SqliteStore with database path: {self.db_path}")

    def connect(self):
        """Connects to the SQLite database and sets up the schema if needed."""
        if self.conn:
            return # Already connected

        print(f"Connecting to SQLite database at: {self.db_path}")
        is_new_db = not os.path.exists(self.db_path)

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row # Access columns by name
            self.cursor = self.conn.cursor()

            # Create tables if they don't exist
            self._initialize_schema()

            if is_new_db:
                print("SQLite database schema initialized.")
            else:
                print("Connected to existing SQLite database.")

        except sqlite3.Error as e:
            print(f"Error connecting to or initializing SQLite database: {e}")
            self.conn = None
            self.cursor = None
            raise # Re-raise the exception

    def _initialize_schema(self):
        """Creates the necessary tables if they don't exist."""
        if not self.cursor:
            raise ConnectionError("Database not connected.")

        try:
            # Conversations table: Stores metadata about each conversation
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                title TEXT,
                create_time REAL, -- Unix timestamp
                model_slug TEXT,
                metadata TEXT -- Store other details as JSON
            );
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error initializing SQLite schema: {e}")
            if self.conn:
                self.conn.rollback()
            raise

    def save_conversation_metadata(self, conv_meta: ConversationMetadata):
        """
        Saves or updates conversation metadata in the SQLite database.

        Args:
            conv_meta: A ConversationMetadata object.

        Returns:
            True if successful, False otherwise.
        """
        if not self.conn or not self.cursor:
            print("Error: Cannot save metadata, database not connected.")
            return False

        try:
            # Ensure metadata is stored as a JSON string
            metadata_json = json.dumps(conv_meta.metadata) if conv_meta.metadata else None

            self.cursor.execute(
                """
                INSERT OR REPLACE INTO conversations (conv_id, title, create_time, model_slug, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conv_meta.conv_id,
                    conv_meta.title,
                    conv_meta.create_time,
                    conv_meta.model_slug,
                    metadata_json
                )
            )
            self.conn.commit()
            # print(f"Conversation metadata saved/updated for conv_id: {conv_meta.conv_id}")
            return True
        except sqlite3.Error as e:
            print(f"Error saving conversation metadata for conv_id {conv_meta.conv_id}: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        except Exception as e:
            print(f"Unexpected error saving conversation metadata: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def get_conversation_metadata(self, conv_id: str) -> Optional[ConversationMetadata]:
        """
        Retrieves conversation metadata from the SQLite database.

        Args:
            conv_id: The ID of the conversation to retrieve.

        Returns:
            A ConversationMetadata object if found, None otherwise.
        """
        if not self.conn or not self.cursor:
            print("Error: Cannot get metadata, database not connected.")
            return None

        try:
            self.cursor.execute("SELECT * FROM conversations WHERE conv_id = ?", (conv_id,))
            row = self.cursor.fetchone()

            if row:
                metadata_json = row["metadata"]
                metadata_dict = json.loads(metadata_json) if metadata_json else None
                return ConversationMetadata(
                    conv_id=row["conv_id"],
                    title=row["title"],
                    create_time=row["create_time"],
                    model_slug=row["model_slug"],
                    metadata=metadata_dict
                )
            else:
                return None
        except sqlite3.Error as e:
            print(f"Error retrieving conversation metadata for conv_id {conv_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error retrieving conversation metadata: {e}")
            return None

    def close(self):
        """Closes the SQLite database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            print("SQLite connection closed.")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing SqliteStore...")
    # Ensure the test runs in a temporary directory or uses a test DB name
    test_db_path = "test_phoenix_memory.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    try:
        with SqliteStore(db_path=test_db_path) as store:
            print("Store connected via context manager.")

            # Create dummy metadata
            meta1 = ConversationMetadata(
                conv_id="test_conv_1",
                title="My First Test",
                model_slug="test_model_v1",
                metadata={"source": "test_script", "tags": ["testing", "example"]}
            )
            meta2 = ConversationMetadata(
                conv_id="test_conv_2",
                title="Another Test Conversation",
                create_time=time.time() - 3600 # An hour ago
            )

            # Save metadata
            print("\nSaving metadata...")
            save_ok1 = store.save_conversation_metadata(meta1)
            save_ok2 = store.save_conversation_metadata(meta2)
            print(f"Save meta1 successful: {save_ok1}")
            print(f"Save meta2 successful: {save_ok2}")

            # Retrieve metadata
            print("\nRetrieving metadata...")
            retrieved_meta1 = store.get_conversation_metadata("test_conv_1")
            retrieved_meta2 = store.get_conversation_metadata("test_conv_2")
            retrieved_meta_none = store.get_conversation_metadata("non_existent_id")

            print(f"Retrieved meta1: {retrieved_meta1}")
            print(f"Retrieved meta2: {retrieved_meta2}")
            print(f"Retrieved non-existent: {retrieved_meta_none}")

            # Verify data
            if retrieved_meta1:
                assert retrieved_meta1.conv_id == meta1.conv_id
                assert retrieved_meta1.title == meta1.title
                assert retrieved_meta1.metadata == meta1.metadata
                print("Meta1 verification PASSED")
            else:
                print("Meta1 verification FAILED")

            if retrieved_meta2:
                assert retrieved_meta2.conv_id == meta2.conv_id
                assert retrieved_meta2.title == meta2.title
                assert retrieved_meta2.metadata is None # meta2 had no extra metadata
                print("Meta2 verification PASSED")
            else:
                print("Meta2 verification FAILED")

            assert retrieved_meta_none is None
            print("Non-existent verification PASSED")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # Clean up the test database file
        if os.path.exists(test_db_path):
            # os.remove(test_db_path)
            print(f"Test database '{test_db_path}' left for inspection.")
        else:
             print(f"Test database '{test_db_path}' not found for cleanup.")

    print("\nSqliteStore testing finished.")
