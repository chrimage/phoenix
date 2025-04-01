# phoenix/adapters/metadata_db/sqlite_client.py - Adapter for SQLite interactions
import sqlite3
import json
import os
from typing import Optional

from phoenix.config import settings
from phoenix.core.models import Conversation # Import Conversation model for type hinting

class SQLiteClient:
    """Wraps SQLite database interactions for storing conversation metadata."""

    def __init__(self, db_path: str = settings.DB_FILE_NAME):
        """
        Initializes the SQLite client and connects to the database.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._conn = None
        self._cursor = None
        # Ensure the directory for the database exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._connect()
        self.setup_database() # Ensure table exists on initialization

    def _connect(self):
        """Establishes the database connection and cursor."""
        try:
            print(f"Connecting to SQLite database at: {self.db_path}")
            self._conn = sqlite3.connect(self.db_path)
            # Use Row factory for dictionary-like access (optional but convenient)
            # self._conn.row_factory = sqlite3.Row
            self._cursor = self._conn.cursor()
            print("SQLite connection established.")
        except sqlite3.Error as e:
            print(f"❌ Error connecting to SQLite database: {e}")
            self._conn = None # Ensure connection is None on error
            self._cursor = None
            raise # Propagate the error

    def setup_database(self):
        """Creates the necessary tables if they don't exist."""
        if not self._cursor:
            print("❌ Cannot setup database, no cursor available.")
            return
        try:
            print("Ensuring 'conversations' table exists...")
            self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                title TEXT,
                create_time REAL, -- Unix timestamp
                model_slug TEXT,
                metadata TEXT -- Store other details as JSON
            );
            """)
            self._conn.commit()
            print("'conversations' table is ready.")
        except sqlite3.Error as e:
            print(f"❌ Error setting up database table: {e}")
            if self._conn:
                self._conn.rollback()
            raise

    def save_conversation_metadata(self, conversation: Conversation) -> bool:
        """
        Saves or updates a conversation's metadata in the database.

        Args:
            conversation: The Conversation object containing the metadata.

        Returns:
            True if successful, False otherwise.
        """
        if not self._cursor or not self._conn:
            print("❌ Cannot save metadata, database connection not available.")
            return False

        try:
            # Serialize metadata dictionary to JSON string for storage
            metadata_json = json.dumps(conversation.metadata or {})

            self._cursor.execute(
                """
                INSERT OR REPLACE INTO conversations
                (conv_id, title, create_time, model_slug, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation.conv_id,
                    conversation.title,
                    conversation.create_time,
                    conversation.model_slug,
                    metadata_json
                )
            )
            self._conn.commit()
            # print(f"Saved/Updated metadata for conversation: {conversation.conv_id[:8]}")
            return True
        except sqlite3.Error as e:
            print(f"❌ Error saving conversation metadata for {conversation.conv_id}: {e}")
            if self._conn:
                self._conn.rollback()
            return False
        except Exception as e:
            print(f"❌ Unexpected error saving conversation metadata: {e}")
            if self._conn:
                self._conn.rollback()
            return False

    def get_conversation_metadata(self, conv_id: str) -> Optional[dict]:
        """
        Retrieves metadata for a specific conversation ID. (Optional method)

        Args:
            conv_id: The ID of the conversation to retrieve.

        Returns:
            A dictionary containing the conversation metadata, or None if not found or error.
        """
        if not self._cursor:
            print("❌ Cannot get metadata, database connection not available.")
            return None
        try:
            self._cursor.execute("SELECT * FROM conversations WHERE conv_id = ?", (conv_id,))
            row = self._cursor.fetchone()
            if row:
                # Assuming column order: conv_id, title, create_time, model_slug, metadata
                metadata_dict = {
                    "conv_id": row[0],
                    "title": row[1],
                    "create_time": row[2],
                    "model_slug": row[3],
                    "metadata": json.loads(row[4] or '{}') # Deserialize JSON
                }
                return metadata_dict
            else:
                return None # Not found
        except sqlite3.Error as e:
            print(f"❌ Error retrieving conversation metadata for {conv_id}: {e}")
            return None
        except json.JSONDecodeError as e:
             print(f"❌ Error decoding metadata JSON for {conv_id}: {e}")
             # Optionally return partial data or None
             return None


    def close(self):
        """Closes the database connection."""
        if self._conn:
            try:
                self._conn.close()
                print("SQLite connection closed.")
                self._conn = None
                self._cursor = None
            except sqlite3.Error as e:
                print(f"❌ Error closing SQLite connection: {e}")

    def __del__(self):
        """Ensures connection is closed when the object is garbage collected."""
        self.close()

# Example Usage (for testing purposes)
if __name__ == "__main__":
    import time
    print("Testing SQLiteClient...")

    # Use a temporary DB for testing
    test_db_path = os.path.join(settings.DATA_DIR, "test_phoenix_memory.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path) # Clean up previous test runs

    try:
        client = SQLiteClient(db_path=test_db_path)

        # Create dummy conversation data
        ts = time.time()
        convo1 = Conversation(
            conv_id="test_conv_001",
            title="First Test Conversation",
            create_time=ts,
            model_slug="test-model-v1",
            metadata={"source": "testing", "user_id": 123}
        )
        convo2 = Conversation(
            conv_id="test_conv_002",
            title="Second Test",
            create_time=ts + 60,
            model_slug="test-model-v2",
            metadata={"tags": ["important", "code"]}
        )

        # Test saving
        print("\n--- Testing Save Metadata ---")
        success1 = client.save_conversation_metadata(convo1)
        success2 = client.save_conversation_metadata(convo2)
        print(f"Save Convo 1 successful: {success1}")
        print(f"Save Convo 2 successful: {success2}")

        # Test retrieving (optional)
        print("\n--- Testing Get Metadata ---")
        retrieved1 = client.get_conversation_metadata("test_conv_001")
        retrieved_nonexistent = client.get_conversation_metadata("non_existent_id")

        print("Retrieved Convo 1:")
        if retrieved1:
            print(json.dumps(retrieved1, indent=2))
        else:
            print("Not found or error.")

        print("\nRetrieved Non-existent:")
        if retrieved_nonexistent:
            print(json.dumps(retrieved_nonexistent, indent=2))
        else:
            print("Not found (as expected).")


    except Exception as ex:
        print(f"An error occurred during testing: {ex}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the test database file
        if os.path.exists(test_db_path):
            # Ensure client connection is closed before removing
            if 'client' in locals() and client._conn:
                 client.close()
            try:
                os.remove(test_db_path)
                print(f"Cleaned up test database: {test_db_path}")
            except OSError as e:
                print(f"Error removing test database: {e}")
