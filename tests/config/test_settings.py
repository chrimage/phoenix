# tests/config/test_settings.py

import pytest
import os
from unittest.mock import patch

# Adjust import path as necessary
# Assuming settings are loaded from phoenix.config.settings
# This might need adjustment if settings are loaded differently (e.g., via a function)
from phoenix.config import settings # Direct import might trigger loading

# --- Test Cases ---

def test_default_settings_loaded():
    """Tests if default settings are loaded correctly."""
    # These assertions depend on the actual default values in your settings
    assert isinstance(settings.DB_FILE_NAME, str)
    assert settings.DB_FILE_NAME == "data/phoenix_memory.db" # Example default
    assert isinstance(settings.CHROMA_DIR, str)
    assert settings.CHROMA_DIR == "chroma_memory" # Example default
    assert isinstance(settings.CHROMA_COLLECTION_NAME, str)
    # Add checks for other important default settings

@patch.dict(os.environ, {
    "GOOGLE_API_KEY": "test_key_from_env",
    "DB_FILE_NAME": "env_db.sqlite",
    "CHROMA_DIR": "/tmp/env_chroma",
    "EMBEDDING_MODEL": "models/embedding-002", # Example override
})
def test_settings_override_from_env():
    """Tests if settings can be overridden by environment variables."""
    # IMPORTANT: How settings are reloaded depends on your implementation.
    # If settings are loaded only once at import time, this test might need
    # to trigger a reload mechanism or test the loading logic directly.
    # This example assumes direct access reflects overrides, which might not be true.

    # Reload settings if your application supports it or test the loading function
    # Example: from phoenix.config import load_settings; loaded_settings = load_settings()
    # Then assert on loaded_settings instead of the imported 'settings' object.

    # Assuming direct import reflects env vars (may need adjustment)
    assert settings.GOOGLE_API_KEY == "test_key_from_env"
    assert settings.DB_FILE_NAME == "env_db.sqlite"
    assert settings.CHROMA_DIR == "/tmp/env_chroma"
    assert settings.EMBEDDING_MODEL == "models/embedding-002"

def test_api_key_presence():
    """Checks if the Google API key is present (might be loaded from .env)."""
    # This test is tricky because the actual key shouldn't be in tests.
    # It might be better to test the *logic* that loads the key,
    # or ensure it's non-empty if loaded.
    if os.getenv("CI"): # Skip in CI if key isn't set there
        pytest.skip("Skipping API key presence check in CI environment")

    # Check if it's loaded (assuming dotenv loading happens)
    assert settings.GOOGLE_API_KEY is not None
    assert isinstance(settings.GOOGLE_API_KEY, str)
    assert len(settings.GOOGLE_API_KEY) > 0 # Basic check

# Add tests for validation logic if your settings use Pydantic or similar
