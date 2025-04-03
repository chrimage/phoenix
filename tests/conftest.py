# tests/conftest.py
# Shared fixtures and configuration for pytest

import pytest
import os
import tempfile
import shutil
from pathlib import Path

# Example fixture for creating a temporary directory for tests
@pytest.fixture(scope="session") # Use 'session' scope for efficiency if needed across many tests
def temp_test_dir():
    """Creates a temporary directory for test artifacts."""
    temp_dir = Path(tempfile.mkdtemp(prefix="phoenix_test_"))
    print(f"Created temporary test directory: {temp_dir}")
    yield temp_dir # Provide the path to the tests
    # Teardown: Remove the directory after the test session finishes
    print(f"Removing temporary test directory: {temp_dir}")
    shutil.rmtree(temp_dir)

# Add other shared fixtures here as needed, e.g.,
# - Mocked LLM adapter
# - Temporary SQLite database connection/path
# - Temporary ChromaDB instance/path
# - Sample conversation data
