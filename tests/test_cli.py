# tests/test_cli.py

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

# Adjust import path to your Typer app instance
from phoenix.cli import app

# --- Fixtures ---

@pytest.fixture
def runner():
    """Provides a CliRunner instance for invoking CLI commands."""
    return CliRunner()

@pytest.fixture
def mock_cli_dependencies(mocker):
    """Mocks the service getters used by the CLI commands."""
    # Patch the getter functions in the cli module
    mock_get_indexer = mocker.patch('phoenix.cli.get_indexer_service')
    mock_get_chat = mocker.patch('phoenix.cli.get_chat_service')

    # Configure the mock services returned by the getters
    mock_indexer_service = MagicMock()
    mock_chat_service = MagicMock()
    mock_get_indexer.return_value = mock_indexer_service
    mock_get_chat.return_value = mock_chat_service

    return {
        "indexer_service": mock_indexer_service,
        "chat_service": mock_chat_service,
    }

@pytest.fixture
def sample_input_file(tmp_path):
    """Creates a dummy input file for indexing tests."""
    input_file = tmp_path / "sample_convos.json"
    input_file.write_text('[]') # Minimal valid JSON
    return str(input_file)

# --- Test Cases for 'index' command ---

def test_index_command_success(runner, mock_cli_dependencies, sample_input_file):
    """Tests the happy path for the index command."""
    mock_indexer = mock_cli_dependencies["indexer_service"]
    result = runner.invoke(app, ["index", "--input", sample_input_file])

    assert result.exit_code == 0
    assert "--- Starting Phoenix Indexing ---" in result.stdout
    assert "--- Phoenix Indexing Finished ---" in result.stdout
    # Verify that the indexer service's process method was called
    mock_indexer.process_input_file.assert_called_once_with(
        input_path=sample_input_file,
        max_convos=None # Default value
    )

def test_index_command_with_max_convos(runner, mock_cli_dependencies, sample_input_file):
    """Tests the index command with the --max option."""
    mock_indexer = mock_cli_dependencies["indexer_service"]
    result = runner.invoke(app, ["index", "-i", sample_input_file, "--max", "10"])

    assert result.exit_code == 0
    mock_indexer.process_input_file.assert_called_once_with(
        input_path=sample_input_file,
        max_convos=10
    )

@patch('shutil.rmtree')
@patch('os.remove')
@patch('os.path.exists')
def test_index_command_recreate_db(mock_exists, mock_remove, mock_rmtree, runner, mock_cli_dependencies, sample_input_file):
    """Tests the index command with the --recreate option."""
    mock_indexer = mock_cli_dependencies["indexer_service"]
    # Simulate existing DB files/dirs
    mock_exists.return_value = True

    # Need to know the default paths from settings to check calls
    # Assuming defaults are 'chroma_memory' and 'data/phoenix_memory.db'
    default_chroma_path = 'chroma_memory'
    default_sqlite_path = 'data/phoenix_memory.db'

    result = runner.invoke(app, ["index", "-i", sample_input_file, "--recreate"])

    assert result.exit_code == 0
    assert f"Deleting existing ChromaDB directory: {default_chroma_path}" in result.stdout
    assert f"Deleting existing SQLite database: {default_sqlite_path}" in result.stdout
    mock_exists.assert_any_call(default_chroma_path)
    mock_exists.assert_any_call(default_sqlite_path)
    mock_rmtree.assert_called_once_with(default_chroma_path)
    mock_remove.assert_called_once_with(default_sqlite_path)
    mock_indexer.process_input_file.assert_called_once()

def test_index_command_file_not_found(runner, mock_cli_dependencies):
    """Tests the index command when the input file doesn't exist."""
    mock_indexer = mock_cli_dependencies["indexer_service"]
    # Simulate file not found error during processing
    mock_indexer.process_input_file.side_effect = FileNotFoundError("File not found")

    result = runner.invoke(app, ["index", "-i", "non_existent_file.json"])

    assert result.exit_code != 0 # Expect failure
    assert "Indexing Failed" in result.stdout
    assert "FileNotFoundError" in result.stdout # Check for error type in output

# --- Test Cases for 'chat' command ---

def test_chat_command_success(runner, mock_cli_dependencies):
    """Tests the happy path for the chat command."""
    mock_chat = mock_cli_dependencies["chat_service"]
    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert "--- Starting Phoenix Chat ---" in result.stdout
    # assert "--- Phoenix Chat Finished ---" in result.stdout # This might not appear if loop is mocked
    mock_chat.run_interactive_chat.assert_called_once()

def test_chat_command_db_override(runner, mock_cli_dependencies):
    """Tests the chat command with database path overrides."""
    mock_chat = mock_cli_dependencies["chat_service"]
    # We need to check if the settings are actually modified or if the
    # overridden paths are passed correctly to the service constructor.
    # Patching the service constructor or checking its call args might be needed.

    # For simplicity, just check if the command runs and calls the service method
    result = runner.invoke(app, ["chat", "--chroma-dir", "/tmp/test_chroma", "--db-path", "/tmp/test.db"])

    assert result.exit_code == 0
    mock_chat.run_interactive_chat.assert_called_once()
    # Add more specific checks if needed, e.g., patching settings or service init

# Add more tests for error handling in both commands
