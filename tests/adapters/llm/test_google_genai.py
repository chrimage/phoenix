# tests/adapters/llm/test_google_genai.py

import pytest
from unittest.mock import patch, MagicMock

# Assuming your adapter class is in phoenix.adapters.llm.google_genai
# Adjust the import path if necessary
from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter

# Example fixture to initialize the adapter (could also be in conftest.py if shared)
@pytest.fixture
def genai_adapter():
    """Provides an instance of the GoogleGenAIAdapter."""
    # You might need to mock environment variables or settings used during init
    with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_api_key'}):
         # Mock the underlying GenerativeModel if needed for initialization
         with patch('google.generativeai.GenerativeModel') as mock_model:
            adapter = GoogleGenAIAdapter()
            # Store the mock for later use in tests if needed
            adapter._mock_model_instance = MagicMock() # Example: mock internal model if accessed directly
            mock_model.return_value = adapter._mock_model_instance
            yield adapter

# --- Test Cases ---

def test_adapter_initialization(genai_adapter):
    """Tests if the adapter initializes correctly."""
    assert isinstance(genai_adapter, GoogleGenAIAdapter)
    # Add more specific initialization checks if needed

@pytest.mark.skip(reason="Requires mocking the API call")
def test_generate_text(genai_adapter, mocker):
    """Tests the text generation functionality (requires mocking)."""
    # Mock the specific method called on the underlying client/model
    mock_generate_content = mocker.patch.object(genai_adapter._mock_model_instance, 'generate_content')
    mock_generate_content.return_value = MagicMock(text="Mocked response text")

    prompt = "Test prompt"
    response = genai_adapter.generate_text(prompt)

    assert response == "Mocked response text"
    mock_generate_content.assert_called_once_with(prompt)

@pytest.mark.skip(reason="Requires mocking the API call")
def test_generate_embedding(genai_adapter, mocker):
    """Tests the embedding generation functionality (requires mocking)."""
    # Mock the embedding function
    mock_embed_content = mocker.patch('google.generativeai.embed_content')
    mock_embed_content.return_value = {'embedding': [0.1, 0.2, 0.3]}

    text_to_embed = "Some text to embed"
    embedding = genai_adapter.generate_embedding(text_to_embed)

    assert embedding == [0.1, 0.2, 0.3]
    mock_embed_content.assert_called_once_with(
        model=genai_adapter.embedding_model_name, # Use the actual model name from the adapter
        content=text_to_embed,
        task_type="retrieval_document" # Or whatever task type is appropriate
    )

# Add more tests for error handling, different parameters, etc.
