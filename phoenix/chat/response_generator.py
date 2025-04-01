# phoenix/chat/response_generator.py - Generates chat responses using LLM
from typing import Optional, Any # Any for chat_session type hint flexibility

from phoenix.config import settings
from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter

def generate_chat_response(
    chat_session: Any, # Expects a google.generativeai.ChatSession object
    prompt: str,
    llm_adapter: GoogleGenAIAdapter,
    temperature: float = settings.CHAT_TEMPERATURE,
    max_output_tokens: Optional[int] = None # Let adapter handle default if None
) -> str:
    """
    Generates a chat response using the provided LLM adapter and chat session.

    Args:
        chat_session: The active chat session object (e.g., from adapter.start_chat).
        prompt: The final prompt string (including context and user query) to send.
        llm_adapter: An instance of the GoogleGenAIAdapter.
        temperature: The generation temperature for this turn.
        max_output_tokens: Optional maximum tokens for the response.

    Returns:
        The generated response text from the AI.
    """
    if not chat_session:
        print("Error: Chat session is not initialized.")
        return "[Error: Chat session not available]"
    if not prompt:
        print("Warning: Empty prompt received for chat response generation.")
        return "[Error: Empty prompt]"

    print("[Generating response...]")
    try:
        # Use the adapter to send the message within the session
        response_text = llm_adapter.send_chat_message(
            chat_session=chat_session,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        return response_text
    except Exception as e:
        # Errors should be handled within the adapter's send_chat_message,
        # but catch unexpected issues here too.
        print(f"Unexpected error during chat response generation: {e}")
        import traceback
        traceback.print_exc()
        return "[Unexpected error generating response]"

# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing Response Generator...")

    # Mock LLM Adapter and Chat Session
    class MockChatSession:
        def __init__(self):
            self.history = []
        def send_message(self, prompt, generation_config):
            print(f"Mock send_message called with prompt: '{prompt[:50]}...'")
            self.history.append({"role": "user", "parts": [{"text": prompt}]})
            response_text = f"Mock response to: {prompt[:30]}..."
            self.history.append({"role": "model", "parts": [{"text": response_text}]})
            # Mock the response object structure expected by the adapter
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            return MockResponse(response_text)

    class MockLLMAdapter:
        def start_chat(self, model_name, history=None):
            print(f"Mock start_chat called for model {model_name}")
            return MockChatSession()

        def send_chat_message(self, chat_session, prompt, temperature, max_output_tokens):
             # Simulate calling the session's method
             mock_response = chat_session.send_message(prompt, generation_config=None)
             return mock_response.text # Adapter extracts text

    mock_adapter = MockLLMAdapter()
    mock_session = mock_adapter.start_chat(settings.CHAT_MODEL)

    # Test generating a response
    print("\nTesting response generation...")
    test_prompt = "This is the assembled context and user query."
    response = generate_chat_response(mock_session, test_prompt, mock_adapter)

    print(f"Generated Response: {response}")
    assert "Mock response to: This is the assembled context" in response
    print("Response generation test PASSED")

    # Test with empty prompt
    print("\nTesting empty prompt...")
    response_empty = generate_chat_response(mock_session, "", mock_adapter)
    print(f"Generated Response (Empty Prompt): {response_empty}")
    assert "[Error: Empty prompt]" in response_empty
    print("Empty prompt test PASSED")

    # Test with no session
    print("\nTesting no session...")
    response_no_session = generate_chat_response(None, test_prompt, mock_adapter)
    print(f"Generated Response (No Session): {response_no_session}")
    assert "[Error: Chat session not available]" in response_no_session
    print("No session test PASSED")


    print("\nResponse Generator testing finished.")
