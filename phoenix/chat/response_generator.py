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
