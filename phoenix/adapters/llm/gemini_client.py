# phoenix/adapters/llm/gemini_client.py - Adapter for Google Generative AI interactions
import google.generativeai as genai
from google.generativeai import types as genai_types # Alias to avoid conflict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Any, Optional
# Import common Google API core exceptions for retry logic
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    # Fallback if google-api-core is not directly available or used differently
    google_exceptions = None
    print("⚠️ Warning: google.api_core.exceptions not found. Retry logic might be less specific.")

from phoenix.config import settings
from phoenix.core.models import Message # Assuming Message model might be useful for history

# --- Constants for Retrying ---
RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=60)
RETRY_ATTEMPTS = 5
# Define specific API errors that are often transient and worth retrying
# Use exceptions from google.api_core if available
if google_exceptions:
    RETRYABLE_ERRORS = (
        google_exceptions.ServiceUnavailable, # 503
        google_exceptions.ResourceExhausted, # 429
        google_exceptions.DeadlineExceeded,  # 504
        google_exceptions.InternalServerError, # 500 (less common to retry, but possible)
        # google.auth.exceptions.TransportError, # If auth issues are transient
    )
else:
    # Fallback to broader exception types if specific ones aren't available
    # This is less ideal as it might retry non-transient errors
    RETRYABLE_ERRORS = (Exception,) # Retry on generic Exception as a last resort

class GeminiClient:
    """Wraps Google Generative AI API interactions."""

    def __init__(self):
        """Initializes the Gemini client and configures the API."""
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GeminiClient requires GOOGLE_API_KEY to be set.")

        try:
            # Use alias for genai types if needed elsewhere, though configure doesn't use it
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            print("Google Generative AI SDK configured.")
            # We initialize models on demand in specific methods
        except Exception as e:
            print(f"❌ Error configuring Google Generative AI SDK: {e}")
            raise

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT,
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        reraise=True # Re-raise the exception if all retries fail
    )
    def generate_text(
        self,
        prompt: str,
        model_name: str = settings.SUMMARY_MODEL_NAME, # Default to summary model
        max_output_tokens: int = 500,
        temperature: float = 0.2 # Default to lower temp for factual tasks
    ) -> str:
        """
        Generates text using a specified Gemini model (e.g., for summaries, insights).

        Args:
            prompt: The input prompt text.
            model_name: The name of the Gemini model to use.
            max_output_tokens: Maximum tokens for the generated response.
            temperature: The generation temperature.

        Returns:
            The generated text content, or an empty string if an error occurs
            that isn't handled by retry, or if the response is blocked.
        """
        print(f"Generating text with model: {model_name} (Temp: {temperature})")
        try:
            model = genai.GenerativeModel(model_name)
            # Use alias for genai types
            generation_config = genai_types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            # Check for valid response parts before accessing text
            if response.parts:
                return response.text.strip()
            else:
                # Handle cases where the response might be blocked or empty
                print(f"⚠️ Warning: Received empty or blocked response from {model_name}.")
                # Log blocking reason if available
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     print(f"   Block Reason: {response.prompt_feedback.block_reason.name}")
                return "" # Return empty string for blocked/empty responses
        except (genai_types.StopCandidateError, genai_types.BlockedPromptError, genai_types.InvalidcontentError) as e:
            # Handle specific non-retryable errors from genai library
            print(f"❌ Gemini API Error (Non-Retryable): {type(e).__name__} - {e}")
            return f"[Error generating text due to API issue: {type(e).__name__}]"
        except Exception as e:
            # Catch other unexpected errors during generation
            print(f"❌ Unexpected error during text generation with {model_name}: {e}")
            # This will trigger tenacity retry if it's a RETRYABLE_ERROR
            raise # Re-raise to allow tenacity to handle retries

    # --- Chat Functionality ---

    def start_chat_session(
        self,
        model_name: str = settings.CHAT_MODEL,
        history: Optional[List[Dict[str, Any]]] = None # Allow starting with history
    ) -> genai.ChatSession:
        """
        Starts a new chat session with the specified model and optional history.

        Args:
            model_name: The chat model to use (e.g., settings.CHAT_MODEL).
            history: Optional list of previous messages in the format expected by genai.

        Returns:
            An initialized genai.ChatSession object.
        """
        print(f"Starting chat session with model: {model_name}")
        model = genai.GenerativeModel(model_name)
        # Ensure history is in the correct format if provided
        formatted_history = history or []
        chat_session = model.start_chat(history=formatted_history)
        return chat_session

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT,
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        reraise=True
    )
    def send_chat_message(
        self,
        chat_session: genai.ChatSession,
        prompt: str,
        temperature: float = settings.CHAT_TEMPERATURE
    ) -> str:
        """
        Sends a message within an existing chat session and returns the response.

        Args:
            chat_session: The active genai.ChatSession object.
            prompt: The user's message/prompt to send.
            temperature: The generation temperature for the chat response.

        Returns:
            The AI's response text, or an error message string.
        """
        print(f"Sending chat message (Temp: {temperature})...")
        try:
            # Define generation config for the chat turn using alias
            chat_config = genai_types.GenerationConfig(
                temperature=temperature
                # max_output_tokens can also be set here if needed
            )
            # Send the message using the session
            response = chat_session.send_message(prompt, generation_config=chat_config)

            if response.parts:
                return response.text.strip()
            else:
                print("⚠️ Warning: Received empty or blocked chat response.")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     print(f"   Block Reason: {response.prompt_feedback.block_reason.name}")
                return "[AI response was empty or blocked]"
        except (genai_types.StopCandidateError, genai_types.BlockedPromptError, genai_types.InvalidcontentError) as e:
            # Handle specific non-retryable errors from genai library
            print(f"❌ Gemini API Error (Non-Retryable) during chat: {type(e).__name__} - {e}")
            # Update chat history with an error message? Or just return error?
            # chat_session.history.append(genai_types.Content(role="model", parts=[genai_types.Part(text=f"[Error: {type(e).__name__}]")]))
            return f"[Error generating response due to API issue: {type(e).__name__}]"
        except Exception as e:
            print(f"❌ Unexpected error sending chat message: {e}")
            raise # Re-raise for tenacity retry

# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("Testing GeminiClient...")
    try:
        client = GeminiClient()

        # Test text generation
        print("\n--- Testing Text Generation ---")
        summary_prompt = "Summarize the concept of Large Language Models in one sentence."
        summary = client.generate_text(summary_prompt, max_output_tokens=100)
        print(f"Generated Summary: {summary}")

        # Test chat
        print("\n--- Testing Chat ---")
        session = client.start_chat_session()
        print("Chat session started.")

        response1 = client.send_chat_message(session, "Hello there!")
        print(f"Phoenix: {response1}")

        response2 = client.send_chat_message(session, "What is the capital of France?")
        print(f"Phoenix: {response2}")

        print("\nChat History:")
        for message in session.history:
             # Ensure parts exist and have text before printing
             text_content = message.parts[0].text if message.parts and message.parts[0].text else "[Empty Content]"
             print(f"- {message.role.capitalize()}: {text_content}")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")
