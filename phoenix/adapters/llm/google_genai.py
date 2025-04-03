# phoenix/adapters/llm/google_genai.py - Adapter for Google Generative AI API (using google-genai SDK)
from google import genai
from google.genai import types
from google.genai import errors as genai_errors # Import SDK-specific errors
# Import common Google API core exceptions for retry logic (still relevant for transport)
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Optional, Dict, Any # Use Any for history type hint in example

from phoenix.config import settings

# Define common retryable errors for the API using google.api_core.exceptions
# These cover lower-level transport/server issues.
RETRYABLE_API_CORE_ERRORS = (
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError, # 500
    google_exceptions.ServiceUnavailable, # 503
    google_exceptions.ResourceExhausted, # 429 (Rate Limiting)
    google_exceptions.Aborted, # Can sometimes indicate a transient state
    # google_exceptions.Unknown, # Generally avoid retrying Unknown
)
# Add specific genai errors if needed for retry, though API core covers most transient ones.
# Example: genai_errors.ResourceExhaustedError might also be caught by google_exceptions.ResourceExhausted
from typing import Any # Import Any for type hinting


class GoogleGenAIAdapter:
    """Adapter for interacting with Google AI models using the google-genai SDK."""

    def __init__(self):
        """Initializes the Google AI Client."""
        try:
            # Client uses GOOGLE_API_KEY environment variable by default
            self.client = genai.Client()
            print("GoogleGenAIAdapter initialized with genai.Client().")
            # Perform a lightweight check, like listing models, to ensure connectivity
            # self.list_models() # Optional: uncomment to test connection on init
        except Exception as e:
            print(f"Error initializing Google GenAI Client: {e}")
            print("Ensure the GOOGLE_API_KEY environment variable is set correctly.")
            # Depending on the application, might want to raise this
            self.client = None # Indicate initialization failure

    def list_models(self):
        """Lists available models (useful for testing configuration)."""
        if not self.client:
            print("Client not initialized.")
            return []
        try:
            print("Listing available Google AI models...")
            models = [m.name for m in self.client.models.list()]
            print(f"Found models: {models}")
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    # Note: _get_model is removed as client.models/client.chats handles model interaction directly.

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_API_CORE_ERRORS), # Retry on API core transport errors
        reraise=True # Re-raise the exception if retries fail
    )
    def generate_text(self,
                      prompt: str,
                      model_name: str,
                      temperature: float = 0.7,
                      max_output_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      generation_config_override: Optional[types.GenerationConfig] = None) -> str:
        """
        Generates text using a specified Google AI model with retry logic.

        Args:
            prompt: The input prompt (will be converted to `types.Content`).
            model_name: The name of the model to use (e.g., settings.SUMMARY_MODEL).
            temperature: The generation temperature.
            max_output_tokens: Maximum tokens for the response.
            stop_sequences: Optional sequences to stop generation.
            generation_config_override: Optional pre-configured GenerationConfig object.

        Returns:
            The generated text content, or an error/warning message.

        Raises:
            google.api_core.exceptions.*: On API errors after retries fail.
            Exception: For other unexpected errors during generation.
        """
        if not self.client:
            return "[Error: GoogleGenAIAdapter client not initialized]"

        try:
            # Pass config parameters directly to avoid GenerationConfig object issues
            generation_params = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "stop_sequences": stop_sequences,
            }
            # Use override if provided, otherwise use individual params
            effective_config = generation_config_override if generation_config_override else generation_params

            # print(f"Sending prompt to {model_name} (first 50 chars): {prompt[:50]}...")
            # Use client.models.generate_content
            response = self.client.models.generate_content(
                model=f'models/{model_name}', # Model name needs 'models/' prefix usually
                contents=prompt, # SDK converts str to list[Content] automatically
                **effective_config # Unpack config dict as keyword arguments
            )
            # print(f"Received response from {model_name}.")

            # Enhanced response handling based on guide examples
            try:
                # Access text directly if available and not blocked
                return response.text.strip()
            except ValueError:
                # Handle cases where accessing .text raises ValueError (e.g., blocked content)
                finish_reason = "UNKNOWN"
                safety_ratings_str = "N/A"
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason_enum = getattr(candidate, 'finish_reason', None)
                    if finish_reason_enum is not None:
                        finish_reason = getattr(finish_reason_enum, 'name', "UNKNOWN")
                    safety_ratings = getattr(candidate, 'safety_ratings', [])
                    safety_ratings_str = str(safety_ratings) if safety_ratings else "None"

                print(f"Warning: Could not access response text from {model_name}. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings_str}")
                # Check prompt feedback if available
                prompt_feedback_str = "N/A"
                if hasattr(response, 'prompt_feedback'):
                    prompt_feedback_str = str(response.prompt_feedback)
                print(f"Prompt Feedback: {prompt_feedback_str}")

                return f"[Warning: Empty or blocked response. Finish Reason: {finish_reason}]"
            except Exception as text_access_error:
                # Catch any other unexpected error during text access
                 print(f"Error accessing response text from {model_name}: {text_access_error}")
                 return f"[Error: Could not process response content. Reason: {text_access_error}]"

        # Removed except genai_errors.GoogleAPIError as it causes AttributeError in current library version
        except google_exceptions.InvalidArgument as e:
            # InvalidArgument is often non-retryable (e.g., bad API key, malformed request)
            print(f"Non-retryable Invalid Argument Error for model {model_name}: {e}")
            return f"[Error generating response due to Invalid Argument: {e}]"
        except RETRYABLE_API_CORE_ERRORS as e:
             # This block might not be strictly necessary due to tenacity handling,
             # but can be useful for logging specific retry attempts if needed.
             print(f"Retryable Google API Core Error ({type(e).__name__}) for model {model_name}. Retrying...")
             raise # Re-raise for tenacity to catch
        except Exception as e:
            print(f"Unexpected error during text generation with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return "[Unexpected error generating response]"

    def start_chat(self, model_name: str) -> Optional[Any]: # Changed type hint
        """
        Starts a new chat session using the specified model via `client.chats.create`.
        History is managed internally by the returned ChatSession object.

        Args:
            model_name: The chat model to use (e.g., settings.CHAT_MODEL).

        Returns:
            A google.genai.ChatSession object, or None if client is not initialized.
        """
        if not self.client:
            print("Error: GoogleGenAIAdapter client not initialized")
            return None

        print(f"Starting chat session with model: models/{model_name}")
        try:
            # Use client.chats.create
            # Model name needs 'models/' prefix usually
            chat_session = self.client.chats.create(model=f'models/{model_name}')
            # History is implicitly managed by the chat_session object
            return chat_session
        except Exception as e:
            print(f"Error starting chat session with {model_name}: {e}")
            return None # Return None on error

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_API_CORE_ERRORS), # Retry on API core transport errors
        reraise=True
    )
    def send_chat_message(self,
                          chat_session: Any, # Changed type hint
                          prompt: str,
                          temperature: float = settings.CHAT_TEMPERATURE,
                          max_output_tokens: Optional[int] = None,
                          generation_config_override: Optional[types.GenerationConfig] = None) -> str:
        """
        Sends a message within an existing chat session with retry logic.

        Args:
            chat_session: The active google.genai.ChatSession object.
            prompt: The user's message/prompt.
            temperature: The generation temperature for this turn.
            max_output_tokens: Optional max tokens for this response.
            generation_config_override: Optional pre-configured GenerationConfig object.

        Returns:
            The AI's response text, or an error/warning message.

        Raises:
            google.api_core.exceptions.*: On API errors after retries fail.
            AttributeError: If chat_session is None.
            Exception: For other unexpected errors during chat.
        """
        if not chat_session:
             return "[Error: Invalid chat session provided]"
            # Removed isinstance check for genai.ChatSession as it causes AttributeError

        try:
            # Pass config parameters directly to avoid GenerationConfig object issues
            generation_params = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                # Add safety_settings if needed directly here
            }
            # Use override if provided, otherwise use individual params
            effective_config = generation_config_override if generation_config_override else generation_params

            # print(f"Sending chat message (first 50 chars): {prompt[:50]}...")
            # Use the send_message method of the ChatSession object
            # Removed config arguments as they cause TypeError in this library version
            response = chat_session.send_message(prompt)
            # print("Received chat response.")

            # Enhanced response handling similar to generate_text
            try:
                return response.text.strip()
            except ValueError:
                finish_reason = "UNKNOWN"
                safety_ratings_str = "N/A"
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason_enum = getattr(candidate, 'finish_reason', None)
                    if finish_reason_enum is not None:
                        finish_reason = getattr(finish_reason_enum, 'name', "UNKNOWN")
                    safety_ratings = getattr(candidate, 'safety_ratings', [])
                    safety_ratings_str = str(safety_ratings) if safety_ratings else "None"
                print(f"Warning: Could not access chat response text. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings_str}")
                prompt_feedback_str = "N/A"
                if hasattr(response, 'prompt_feedback'):
                    prompt_feedback_str = str(response.prompt_feedback)
                print(f"Prompt Feedback: {prompt_feedback_str}")
                return f"[Warning: Empty or blocked chat response. Finish Reason: {finish_reason}]"
            except Exception as text_access_error:
                 print(f"Error accessing chat response text: {text_access_error}")
                 return f"[Error: Could not process chat response content. Reason: {text_access_error}]"

        # Removed except genai_errors.GoogleAPIError as it causes AttributeError in current library version
        except google_exceptions.InvalidArgument as e:
             print(f"Non-retryable Invalid Argument Error during chat: {e}")
             return f"[Error generating response due to Invalid Argument: {e}]"
        except RETRYABLE_API_CORE_ERRORS as e:
             print(f"Retryable Google API Core Error ({type(e).__name__}) during chat. Retrying...")
             raise # Re-raise for tenacity
        except Exception as e:
            print(f"Unexpected error sending chat message: {e}")
            import traceback
            traceback.print_exc()
            return "[Unexpected error generating response]"

# Example Usage (for testing - updated for new SDK patterns)
if __name__ == "__main__":
    print("Testing GoogleGenAIAdapter (using google-genai SDK)...")

    # Check if the API key is likely configured (Client init relies on env var)
    if not settings.GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not found in settings. Assuming it's set as an environment variable for genai.Client().")
        # Allow proceeding, as Client() might still work if env var is set

    adapter = GoogleGenAIAdapter()

    if not adapter.client:
        print("Skipping adapter tests: Client initialization failed.")
    else:
        # Optional: Test model listing
        # adapter.list_models()

        # Test text generation (using flash model for speed/cost)
        print("\nTesting text generation...")
        try:
            # Use a model name expected by the new SDK, e.g., 'gemini-1.5-flash' or 'gemini-2.0-flash'
            # Ensure settings.SUMMARY_MODEL is updated accordingly if needed.
            # Defaulting to a known good model if setting isn't specific.
            summary_model = settings.SUMMARY_MODEL or "gemini-2.0-flash"
            print(f"Using summary model: {summary_model}")

            summary_prompt = "Summarize the concept of a vector database in one sentence."
            summary = adapter.generate_text(
                prompt=summary_prompt,
                model_name=summary_model,
                temperature=0.2,
                max_output_tokens=100
            )
            print(f"Generated Summary: {summary}")
            assert isinstance(summary, str) and len(summary) > 10 and not summary.startswith("[")
            print("Text generation test PASSED")
        except Exception as e:
            print(f"Text generation test FAILED: {e}")
            import traceback
            traceback.print_exc()


        # Test chat
        print("\nTesting chat session...")
        try:
            # Use a model name expected by the new SDK, e.g., 'gemini-1.5-flash' or 'gemini-2.0-flash'
            # Ensure settings.CHAT_MODEL is updated accordingly if needed.
            chat_model = settings.CHAT_MODEL or "gemini-2.0-flash"
            print(f"Using chat model: {chat_model}")

            # Start chat session - history is managed internally now
            chat_session = adapter.start_chat(model_name=chat_model)

            if not chat_session:
                 raise Exception("Failed to start chat session.")

            print("Chat session started.")

            # Send initial messages to build history if needed (example)
            print("Sending initial message...")
            initial_response = adapter.send_chat_message(chat_session, "Briefly, what is Python?")
            print(f"Initial Response: {initial_response}")
            assert isinstance(initial_response, str) and len(initial_response) > 0 and not initial_response.startswith("[")

            print("Sending second message...")
            response1 = adapter.send_chat_message(chat_session, "What is its main advantage?")
            print(f"Chat Response 1: {response1}")
            assert isinstance(response1, str) and len(response1) > 0 and not response1.startswith("[")

            print("Sending third message...")
            response2 = adapter.send_chat_message(chat_session, "What is the capital of France?")
            print(f"Chat Response 2: {response2}")
            assert isinstance(response2, str) and "Paris" in response2 and not response2.startswith("[")

            print("Chat session test PASSED")
            print("\nFinal Chat History (from session object):")
            # Access history from the session object
            if hasattr(chat_session, 'history'):
                for message in chat_session.history:
                     # Accessing parts safely
                     part_text = "[No Text Part]"
                     if hasattr(message, 'parts') and message.parts:
                         # Assuming simple text parts for this example
                         first_part = message.parts[0]
                         if hasattr(first_part, 'text'):
                             part_text = first_part.text
                         elif hasattr(first_part, 'data'): # Handle potential inline data/other types
                             part_text = f"[Data Part: {first_part.mime_type}]"
                         # Add more checks for other Part types if necessary

                     print(f"- {message.role}: {part_text[:80]}...")
            else:
                print("Could not retrieve chat history from session object.")

        except Exception as e:
            print(f"Chat session test FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\nGoogleGenAIAdapter testing finished.")
