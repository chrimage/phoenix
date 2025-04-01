# phoenix/adapters/llm/google_genai.py - Adapter for Google Generative AI API
import google.generativeai as genai
from google.generativeai import types as genai_types # Alias for most types
# Import common Google API core exceptions for retry logic
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Optional, Dict, Any # Use Any for history type hint

from phoenix.config import settings

# Configure the library globally on import
try:
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    print(f"Google GenAI configured with API key.")
except Exception as e:
    print(f"Error configuring Google GenAI: {e}")
    # Depending on the application, might want to raise this or handle differently

# Define common retryable errors for the API using google.api_core.exceptions
RETRYABLE_ERRORS = (
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
    google_exceptions.ServiceUnavailable, # Often used for transient server issues
    google_exceptions.ResourceExhausted, # Typically indicates rate limiting
    google_exceptions.Aborted, # Can sometimes indicate a transient state
    # google_exceptions.Unknown, # Consider if needed, might retry too broadly
)

class GoogleGenAIAdapter:
    """Adapter for interacting with Google Generative AI models."""

    def __init__(self):
        # Models are initialized on demand to potentially use different models
        self._models: Dict[str, genai.GenerativeModel] = {}
        print("GoogleGenAIAdapter initialized.")

    def _get_model(self, model_name: str) -> genai.GenerativeModel:
        """Gets or initializes a GenerativeModel instance."""
        if model_name not in self._models:
            print(f"Initializing Google GenAI model: {model_name}")
            try:
                self._models[model_name] = genai.GenerativeModel(model_name)
            except Exception as e:
                print(f"Failed to initialize model {model_name}: {e}")
                raise # Re-raise critical initialization error
        return self._models[model_name]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        reraise=True # Re-raise the exception if retries fail
    )
    def generate_text(self,
                      prompt: str,
                      model_name: str,
                      temperature: float = 0.7,
                      max_output_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      generation_config_override: Optional[genai_types.GenerationConfig] = None) -> str:
        """
        Generates text using a specified Google GenAI model with retry logic.

        Args:
            prompt: The input prompt.
            model_name: The name of the model to use (e.g., settings.SUMMARY_MODEL).
            temperature: The generation temperature.
            max_output_tokens: Maximum tokens for the response.
            stop_sequences: Optional sequences to stop generation.
            generation_config_override: Optional pre-configured GenerationConfig object.

        Returns:
            The generated text content.

        Raises:
            Various google.api_core.exceptions on API errors after retries.
            Exception: For other unexpected errors.
        """
        try:
            model = self._get_model(model_name)
            # Use override if provided, otherwise create config
            generation_config = generation_config_override or genai_types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stop_sequences=stop_sequences
            )

            # print(f"Sending prompt to {model_name} (first 50 chars): {prompt[:50]}...")
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            # print(f"Received response from {model_name}.")

            # Check for valid response text
            if hasattr(response, 'text') and response.text:
                 return response.text.strip()
            else:
                 # Handle cases where response might be blocked or empty without raising specific exceptions initially
                 # Check finish_reason if available
                 finish_reason = "UNKNOWN"
                 safety_ratings = []
                 if hasattr(response, 'candidates') and response.candidates:
                     candidate = response.candidates[0]
                     # Access finish_reason safely, checking if it exists and has a name attribute
                     finish_reason_enum = getattr(candidate, 'finish_reason', None)
                     if finish_reason_enum is not None:
                         finish_reason = getattr(finish_reason_enum, 'name', "UNKNOWN")
                     safety_ratings = getattr(candidate, 'safety_ratings', [])

                 print(f"Warning: Empty response text received from {model_name}. Finish Reason: {finish_reason}")
                 # Optionally log safety ratings if needed
                 # if safety_ratings: print(f"Safety Ratings: {safety_ratings}")
                 return f"[Warning: Empty response received. Finish Reason: {finish_reason}]"


        except (genai_types.BlockedPromptException, genai_types.StopCandidateException) as e:
             # Specific non-retryable exceptions from the library if they exist (check library docs)
             print(f"Non-retryable Gemini API Error ({type(e).__name__}) for model {model_name}: {e}")
             return f"[Error generating response due to API issue: {type(e).__name__}]"
        except google_exceptions.InvalidArgument as e:
            # InvalidArgument is often non-retryable
            print(f"Non-retryable Invalid Argument Error for model {model_name}: {e}")
            return f"[Error generating response due to Invalid Argument: {e}]"
        except RETRYABLE_ERRORS as e:
             # This block might not be strictly necessary due to tenacity handling,
             # but can be useful for logging specific retry attempts if needed.
             print(f"Retryable Google API Error ({type(e).__name__}) for model {model_name}. Retrying...")
             raise # Re-raise for tenacity to catch
        except Exception as e:
            print(f"Unexpected error during text generation with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return error string for unexpected issues
            return "[Unexpected error generating response]"

    def start_chat(self, model_name: str, history: Optional[List[Any]] = None): # Use Any for history type hint
        """
        Starts a new chat session with the specified model.

        Args:
            model_name: The chat model to use (e.g., settings.CHAT_MODEL).
            history: Optional initial chat history (list of Content-like objects/dicts).

        Returns:
            A google.generativeai.ChatSession object.
        """
        model = self._get_model(model_name)
        print(f"Starting chat session with model: {model_name}")
        # The library handles conversion of dicts/Content objects internally for history
        return model.start_chat(history=history or [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        reraise=True
    )
    def send_chat_message(self,
                          chat_session, # Should be a genai.ChatSession object
                          prompt: str,
                          temperature: float = settings.CHAT_TEMPERATURE,
                          max_output_tokens: Optional[int] = None,
                          generation_config_override: Optional[genai_types.GenerationConfig] = None) -> str:
        """
        Sends a message within an existing chat session with retry logic.

        Args:
            chat_session: The active ChatSession object.
            prompt: The user's message/prompt.
            temperature: The generation temperature for this turn.
            max_output_tokens: Optional max tokens for this response.
            generation_config_override: Optional pre-configured GenerationConfig object.

        Returns:
            The AI's response text.
        """
        try:
            # Use override if provided, otherwise create config
            generation_config = generation_config_override or genai_types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            # print(f"Sending chat message (first 50 chars): {prompt[:50]}...")
            response = chat_session.send_message(prompt, generation_config=generation_config)
            # print("Received chat response.")

            # Check for valid response text
            if hasattr(response, 'text') and response.text:
                 return response.text.strip()
            else:
                 finish_reason = "UNKNOWN"
                 safety_ratings = []
                 if hasattr(response, 'candidates') and response.candidates:
                     candidate = response.candidates[0]
                     # Access finish_reason safely, checking if it exists and has a name attribute
                     finish_reason_enum = getattr(candidate, 'finish_reason', None)
                     if finish_reason_enum is not None:
                         finish_reason = getattr(finish_reason_enum, 'name', "UNKNOWN")
                     safety_ratings = getattr(candidate, 'safety_ratings', [])
                 print(f"Warning: Empty chat response text received. Finish Reason: {finish_reason}")
                 return f"[Warning: Empty response received. Finish Reason: {finish_reason}]"

        except (genai_types.BlockedPromptException, genai_types.StopCandidateException) as e:
             print(f"Non-retryable Gemini API Error ({type(e).__name__}) during chat: {e}")
             return f"[Error generating response due to API issue: {type(e).__name__}]"
        except google_exceptions.InvalidArgument as e:
             print(f"Non-retryable Invalid Argument Error during chat: {e}")
             return f"[Error generating response due to Invalid Argument: {e}]"
        except RETRYABLE_ERRORS as e:
             print(f"Retryable Google API Error ({type(e).__name__}) during chat. Retrying...")
             raise # Re-raise for tenacity
        except Exception as e:
            print(f"Unexpected error sending chat message: {e}")
            import traceback
            traceback.print_exc()
            return "[Unexpected error generating response]"

# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing GoogleGenAIAdapter...")

    if not settings.GOOGLE_API_KEY:
        print("Skipping adapter tests: GOOGLE_API_KEY not set.")
    else:
        adapter = GoogleGenAIAdapter()

        # Test text generation (using flash model for speed/cost)
        print("\nTesting text generation...")
        try:
            summary_prompt = "Summarize the concept of a vector database in one sentence."
            summary = adapter.generate_text(
                prompt=summary_prompt,
                model_name=settings.SUMMARY_MODEL, # Use flash model
                temperature=0.2,
                max_output_tokens=100
            )
            print(f"Generated Summary: {summary}")
            assert isinstance(summary, str) and len(summary) > 10 and not summary.startswith("[")
            print("Text generation test PASSED")
        except Exception as e:
            print(f"Text generation test FAILED: {e}")

        # Test chat
        print("\nTesting chat session...")
        try:
            # Example history (list of dicts, library handles conversion)
            initial_history = [
                {'role': 'user', 'parts': [{'text': 'Briefly, what is Python?'}]},
                {'role': 'model', 'parts': [{'text': 'Python is a popular programming language.'}]}
            ]
            chat_session = adapter.start_chat(model_name=settings.CHAT_MODEL, history=initial_history)
            print("Chat session started with history.")

            response1 = adapter.send_chat_message(chat_session, "What is its main advantage?")
            print(f"Chat Response 1: {response1}")
            assert isinstance(response1, str) and len(response1) > 0 and not response1.startswith("[")

            response2 = adapter.send_chat_message(chat_session, "What is the capital of France?")
            print(f"Chat Response 2: {response2}")
            assert isinstance(response2, str) and "Paris" in response2 and not response2.startswith("[")

            print("Chat session test PASSED")
            print("\nFinal Chat History:")
            # Access history from the session object
            if hasattr(chat_session, 'history'):
                for message in chat_session.history:
                     # Accessing parts directly assuming simple text parts for example
                     if hasattr(message, 'parts') and message.parts:
                         # Check if parts[0] has a text attribute
                         part_text = getattr(message.parts[0], 'text', '[No Text Part]')
                         print(f"- {message.role}: {part_text[:80]}...")
                     else:
                         print(f"- {message.role}: [No Parts]")
            else:
                print("Could not retrieve chat history.")


        except Exception as e:
            print(f"Chat session test FAILED: {e}")

    print("\nGoogleGenAIAdapter testing finished.")
