# phoenix/indexing/summarizer.py - Conversation Summarization Logic
import uuid
import time
from typing import Optional

from phoenix.config import settings
from phoenix.core.models import Summary, ParsedConversation, ConversationMetadata, Message
from phoenix.core.utils import estimate_tokens
from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter

def generate_conversation_summary(
    parsed_conv: ParsedConversation,
    llm_adapter: GoogleGenAIAdapter,
    min_tokens_for_summary: int = settings.MIN_TOKENS_FOR_SUMMARY,
    target_summary_tokens: int = settings.TARGET_CONV_SUMMARY_TOKENS,
    summary_model_name: str = settings.SUMMARY_MODEL
) -> Optional[Summary]:
    """
    Generates a summary for a conversation if it meets the minimum token threshold.

    Args:
        parsed_conv: The ParsedConversation object.
        llm_adapter: An instance of the GoogleGenAIAdapter.
        min_tokens_for_summary: Minimum total tokens in the conversation to trigger summary generation.
        target_summary_tokens: The desired approximate token count for the generated summary.
        summary_model_name: The name of the LLM model to use for summarization.

    Returns:
        A Summary object if a summary was generated, None otherwise.
    """
    # Combine all messages into a single text for summarization
    full_conv_text = "\n".join([f"{msg.role}: {msg.content}" for msg in parsed_conv.messages])
    conv_tokens = estimate_tokens(full_conv_text)

    if conv_tokens <= min_tokens_for_summary:
        # print(f"Conversation {parsed_conv.metadata.conv_id} below token threshold ({conv_tokens} <= {min_tokens_for_summary}). Skipping summary.")
        return None

    print(f"  - Conversation {parsed_conv.metadata.conv_id} exceeds token threshold ({conv_tokens} > {min_tokens_for_summary}). Generating summary...")
    summary_prompt = f"Summarize the key topics, decisions, and outcomes of the following conversation concisely (target ~{target_summary_tokens} tokens):\n\n---\n{full_conv_text}\n---\n\nSummary:"

    try:
        # Use the adapter to generate the summary
        summary_text = llm_adapter.generate_text(
            prompt=summary_prompt,
            model_name=summary_model_name,
            temperature=0.2, # Lower temperature for factual summaries
            max_output_tokens=int(target_summary_tokens * 1.5) # Allow some buffer
        )

        if summary_text and not summary_text.startswith("[Error"):
            summary_token_count = estimate_tokens(summary_text)
            print(f"  - Generated conversation summary ({summary_token_count} tokens)")

            summary_doc = Summary(
                doc_id=f"summary_{parsed_conv.metadata.conv_id}",
                text=summary_text,
                original_conv_id=parsed_conv.metadata.conv_id,
                summary_token_count=summary_token_count,
                create_time=parsed_conv.metadata.create_time, # Use conversation creation time
                model_slug=parsed_conv.metadata.model_slug, # Inherit model from conv
                title=parsed_conv.metadata.title
                # Metadata will be added by ChromaStore using update_metadata
            )
            return summary_doc
        else:
            print(f"  - Summary generation failed or returned an error message: {summary_text}")
            return None
    except Exception as e:
        print(f"  - ERROR generating conversation summary for {parsed_conv.metadata.conv_id}: {e}")
        return None

# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing Summarizer...")

    # Mock LLM Adapter
    class MockLLMAdapter:
        def generate_text(self, prompt, model_name, temperature, max_output_tokens):
            print(f"Mock generate_text called for model {model_name}")
            if "Summarize" in prompt:
                return "This is a mock summary of the conversation about testing."
            return "[Mock Error]"

    mock_adapter = MockLLMAdapter()

    # Create dummy ParsedConversation data
    meta_long = ConversationMetadata(conv_id="test_summary_conv_long", title="Long Summary Test")
    messages_long = [Message(role="user", content="Word " * 500)] # Exceeds default min_tokens
    parsed_conv_long = ParsedConversation(metadata=meta_long, messages=messages_long)

    meta_short = ConversationMetadata(conv_id="test_summary_conv_short", title="Short Summary Test")
    messages_short = [Message(role="user", content="Word " * 50)] # Below default min_tokens
    parsed_conv_short = ParsedConversation(metadata=meta_short, messages=messages_short)

    # Test long conversation (should generate summary)
    print("\nTesting long conversation...")
    summary_long = generate_conversation_summary(parsed_conv_long, mock_adapter)
    if summary_long:
        print(f"Generated Summary (Long): {summary_long.text}")
        assert isinstance(summary_long, Summary)
        assert summary_long.text == "This is a mock summary of the conversation about testing."
        assert summary_long.original_conv_id == "test_summary_conv_long"
        print("Long conversation test PASSED")
    else:
        print("Long conversation test FAILED (No summary generated)")

    # Test short conversation (should skip summary)
    print("\nTesting short conversation...")
    summary_short = generate_conversation_summary(parsed_conv_short, mock_adapter)
    if summary_short is None:
        print("Short conversation test PASSED (Summary skipped as expected)")
    else:
        print(f"Short conversation test FAILED (Summary generated unexpectedly: {summary_short})")

    print("\nSummarizer testing finished.")
