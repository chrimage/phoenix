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
