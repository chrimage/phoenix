# phoenix/chat/context_builder.py - Assembles context for chat prompts
from typing import List, Dict, Optional
from datetime import datetime
import time # Needed for synthesize_user_model example

from phoenix.config import settings
from phoenix.core.utils import estimate_tokens
from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter # Needed for synthesize_user_model

def assemble_chat_context(retrieved_chunks: List[Dict],
                          retrieved_insights: List[Dict],
                          max_tokens: int = settings.CONTEXT_TOKEN_LIMIT) -> str:
    """
    Formats retrieved chunks and insights into a context string within token limits.

    Args:
        retrieved_chunks: List of dictionaries representing relevant chunks.
        retrieved_insights: List of dictionaries representing relevant insights.
        max_tokens: The maximum estimated tokens allowed for the context string.

    Returns:
        A formatted context string, or "No relevant context found."
    """
    total_tokens = 0
    context_parts = []
    # Slightly bias token budget towards chunks as they contain more direct history
    max_tokens_insights = max_tokens // 3
    max_tokens_chunks = max_tokens - max_tokens_insights

    # --- Process Insight Notes First ---
    if retrieved_insights:
        insights_context = "--- USER MODEL AND CONVERSATION INSIGHTS ---\n\n"
        insights_tokens = estimate_tokens(insights_context)
        processed_insight_ids = set()

        for insight in retrieved_insights:
            insight_id = insight.get('insight_id')
            if not insight_id or insight_id in processed_insight_ids:
                continue

            insight_text = insight.get('insight_text', '')
            insight_token_count = estimate_tokens(insight_text)

            if insights_tokens + insight_token_count <= max_tokens_insights:
                insights_context += f"• {insight_text}\n" # Removed extra newline
                insights_tokens += insight_token_count
                processed_insight_ids.add(insight_id)
            else:
                break # Stop adding insights if token limit is reached

        # Only add the insights section if any insights were actually added
        if insights_tokens > estimate_tokens("--- USER MODEL AND CONVERSATION INSIGHTS ---\n\n"):
            context_parts.append(insights_context.strip()) # Strip trailing newline
            total_tokens += insights_tokens

    # --- Process Regular Chunks and Summaries ---
    if retrieved_chunks:
        chunks_context = "--- RELEVANT PREVIOUS CONVERSATION DETAILS ---\n\n"
        chunks_tokens = estimate_tokens(chunks_context)

        # Sort chunks - prioritize summaries first, then by similarity
        summaries = sorted(
            [c for c in retrieved_chunks if c.get('doc_type') == 'conversation_summary'],
            key=lambda x: x.get('similarity', 0.0), reverse=True
        )
        regular_chunks = sorted(
            [c for c in retrieved_chunks if c.get('doc_type') != 'conversation_summary'],
             key=lambda x: x.get('similarity', 0.0), reverse=True
        )

        processed_chunk_ids = set()

        # Process summaries first
        for summary in summaries:
            chunk_id = summary.get('chunk_id') or summary.get('summary_id') # Handle potential ID variations
            if not chunk_id or chunk_id in processed_chunk_ids: continue

            chunk_text = summary.get('chunk_text') or summary.get('text')
            if not chunk_text: continue

            # Use stored token count if available, otherwise estimate
            chunk_tokens = summary.get('summary_token_count') or estimate_tokens(chunk_text)
            conv_id_short = summary.get('original_conv_id', 'unknown')[:8]
            title = summary.get('title', 'No title')
            create_time_str = datetime.fromtimestamp(summary.get('create_time', 0)).strftime('%Y-%m-%d') if summary.get('create_time') else 'unknown date'

            if chunks_tokens + chunk_tokens <= max_tokens_chunks:
                chunks_context += f"[Summary of Conversation {conv_id_short} ({title}) from {create_time_str}]\n{chunk_text}\n\n"
                chunks_tokens += chunk_tokens
                processed_chunk_ids.add(chunk_id)
            else:
                break # Stop adding summaries if limit reached

        # Process regular chunks
        for chunk in regular_chunks:
            chunk_id = chunk.get('chunk_id')
            if not chunk_id or chunk_id in processed_chunk_ids: continue

            chunk_text = chunk.get('chunk_text') or chunk.get('text')
            if not chunk_text: continue

            chunk_tokens = chunk.get('token_count') or estimate_tokens(chunk_text)
            conv_id_short = chunk.get('conv_id', 'unknown')[:8]
            start_time_str = datetime.fromtimestamp(chunk.get('start_time', 0)).strftime('%Y-%m-%d %H:%M') if chunk.get('start_time') else 'unknown time'

            if chunks_tokens + chunk_tokens <= max_tokens_chunks:
                chunks_context += f"----\n[From Conversation {conv_id_short} around {start_time_str}]\n{chunk_text}\n\n"
                chunks_tokens += chunk_tokens
                processed_chunk_ids.add(chunk_id)
            else:
                # Try adding a truncated version if there's significant space left
                remaining_tokens = max_tokens_chunks - chunks_tokens
                if remaining_tokens > 50: # Need at least 50 tokens space
                    # Simple truncation based on character estimate (adjust multiplier as needed)
                    allowed_chars = int(remaining_tokens * 3.5)
                    truncated_text = chunk_text[:allowed_chars].strip() + "..."
                    truncated_tokens = estimate_tokens(truncated_text)
                    if truncated_tokens > 0:
                        chunks_context += f"----\n[From Conversation {conv_id_short} around {start_time_str} (truncated)]\n{truncated_text}\n\n"
                        chunks_tokens += truncated_tokens
                        processed_chunk_ids.add(chunk_id)
                break # Stop adding chunks

        # Only add the chunks section if any were actually added
        if chunks_tokens > estimate_tokens("--- RELEVANT PREVIOUS CONVERSATION DETAILS ---\n\n"):
            context_parts.append(chunks_context.strip())
            total_tokens += chunks_tokens

    # Combine all parts into a single context string
    full_context = "\n\n".join(context_parts) # Separate sections with double newline

    return full_context if total_tokens > 0 else "No relevant context found."


def synthesize_user_model(retrieved_insights: List[Dict], llm_adapter: GoogleGenAIAdapter) -> str:
    """
    Generates a synthesized user model summary from relevant insight notes using an LLM.

    Args:
        retrieved_insights: List of dictionaries representing relevant insights about the user.
        llm_adapter: An instance of the GoogleGenAIAdapter.

    Returns:
        A string containing the synthesized user profile, or an empty string if no insights
        or an error occurs.
    """
    if not retrieved_insights:
        return ""

    # Create prompt for synthesis
    prompt = "Based on the following insights and facts about the user, create a brief summary of the user's profile including potential interests, preferences, knowledge levels, and explicitly stated goals or facts. Focus on stable information.\n\nInsights/Facts:\n"
    insight_count = 0
    for insight in retrieved_insights:
        insight_text = insight.get('insight_text')
        if insight_text:
            prompt += f"- {insight_text}\n"
            insight_count += 1

    if insight_count == 0:
        return "" # No actual insight text found

    prompt += "\nSynthesized User Profile Summary:"

    # Call the LLM for synthesis
    try:
        # Use the adapter
        response_text = llm_adapter.generate_text(
            prompt=prompt,
            model_name=settings.SUMMARY_MODEL, # Use a faster/cheaper model for synthesis
            temperature=0.2,
            max_output_tokens=300 # Limit summary length
        )

        if response_text and not response_text.startswith("[Error"):
            return response_text.strip()
        else:
            print(f"User model synthesis failed or returned error: {response_text}")
            return "" # Return empty on failure/error message

    except Exception as e:
        print(f"Unexpected error generating user model synthesis: {e}")
        return "" # Return empty on unexpected error


# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing Context Builder...")

    # --- Test assemble_chat_context ---
    print("\nTesting assemble_chat_context...")
    mock_chunks = [
        {'doc_type': 'chunk', 'chunk_id': 'c1', 'text': 'This is the first chunk about topic A.', 'similarity': 0.9, 'conv_id': 'conv1', 'start_time': time.time()-100},
        {'doc_type': 'conversation_summary', 'summary_id': 's1', 'text': 'Summary of conversation conv2 about topic B.', 'similarity': 0.85, 'original_conv_id': 'conv2', 'create_time': time.time()-200, 'title': 'Topic B Discussion'},
        {'doc_type': 'chunk', 'chunk_id': 'c2', 'text': 'Second chunk, also related to topic A.', 'similarity': 0.8, 'conv_id': 'conv1', 'start_time': time.time()-90},
        {'doc_type': 'chunk', 'chunk_id': 'c3', 'text': 'A much longer chunk designed to test truncation ' + ('word ' * 100), 'similarity': 0.7, 'conv_id': 'conv3', 'start_time': time.time()-50},
    ]
    mock_insights = [
        {'insight_id': 'i1', 'insight_text': 'User is interested in topic A.', 'similarity': 0.95},
        {'insight_id': 'i2', 'insight_text': 'User prefers short answers.', 'similarity': 0.75},
        {'insight_id': 'i3', 'insight_text': 'Fact: User project is called Phoenix.', 'similarity': 0.7},
    ]

    # Test with ample token limit
    context_full = assemble_chat_context(mock_chunks, mock_insights, max_tokens=2000)
    print("\nContext (Full Tokens):")
    print(context_full)
    assert "USER MODEL AND CONVERSATION INSIGHTS" in context_full
    assert "User is interested in topic A." in context_full
    assert "Fact: User project is called Phoenix." in context_full
    assert "RELEVANT PREVIOUS CONVERSATION DETAILS" in context_full
    assert "[Summary of Conversation conv2" in context_full
    assert "This is the first chunk about topic A." in context_full
    assert "Second chunk, also related to topic A." in context_full
    assert "A much longer chunk" in context_full
    assert "..." not in context_full # Should not be truncated

    # Test with limited token limit (force truncation)
    context_limited = assemble_chat_context(mock_chunks, mock_insights, max_tokens=100) # Low limit
    print("\nContext (Limited Tokens - Expect Truncation):")
    print(context_limited)
    assert "USER MODEL AND CONVERSATION INSIGHTS" in context_limited
    assert "User is interested in topic A." in context_limited # High similarity insight should fit
    # assert "User prefers short answers." not in context_limited # Lower similarity insight might be cut
    assert "RELEVANT PREVIOUS CONVERSATION DETAILS" in context_limited
    assert "[Summary of Conversation conv2" in context_limited # Summary has high similarity
    assert "This is the first chunk about topic A." in context_limited # High similarity chunk
    # assert "Second chunk" not in context_limited # Lower similarity chunk likely cut
    assert "..." in context_limited or "truncated" in context_limited # Expect truncation marker if long chunk was attempted

    # Test with no context
    context_none = assemble_chat_context([], [], max_tokens=1000)
    print(f"\nContext (None): '{context_none}'")
    assert context_none == "No relevant context found."

    # --- Test synthesize_user_model ---
    print("\nTesting synthesize_user_model...")

    class MockLLMAdapterSynth:
        def generate_text(self, prompt, model_name, temperature, max_output_tokens):
            print(f"Mock generate_text called for model {model_name}")
            if "Insights/Facts:" in prompt:
                return "Synthesized profile: User likes topic A and project Phoenix."
            return "[Mock Synth Error]"

    mock_adapter_synth = MockLLMAdapterSynth()

    user_model = synthesize_user_model(mock_insights, mock_adapter_synth)
    print(f"Synthesized User Model: {user_model}")
    assert "User likes topic A" in user_model
    assert "project Phoenix" in user_model

    user_model_empty = synthesize_user_model([], mock_adapter_synth)
    print(f"Synthesized User Model (Empty Input): '{user_model_empty}'")
    assert user_model_empty == ""

    print("\nContext Builder testing finished.")
