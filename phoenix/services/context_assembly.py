# phoenix/services/context_assembly.py - Logic for assembling chat context
from typing import List, Dict, Any
from datetime import datetime

from phoenix.config import settings
from phoenix.core import text_processing # For estimate_tokens

# Import adapters if needed for synthesis (though synthesis logic is removed for now)
# from phoenix.adapters.llm.gemini_client import GeminiClient

class ContextAssembler:
    """Handles assembling context strings for the chat prompt."""

    def __init__(self):
        # If synthesis is re-introduced, the LLM client might be needed here
        # self.llm_client = llm_client
        print("🛠️ Context Assembler Initialized 🛠️")
        pass # No specific initialization needed for basic assembly

    def assemble_chat_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        retrieved_insights: List[Dict[str, Any]],
        max_tokens: int = settings.CONTEXT_TOKEN_LIMIT
    ) -> str:
        """
        Formats retrieved chunks and insights into a context string within token limits.

        Args:
            retrieved_chunks: List of dictionaries representing retrieved chunks/summaries
                              (expected keys: 'id', 'text', 'metadata', 'similarity').
            retrieved_insights: List of dictionaries representing retrieved insights
                                (expected keys: 'id', 'text', 'metadata', 'similarity').
            max_tokens: The maximum estimated tokens allowed for the context string.

        Returns:
            A formatted context string, or a message indicating no context was found.
        """
        total_tokens = 0
        context_parts = []
        # Slightly bias token budget towards chunks/summaries as they are often longer
        max_tokens_insights = int(max_tokens * 0.4)
        max_tokens_chunks = max_tokens - max_tokens_insights

        # --- Process Insight Notes ---
        if retrieved_insights:
            insights_header = "--- USER MODEL AND CONVERSATION INSIGHTS ---\n"
            insights_context = insights_header
            insights_tokens = text_processing.estimate_tokens(insights_context)
            processed_insight_ids = set()

            # Sort insights by similarity (highest first) - assuming input is sorted
            for insight in retrieved_insights:
                insight_id = insight.get('id')
                if not insight_id or insight_id in processed_insight_ids:
                    continue # Skip if no ID or already processed

                insight_text = insight.get('text', '')
                insight_token_count = text_processing.estimate_tokens(insight_text)

                if insights_tokens + insight_token_count <= max_tokens_insights:
                    insights_context += f"• {insight_text}\n" # Add bullet point
                    insights_tokens += insight_token_count
                    processed_insight_ids.add(insight_id)
                else:
                    break # Stop adding insights if token limit is reached

            # Only add the insights section if any insights were actually added
            if insights_context != insights_header:
                context_parts.append(insights_context.strip() + "\n") # Add trailing newline
                total_tokens += insights_tokens

        # --- Process Regular Chunks and Summaries ---
        if retrieved_chunks:
            chunks_header = "--- RELEVANT PREVIOUS CONVERSATION DETAILS ---\n"
            chunks_context = chunks_header
            chunks_tokens = text_processing.estimate_tokens(chunks_context)

            # Sort chunks - prioritize summaries first, then by similarity (assuming input is sorted)
            summaries = []
            regular_chunks = []
            for chunk in retrieved_chunks:
                metadata = chunk.get('metadata', {})
                if metadata.get('doc_type') == 'conversation_summary':
                    summaries.append(chunk)
                else:
                    regular_chunks.append(chunk)

            # Process summaries first
            for summary in summaries:
                summary_text = summary.get('text', '')
                metadata = summary.get('metadata', {})
                # Use stored token count if available, otherwise estimate
                summary_tokens = metadata.get('summary_token_count') or text_processing.estimate_tokens(summary_text)
                conv_id_short = metadata.get('original_conv_id', 'unknown')[:8]
                title = metadata.get('title', 'No Title')

                if chunks_tokens + summary_tokens <= max_tokens_chunks:
                    chunks_context += f"\n[Summary of Conversation {conv_id_short} - '{title}']\n{summary_text}\n"
                    chunks_tokens += summary_tokens
                else:
                    break # Stop adding summaries if limit reached

            # Process regular chunks (if space permits)
            for chunk in regular_chunks:
                chunk_text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                chunk_tokens = metadata.get('token_count') or text_processing.estimate_tokens(chunk_text)
                conv_id_short = metadata.get('conv_id', 'unknown')[:8]
                start_time_ts = metadata.get('start_time')
                start_date_str = datetime.fromtimestamp(start_time_ts).strftime('%Y-%m-%d') if start_time_ts else 'unknown date'

                # Check if adding this chunk exceeds the limit
                if chunks_tokens + chunk_tokens <= max_tokens_chunks:
                    chunks_context += f"\n----\n[From Conversation {conv_id_short} around {start_date_str}]\n{chunk_text}\n"
                    chunks_tokens += chunk_tokens
                else:
                    # Try adding a truncated version if there's significant space left
                    remaining_tokens = max_tokens_chunks - chunks_tokens
                    if remaining_tokens > 50: # Need at least 50 tokens space
                        # Estimate allowed characters (very rough)
                        allowed_chars = int(remaining_tokens * 4)
                        truncated_text = chunk_text[:allowed_chars].rsplit(' ', 1)[0] + "..." # Truncate at word boundary
                        truncated_tokens = text_processing.estimate_tokens(truncated_text)
                        if chunks_tokens + truncated_tokens <= max_tokens_chunks:
                             chunks_context += f"\n----\n[From Conversation {conv_id_short} around {start_date_str} (truncated)]\n{truncated_text}\n"
                             chunks_tokens += truncated_tokens
                    break # Stop adding chunks (either added truncated or no space)

            # Only add the chunks section if any were actually added
            if chunks_context != chunks_header:
                context_parts.append(chunks_context.strip() + "\n") # Add trailing newline
                total_tokens += chunks_tokens

        # Combine all parts into a single context string
        if not context_parts:
            return "No relevant context found."
        else:
            # Join with double newline for better separation between sections
            return "\n".join(context_parts).strip()

    # Note: synthesize_user_model logic was removed as it wasn't fully implemented
    # in the original chat script and adds complexity. Can be added back later if needed.

# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("Testing ContextAssembler...")
    assembler = ContextAssembler()

    # Dummy data
    dummy_insights = [
        {'id': 'ins_1', 'text': 'User is interested in Python programming.', 'metadata': {}, 'similarity': 0.9},
        {'id': 'ins_2', 'text': 'User prefers concise code examples.', 'metadata': {}, 'similarity': 0.85},
        {'id': 'ins_3', 'text': 'AI sometimes misunderstands complex queries.', 'metadata': {}, 'similarity': 0.8},
    ]
    dummy_chunks = [
        {'id': 'sum_1', 'text': 'This conversation discussed refactoring the codebase using vertical slices.', 'metadata': {'doc_type': 'conversation_summary', 'original_conv_id': 'conv_abc', 'title': 'Refactoring Plan'}, 'similarity': 0.95},
        {'id': 'chk_1', 'text': 'The user asked about specific adapter implementations. The assistant provided examples for ChromaDB.', 'metadata': {'doc_type': 'conversation_chunk', 'conv_id': 'conv_xyz', 'start_time': 1678886400.0, 'token_count': 20}, 'similarity': 0.88},
        {'id': 'chk_2', 'text': 'We also talked about setting up the configuration file and loading environment variables correctly.', 'metadata': {'doc_type': 'conversation_chunk', 'conv_id': 'conv_xyz', 'start_time': 1678886500.0, 'token_count': 25}, 'similarity': 0.86},
        {'id': 'chk_3', 'text': 'This is a much longer chunk designed to test truncation. It contains many words and sentences to see if the context assembler correctly shortens it when the token limit is approached. We need enough text here to potentially exceed a reasonable limit when combined with other chunks and insights.', 'metadata': {'doc_type': 'conversation_chunk', 'conv_id': 'conv_lmn', 'start_time': 1678887000.0, 'token_count': 60}, 'similarity': 0.80},
    ]

    print("\n--- Test Case 1: Sufficient Token Limit ---")
    context1 = assembler.assemble_chat_context(dummy_chunks, dummy_insights, max_tokens=1000)
    print(context1)
    print(f"\nEstimated Tokens: {text_processing.estimate_tokens(context1)}")

    print("\n--- Test Case 2: Limited Token Limit (Testing Truncation) ---")
    context2 = assembler.assemble_chat_context(dummy_chunks, dummy_insights, max_tokens=100) # Reduced limit
    print(context2)
    print(f"\nEstimated Tokens: {text_processing.estimate_tokens(context2)}")

    print("\n--- Test Case 3: No Insights ---")
    context3 = assembler.assemble_chat_context(dummy_chunks, [], max_tokens=1000)
    print(context3)
    print(f"\nEstimated Tokens: {text_processing.estimate_tokens(context3)}")

    print("\n--- Test Case 4: No Chunks ---")
    context4 = assembler.assemble_chat_context([], dummy_insights, max_tokens=1000)
    print(context4)
    print(f"\nEstimated Tokens: {text_processing.estimate_tokens(context4)}")

    print("\n--- Test Case 5: No Context ---")
    context5 = assembler.assemble_chat_context([], [], max_tokens=1000)
    print(context5)
    print(f"\nEstimated Tokens: {text_processing.estimate_tokens(context5)}")
