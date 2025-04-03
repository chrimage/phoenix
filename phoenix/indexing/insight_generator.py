# phoenix/indexing/insight_generator.py - Insight/Fact Generation Logic
import uuid
import time
from typing import List, Optional, Tuple

from phoenix.config import settings
from phoenix.core.models import InsightNote, ParsedConversation, Message, ConversationMetadata
from phoenix.core.utils import estimate_tokens
from phoenix.adapters.llm.google_genai import GoogleGenAIAdapter

def generate_insight_notes(
    parsed_conv: ParsedConversation,
    llm_adapter: GoogleGenAIAdapter,
    min_tokens_for_insights: int = settings.MIN_TOKENS_FOR_SUMMARY, # Use same threshold as summary
    target_insight_tokens: int = settings.TARGET_INSIGHT_NOTE_TOKENS,
    max_insights: int = settings.MAX_INSIGHT_NOTES_PER_CONV,
    insight_model_name: str = settings.SUMMARY_MODEL # Often same model as summary
) -> List[InsightNote]:
    """
    Generates insight notes and facts from a conversation using an LLM.

    Args:
        parsed_conv: The ParsedConversation object.
        llm_adapter: An instance of the GoogleGenAIAdapter.
        min_tokens_for_insights: Minimum conversation tokens to trigger insight generation.
        target_insight_tokens: Approximate target token length per insight.
        max_insights: Maximum number of insights/facts to generate.
        insight_model_name: The LLM model to use for generation.

    Returns:
        A list of InsightNote objects, or an empty list if none were generated or an error occurred.
    """
    # Combine all messages into a single text
    full_conv_text = "\n".join([f"{msg.role}: {msg.content}" for msg in parsed_conv.messages])
    conv_tokens = estimate_tokens(full_conv_text)

    if conv_tokens <= min_tokens_for_insights:
        # print(f"Conversation {parsed_conv.metadata.conv_id} below token threshold ({conv_tokens} <= {min_tokens_for_insights}). Skipping insights.")
        return []

    print(f"  - Generating insight notes/facts for conversation {parsed_conv.metadata.conv_id}...")

    # Revised prompt to extract insights AND key facts
    prompt = f"""Analyze the following conversation.

Extract important, relatively stable insights about:
1. Recurring user interests, preferences, or demonstrated knowledge/skill levels. Avoid temporary goals unless they represent a significant shift or decision.
2. Key capabilities or limitations demonstrated by the AI assistant during the interaction.
3. Core topics discussed repeatedly, established facts, significant decisions made, or explicit action items agreed upon.

ALSO, extract any specific key facts mentioned, such as:
- User's name (if stated)
- Specific project names, tools, or locations mentioned
- Explicitly stated goals or preferences not covered by general insights

For each insight OR fact:
- Make it a single, concise, standalone statement.
- Start the line with "Insight:" for general insights.
- Start the line with "Fact:" for specific factual details.
- Ensure it's likely to remain relevant beyond the immediate context of this conversation.
- Avoid speculating on short-term intentions. Focus on observed patterns or explicit statements.

Example Format:
Insight: User prefers concise code examples.
Fact: User mentioned their name is Chris.
Insight: AI demonstrated image generation capabilities.
Fact: Project discussed is named 'LyricVideoMaker'.

Provide up to {max_insights} insights/facts, prioritizing the most significant and stable ones.

Conversation:
---
{full_conv_text}
---

Insights and Facts:"""

    # Estimate max output tokens needed
    max_output_tokens = int(target_insight_tokens * max_insights * 2.5) # Generous buffer for formatting/prefixes

    try:
        response_text = llm_adapter.generate_text(
            prompt=prompt,
            model_name=insight_model_name,
            temperature=0.2, # Lower temperature for factual extraction
            max_output_tokens=max_output_tokens
        )

        if not response_text or response_text.startswith("[Error"):
            print(f"  - Insight generation failed or returned an error: {response_text}")
            return []

        # Parse the response
        generated_insights: List[InsightNote] = []
        insight_blocks = [block.strip() for block in response_text.split("\n\n") if block.strip()]

        for block in insight_blocks:
            lines = block.split("\n")
            # Check if the block starts with "Insight:" or "Fact:"
            if lines and (lines[0].startswith("Insight:") or lines[0].startswith("Fact:")):
                # Extract the text after the prefix
                insight_or_fact_text = lines[0].split(":", 1)[1].strip()
                if insight_or_fact_text:
                    insight_token_count = estimate_tokens(insight_or_fact_text)
                    insight_doc = InsightNote(
                        doc_id=f"insight_{parsed_conv.metadata.conv_id}_{uuid.uuid4()}",
                        text=insight_or_fact_text,
                        original_conv_id=parsed_conv.metadata.conv_id,
                        insight_token_count=insight_token_count,
                        create_time=parsed_conv.metadata.create_time, # Use conversation time
                        model_slug=parsed_conv.metadata.model_slug, # Inherit model
                        title=parsed_conv.metadata.title
                        # Metadata added by ChromaStore
                    )
                    generated_insights.append(insight_doc)

        if generated_insights:
            print(f"  - Generated {len(generated_insights)} insights/facts")
            # Print preview
            # print("\n  --- INSIGHTS/FACTS PREVIEW ---")
            # preview_count = min(3, len(generated_insights))
            # for i in range(preview_count):
            #     print(f"  - {generated_insights[i].text}")
            # print("  ----------------------------\n")
        else:
            print("  - No insights/facts parsed from LLM response.")

        return generated_insights

    except Exception as e:
        print(f"  - ERROR generating insight notes for {parsed_conv.metadata.conv_id}: {e}")
        return []
