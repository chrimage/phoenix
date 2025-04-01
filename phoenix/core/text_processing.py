# phoenix/core/text_processing.py - Text processing functions like chunking and token estimation
import spacy
import json
import uuid
from typing import List, Dict, Tuple

# Import core models and configuration settings
from phoenix.core.models import Conversation, Message, Chunk
from phoenix.config import settings

# --- spaCy Model Loading ---
nlp = None
try:
    nlp = spacy.load(settings.SPACY_MODEL)
    print(f"spaCy model '{settings.SPACY_MODEL}' loaded successfully.")
except OSError:
    print(f"❌ Error: spaCy model '{settings.SPACY_MODEL}' not found.")
    print("Please run the following command in your terminal:")
    print(f"  python -m spacy download {settings.SPACY_MODEL}")
    # In a real application, you might want to raise an exception or exit
    # raise RuntimeError(f"spaCy model '{settings.SPACY_MODEL}' not found. Please download it.")
except Exception as e:
    print(f"❌ An unexpected error occurred loading the spaCy model: {e}")
    # raise RuntimeError(f"Failed to load spaCy model: {e}")

# --- Token Estimation ---
def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string using whitespace splitting."""
    if not text:
        return 0
    return len(text.split())

# --- Conversation Chunking ---
def chunk_conversation(conversation: Conversation) -> List[Chunk]:
    """Splits a conversation into overlapping chunks based on sentences."""
    if not nlp:
        print("❌ Error: spaCy model not loaded. Cannot perform chunking.")
        return []
    if not conversation or not conversation.messages:
        return []

    chunks: List[Chunk] = []
    all_sentences: List[Tuple[str, str, float]] = [] # (sentence_text, msg_id, timestamp)

    # Use spaCy to split messages into sentences, preserving metadata
    for msg in conversation.messages:
        if msg.content and isinstance(msg.content, str):
            try:
                doc = nlp(msg.content)
                for sent in doc.sents:
                    # Store sentence text along with original message ID and timestamp
                    all_sentences.append((sent.text, msg.msg_id, msg.timestamp))
            except Exception as e:
                print(f"⚠️ Warning: Error processing message {msg.msg_id} with spaCy: {e}")
                # Optionally, add the whole message content as a single "sentence"
                # all_sentences.append((msg.content, msg.msg_id, msg.timestamp))

    if not all_sentences:
        print(f"⚠️ Warning: No sentences extracted from conversation {conversation.conv_id}.")
        return [] # No text content to chunk

    current_chunk_sentences: List[Tuple[str, str, float]] = []
    current_chunk_tokens = 0
    current_message_ids = set()
    start_time = all_sentences[0][2] # Timestamp of the first sentence

    for i, (sent_text, msg_id, timestamp) in enumerate(all_sentences):
        # Rough token estimation for the sentence
        sent_tokens = estimate_tokens(sent_text)

        # Check if adding this sentence exceeds target size (allow overshoot)
        # Use settings.TARGET_CHUNK_TOKENS
        exceeds_limit = current_chunk_sentences and \
                        (current_chunk_tokens + sent_tokens > settings.TARGET_CHUNK_TOKENS * 1.2)

        if exceeds_limit:
            # Finalize the previous chunk
            chunk_id = f"chunk_{conversation.conv_id}_{uuid.uuid4()}"
            chunk_text = " ".join(s[0] for s in current_chunk_sentences).strip()
            end_time = current_chunk_sentences[-1][2] # Timestamp of last sentence in chunk

            # Create Chunk object using the dataclass
            chunk_obj = Chunk(
                chunk_id=chunk_id,
                conv_id=conversation.conv_id,
                chunk_text=chunk_text,
                start_time=start_time,
                end_time=end_time,
                message_ids=sorted(list(current_message_ids)), # Store as list
                token_count=current_chunk_tokens,
                # Inherit metadata (optional, can be added later or looked up)
                title=conversation.title,
                model_slug=conversation.model_slug
            )
            chunks.append(chunk_obj)

            # Start new chunk with overlap (use settings.SENTENCE_OVERLAP)
            overlap_start_index = max(0, len(current_chunk_sentences) - settings.SENTENCE_OVERLAP)
            current_chunk_sentences = current_chunk_sentences[overlap_start_index:]
            current_chunk_tokens = sum(estimate_tokens(s[0]) for s in current_chunk_sentences)
            current_message_ids = set(s[1] for s in current_chunk_sentences)
            start_time = current_chunk_sentences[0][2] if current_chunk_sentences else timestamp # Update start time

        # Add current sentence to the chunk buffer
        current_chunk_sentences.append((sent_text, msg_id, timestamp))
        current_chunk_tokens += sent_tokens
        current_message_ids.add(msg_id)
        # Ensure start_time reflects the actual earliest time in the current chunk buffer
        if len(current_chunk_sentences) == 1:
             start_time = timestamp

    # Add the last remaining chunk
    if current_chunk_sentences:
        chunk_id = f"chunk_{conversation.conv_id}_{uuid.uuid4()}"
        chunk_text = " ".join(s[0] for s in current_chunk_sentences).strip()
        end_time = current_chunk_sentences[-1][2] # Timestamp of last sentence

        chunk_obj = Chunk(
            chunk_id=chunk_id,
            conv_id=conversation.conv_id,
            chunk_text=chunk_text,
            start_time=start_time,
            end_time=end_time,
            message_ids=sorted(list(current_message_ids)),
            token_count=current_chunk_tokens,
            title=conversation.title,
            model_slug=conversation.model_slug
        )
        chunks.append(chunk_obj)

    return chunks
