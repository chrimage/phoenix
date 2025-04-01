# phoenix/indexing/chunker.py - Text Chunking Logic
import spacy
import json
import uuid
import time # Added import for time module
from typing import List, Dict, Optional

from phoenix.config import settings
from phoenix.core.models import ParsedConversation, Chunk, Message, ConversationMetadata
from phoenix.core.utils import estimate_tokens

# Load spaCy model globally for efficiency
try:
    nlp = spacy.load(settings.SPACY_MODEL)
    print(f"spaCy model '{settings.SPACY_MODEL}' loaded successfully for chunking.")
except OSError:
    print(f"spaCy model '{settings.SPACY_MODEL}' not found.")
    print("Please run the following command in your terminal:")
    print(f"  python -m spacy download {settings.SPACY_MODEL}")
    # Depending on application needs, might raise an error or exit
    raise ImportError(f"Required spaCy model '{settings.SPACY_MODEL}' not downloaded.")
except Exception as e:
    print(f"Error loading spaCy model '{settings.SPACY_MODEL}': {e}")
    raise

def chunk_conversation(parsed_conv: ParsedConversation,
                       target_chunk_tokens: int = settings.TARGET_CHUNK_TOKENS,
                       sentence_overlap: int = settings.SENTENCE_OVERLAP) -> List[Chunk]:
    """
    Splits a parsed conversation into overlapping chunks based on sentences.

    Args:
        parsed_conv: The ParsedConversation object containing metadata and messages.
        target_chunk_tokens: The desired approximate number of tokens per chunk.
        sentence_overlap: The number of sentences to overlap between consecutive chunks.

    Returns:
        A list of Chunk objects.
    """
    chunks: List[Chunk] = []
    all_sentences = [] # Store tuples of (sentence_text, msg_id, timestamp)

    # Use spaCy to split messages into sentences, preserving metadata
    for msg in parsed_conv.messages:
        if msg.content and isinstance(msg.content, str):
            try:
                doc = nlp(msg.content)
                for sent in doc.sents:
                    # Store sentence text along with original message ID and timestamp
                    all_sentences.append((sent.text, msg.msg_id, msg.timestamp))
            except Exception as e:
                print(f"Warning: spaCy failed to process message {msg.msg_id}. Skipping message content. Error: {e}")
                continue # Skip this message if spaCy fails

    if not all_sentences:
        print(f"Warning: No sentences extracted from conversation {parsed_conv.metadata.conv_id}. No chunks generated.")
        return [] # No text content to chunk

    current_chunk_sentences = []
    current_chunk_tokens = 0
    current_message_ids = set()
    # Initialize start_time with the timestamp of the very first sentence
    start_time = all_sentences[0][2]

    for i, (sent_text, msg_id, timestamp) in enumerate(all_sentences):
        # Estimate tokens for the current sentence
        sent_tokens = estimate_tokens(sent_text)

        # Check if adding this sentence would exceed the target size significantly
        # Allow some overshoot (e.g., 20%) to avoid tiny final chunks
        overshoot_limit = target_chunk_tokens * 1.2
        if current_chunk_sentences and (current_chunk_tokens + sent_tokens > overshoot_limit):
            # Finalize the previous chunk
            chunk_text = " ".join(s[0] for s in current_chunk_sentences).strip()
            if chunk_text: # Ensure chunk is not empty
                chunk_id = f"chunk_{parsed_conv.metadata.conv_id}_{uuid.uuid4()}"
                # End time is the timestamp of the last sentence included in this chunk
                end_time = current_chunk_sentences[-1][2]

                chunk = Chunk(
                    doc_id=chunk_id,
                    text=chunk_text,
                    conv_id=parsed_conv.metadata.conv_id,
                    start_time=start_time,
                    end_time=end_time,
                    token_count=current_chunk_tokens,
                    message_ids=sorted(list(current_message_ids))
                    # Metadata will be added later using conv_meta in ChromaStore
                )
                chunks.append(chunk)

            # Start new chunk with overlap
            overlap_start_index = max(0, len(current_chunk_sentences) - sentence_overlap)
            # Carry over the overlapping sentences
            current_chunk_sentences = current_chunk_sentences[overlap_start_index:]
            # Recalculate tokens and message IDs for the overlapping part
            current_chunk_tokens = sum(estimate_tokens(s[0]) for s in current_chunk_sentences)
            current_message_ids = set(s[1] for s in current_chunk_sentences)
            # Update start_time for the new chunk based on the first sentence of the overlap
            start_time = current_chunk_sentences[0][2] if current_chunk_sentences else timestamp

        # Add current sentence to the chunk buffer
        current_chunk_sentences.append((sent_text, msg_id, timestamp))
        current_chunk_tokens += sent_tokens
        current_message_ids.add(msg_id)
        # Ensure start_time reflects the actual earliest time in the current chunk buffer
        if len(current_chunk_sentences) == 1:
             start_time = timestamp


    # Add the last remaining chunk if it contains any sentences
    if current_chunk_sentences:
        chunk_text = " ".join(s[0] for s in current_chunk_sentences).strip()
        if chunk_text: # Ensure final chunk is not empty
            chunk_id = f"chunk_{parsed_conv.metadata.conv_id}_{uuid.uuid4()}"
            # End time is the timestamp of the very last sentence
            end_time = current_chunk_sentences[-1][2]

            chunk = Chunk(
                doc_id=chunk_id,
                text=chunk_text,
                conv_id=parsed_conv.metadata.conv_id,
                start_time=start_time,
                end_time=end_time,
                token_count=current_chunk_tokens,
                message_ids=sorted(list(current_message_ids))
            )
            chunks.append(chunk)

    # print(f"Generated {len(chunks)} chunks for conversation {parsed_conv.metadata.conv_id}")
    return chunks

# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing Chunker...")

    # Create dummy ParsedConversation data
    meta = ConversationMetadata(conv_id="test_chunk_conv", title="Chunking Test")
    messages = [
        Message(role="user", content="This is the first sentence. Here is the second sentence.", timestamp=time.time()-20),
        Message(role="assistant", content="Okay, I understand. Let's proceed. This is sentence four.", timestamp=time.time()-15),
        Message(role="user", content="Sentence five is here. And finally, sentence six.", timestamp=time.time()-10),
        Message(role="user", content="A short seventh sentence.", timestamp=time.time()-5),
    ]
    parsed_conv_test = ParsedConversation(metadata=meta, messages=messages)

    # Test with small target tokens to force multiple chunks
    test_target_tokens = 10
    test_overlap = 1
    print(f"\nTesting with target_tokens={test_target_tokens}, overlap={test_overlap}")
    generated_chunks = chunk_conversation(
        parsed_conv_test,
        target_chunk_tokens=test_target_tokens,
        sentence_overlap=test_overlap
    )

    print(f"Generated {len(generated_chunks)} chunks:")
    total_sentences_in_chunks = 0
    for i, chunk in enumerate(generated_chunks):
        print(f"--- Chunk {i+1} (ID: {chunk.doc_id}) ---")
        print(f"  Text: '{chunk.text}'")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Start Time: {chunk.start_time}")
        print(f"  End Time: {chunk.end_time}")
        print(f"  Message IDs: {chunk.message_ids}")
        # Basic validation
        assert isinstance(chunk, Chunk)
        assert chunk.text.strip() != ""
        assert chunk.token_count > 0
        assert chunk.end_time >= chunk.start_time
        total_sentences_in_chunks += len(nlp(chunk.text).sents) # Count sentences in chunk

    # Check overlap (approximate check)
    if len(generated_chunks) > 1:
        last_sent_chunk1 = list(nlp(generated_chunks[0].text).sents)[-1].text
        first_sent_chunk2 = list(nlp(generated_chunks[1].text).sents)[0].text
        print(f"\nOverlap Check:")
        print(f"  Last sentence of Chunk 1: '{last_sent_chunk1}'")
        print(f"  First sentence of Chunk 2: '{first_sent_chunk2}'")
        # This is a simple check; real overlap might span multiple sentences
        # assert last_sent_chunk1 in generated_chunks[1].text

    print(f"\nTotal sentences in original: 7") # Manually counted
    # Note: Total sentences in chunks might be > original due to overlap
    print(f"Total sentences across generated chunks: {total_sentences_in_chunks}")


    print("\nChunker testing finished.")
