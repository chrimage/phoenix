# tests/chat/test_context_builder.py

import pytest

# Adjust import path as necessary
from phoenix.chat.context_builder import assemble_chat_context

# --- Test Cases ---

def test_assemble_chat_context_empty():
    """Tests context assembly with no retrieved documents."""
    user_query = "What is the weather?"
    retrieved_docs = []
    expected_context = f"User Query: {user_query}\n\nRetrieved Context:\n---" # Or adjust based on actual implementation
    context = assemble_chat_context(user_query, retrieved_docs)
    assert context.strip() == expected_context.strip()

def test_assemble_chat_context_with_docs():
    """Tests context assembly with some retrieved documents."""
    user_query = "Tell me about project Phoenix."
    retrieved_docs = [
        "Document 1: Phoenix is a project about memory.",
        "Document 2: It uses ChromaDB and SQLite.",
        "Document 3: The goal is to chat with conversation history."
    ]
    # This expected format depends heavily on the implementation details
    expected_context = f"""User Query: {user_query}

Retrieved Context:
---
Document 1: Phoenix is a project about memory.
---
Document 2: It uses ChromaDB and SQLite.
---
Document 3: The goal is to chat with conversation history.
---"""
    context = assemble_chat_context(user_query, retrieved_docs)
    assert context.strip() == expected_context.strip()

def test_assemble_chat_context_long_query():
    """Tests context assembly with a longer user query."""
    user_query = "Can you explain the detailed architecture of the indexing pipeline, including the parser, chunker, summarizer, and insight generator components, and how they interact with the storage layers?"
    retrieved_docs = ["Doc A", "Doc B"]
    # Just check that the query is included correctly
    context = assemble_chat_context(user_query, retrieved_docs)
    assert f"User Query: {user_query}" in context
    assert "Doc A" in context
    assert "Doc B" in context

# Add more tests for edge cases, different formatting, very long docs, etc.
