# Chroma API Guide for LLMs: Building Vector Database Applications

This guide is specifically designed to help large language models effectively use Chroma, an open-source AI application database that makes knowledge, facts, and skills pluggable for LLMs. It provides a structured approach to understanding the capabilities, features, and best practices of the Chroma API.

## Table of Contents
1. [Introduction to Chroma](#introduction-to-chroma)
2. [Installation and Setup](#installation-and-setup)
3. [Core Concepts](#core-concepts)
4. [Running Chroma](#running-chroma)
5. [Working with Collections](#working-with-collections)
6. [Adding and Managing Data](#adding-and-managing-data)
7. [Querying Data](#querying-data)
8. [Embedding Functions](#embedding-functions)
9. [Metadata Filtering](#metadata-filtering)
10. [Best Practices](#best-practices)
11. [Example Patterns](#example-patterns)

## Introduction to Chroma

Chroma is the open-source AI application database, designed to make it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs. It provides everything needed for retrieval applications:

- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

Chroma can run as a server and provides Python and JavaScript/TypeScript client SDKs. It's licensed under Apache 2.0.

## Installation and Setup

### Python Installation

```python
# Install Chroma with pip
pip install chromadb
```

### JavaScript/TypeScript Installation

```bash
# With yarn
yarn add chromadb chromadb-default-embed

# With npm
npm install chromadb chromadb-default-embed

# With pnpm
pnpm add chromadb chromadb-default-embed
```

## Core Concepts

### Key Components

- **Client**: The entry point for interacting with Chroma, responsible for managing collections and database operations
- **Collections**: Where embeddings, documents, and metadata are stored and indexed
- **Embedding Functions**: Functions that convert text or other data into vector embeddings
- **Documents**: Text content that can be stored and retrieved
- **Embeddings**: Vector representations of documents or data
- **Metadata**: Additional information associated with documents and embeddings, enabling filtering
- **Queries**: Methods to search for similar documents based on text or embeddings

## Running Chroma

Chroma can run in different modes, each suitable for different use cases:

### Ephemeral Client (In-Memory)

Perfect for exploration, testing, and notebooks. Data is lost when the program terminates.

```python
import chromadb
chroma_client = chromadb.Client()
```

### Persistent Client

For applications that need data persistence between runs.

```python
import chromadb
chroma_client = chromadb.PersistentClient(path="/path/to/save/to")
```

If no path is provided, data is stored in a directory named `.chroma`.

### Client-Server Mode

For production deployments or when you need to access Chroma from multiple applications.

First, start the Chroma server:
```bash
chroma run --path /db_path
```

Then connect to it:
```python
import chromadb
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
```

For asynchronous applications, Chroma also provides an async HTTP client:
```python
import asyncio
import chromadb

async def main():
    client = await chromadb.AsyncHttpClient()
    # Use client as normal, with await for operations
    
asyncio.run(main())
```

## Working with Collections

Collections are where you store embeddings, documents, and metadata.

### Creating Collections

```python
# Create a new collection
collection = chroma_client.create_collection(name="my_collection")

# Create a collection with a custom embedding function
collection = chroma_client.create_collection(
    name="my_collection",
    embedding_function=embedding_function
)

# Create a collection with metadata
collection = chroma_client.create_collection(
    name="my_collection",
    metadata={
        "description": "My first Chroma collection",
        "created": "2025-03-28"
    }
)
```

### Managing Collections

```python
# Get an existing collection
collection = chroma_client.get_collection(name="my_collection")

# Get a collection if it exists, or create it if it doesn't
collection = chroma_client.get_or_create_collection(name="my_collection")

# Delete a collection and all associated data
chroma_client.delete_collection(name="my_collection")

# Get information about a collection
collection.peek()  # Returns first 10 items
collection.count() # Returns number of items
collection.modify(name="new_name") # Rename collection
```

## Adding and Managing Data

### Adding Documents

Chroma can automatically embed documents for you:

```python
collection.add(
    documents=["This is a document about pineapple", "This is a document about oranges"],
    ids=["id1", "id2"]
)
```

You can also add metadata associated with each document:

```python
collection.add(
    documents=["Chapter 1 content...", "Chapter 2 content..."],
    metadatas=[{"chapter": "1", "topic": "introduction"}, {"chapter": "2", "topic": "methodology"}],
    ids=["doc1", "doc2"]
)
```

### Adding Pre-computed Embeddings

If you've already embedded your documents, you can add the embeddings directly:

```python
collection.add(
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4]],
    documents=["doc1", "doc2"],
    metadatas=[{"source": "book"}, {"source": "article"}],
    ids=["id1", "id2"]
)
```

You can also store only embeddings without documents:

```python
collection.add(
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4]],
    metadatas=[{"source": "book"}, {"source": "article"}],
    ids=["id1", "id2"]
)
```

### Updating Data

To update existing documents or embeddings, use `update` or `upsert`:

```python
# Update existing documents (will fail if IDs don't exist)
collection.update(
    documents=["Updated document content"],
    metadatas=[{"updated": True}],
    ids=["id1"]
)

# Update if exists, add if not (ideal for incremental updates)
collection.upsert(
    documents=["New or updated document content"],
    metadatas=[{"updated": True}],
    ids=["id1"]
)
```

### Deleting Data

```python
# Delete by IDs
collection.delete(ids=["id1", "id2"])

# Delete by metadata filter
collection.delete(where={"source": "book"})
```

## Querying Data

### Basic Query

```python
# Query by text (Chroma will embed this for you)
results = collection.query(
    query_texts=["This is a query about tropical fruits"],
    n_results=2
)

# Query by embeddings
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=5
)
```

### Query Results

Query results are returned as a dictionary with the following keys:

```python
{
  'documents': [['doc1', 'doc2', ...]],  # List of document texts
  'ids': [['id1', 'id2', ...]],          # List of document IDs
  'distances': [[0.1, 0.2, ...]],        # List of distances from query
  'metadatas': [[{...}, {...}, ...]],    # List of metadata dictionaries
  'embeddings': None                     # Embeddings (None by default)
}
```

### Selecting Return Data

You can control which data is returned:

```python
# Only return documents and IDs
results = collection.query(
    query_texts=["Query text"],
    include=["documents"],
    n_results=5
)

# Return everything including embeddings
results = collection.query(
    query_texts=["Query text"],
    include=["documents", "metadatas", "distances", "embeddings"],
    n_results=5
)
```

### Getting Data by ID

```python
# Get items by ID
items = collection.get(ids=["id1", "id2"])

# Get all items
all_items = collection.get()

# Get items with filter
filtered_items = collection.get(where={"chapter": "1"})
```

## Embedding Functions

Chroma uses embedding functions to convert text into vector embeddings. You can use the built-in options or create your own.

### Default Embedding Function

By default, Chroma uses the Sentence Transformers `all-MiniLM-L6-v2` model:

```python
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
```

### Sentence Transformers Embedding Function

You can specify a different Sentence Transformers model:

```python
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

### OpenAI Embedding Function

```python
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-ada-002"
)
```

### Google Generative AI Embedding Function

```python
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key="your-api-key",
    model_name="models/embedding-001"
)
```

### Cohere Embedding Function

```python
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key="your-api-key",
    model_name="embed-english-v2.0"
)
```

### Custom Embedding Function

You can create your own embedding function:

```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Implement your embedding logic here
        return embeddings
```

## Metadata Filtering

Chroma supports filtering queries by metadata using a where filter:

### Basic Filters

```python
# Equal to
collection.query(
    query_texts=["Query text"],
    where={"metadata_field": "value"}
)

# Comparison operators
collection.query(
    query_texts=["Query text"],
    where={"numeric_field": {"$gt": 10}}  # Greater than 10
)
```

### Supported Operators

- `$eq` - Equal to (string, int, float)
- `$ne` - Not equal to (string, int, float)
- `$gt` - Greater than (int, float)
- `$gte` - Greater than or equal to (int, float)
- `$lt` - Less than (int, float)
- `$lte` - Less than or equal to (int, float)
- `$in` - Value is in a list (string, int, float, bool)
- `$nin` - Value is not in a list (string, int, float, bool)

### Logical Operators

```python
# AND operator
collection.query(
    query_texts=["Query text"],
    where={
        "$and": [
            {"field1": {"$gt": 10}},
            {"field2": "value"}
        ]
    }
)

# OR operator
collection.query(
    query_texts=["Query text"],
    where={
        "$or": [
            {"field1": {"$gt": 10}},
            {"field2": "value"}
        ]
    }
)
```

### Inclusion Operators

```python
# IN operator
collection.query(
    query_texts=["Query text"],
    where={"category": {"$in": ["science", "technology", "math"]}}
)

# NOT IN operator
collection.query(
    query_texts=["Query text"],
    where={"category": {"$nin": ["fiction", "history"]}}
)
```

## Best Practices

### Collection Organization

1. **Logical Grouping**: Create collections based on logical groups of data
2. **Size Management**: Keep collection sizes manageable (smaller collections perform better)
3. **Metadata Schema**: Plan your metadata schema in advance for efficient filtering

### Performance Optimization

1. **Batch Operations**: Add documents in batches rather than one at a time
2. **Query Size**: Limit `n_results` to only what you need
3. **Include Parameter**: Only request the data you need using the `include` parameter
4. **Persistent Client**: Use the persistent client for production workloads
5. **Client-Server Mode**: For production deployments, use client-server mode

### Data Management

1. **Unique IDs**: Use a consistent ID scheme for your documents
2. **Metadata Design**: Include meaningful metadata to enable powerful filtering
3. **Document Size**: Keep documents at an appropriate size for your embedding model
4. **Upsert**: Use `upsert` for incremental updates to avoid duplicates

## Example Patterns

### RAG (Retrieval-Augmented Generation) Implementation

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize client and embedding function
client = chromadb.PersistentClient(path="./rag_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()

# Create or get collection
collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_function
)

# Add documents to knowledge base
collection.add(
    documents=[
        "Jupiter is the largest planet in our solar system.",
        "Mars is known as the red planet.",
        "Venus is the hottest planet in our solar system."
    ],
    metadatas=[
        {"topic": "astronomy", "planet": "jupiter"},
        {"topic": "astronomy", "planet": "mars"},
        {"topic": "astronomy", "planet": "venus"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# RAG query function
def rag_query(user_question, n_results=3):
    # Retrieve relevant documents
    results = collection.query(
        query_texts=[user_question],
        n_results=n_results
    )
    
    # Format results for LLM context
    context = "\n".join(results["documents"][0])
    
    # Construct prompt with context and question
    prompt = f"""
    Based on the following information:
    {context}
    
    Please answer the question: {user_question}
    """
    
    # Send prompt to LLM (implementation depends on your LLM setup)
    # llm_response = call_llm_api(prompt)
    
    return prompt  # In a real implementation, return llm_response

# Example usage
query_result = rag_query("Which planet is the largest?")
print(query_result)
```

### Document Chunking and Embedding

```python
import chromadb
from chromadb.utils import embedding_functions
import uuid

# Initialize Chroma
client = chromadb.PersistentClient(path="./document_store")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
collection = client.get_or_create_collection("documents", embedding_function=embedding_function)

def chunk_document(text, chunk_size=1000, overlap=200):
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Make sure we don't cut words in the middle
        if end < len(text):
            # Find the last period or space before the end
            last_period = text.rfind('.', start, end)
            last_space = text.rfind(' ', start, end)
            
            if last_period > start + chunk_size // 2:
                end = last_period + 1  # Include the period
            elif last_space > start + chunk_size // 2:
                end = last_space + 1  # Include the space
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def process_document(doc_text, doc_metadata=None):
    """Process a document: chunk it and add to Chroma."""
    chunks = chunk_document(doc_text)
    
    # Create IDs and metadata for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Create metadata for each chunk
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = {"chunk_index": i, "chunk_count": len(chunks)}
        
        # Add document metadata if provided
        if doc_metadata:
            chunk_metadata.update(doc_metadata)
            
        metadatas.append(chunk_metadata)
    
    # Add chunks to collection
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    return ids

# Example usage
sample_document = """
This is a long document that would be split into multiple chunks.
It contains information about various topics that might be relevant for retrieval.
The chunking strategy ensures that we don't cut sentences in the middle, which helps maintain context.
Overlapping chunks help ensure that information that spans chunk boundaries isn't lost during retrieval.
"""

doc_metadata = {
    "title": "Sample Document",
    "author": "Chroma Guide",
    "date": "2025-03-28"
}

chunk_ids = process_document(sample_document, doc_metadata)
print(f"Document processed into {len(chunk_ids)} chunks")
```

### Multimodal Search Implementation

```python
import chromadb
import numpy as np
from chromadb.utils import embedding_functions

# Initialize client
client = chromadb.PersistentClient(path="./multimodal_db")

# Create a collection for text
text_ef = embedding_functions.SentenceTransformerEmbeddingFunction()
text_collection = client.get_or_create_collection("text_data", embedding_function=text_ef)

# In a real implementation, you would have an image embedding function
# This is a placeholder example
class MockImageEmbeddingFunction:
    def __call__(self, input_images):
        # Simulate creating embeddings (in real implementation, use a vision model)
        return [np.random.rand(384).tolist() for _ in input_images]

image_ef = MockImageEmbeddingFunction()
image_collection = client.get_or_create_collection("image_data")

# Add text data
text_collection.add(
    documents=[
        "A red sports car on a racetrack",
        "A blue sedan parked in a driveway",
        "A green SUV driving through the mountains"
    ],
    metadatas=[
        {"type": "text", "subject": "car", "color": "red"},
        {"type": "text", "subject": "car", "color": "blue"},
        {"type": "text", "subject": "car", "color": "green"}
    ],
    ids=["text1", "text2", "text3"]
)

# Add image data (in a real implementation, these would be image files or URLs)
image_collection.add(
    embeddings=image_ef(["image1.jpg", "image2.jpg", "image3.jpg"]),
    metadatas=[
        {"type": "image", "subject": "car", "color": "red"},
        {"type": "image", "subject": "car", "color": "blue"},
        {"type": "image", "subject": "car", "color": "green"}
    ],
    ids=["img1", "img2", "img3"]
)

# Function to perform multimodal search
def multimodal_search(query, filter_criteria=None):
    results = {}
    
    # Search text collection
    text_results = text_collection.query(
        query_texts=[query],
        where=filter_criteria,
        n_results=3
    )
    results["text"] = text_results
    
    # In a real implementation, decide whether to search by text or image
    # For example purposes, we're just searching both
    
    # For image search, you would typically:
    # 1. Generate an embedding from the query text or image
    # 2. Search the image collection with that embedding
    
    # This is a simplified example
    if filter_criteria:
        image_results = image_collection.get(where=filter_criteria)
    else:
        # In a real implementation, you'd convert the text query to an image embedding
        # and search with that embedding
        image_results = {"ids": [["img1", "img2", "img3"]]}
    
    results["images"] = image_results
    
    return results

# Example usage
search_results = multimodal_search("red car", {"color": "red"})
print(search_results)
```

## Final Notes

When using Chroma for retrieval applications:

1. **Data Quality**: The quality of your embeddings depends on the quality of your data
2. **Embedding Models**: Choose embedding models appropriate for your domain and task
3. **Chunking Strategy**: Develop a good chunking strategy for larger documents
4. **Metadata Strategy**: Design your metadata schema to support your filtering needs
5. **Persistence**: Use persistent storage for production workloads
6. **Error Handling**: Implement proper error handling in your code
7. **Performance Monitoring**: Monitor performance metrics in production

This guide provides a comprehensive overview of using Chroma for vector database applications. By following these patterns and best practices, you can effectively leverage Chroma's capabilities for retrieval-augmented generation, semantic search, and other AI applications.