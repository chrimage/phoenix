# phoenix/core/models.py - Pydantic Data Models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import time

class Message(BaseModel):
    """Represents a single message within a conversation."""
    msg_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4()}")
    role: str # E.g., "user", "assistant", "system"
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None # For any extra message-specific data

class ConversationMetadata(BaseModel):
    """Metadata associated with a full conversation."""
    conv_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4()}")
    title: Optional[str] = "Untitled Conversation"
    create_time: float = Field(default_factory=time.time) # Unix timestamp
    model_slug: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None # Store other details as JSON (e.g., original OpenAI ID)

class BaseDocument(BaseModel):
    """Base model for documents stored in the vector database."""
    doc_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4()}")
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None # Embedding vector, usually handled by ChromaDB

class Chunk(BaseDocument):
    """Represents a text chunk derived from a conversation."""
    doc_type: str = "chunk"
    conv_id: str
    start_time: float
    end_time: float
    token_count: Optional[int] = None
    message_ids: List[str] = Field(default_factory=list) # IDs of messages included in the chunk

    # Add relevant fields from ConversationMetadata to metadata for easier filtering/retrieval
    def update_metadata(self, conv_meta: ConversationMetadata):
        self.metadata.update({
            "conv_id": self.conv_id,
            "title": conv_meta.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "token_count": self.token_count,
            "message_count": len(self.message_ids),
            "model_slug": conv_meta.model_slug,
            "doc_type": self.doc_type
        })

class Summary(BaseDocument):
    """Represents a summary of a conversation."""
    doc_type: str = "conversation_summary"
    original_conv_id: str
    summary_token_count: Optional[int] = None
    create_time: float = Field(default_factory=time.time)
    model_slug: Optional[str] = None
    title: Optional[str] = None

    def update_metadata(self):
        self.metadata.update({
            "doc_type": self.doc_type,
            "original_conv_id": self.original_conv_id,
            "summary_token_count": self.summary_token_count,
            "title": self.title,
            "create_time": self.create_time,
            "model_slug": self.model_slug
        })


class InsightNote(BaseDocument):
    """Represents an insight or fact extracted from a conversation."""
    doc_type: str = "insight_note"
    original_conv_id: str
    insight_token_count: Optional[int] = None
    create_time: float = Field(default_factory=time.time)
    model_slug: Optional[str] = None
    title: Optional[str] = None
    # tags: List[str] = Field(default_factory=list) # Tags removed as per latest logic

    def update_metadata(self):
        self.metadata.update({
            "doc_type": self.doc_type,
            # "tags": json.dumps(self.tags), # Tags removed
            "original_conv_id": self.original_conv_id,
            "insight_token_count": self.insight_token_count,
            "title": self.title,
            "create_time": self.create_time,
            "model_slug": self.model_slug
        })

class ParsedConversation(BaseModel):
    """Structure holding parsed conversation data before indexing."""
    metadata: ConversationMetadata
    messages: List[Message]
