# phoenix/core/models.py - Core data structures for Phoenix
import time # Moved import to the top
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Message:
    """Represents a single message within a conversation."""
    msg_id: str
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: float # Unix timestamp

@dataclass
class Conversation:
    """Represents a full conversation thread."""
    conv_id: str
    title: Optional[str] = None
    create_time: float = field(default_factory=time.time) # Unix timestamp
    model_slug: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) # Store other details like original_id, update_time
    # Optional fields for storing generated summary/insights directly, though often stored separately
    # summary: Optional[str] = None
    # insight_notes: List['InsightNote'] = field(default_factory=list)

@dataclass
class Chunk:
    """Represents a text chunk derived from a conversation for embedding."""
    chunk_id: str
    conv_id: str
    chunk_text: str
    start_time: float # Timestamp of the first message contributing to the chunk
    end_time: float   # Timestamp of the last message contributing to the chunk
    message_ids: List[str] # IDs of messages included in this chunk
    token_count: Optional[int] = None # Estimated token count
    # Metadata inherited from conversation can be added here or looked up via conv_id
    title: Optional[str] = None
    model_slug: Optional[str] = None

@dataclass
class InsightNote:
    """Represents a generated insight or fact about a conversation or user."""
    insight_id: str
    original_conv_id: str
    insight_text: str
    doc_type: str = "insight_note" # To distinguish in vector DB
    insight_token_count: Optional[int] = None
    # Metadata inherited from conversation
    title: Optional[str] = None
    create_time: float = field(default_factory=time.time)
    model_slug: Optional[str] = None
    # tags: List[str] = field(default_factory=list) # Tags removed as per latest indexer logic

@dataclass
class ConversationSummary:
    """Represents a generated summary of a conversation."""
    summary_id: str
    original_conv_id: str
    summary_text: str
    doc_type: str = "conversation_summary" # To distinguish in vector DB
    summary_token_count: Optional[int] = None
    # Metadata inherited from conversation
    title: Optional[str] = None
    create_time: float = field(default_factory=time.time)
    model_slug: Optional[str] = None

# Removed redundant import from the end
