# phoenix/adapters/parsers/openai_parser.py - Parser for OpenAI conversation exports
import json
import uuid
import time
from typing import Dict, List, Optional, Any

# Import core models
from phoenix.core.models import Conversation, Message

class OpenAIParser:
    """Parser for OpenAI conversation exports (conversations.json)."""

    def parse_conversation_data(self, data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Parses a single conversation entry from the conversations.json list
        into a Conversation object.

        Args:
            data: A dictionary representing a single conversation entry from the JSON.

        Returns:
            A Conversation object if parsing is successful, otherwise None.
        """
        if not isinstance(data, dict) or not data.get('mapping'):
            print(f"Skipping invalid conversation data structure: {str(data)[:100]}...")
            return None # Skip items that aren't valid conversation dicts

        conv_id = data.get('id', f"gen_{uuid.uuid4()}") # Generate ID if missing
        title = data.get('title', 'Untitled Conversation')
        create_time = data.get('create_time') # Keep as original timestamp (float)
        model_slug = data.get('current_model_slug') or data.get('model_slug') or "unknown"

        # Extract messages in chronological order
        messages: List[Message] = self._extract_thread(data.get('mapping', {}), data.get('current_node'))

        if not messages:
            # print(f"Warning: No messages extracted for conversation '{title}' (ID: {conv_id}). Skipping.")
            # Don't skip the whole conversation, just proceed without messages if necessary
            # Allow creating a Conversation object even without messages, metadata might still be useful
            pass # Continue processing even if messages are empty

        # Basic metadata from the original export
        metadata = {
            "original_id": data.get("id"),
            "update_time": data.get("update_time"),
            # Add any other top-level fields from 'data' you want to preserve
        }

        # Create and return the Conversation object
        return Conversation(
            conv_id=conv_id,
            title=title,
            create_time=float(create_time) if create_time else time.time(),
            model_slug=model_slug,
            messages=messages,
            metadata=metadata
        )

    def _extract_thread(self, mapping: Dict[str, Any], current_node_id: Optional[str]) -> List[Message]:
        """
        Extracts the main message thread leading to the current node.

        Args:
            mapping: The 'mapping' dictionary from the conversation data.
            current_node_id: The ID of the 'current_node' in the conversation.

        Returns:
            A list of Message objects in chronological order.
        """
        if not mapping or not current_node_id or current_node_id not in mapping:
            # Attempt to find a root node if current_node_id is invalid or missing
            roots = [nid for nid, node in mapping.items() if not node.get('parent')]
            if not roots:
                print(f"⚠️ Warning: Could not determine starting node for thread extraction.")
                return []
            # If multiple roots, maybe pick the one with the latest timestamp? For now, pick first.
            current_node_id = roots[0]
            print(f"⚠️ Warning: Invalid 'current_node'. Starting extraction from root node {current_node_id}.")
            # Consider more robust root finding if needed

        thread: List[Message] = []
        node_id = current_node_id

        while node_id:
            node = mapping.get(node_id)
            if not node:
                print(f"⚠️ Warning: Node ID '{node_id}' not found in mapping during thread extraction.")
                break # Should not happen in valid data, but break defensively

            message_data = node.get('message')
            # Only include nodes with actual message content
            if message_data and message_data.get("content"):
                role = message_data.get("author", {}).get("role", "system")
                # Normalize unknown roles to 'system' or handle as needed
                if role not in ["user", "assistant", "system", "tool"]: # Added 'tool' role
                    # print(f"Normalizing unknown role '{role}' to 'system'.")
                    role = "system"

                content_data = message_data.get("content", {})
                content_type = content_data.get("content_type", "text")
                text_content = ""

                # Extract text based on content type
                if content_type == "text":
                    parts = content_data.get("parts", [])
                    # Handle cases where parts might be None or not a list
                    if isinstance(parts, list) and parts and isinstance(parts[0], str):
                        text_content = parts[0]
                    elif isinstance(content_data.get("text"), str): # Fallback for older formats?
                         text_content = content_data["text"]
                elif content_type == "code": # Handle code blocks
                     text_content = f"```\n{content_data.get('text', '')}\n```"
                # Add handling for other types like multimodal later if needed
                elif content_type in ["tether_quote", "tether_browsing_display", "multimodal_text", "system_error"]:
                     # Try to extract text - might need refinement based on actual data examples
                     parts = content_data.get("parts", [])
                     text_value = content_data.get("text") # Some types might just have 'text'

                     if isinstance(parts, list) and parts:
                         # Simple extraction, might need adjustment for complex parts
                         extracted_parts = [str(p) for p in parts if isinstance(p, (str, int, float))]
                         text_content = " ".join(extracted_parts)
                     elif isinstance(text_value, str):
                          text_content = text_value
                     else:
                          # Fallback representation if text extraction is complex/unclear
                          text_content = f"[{content_type} content]"
                else:
                     # Fallback for completely unknown or unhandled types
                     text_content = f"[{content_type} content omitted]"


                # Only add if there's non-whitespace content
                if text_content and text_content.strip():
                    msg_time = message_data.get('create_time')
                    msg_id = message_data.get("id", f"msg_{uuid.uuid4()}")

                    # Create Message object
                    message_obj = Message(
                        msg_id=msg_id,
                        role=role,
                        content=text_content.strip(),
                        timestamp=float(msg_time) if msg_time else time.time() # Fallback timestamp
                    )
                    thread.append(message_obj)

            # Move up the chain to the parent node
            node_id = node.get('parent')

        return thread[::-1] # Reverse to get chronological order (oldest first)

# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("Testing OpenAIParser...")

    # Example minimal conversation data structure
    test_data = {
        "id": "conv_test_123",
        "title": "Test Parsing",
        "create_time": 1678886400.0,
        "current_node": "node_2",
        "mapping": {
            "node_1": {
                "id": "node_1",
                "message": {
                    "id": "msg_abc",
                    "author": {"role": "user"},
                    "create_time": 1678886401.0,
                    "content": {"content_type": "text", "parts": ["Hello Phoenix!"]}
                },
                "parent": None,
                "children": ["node_2"]
            },
            "node_2": {
                "id": "node_2",
                "message": {
                    "id": "msg_def",
                    "author": {"role": "assistant"},
                    "create_time": 1678886405.0,
                    "content": {"content_type": "text", "parts": ["Hello User! How can I help?"]}
                },
                "parent": "node_1",
                "children": []
            }
        }
    }

    parser = OpenAIParser()
    parsed_conversation = parser.parse_conversation_data(test_data)

    if parsed_conversation:
        print("\nParsed Conversation Object:")
        print(f"  ID: {parsed_conversation.conv_id}")
        print(f"  Title: {parsed_conversation.title}")
        print(f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(parsed_conversation.create_time))}")
        print(f"  Metadata: {parsed_conversation.metadata}")
        print(f"  Messages ({len(parsed_conversation.messages)}):")
        for msg in parsed_conversation.messages:
            print(f"    - [{msg.role}] {msg.content[:50]}...")
    else:
        print("\nParsing failed.")

    # Test invalid data
    print("\nTesting invalid data:")
    invalid_data = {"foo": "bar"}
    result_invalid = parser.parse_conversation_data(invalid_data)
    print(f"Result for invalid data: {result_invalid}")

    print("\nTesting data with no messages:")
    no_message_data = {
        "id": "conv_test_456",
        "title": "No Messages Test",
        "create_time": 1678886500.0,
        "current_node": "node_x",
        "mapping": {
             "node_x": { "id": "node_x", "message": None, "parent": None, "children": [] }
        }
    }
    result_no_msg = parser.parse_conversation_data(no_message_data)
    if result_no_msg:
         print(f"Parsed conversation with no messages (ID: {result_no_msg.conv_id}), Message count: {len(result_no_msg.messages)}")
    else:
         print("Parsing failed for no-message data.")
