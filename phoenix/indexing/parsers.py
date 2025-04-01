# phoenix/indexing/parsers.py - Conversation Data Parsers
import json
import uuid
import time
from typing import Dict, List, Optional

from phoenix.core.models import Message, ConversationMetadata, ParsedConversation

class OpenAIParser:
    """Parser for OpenAI conversation exports (conversations.json format)."""

    def parse_conversation_data(self, data: Dict) -> Optional[ParsedConversation]:
        """
        Parses a single raw conversation entry from the OpenAI export list.

        Args:
            data: A dictionary representing a single conversation from the export.

        Returns:
            A ParsedConversation object containing metadata and messages, or None if parsing fails.
        """
        if not isinstance(data, dict) or 'mapping' not in data:
            # print(f"Skipping invalid conversation data structure: {str(data)[:100]}...")
            return None # Skip items that aren't valid conversation dicts

        conv_id = data.get('id', f"conv_{uuid.uuid4()}") # Use 'conv_' prefix for consistency
        title = data.get('title', 'Untitled Conversation')
        create_time = data.get('create_time') # Keep as original timestamp (float)
        model_slug = data.get('current_model_slug') or data.get('model_slug') or "unknown"

        # Extract messages in chronological order
        messages_data = self._extract_thread(data.get('mapping', {}), data.get('current_node'))

        if not messages_data:
            # print(f"Warning: No messages extracted for conversation '{title}' (ID: {conv_id}). Skipping.")
            return None

        # Create Message objects
        messages = [Message(**msg_data) for msg_data in messages_data]

        # Basic metadata from the source file
        source_metadata = {
            "original_openai_id": data.get("id"),
            "openai_update_time": data.get("update_time"),
            "openai_current_node": data.get("current_node"),
            # Add other potentially useful raw fields if needed
        }

        # Create ConversationMetadata object
        conv_metadata = ConversationMetadata(
            conv_id=conv_id,
            title=title,
            create_time=float(create_time) if create_time else time.time(),
            model_slug=model_slug,
            metadata=source_metadata
        )

        return ParsedConversation(metadata=conv_metadata, messages=messages)

    def _extract_thread(self, mapping: Dict, current_node_id: Optional[str]) -> List[Dict]:
        """
        Extracts the main message thread leading to the current node.
        Returns a list of dictionaries, ready to be converted to Message objects.
        """
        if not mapping or not current_node_id or current_node_id not in mapping:
            # Attempt to find a root node if current_node_id is invalid or missing
            roots = [nid for nid, node_info in mapping.items() if not node_info.get('parent')]
            if not roots:
                return []
            # If multiple roots, maybe pick the one with the latest timestamp? For now, pick first.
            # A more robust approach might be needed depending on data variations.
            current_node_id = roots[0]
            # print(f"Warning: Invalid/missing current_node_id. Using root node {current_node_id}.")

        thread_data: List[Dict] = []
        node_id = current_node_id

        while node_id:
            node = mapping.get(node_id)
            if not node:
                # print(f"Warning: Node ID {node_id} not found in mapping. Stopping thread extraction.")
                break # Should not happen in valid data

            message_data = node.get('message')
            # Only include nodes with actual message content and author role
            if message_data and message_data.get("content") and message_data.get("author", {}).get("role"):
                role = message_data["author"]["role"]
                if role not in ["user", "assistant", "system", "tool"]: # Allow 'tool' role
                    # print(f"Warning: Normalizing unknown role '{role}' to 'system' for message {message_data.get('id')}.")
                    role = "system" # Normalize unknown roles

                content_data = message_data.get("content", {})
                text_content = self._extract_text_from_content(content_data)

                if text_content is not None and text_content.strip(): # Only add if there's non-whitespace content
                    msg_time = message_data.get('create_time')
                    msg_id = message_data.get("id", f"msg_{uuid.uuid4()}") # Generate ID if missing

                    thread_data.append({
                        "msg_id": msg_id,
                        "role": role,
                        "content": text_content,
                        "timestamp": float(msg_time) if msg_time else time.time(), # Fallback timestamp
                        "metadata": { # Store original content structure for reference if needed
                            "openai_content_type": content_data.get("content_type"),
                            "openai_message_metadata": message_data.get("metadata")
                        }
                    })

            node_id = node.get('parent') # Move up the chain

        return thread_data[::-1] # Reverse to get chronological order

    def _extract_text_from_content(self, content_data: Dict) -> Optional[str]:
        """Extracts usable text from various OpenAI content types."""
        content_type = content_data.get("content_type", "text")
        text_content = None

        try:
            if content_type == "text":
                parts = content_data.get("parts", [])
                if parts and isinstance(parts[0], str):
                    text_content = parts[0]
            elif content_type == "code":
                 text_content = f"```\n{content_data.get('text', '')}\n```"
            elif content_type == "tether_quote":
                 text_content = f"> {content_data.get('text', '')}" # Basic quote formatting
            elif content_type in ["tether_browsing_display", "multimodal_text"]:
                 # Try to extract text - might need refinement based on actual data variations
                 parts = content_data.get("parts", [])
                 extracted_parts = []
                 if isinstance(parts, list) and parts:
                     for part in parts:
                         if isinstance(part, str):
                             extracted_parts.append(part)
                         elif isinstance(part, dict) and 'text' in part: # Simple handling for dict parts with text
                             extracted_parts.append(part['text'])
                         # Add more specific handling for image parts etc. if needed later
                 elif isinstance(content_data.get("text"), str):
                      extracted_parts.append(content_data["text"])

                 if extracted_parts:
                     text_content = "\n".join(extracted_parts)
                 else:
                     # Fallback if no text parts found
                     text_content = f"[{content_type} content]"
            # Add specific handlers for other types like system messages, tool calls/outputs if needed
            elif content_type == "system_error":
                text_content = f"[System Error: {content_data.get('text')}]"
            # Default for unhandled types - maybe log or return placeholder
            else:
                 # print(f"Warning: Unhandled content type '{content_type}'. Omitting content.")
                 text_content = f"[{content_type} content omitted]"

        except Exception as e:
            # print(f"Error extracting text from content_type '{content_type}': {e}")
            text_content = f"[Error processing {content_type} content]"

        return text_content


# Example Usage (for testing)
if __name__ == "__main__":
    print("Testing OpenAIParser...")

    # Load sample data (assuming sample_conversations.json exists in the project root)
    sample_file = "../../sample_conversations.json" # Adjust path relative to this file
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            all_sample_data = json.load(f)
        print(f"Loaded {len(all_sample_data)} conversations from {sample_file}")

        parser = OpenAIParser()
        parsed_count = 0
        for i, raw_conv in enumerate(all_sample_data[:5]): # Test first 5
            print(f"\n--- Parsing Sample Conversation {i+1} ---")
            parsed_conv_obj = parser.parse_conversation_data(raw_conv)

            if parsed_conv_obj:
                parsed_count += 1
                print(f"  Conv ID: {parsed_conv_obj.metadata.conv_id}")
                print(f"  Title: {parsed_conv_obj.metadata.title}")
                print(f"  Model: {parsed_conv_obj.metadata.model_slug}")
                print(f"  Message Count: {len(parsed_conv_obj.messages)}")
                if parsed_conv_obj.messages:
                    print(f"  First Message Role: {parsed_conv_obj.messages[0].role}")
                    print(f"  First Message Content: {parsed_conv_obj.messages[0].content[:80]}...")
                    print(f"  Last Message Role: {parsed_conv_obj.messages[-1].role}")
                    print(f"  Last Message Content: {parsed_conv_obj.messages[-1].content[:80]}...")
                # Add more assertions if needed
                assert isinstance(parsed_conv_obj, ParsedConversation)
                assert isinstance(parsed_conv_obj.metadata, ConversationMetadata)
                assert isinstance(parsed_conv_obj.messages, list)
                if parsed_conv_obj.messages:
                    assert isinstance(parsed_conv_obj.messages[0], Message)
            else:
                print("  Parsing failed or resulted in None.")

        print(f"\nSuccessfully parsed {parsed_count} out of 5 sample conversations.")

    except FileNotFoundError:
        print(f"ERROR: Sample file not found at {sample_file}. Cannot run parser test.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    print("\nOpenAIParser testing finished.")
