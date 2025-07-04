import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

# USER_DATA_PATH is passed as an argument (user_data_path)

def _serialize_messages(messages: list) -> list[dict]:
    """Converts a list of BaseMessage objects to a list of serializable dictionaries."""
    serialized_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            serialized_messages.append({"type": "ai", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            serialized_messages.append({"type": "human", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            serialized_messages.append({"type": "system", "content": msg.content})
        elif isinstance(msg, dict) and "type" in msg and "content" in msg:
            # Already serialized, pass through
            serialized_messages.append(msg)
        else:
            # Fallback for other message types or if already a dict but not our format
            serialized_messages.append({"type": "unknown", "content": str(msg)})
    return serialized_messages

def _deserialize_messages(serialized_messages: list[dict]) -> list[BaseMessage]:
    """Converts a list of serialized message dictionaries back to BaseMessage objects."""
    messages = []
    for msg_data in serialized_messages:
        msg_type = msg_data.get("type")
        content = msg_data.get("content")
        if msg_type == "ai":
            messages.append(AIMessage(content=content))
        elif msg_type == "human":
            messages.append(HumanMessage(content=content))
        elif msg_type == "system":
            messages.append(SystemMessage(content=content))
        else:
            # Fallback or if it's an unknown type, store as a generic message or raise error
            # For now, let's just recreate as AIMessage if type is unknown, to avoid breaking things.
            # A more robust solution might involve a default message type or logging a warning.
            print(f"Warning: Unknown message type '{msg_type}' during deserialization. Content: {content}")
            messages.append(AIMessage(content=f"({msg_type or 'unknown'}): {content}")) # Preserve content
    return messages

def load_user_data(user_data_path: str):
    """Utility to load or initialize user data from the given path."""
    try:
        with open(user_data_path, 'r') as file:
            data = json.load(file)
            if 'messages' in data and isinstance(data['messages'], list):
                data['messages'] = _deserialize_messages(data['messages'])
            return data
    except FileNotFoundError:
        # Return a default structure if the file doesn't exist
        # Ensure this default structure matches IdiomaticState expectations
        return {
            "name": "",
            "score": 0,
            "history": [],
            "repetition_schedule": {},
            "messages": [],
            "last_question": None,
            "user_level": "intermediate",
            "category": "general",
            "finished": False # Ensure all IdiomaticState fields have defaults
        }

def save_user_data(user_data, user_data_path: str):
    """Saves user data to the given path, serializing messages if necessary."""
    if 'messages' in user_data and isinstance(user_data['messages'], list):
        # Ensure messages are serializable
        user_data_copy = user_data.copy() # Avoid modifying the original dict in-memory if it's used elsewhere
        user_data_copy['messages'] = _serialize_messages(user_data_copy['messages'])
    else:
        user_data_copy = user_data

    with open(user_data_path, 'w') as file:
        json.dump(user_data_copy, file, indent=4) # Added indent for readability
