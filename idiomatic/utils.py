# This file will house data persistence functions (load_user_data, save_user_data)
# and other general utilities, such as display fallbacks.

import json
from .config import IdiomaticConfig # Assuming IdiomaticConfig is in config.py at the same level

# IPython display fallbacks
try:
    from IPython.display import Markdown as IPMarkdown, display as IPDisplay
    # Use a basic print for display in non-notebook environments if preferred,
    # or let IPDisplay handle it if it can.
    # For simplicity, we can create wrappers that are safe to call.
    def display(obj):
        """Displays object, using IPython.display if available."""
        IPDisplay(obj)

    def Markdown(text: str):
        """Formats text as Markdown, using IPython.display.Markdown if available."""
        return IPMarkdown(text)

except ImportError:
    print("IPython.display not found. Using basic print for display and Markdown.")
    def display(obj): # Basic fallback
        """Basic print fallback for display."""
        print(str(obj))

    def Markdown(text: str): # Basic fallback
        """Basic string passthrough for Markdown if IPython is not available."""
        return text # Or could add markdown-like syntax if needed: f"**Markdown:**\n{text}"

# Data Persistence
def load_user_data(config: IdiomaticConfig) -> dict:
    """Utility to load or initialize user data based on config."""
    try:
        with open(config.user_data_path, 'r') as file:
            user_data = json.load(file)
            # Ensure all expected top-level keys for IdiomaticState are present
            # This helps prevent KeyErrors later if loading older/incomplete data
            # The keys are based on IdiomaticState, which will be defined in agent.py
            # For now, using a common set. This might need adjustment once IdiomaticState is moved.
            expected_keys = {"name": "", "user_level": "intermediate", "category": "general",
                             "score": 0, "history": [], "repetition_schedule": {},
                             "finished": False, "last_question": None, "messages": []}
            for key, default_value in expected_keys.items():
                if key not in user_data:
                    user_data[key] = default_value
            return user_data
    except FileNotFoundError:
        return {"name": "", "user_level": "intermediate", "category": "general",
                "score": 0, "history": [], "repetition_schedule": {},
                "finished": False, "last_question": None, "messages": []}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {config.user_data_path}. Initializing new data.")
        return {"name": "", "user_level": "intermediate", "category": "general",
                "score": 0, "history": [], "repetition_schedule": {},
                "finished": False, "last_question": None, "messages": []}


def save_user_data(config: IdiomaticConfig, user_data_state: dict):
    """Saves relevant parts of user data state based on config."""
    # Keys to save, should align with IdiomaticState structure eventually.
    # For now, using the known important fields.
    fields_to_save = ["name", "user_level", "category", "score", "history",
                      "repetition_schedule", "finished", "last_question"]

    data_to_save = {}
    for key in fields_to_save:
        data_to_save[key] = user_data_state.get(key) # Use .get() for safety

    with open(config.user_data_path, 'w') as file:
        json.dump(data_to_save, file, indent=2)
