import json

# USER_DATA_PATH is passed as an argument (user_data_path)

def load_user_data(user_data_path: str):
    """Utility to load or initialize user data from the given path."""
    try:
        with open(user_data_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Return a default structure if the file doesn't exist
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
    """Saves user data to the given path."""
    with open(user_data_path, 'w') as file:
        json.dump(user_data, file)
