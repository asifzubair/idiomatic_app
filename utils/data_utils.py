import json

USER_DATA_PATH = 'user_data.json'

def load_user_data():
    """Utility to load or initialize user data"""
    try:
        with open(USER_DATA_PATH, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"name": "", "score": 0, "history": [], "repetition_schedule": {}}

def save_user_data(user_data):
    with open(USER_DATA_PATH, 'w') as file:
        json.dump(user_data, file)
