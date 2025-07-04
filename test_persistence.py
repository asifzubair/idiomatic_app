import os
import json
from idiomatic.utils.data_utils import save_user_data, load_user_data
from idiomatic.core.schemas import IdiomaticState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Define a temporary file path for testing
TEST_USER_DATA_PATH = "test_user_data.json"

def run_test():
    print("--- Starting Persistence Test ---")

    # 1. Create a sample IdiomaticState
    sample_state: IdiomaticState = {
        "messages": [
            HumanMessage(content="Hello Idiomatic!"),
            AIMessage(content="Hello User! How can I help?"),
            SystemMessage(content="System is ready."),
            HumanMessage(content="What does 'break a leg' mean?"),
            AIMessage(content="'Break a leg' means good luck!")
        ],
        "name": "Test User",
        "user_level": "intermediate",
        "category": "general",
        "score": 10,
        "history": ["some history item"],
        "repetition_schedule": {"break a leg": {"last_seen": "2023-01-01T12:00:00Z", "success": True}},
        "finished": False,
        "last_question": {"idiom": "break a leg", "question": "What is it?", "options": [], "answer": "a"}
    }
    print(f"\n1. Original Sample State (first message type): {type(sample_state['messages'][0])}")

    try:
        # 2. Call save_user_data
        try:
            print(f"\n2. Saving data to {TEST_USER_DATA_PATH}...")
            save_user_data(sample_state, TEST_USER_DATA_PATH)
            print("Save successful.")
        except Exception as e:
            print(f"Error during save_user_data: {e}")
            return # Exit run_test if save fails

        # 3. Print the content of the JSON file
        try:
            print(f"\n3. Content of {TEST_USER_DATA_PATH}:")
            with open(TEST_USER_DATA_PATH, 'r') as f:
                raw_content = f.read()
                print(raw_content)
                # Basic check for serialized message structure
                saved_data = json.loads(raw_content)
                if not (saved_data and 'messages' in saved_data and isinstance(saved_data['messages'], list) and \
                        all(isinstance(m, dict) and 'type' in m and 'content' in m for m in saved_data['messages'])):
                    print("ERROR: Saved messages are not in the expected dictionary format.")
                    return # Exit run_test if format is wrong
                print("Serialized messages format looks OK.")

        except Exception as e:
            print(f"Error reading or verifying {TEST_USER_DATA_PATH}: {e}")
            return # Exit run_test if read/verify fails

        # 4. Call load_user_data
        loaded_state = None
        try:
            print(f"\n4. Loading data from {TEST_USER_DATA_PATH}...")
            loaded_state = load_user_data(TEST_USER_DATA_PATH)
            print("Load successful.")
        except Exception as e:
            print(f"Error during load_user_data: {e}")
            return # Exit run_test if load fails

        # 5. Print the loaded state and verify messages
        if loaded_state:
            print("\n5. Loaded State Verification:")
            print(f"  Name: {loaded_state.get('name')}")
            print(f"  Score: {loaded_state.get('score')}")
            if 'messages' in loaded_state and loaded_state['messages']:
                first_loaded_message = loaded_state['messages'][0]
                print(f"  First loaded message type: {type(first_loaded_message)}")
                print(f"  First loaded message content: {first_loaded_message.content}")

                # Verify all messages are reconstructed correctly
                original_messages = sample_state['messages']
                loaded_messages = loaded_state['messages']
                if len(original_messages) == len(loaded_messages) and \
                   all(isinstance(loaded_msg, type(original_msg)) and \
                       loaded_msg.content == original_msg.content \
                       for original_msg, loaded_msg in zip(original_messages, loaded_messages)):
                    print("  SUCCESS: All messages correctly deserialized to their original types and content.")
                else:
                    print("  ERROR: Messages were not correctly deserialized.")
                    print("  Original messages:")
                    for i, msg in enumerate(original_messages):
                        print(f"    {i}: type={type(msg)}, content='{msg.content}'")
                    print("  Loaded messages:")
                    for i, msg in enumerate(loaded_messages):
                        print(f"    {i}: type={type(msg)}, content='{msg.content}'")
            else:
                print("  ERROR: Messages not found in loaded state or empty.")
        else:
            print("  ERROR: Loaded state is None.")

    finally:
        # 6. Clean up
        if os.path.exists(TEST_USER_DATA_PATH):
            os.remove(TEST_USER_DATA_PATH)
            print(f"\n6. Cleaned up {TEST_USER_DATA_PATH}.")

    print("\n--- Persistence Test Finished ---")

if __name__ == "__main__":
    run_test()
