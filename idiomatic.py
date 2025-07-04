## Imports
import os
import json
from datetime import datetime
from functools import partial

from IPython.display import Markdown, display
from langgraph.graph import END # Will be used by create_graph if not directly
from langchain_core.messages import AIMessage # For evaluate_quiz_answer
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.api_core import retry

# Local module imports
from core.schemas import IdiomaticState
from utils.data_utils import load_user_data, save_user_data
from generation.qna_generation import generate_idiom_question
# tools.idiom_tools contains tool functions and the raw tool_node
from tools.idiom_tools import show_score, explain_last_question, lookup_idiom, quit_session
from tools.idiom_tools import tool_node as base_tool_node # The ToolNode made from original tools
from orchestration.agent import chatbot_node, get_user_input, route_logic
from orchestration.graph import create_graph


# --- LLM and Client Initialization ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Retry mechanism for GenAI client
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

# Orchestration LLM (for chat, intent, tool use decisions)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Or "gemini-pro"
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7 # Lower temperature for more predictable routing/tool use
)

# Client for dedicated Q&A generation (can use different model/params)
# Using gemini-1.5-flash as per original generate_idiom_question
# The model name is already in generate_idiom_question, so client is just the API access
gen_client = genai.Client(api_key=GOOGLE_API_KEY)


## Data Persistence

"""
Initialize user data storage. Since this is a colab, we just use a JSON file for persistence
More formally, we could've used a sqlite3 db etc.
"""

# Moved to utils.data_utils

## Agent Workflow
# (IdiomaticState is imported from core.schemas)

## Question & Answer Generation

"""
This code block illustrates the following aspects of genAI capabilities:
 1. Structured output/JSON mode/controlled generation
 2. Few-shot prompting
 3. Optimizing model params for diverse output
"""

# Moved to generation.qna_generation

# evaluate_quiz_answer remains here or could be moved to a more specific "actions" module.
# For now, it's part of the main script's responsibility.
def evaluate_quiz_answer(state: IdiomaticState) -> IdiomaticState: # Use IdiomaticState
    print("--- Evaluating Answer ---")
    user_input = state["messages"][-1].content.strip().lower()
    correct = state["last_question"]["answer"].lower() if state.get("last_question") else None

    result_message = ""
    if user_input == correct:
        result_message = "✅ Correct!"
        state["score"] += 1
        success = True
    else:
        result_message = f"❌ Incorrect! The correct answer was '{correct}'."
        success = False
    
    display(Markdown(result_message))
    state['messages'].append(AIMessage(content=result_message))

    idiom = state["last_question"]["idiom"]
    state["repetition_schedule"][idiom] = {
        "last_seen": datetime.utcnow().isoformat(),
        "success": success
    }

    return state


## Tools

# Moved to tools.idiom_tools
# llm_with_tools will be created in the main script or graph setup.


## Orchestration LLM & Routing

# Moved to orchestration.agent and orchestration.graph

if True: # Placeholder for the main execution block, which will be refactored
    print("\nStarting Idiomatic Chatbot...")
    # Load initial data
    user_data = load_user_data()
    fields_to_save = ["name", "score", "history", "repetition_schedule"]
    initial_state = IdiomaticState(
        messages=[], # Start with empty messages for the graph
        name=user_data.get("name", ""),
        user_level=user_data.get("user_level", ""),
        category=user_data.get("category", "general"),
        score=user_data.get("score", 0),
        history=user_data.get("history", []),
        repetition_schedule=user_data.get("repetition_schedule", {}),
        finished=False,
        last_question=None # Initialize as None
    )

    # Run the graph stream
    try:
        # Use stream method for interactive input/output
        final_state = None
        for event in app.stream(initial_state, {"recursion_limit": 100}):
            # The stream automatically handles invoking nodes based on edges
            # We primarily care about the final state or handling specific events if needed
            # print(f"Graph Event: {event}") # For debugging graph flow
            # Extract the latest state from the event if needed for saving
                for key, value in event.items():
                    # Check if the value is a dictionary (likely represents node output with state)
                    if isinstance(value, dict) and 'messages' in value:
                        final_state = value # Capture the latest state snapshot
                        # Check if the 'finished' flag was set in this state update
                        if final_state.get('finished'):
                            print("--- Finishing Loop (detected finished flag) ---")
                            break # Exit the loop if finished flag is set


                if final_state and final_state.get('finished'):
                    break # Ensure loop terminates if finished flag detected


        print("\nSession Ended.")
        # Save final state if available
        if final_state:
                print("Saving final user data...")
                save_user_data({k:v for k, v in final_state.items() if k in fields_to_save})
        else:
                # If loop ended without final state, try saving initial (might have been modified)
                print("Saving user data (final state not captured)...")
                save_user_data({k:v for k, v in initial_state.items() if k in fields_to_save})


    except Exception as e:
        print(f"\nAn error occurred during the chat session: {e}")
        # Attempt to save current state on error
        print("Attempting to save current user data on error...")
        # If final_state has data, save it, otherwise save the initial_state which might have been updated
        save_user_data({k:v for k, v in final_state.items() if k in fields_to_save} if final_state else initial_state)
    finally:
            print("Idiomatic chatbot finished.")

else:
    print("Graph compilation failed. Cannot run the application.")