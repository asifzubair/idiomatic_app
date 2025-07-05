# This file will contain the run_app function and other core logic.
import json
import os
from datetime import datetime
from dataclasses import dataclass # Keep for IdiomQuizItem for now
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from google import genai # As per original script
from google.genai import types # As per original script
from google.api_core import retry # As per original script

# Imports from our package
from .config import IdiomaticConfig # Assuming config is in the same package
# utils will be imported when its functions are moved

# Global LLM clients - to be initialized in run_app
llm = None
client = None
llm_with_tools = None

# app_config will be initialized in run_app and can be imported by other modules
app_config: IdiomaticConfig | None = None


# Import utilities from .utils
# display and Markdown will be used by functions that are being moved out of app.py,
# so they will import from utils directly.
# load_user_data and save_user_data will be used by run_app here.
from .utils import load_user_data, save_user_data
# Functions that need display/Markdown will do: from .utils import display, Markdown

# Agent Workflow State (IdiomaticState) is now defined in idiomatic.agent
# Q&A Generation logic is now in idiomatic.qna_generation
# Tool definitions are now in idiomatic.tools
# Agent nodes (chatbot_node, get_user_input) and system prompt are in idiomatic.agent
# Graph routing logic (route_logic) and graph construction will be in idiomatic.graph


# Placeholder for run_app - will be filled progressively
# app_config definition remains as it's initialized by run_app and used by other modules.
# The global llm, client, llm_with_tools also remain defined here and initialized by run_app.

def run_app(config: IdiomaticConfig):
    global llm, client, llm_with_tools, app_config

    app_config = config # Store config globally for access by nodes

    if not config.google_api_key:
        print("Error: GOOGLE_API_KEY is not set. Please set it in your environment or config.")
        return

    os.environ["GOOGLE_API_KEY"] = config.google_api_key

    # Initialize Google GenAI Client as per original script
    try:
        # Retry logic from original script
        if hasattr(genai, 'errors') and hasattr(genai.errors, 'APIError') and hasattr(genai, 'models') and hasattr(genai.models, 'Models'):
            is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and hasattr(e, 'code') and e.code in {429, 503})
            if callable(getattr(genai.models.Models, 'generate_content', None)):
                 genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)
            else:
                print("Warning: genai.models.Models.generate_content not found or not callable. Retry logic for it may not apply.")
        else:
            print("Warning: genai.errors.APIError or genai.models.Models not found. Retry logic may not apply as expected.")

        llm = ChatGoogleGenerativeAI(
            model=config.orchestration_llm_config.model_name,
            google_api_key=config.google_api_key,
            temperature=config.orchestration_llm_config.temperature
        )
        client = genai.Client(api_key=config.google_api_key)

        print(f"Idiomatic App starting with user: {config.user_name or 'New User'}")
        print(f"Orchestration LLM: {config.orchestration_llm_config.model_name}, Quiz LLM: {config.quiz_generation_config.model_name}")

    except Exception as e:
        print(f"Error during LLM initialization: {e}")
        return

    # Import tools from .tools module
    from .tools import defined_tools

    if llm:
        llm_with_tools = llm.bind_tools(defined_tools)
    else:
        print("Error: LLM not initialized, cannot bind tools.")
        return

    print("LLMs and tools initialized.")

    # Import necessary components for graph execution
    # IdiomaticState will be needed for initial state construction
    from .agent import IdiomaticState
    # Graph building and compilation will be handled by a function from graph.py
    # For now, the graph-related parts (nodes, route_logic, graph_builder) are still below
    # but will be moved to graph.py

    # These definitions and the graph building logic below will be moved to appropriate modules.
    # For now, route_logic and graph building remain here to be moved to graph.py next.

    # # Example nodes that would be imported from their respective files:
    # from .qna_generation import generate_idiom_question, evaluate_quiz_answer
    # from .agent import chatbot_node, get_user_input
    # from .tools import tool_node as graph_tool_node # tool_node is an instance from tools.py

    # # Routing logic (to be moved to graph.py)
    # def route_logic(state: IdiomaticState) -> Literal["tools", "evaluate_quiz", "chatbot_node", "generate_question", "__end__"]:
    #     """Decides the next step based on the last message."""
    #     print(f"--- Routing (Finished: {state.get('finished')}) ---")
    #     if state.get("finished"):
    #         print("--- Routing to END ---")
    #         return END

    #     last_message = state["messages"][-1] if state["messages"] else None

    #     if isinstance(last_message, AIMessage) and "Let's start" in last_message.content:
    #         print("--- Routing: Initial Setup -> Generate Question ---")
    #         return "generate_question"

    #     if isinstance(last_message, AIMessage) and last_message.tool_calls:
    #         print("--- Routing: AIMessage with Tool Calls -> Tools Node ---")
    #         return "tools"

    #     if isinstance(last_message, ToolMessage):
    #         print("--- Routing: ToolMessage -> Chatbot Node (to process result) ---")
    #         return "chatbot_node"

    #     if isinstance(last_message, HumanMessage):
    #         content = last_message.content.strip().lower()
    #         if content in {"a", "b", "c", "d"} and state.get("last_question"):
    #             print("--- Routing: Human Answer -> Evaluate Quiz ---")
    #             return "evaluate_quiz"
    #         else:
    #             print("--- Routing: Human Command/Chat -> Chatbot Node ---")
    #             return "chatbot_node"

    #     if isinstance(last_message, AIMessage) and not last_message.tool_calls:
    #         print("--- Routing: AIMessage (No Tool Call/Eval Result) -> Generate Question ---")
    #         return "generate_question"

    #     print("--- Routing: Fallback -> Chatbot Node (or consider Get Input) ---")
    #     return "chatbot_node"


    # print("Node functions and routing logic defined.") # This will be in graph.py

    # Graph Definition & Execution will be handled by graph.py
    from .graph import create_graph

    try:
        app_graph = create_graph()
        # print("\nGraph compiled successfully. Starting Idiomatic Chatbot...") # create_graph will print this
    except Exception as e:
        print(f"Error creating graph: {e}")
        return

    # Initial State Setup - remains in run_app
    # Load user data using the path from config (app_config is global here)
    user_data = load_user_data(app_config)

    initial_state_params = {
        "messages": [], # Start with empty messages for the graph stream
        "name": app_config.user_name or user_data.get("name", ""),
        "user_level": app_config.user_level or user_data.get("user_level", "intermediate"),
        "category": app_config.idiom_category or user_data.get("category", "general"),
        "score": user_data.get("score", 0),
        "history": user_data.get("history", []),
        "repetition_schedule": user_data.get("repetition_schedule", {}),
        "finished": False,
        "last_question": user_data.get("last_question"), # Load last question if available
        # "config_obj": app_config # Removed from state
    }
    # Filter out None values from user_data for last_question if it's not set
    if initial_state_params["last_question"] is None:
        del initial_state_params["last_question"] # LangGraph might not like None for a TypedDict field if not optional

    initial_state = IdiomaticState(**{k: v for k, v in initial_state_params.items() if k in IdiomaticState.__annotations__})


    # Run the graph stream
    final_graph_state = None
    try:
        for event_idx, event in enumerate(app_graph.stream(initial_state, {"recursion_limit": 100})):
            # print(f"\nGraph Event {event_idx}: {list(event.keys())}")
            for key, value in event.items():
                # print(f"  Node '{key}' output: {value}")
                if isinstance(value, dict) and 'messages' in value: # Heuristic for a state update
                    final_graph_state = value # Capture the latest full state snapshot
                    if final_graph_state.get('finished'):
                        print("--- Finishing Loop (detected finished flag in streamed state) ---")
                        break
            if final_graph_state and final_graph_state.get('finished'):
                break

        print("\nSession Ended.")

    except Exception as e:
        print(f"\nAn error occurred during the chat session: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final state if available, otherwise save the potentially modified initial_state
        # The chatbot_node already saves when quit_session_signal is processed.
        # This is a fallback save.
        state_to_save_on_exit = final_graph_state if final_graph_state else initial_state
        if state_to_save_on_exit and not state_to_save_on_exit.get("finished"): # Avoid double save if already saved by quit
             print("Saving user data on exit/error...")
             save_user_data(app_config, state_to_save_on_exit)
        elif not state_to_save_on_exit:
             print("No valid state to save on exit.")

        print("Idiomatic chatbot finished.")
