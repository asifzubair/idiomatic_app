## Imports
import os
import json
from datetime import datetime
from functools import partial

from IPython.display import Markdown, display
from langgraph.graph import END # Will be used by create_graph if not directly
from langchain_core.messages import AIMessage # For evaluate_quiz_answer
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.api_core import retry # type: ignore

# Local module imports (now relative)
from .core.schemas import IdiomaticState
from .utils.data_utils import load_user_data, save_user_data
from .generation.qna_generation import generate_idiom_question
from .tools.idiom_tools import show_score, explain_last_question, lookup_idiom, quit_session # Or import tools_list
from .orchestration.agent import chatbot_node, get_user_input, route_logic
from .orchestration.graph import create_graph
from .config import IdiomaticConfig


# --- LLM and Client Initialization will be handled in run_app using config ---
# Global llm and gen_client will be removed or scoped differently.
# For simplicity, they will be initialized in run_app and passed around.


## Data Persistence
# (load_user_data and save_user_data will be modified to take user_data_path)

"""
Initialize user data storage. Since this is a colab, we just use a JSON file for persistence
More formally, we could've used a sqlite3 db etc.
"""

# Moved to utils.data_utils

## Agent Workflow
# (IdiomaticState is imported from core.schemas)

## Question & Answer Generation
# (generate_idiom_question will be modified to take model_name, temp, etc. from config)

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


# --- Main Application Setup and Execution ---

def run_app(config: IdiomaticConfig):
    """Initializes components, creates the graph, and runs the application stream."""

    if not config.google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the configuration or environment.")

    # Ensure GenAI client retry mechanism is set up
    if not hasattr(genai.models.Models.generate_content, "_retry_predicate"):
        is_retriable_for_models = lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503}
        genai.models.Models.generate_content = retry.Retry(predicate=is_retriable_for_models)(genai.models.Models.generate_content)

    # 1. Initialize LLMs and Clients using config
    orchestrator_llm = ChatGoogleGenerativeAI(
        model=config.orchestrator_llm_model,
        google_api_key=config.google_api_key,
        temperature=config.orchestrator_llm_temperature
    )

    qna_gen_client = genai.Client(api_key=config.google_api_key)

    # 2. Bind LLM to tools that need it
    bound_explain_tool = partial(explain_last_question, llm_instance=orchestrator_llm)
    bound_lookup_tool = partial(lookup_idiom, llm_instance=orchestrator_llm)
    final_tools_list = [show_score, bound_explain_tool, bound_lookup_tool, quit_session]

    # 3. Create the LLM bound with these tools
    llm_with_tools_for_agent = orchestrator_llm.bind_tools(final_tools_list)

    from langgraph.prebuilt import ToolNode
    configured_tool_node = ToolNode(final_tools_list)

    # 4. Prepare node functions (anticipating changes in Step 4 of plan)
    configured_load_user_data = partial(load_user_data, user_data_path=config.user_data_path)
    configured_save_user_data = partial(save_user_data, user_data_path=config.user_data_path)

    configured_generate_idiom_question = partial(generate_idiom_question,
                                                 client_instance=qna_gen_client,
                                                 model_name=config.qna_generator_model,
                                                 temperature=config.qna_generator_temperature,
                                                 top_p=config.qna_generator_top_p,
                                                 max_output_tokens=config.qna_max_output_tokens)

    configured_chatbot_node = partial(chatbot_node,
                                      llm_with_tools_instance=llm_with_tools_for_agent,
                                      save_user_data_func=configured_save_user_data)

    agent_nodes_map = {
        'chatbot_node': configured_chatbot_node,
        'get_user_input': get_user_input,
        'route_logic': route_logic,
    }

    generation_nodes_map = {
        'generate_idiom_question': configured_generate_idiom_question,
        'evaluate_quiz_answer': evaluate_quiz_answer
    }

    # 5. Create the graph
    app = create_graph(
        llm_instance=orchestrator_llm,
        client_instance=qna_gen_client,
        tools_list=final_tools_list,
        configured_tool_node=configured_tool_node,
        schema=IdiomaticState,
        agent_nodes=agent_nodes_map,
        generation_nodes=generation_nodes_map,
        data_utility_nodes={'save_user_data': configured_save_user_data}
    )

    if not app:
        print("Graph compilation failed. Cannot run the application.")
        return

    print("\nStarting Idiomatic Chatbot...")
    current_user_data = configured_load_user_data()

    fields_to_save = ["name", "score", "history", "repetition_schedule", "user_level", "category", "messages", "last_question", "finished"]

    initial_state = IdiomaticState(
        messages=current_user_data.get("messages", []),
        name=current_user_data.get("name", config.user_name or ""),
        user_level=current_user_data.get("user_level", config.user_level),
        category=current_user_data.get("category", config.user_category),
        score=current_user_data.get("score", 0),
        history=current_user_data.get("history", []),
        repetition_schedule=current_user_data.get("repetition_schedule", {}),
        finished=False,
        last_question=current_user_data.get("last_question", None)
    )

    if initial_state["name"] and not initial_state["messages"]:
         initial_state["messages"] = [AIMessage(content=f"Welcome back, {initial_state['name']}! Let's continue.")]
    elif not initial_state["name"] and config.user_name:
        initial_state["name"] = config.user_name

    if not initial_state.get("messages"):
        initial_state["messages"] = []

    try:
        final_graph_state = None
        for event in app.stream(initial_state, {"recursion_limit": 100}):
            for key, value in event.items():
                if isinstance(value, dict) and 'messages' in value:
                    final_graph_state = value
                    if final_graph_state.get('finished'):
                        print("--- Finishing Loop (detected finished flag in stream) ---")
                        break
            if final_graph_state and final_graph_state.get('finished'):
                break

        print("\nSession Ended.")
        if final_graph_state:
            print("Saving final user data...")
            configured_save_user_data({k: v for k, v in final_graph_state.items() if k in fields_to_save})
        else:
            print("Saving current user data (final graph state not captured)...")
            configured_save_user_data({k:v for k, v in initial_state.items() if k in fields_to_save})

    except Exception as e:
        print(f"\nAn error occurred during the chat session: {e}")
        print("Attempting to save current user data on error...")
        current_save_state = final_graph_state if final_graph_state else initial_state
        configured_save_user_data({k:v for k, v in current_save_state.items() if k in fields_to_save})
    finally:
        print("Idiomatic chatbot finished.")

if __name__ == "__main__":
    app_config = IdiomaticConfig()
    if not app_config.google_api_key:
        print("Fatal Error: GOOGLE_API_KEY is not set in environment or config. Application cannot start.")
    else:
        run_app(app_config)