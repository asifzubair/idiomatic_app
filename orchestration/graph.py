from langgraph.graph import StateGraph, END
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.schemas import IdiomaticState # For type hinting

# Will import these from their new locations
# from .agent import chatbot_node, get_user_input, route_logic
# from ..tools.idiom_tools import tool_node as imported_tool_node # Renamed to avoid conflict
# from ..generation.qna_generation import generate_idiom_question
# from ..core.schemas import IdiomaticState
# from ..utils.data_utils import save_user_data # If chatbot_node doesn't get it directly

# For now, define a function that takes these as arguments.

def create_graph(llm_instance, client_instance, tools_list, configured_tool_node, schema: 'IdiomaticState', agent_nodes, generation_nodes, data_utility_nodes):
    """
    Creates and compiles the LangGraph application.
    Dependencies (like llm, client, specific tool functions) should be configured
    and passed into the nodes before they are passed to this function.
    """

    # Unpack node functions from the passed dictionaries
    chatbot_node_func = agent_nodes['chatbot_node']
    get_user_input_func = agent_nodes['get_user_input']
    route_logic_func = agent_nodes['route_logic']

    generate_idiom_question_func = generation_nodes['generate_idiom_question']
    # evaluate_quiz_answer_func will likely remain in the main file or be passed if moved
    # For now, assuming evaluate_quiz_answer is also passed via a dictionary
    evaluate_quiz_answer_func = generation_nodes.get('evaluate_quiz_answer')


    # The tool functions needing llm_instance (explain_last_question, lookup_idiom)
    # should be curried with llm_instance before being added to tools_list
    # and before tool_node is created.
    # This should happen in the main script before calling create_graph.

    # Example of how nodes that need llm or client could be wrapped if not handled by direct passing:
    # wrapped_generate_question = partial(generate_idiom_question_func, client_instance=client_instance)
    # wrapped_chatbot_node = partial(chatbot_node_func, llm_with_tools_instance=llm_with_tools_instance, save_user_data_func=save_user_data)
    # These partials would be created in the main script and passed into this function.
    # For now, this function expects the nodes to be already configured.


    graph_builder = StateGraph(schema)

    graph_builder.add_node("chatbot_node", chatbot_node_func) # Expects already configured chatbot_node
    graph_builder.add_node("tools", configured_tool_node)
    graph_builder.add_node("generate_question", generate_idiom_question_func) # Expects configured generate_idiom_question
    if evaluate_quiz_answer_func:
        graph_builder.add_node("evaluate_quiz", evaluate_quiz_answer_func)
    graph_builder.add_node("get_input", get_user_input_func)

    graph_builder.set_entry_point("chatbot_node")

    graph_builder.add_conditional_edges(
        "chatbot_node",
        route_logic_func,
        {
            "tools": "tools",
            "generate_question": "generate_question",
            "chatbot_node": "get_input", # Original had this, check if it's correct route
            END: END
        }
    )

    graph_builder.add_conditional_edges(
        "get_input",
        route_logic_func,
        {
            "evaluate_quiz": "evaluate_quiz",
            "chatbot_node": "chatbot_node",
            END: END
        }
    )

    graph_builder.add_edge("tools", "chatbot_node")
    if evaluate_quiz_answer_func: # Only add edge if node exists
        graph_builder.add_edge("evaluate_quiz", "generate_question")
    graph_builder.add_edge("generate_question", "get_input")

    try:
        app = graph_builder.compile()
        print("Graph compiled successfully.")
        return app
    except Exception as e:
        print(f"Error compiling graph: {e}")
        return None
