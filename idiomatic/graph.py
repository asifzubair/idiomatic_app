from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage # For route_logic

# Import node functions and state definition
from .agent import IdiomaticState, chatbot_node, get_user_input
from .qna_generation import generate_idiom_question, evaluate_quiz_answer
from .tools import tool_node # This is the configured ToolNode instance

# Routing logic
def route_logic(state: IdiomaticState) -> Literal["tools", "evaluate_quiz", "chatbot_node", "generate_question", "__end__"]:
    """Decides the next step based on the last message."""
    # This function was moved from app.py, its logic remains the same.
    # It needs IdiomaticState, AIMessage, HumanMessage, ToolMessage, END.
    print(f"--- Routing (Finished: {state.get('finished')}) (from graph.py) ---")
    if state.get("finished"):
        print("--- Routing to END (from graph.py) ---")
        return END

    last_message = state["messages"][-1] if state["messages"] else None

    if isinstance(last_message, AIMessage) and "Let's start" in last_message.content:
        print("--- Routing: Initial Setup -> Generate Question (from graph.py) ---")
        return "generate_question"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("--- Routing: AIMessage with Tool Calls -> Tools Node (from graph.py) ---")
        return "tools"

    if isinstance(last_message, ToolMessage):
        print("--- Routing: ToolMessage -> Chatbot Node (from graph.py) ---")
        return "chatbot_node"

    if isinstance(last_message, HumanMessage):
        content = last_message.content.strip().lower()
        if content in {"a", "b", "c", "d"} and state.get("last_question"):
            print("--- Routing: Human Answer -> Evaluate Quiz (from graph.py) ---")
            return "evaluate_quiz"
        else:
            print("--- Routing: Human Command/Chat -> Chatbot Node (from graph.py) ---")
            return "chatbot_node"

    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        print("--- Routing: AIMessage (No Tool Call/Eval Result) -> Generate Question (from graph.py) ---")
        return "generate_question"

    print("--- Routing: Fallback -> Chatbot Node (from graph.py) ---")
    return "chatbot_node"

def create_graph():
    """Creates and compiles the LangGraph StateGraph for the Idiomatic application."""

    graph_builder = StateGraph(IdiomaticState)

    # Add nodes
    # Nodes are imported from their respective modules
    graph_builder.add_node("chatbot_node", chatbot_node)
    graph_builder.add_node("tools", tool_node) # tool_node is the instance from tools.py
    graph_builder.add_node("generate_question", generate_idiom_question)
    graph_builder.add_node("evaluate_quiz", evaluate_quiz_answer)
    graph_builder.add_node("get_input", get_user_input)

    # Set entry point
    graph_builder.set_entry_point("chatbot_node")

    # Add conditional edges
    graph_builder.add_conditional_edges(
        "chatbot_node",
        route_logic,
        {
            "tools": "tools",
            "generate_question": "generate_question",
            "chatbot_node": "chatbot_node",
            "__end__": END  # Ensure END is correctly mapped if route_logic returns it
        }
    )

    graph_builder.add_conditional_edges(
        "get_input",
        route_logic,
        {
            "evaluate_quiz": "evaluate_quiz",
            "chatbot_node": "chatbot_node",
            "__end__": END
        }
    )

    # Add static edges
    graph_builder.add_edge("tools", "chatbot_node")
    graph_builder.add_edge("evaluate_quiz", "generate_question")
    graph_builder.add_edge("generate_question", "get_input")

    # Compile the graph
    app_graph = graph_builder.compile()
    print("\nGraph compiled successfully (from graph.py).")
    return app_graph
