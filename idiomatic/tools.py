from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage

# llm will be imported from .app where it's initialized by run_app
# This relies on app.py making 'llm' available as a module-level variable.
# from .app import llm as global_llm # Example of how it would be imported

# Placeholder for the actual llm that will be imported from app.py
# This will be populated by app.run_app()
llm = None

@tool
def show_score(score: int) -> str:
    """Show the user's current quiz score."""
    return f"📈 Your score is {score}."

@tool
def explain_last_question(idiom: str) -> str:
    """Explain the last idiom that was part of a question."""
    # Access the llm that should have been initialized in app.py's run_app
    from .app import llm as runtime_llm
    if not runtime_llm:
        return "Error: LLM for explain_last_question not initialized."

    print(f"--- Explaining Idiom: {idiom} (from tools.py) ---")
    try:
        response = runtime_llm.invoke([
            SystemMessage(content="Explain this English idiom simply and clearly, including an example sentence."),
            HumanMessage(content=idiom)
        ])
        explanation = response.content.strip()
        return f"**Explanation for '{idiom}':**\n{explanation}"
    except Exception as e:
        print(f"Error invoking LLM for explanation: {e}")
        return f"Sorry, I couldn't generate an explanation for '{idiom}' right now."

@tool
def lookup_idiom(query: str) -> str:
    """Find a natural idiom for a given user query/context or explain a requested idiom."""
    from .app import llm as runtime_llm
    if not runtime_llm:
        return "Error: LLM for lookup_idiom not initialized."

    print(f"--- Looking up/Explaining Idiom from query: {query} (from tools.py) ---")
    try:
        response = runtime_llm.invoke([
            SystemMessage(content="If the user provides a situation, suggest a relevant English idiom. If the user provides an idiom, explain it simply and clearly with an example."),
            HumanMessage(content=query)
        ])
        result = response.content.strip()
        return f"**Regarding '{query}':**\n{result}"
    except Exception as e:
         print(f"Error invoking LLM for lookup/explanation: {e}")
         return f"Sorry, I couldn't process your request for '{query}' right now."

@tool
def quit_session() -> str:
    """Quit the session and signal to end the application."""
    print("--- Quitting Session (from tools.py) ---")
    return "QUIT_SESSION_SIGNAL"

defined_tools = [show_score, explain_last_question, lookup_idiom, quit_session]
tool_node = ToolNode(defined_tools)

# Note: llm_with_tools (llm.bind_tools(defined_tools)) will be created in app.py
# as it requires the 'llm' instance which is also initialized in app.py.
# tools.py provides the 'defined_tools' list and the 'tool_node'.
