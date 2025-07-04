from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

# llm_instance is passed to tools that need it via functools.partial in main.py

@tool
def show_score(score: int) -> str:
    """Show the user's current quiz score."""
    return f"📈 Your score is {score}."

@tool
def explain_last_question(idiom: str, llm_instance) -> str:
    """Explain the last idiom that was part of a question."""
    print(f"--- Explaining Idiom: {idiom} ---")
    try:
        response = llm_instance.invoke([
            SystemMessage(content="Explain this English idiom simply and clearly, including an example sentence."),
            HumanMessage(content=idiom)
        ])
        explanation = response.content.strip()
        return f"**Explanation for '{idiom}':**\n{explanation}"
    except Exception as e:
        print(f"Error invoking LLM for explanation: {e}")
        return f"Sorry, I couldn't generate an explanation for '{idiom}' right now."

@tool
def lookup_idiom(query: str, llm_instance) -> str:
    """Find a natural idiom for a given user query/context or explain a requested idiom."""
    print(f"--- Looking up/Explaining Idiom from query: {query} ---")
    try:
        response = llm_instance.invoke([
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
    print("--- Quitting Session ---")
    return "QUIT_SESSION_SIGNAL"

# The list of raw tool functions.
# These are bound with llm_instance and then used to create ToolNode in main.py
tools_list = [show_score, explain_last_question, lookup_idiom, quit_session]

# The ToolNode itself is now created in main.py after tools are potentially bound with llm_instance.
# So, tool_node definition is removed from here.
# This file now primarily exports the raw tool functions.
