from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

# Placeholder for llm, will be initialized in the main script and passed or imported
# For now, tools requiring llm will need it in their scope.

@tool
def show_score(score: int) -> str:
    """Show the user's current quiz score."""
    return f"📈 Your score is {score}."

@tool
def explain_last_question(idiom: str, llm_instance) -> str: # Added llm_instance
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
def lookup_idiom(query: str, llm_instance) -> str: # Added llm_instance
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

tools = [show_score, explain_last_question, lookup_idiom, quit_session]
# The ToolNode is a simple wrapper, it does not require the LLM instance itself.
# The LLM is bound to tools *before* creating the ToolNode if specific tool functions need the LLM.
# However, the current tool functions expect llm_instance to be passed during their call,
# which means the graph logic will need to handle passing this.
# For simplicity in ToolNode creation, we'll define it here.
# The actual binding of llm_instance to the tool functions will be handled
# when these tools are called by the agent/graph.

tool_node = ToolNode(tools)
