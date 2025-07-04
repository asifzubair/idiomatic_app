## Imports 

import json
import random
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Annotated, Literal
from typing_extensions import TypedDict
from IPython.display import Markdown, display

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from google import genai
from google.genai import types
from google.api_core import retry


# Initialize Google GenAI Client
# from kaggle_secrets import UserSecretsClient
# GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

# For conversational tasks (intent detection, explanation)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# For heavy generation (question generation)
client = genai.Client(api_key=GOOGLE_API_KEY)


## Data Persistence

"""
Initialize user data storage. Since this is a colab, we just use a JSON file for persistence
More formally, we could've used a sqlite3 db etc.
"""

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

user_data = load_user_data()

## Agent Workflow

class IdiomaticState(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    user_level: str
    category: str
    score: int
    history: list[str]
    repetition_schedule: dict
    finished: bool
    last_question: dict


## Question & Answer Generation

"""
This code block illustrates the following aspects of genAI capabilities:
 1. Structured output/JSON mode/controlled generation
 2. Few-shot prompting
 3. Optimizing model params for diverse output
"""

GENERATE_QnA_PROMPT="""
    You are teaching assistant tasked with teaching students english idioms through question and answer.
    When prompted generate a random question tasked with teaching an idiom. Questions MUST be multiple choice.
    Your output will be a dict of the form:
    {
      "idiom": <idiom being tested>,
      "question": <question that is being asked>,
      "answer": <the expected answer>
    }
    The output should strictly follow the above format. Do not include any extra text besides what is in the dictionary. 
    
    Here are a few examples of what can be asked:

    EXAMPLE:
    ```
    {
      "idiom": "bite the bullet",
      "question":
      Q. 'Bite the bullet' means:\n
       a. to do something unpleasant that is unavoidable\n
       b. to react harshly under pressure\n
       c. to avoid making a tough decision\n
       d. to get into a fight,
      "answer": "a"
    }
    ```

    EXAMPLE:
    ```
    {
      "idiom": "go belly up",
      "question":
      Q. The startup was burning through cash, and unless they found an investor soon, they would ________.\n
       a. hit the hay\n
       b. go belly up\n
       c. pull someone’s leg\n
       d. throw in the towel,
      "answer": "b"
    }
    ```

    EXAMPLE:
    ```
    {
      "idiom": "circle the wagons",
      "question":
       Q. When the startup faced heavy media scrutiny, the team decided to ________ and protect their CEO.\n
        a. take the high road\n
        b. throw in the towel\n
        c. circle the wagons\n
        d. shoot from the hip,
      "answer": "c"
    }
    ```
"""

@dataclass
class IdiomQuizItem:
  """Schema for an idiom question and answer"""
  idiom: str     # The idiom being tested
  question: str  # The question about the idiom
  answer: str    # The expected answer

GENERATE_QnA_CONFIG = types.GenerateContentConfig(
    max_output_tokens=200,
    temperature=1.5,
    top_p=0.95,
    response_mime_type="application/json",
    response_schema=IdiomQuizItem
)

def generate_idiom_question(state: IdiomaticState) -> IdiomaticState:
    """Use Gemini to dynamically generate an idiom and an associated Q/A"""
    print("--- Generating Question ---")
    difficulty = state["user_level"]
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=GENERATE_QnA_PROMPT,
        config=GENERATE_QnA_CONFIG,
    )
    qna = json.loads(response.text)
    state["last_question"] = qna
    state["history"].append(qna["idiom"])
    display(Markdown(qna["question"]))
    return state

def evaluate_quiz_answer(state: IdiomaticState) -> IdiomaticState:
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

@tool
def show_score(score: int) -> str:
    """Show the user's current quiz score."""
    return f"📈 Your score is {score}."

@tool
def explain_last_question(idiom: str) -> str:
    """Explain the last idiom that was part of a question."""
    print(f"--- Explaining Idiom: {idiom} ---")
    try:
        response = llm.invoke([
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
    print(f"--- Looking up/Explaining Idiom from query: {query} ---")
    try:
        # Ask the LLM to either find an idiom for the context OR explain the idiom if the query *is* an idiom
        response = llm.invoke([
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
    # This message signals the graph to terminate.
    # The chatbot node will detect this specific message from the ToolNode.
    return "QUIT_SESSION_SIGNAL"

tools = [show_score, explain_last_question, lookup_idiom, quit_session]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)


## Orchestration LLM & Routing

IDIOMATIC_BOT_SYSINT = (
  "You are Idiomatic, a friendly and intelligent English language tutor designed to help non-native speakers master idioms through interactive learning. "
  "Your style is warm, supportive, and concise. You adapt to natural language and understand user intent even when it's not phrased exactly. "
  "Your goals: "
  " - Teach idioms through multiple choice questions generated by the system. "
  " - **You MUST use the provided tools when the user's intent matches.** Specifically: "
  "   * User asks for score ('How am I doing?', 'What's my score?'): **MUST call `show_score` tool.** The current score is available in the state. "
  "   * User asks for explanation of the last question ('explain', 'why?', 'explain that'): **MUST call `explain_last_question` tool.** The idiom is available in the state's last_question. "
  "   * User asks to lookup or define an idiom ('What does X mean?', 'lookup Y', 'idiom for Z?'): **MUST call `lookup_idiom` tool** with the user's query. "
  "   * User wants to quit ('Stop', 'I'm done', 'quit', 'exit'): **MUST call `quit_session` tool.** "
  " - Do NOT invent functionality or apologize for missing tools. The tools ARE available. "
  " - Do NOT call tools unnecessarily. If the user is just chatting or providing a quiz answer (a, b, c, d), respond naturally or let the workflow handle the answer. "
  " - Be flexible, encouraging, and gently correct users when needed. "
  "\n\n"
  "Current Score: {score}. Last Question Idiom: {last_idiom}."
)

def chatbot_node(state: IdiomaticState) -> IdiomaticState:
    """Handles initial setup, invokes LLM for routing/chat/tools, and checks for quit signal."""

    # 1. Initial Setup (if name is not set)
    if not state.get("name"):
        print("--- Initial Setup ---")
        # Use input() for Colab/terminal interaction
        user_name = input("👋 Welcome to Idiomatic! What's your name? ")
        level_choice = input("Skill level (a) beginner / (b) intermediate / (c) advanced: ").strip().lower()
        # Map choice to a descriptive level
        level_map = {'a': 'beginner', 'b': 'intermediate', 'c': 'advanced'}
        level = level_map.get(level_choice, 'intermediate') # Default to intermediate

        category = input(f"Preferred idiom category (e.g., business, animals, food) [default: general]: ") or "general"

        state.update({
            "name": user_name,
            "user_level": level,
            "category": category,
            "messages": [AIMessage(content=f"Hello {user_name}! 👋 Let's start with some {category} idioms at the {level} level. I'll ask you multiple-choice questions. You can also ask me to 'explain', show your 'score', 'lookup' an idiom, or 'quit'.")] # Initial message
        })
        display(Markdown(state['messages'][-1].content))
        # Initial state setup complete, next node should be 'generate_question'
        # We'll handle this transition in the routing logic.
        return state # Return immediately after setup

    # 2. Process Last Message (Tool Result or Human Input)
    last_message = state["messages"][-1] if state["messages"] else None

    # Check if the last message is a ToolMessage with the quit signal
    if isinstance(last_message, ToolMessage) and last_message.content == "QUIT_SESSION_SIGNAL":
        print("--- Quit Signal Received ---")
        final_message = "👋 Thanks for learning with Idiomatic! Your progress is saved."
        state["messages"].append(AIMessage(content=final_message))
        display(Markdown(final_message))
        state["finished"] = True
        save_user_data(state) # Save progress on quit
        return state

    # If the last message wasn't the quit signal, invoke LLM with history
    # This handles: Tool results (like score/explanation), human commands, or general chat
    try:
        print("--- Invoking LLM with Tools ---")
        # Pass score and last_question idiom to the LLM via state implicitly or explicitly in prompt if needed
        # The LLM already has access to tools that use state elements (score, last_question)
        response = llm_with_tools.invoke(
            [SystemMessage(content=IDIOMATIC_BOT_SYSINT)] + state["messages"] # Pass full history
        )
        state["messages"].append(response)

        # Display the response unless it contains tool calls (handled by ToolNode)
        if not response.tool_calls:
            display(Markdown(f"**Idiomatic:** {response.content}"))

    except Exception as e:
        print(f"Error invoking LLM in chatbot_node: {e}")
        # Add an error message to the state
        error_msg = "Sorry, I encountered an error. Please try again."
        state["messages"].append(AIMessage(content=error_msg))
        display(Markdown(error_msg))


    return state

def get_user_input(state: IdiomaticState) -> IdiomaticState:
    """Prompts the user for input and adds it as a HumanMessage."""
    print("--- Waiting for User Input ---")
    # Check if the last message was an AIMessage indicating a question was asked
    # or if it was a tool response that requires a follow-up action from the user.
    prompt_message = "Your answer (a/b/c/d) or command: \n"
    # You could customize the prompt based on the last AI message if needed

    user_input = input(prompt_message).strip()
    state["messages"].append(HumanMessage(content=user_input))
    return state

def route_logic(state: IdiomaticState) -> Literal["tools", "evaluate_quiz", "chatbot_node", "generate_question", "__end__"]:
    """Decides the next step based on the last message."""
    print(f"--- Routing (Finished: {state.get('finished')}) ---")
    if state.get("finished"):
        print("--- Routing to END ---")
        return END # Use END from langgraph.graph

    last_message = state["messages"][-1] if state["messages"] else None

    # 1. After Initial Setup -> Generate Question
    if isinstance(last_message, AIMessage) and "Let's start" in last_message.content:
         print("--- Routing: Initial Setup -> Generate Question ---")
         return "generate_question"

    # 2. After LLM response with Tool Calls -> Tools Node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("--- Routing: AIMessage with Tool Calls -> Tools Node ---")
        return "tools"

    # 3. After Tool Node (which adds ToolMessage) -> Chatbot Node (to process result)
    if isinstance(last_message, ToolMessage):
         print("--- Routing: ToolMessage -> Chatbot Node ---")
         return "chatbot_node" # Chatbot node handles tool results (incl. quit signal)

    # 4. After Human provides quiz answer -> Evaluate Quiz
    if isinstance(last_message, HumanMessage):
        content = last_message.content.strip().lower()
        if content in {"a", "b", "c", "d"} and state.get("last_question"):
             print("--- Routing: Human Answer -> Evaluate Quiz ---")
             return "evaluate_quiz"
        else:
             # It's a command or chat -> Chatbot Node (to process with LLM)
             print("--- Routing: Human Command/Chat -> Chatbot Node ---")
             return "chatbot_node"

    # 5. After evaluation or non-tool AI response -> Generate Question
    # This covers the case after evaluate_quiz adds its result message,
    # or after the chatbot gives an explanation/score/chat response.
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
         print("--- Routing: AIMessage (Eval Result/Chat/Explanation) -> Generate Question ---")
         return "generate_question"

    # Default fallback (should ideally not be reached often)
    print("--- Routing: Fallback -> Generate Question (or consider END/error) ---")
    # If unsure, maybe ask a new question? Or prompt user again?
    return "generate_question"

graph_builder = StateGraph(IdiomaticState)

graph_builder.add_node("chatbot_node", chatbot_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("generate_question", generate_idiom_question)
graph_builder.add_node("evaluate_quiz", evaluate_quiz_answer)
graph_builder.add_node("get_input", get_user_input) # Add the input node

# Entry point is the chatbot node (handles initial setup)
graph_builder.set_entry_point("chatbot_node")

# Conditional Edges based on Router
graph_builder.add_conditional_edges(
    "chatbot_node", # Source node for routing decisions *after* chatbot processes input/tool result
    route_logic,
    {
        "tools": "tools",
        "generate_question": "generate_question", # Route to generate after setup or tool explanation
        "chatbot_node": "get_input", # Should not loop back directly, get input first
         END: END
    }
)

graph_builder.add_conditional_edges(
    "get_input", # Source node for routing decisions *after* getting user input
    route_logic,
    {
        "evaluate_quiz": "evaluate_quiz",
        "chatbot_node": "chatbot_node", # Route to chatbot to process command/chat
         END: END # Should not happen here unless user types quit command directly
    }
)


# Edges from specific task nodes
graph_builder.add_edge("tools", "chatbot_node")          # After tools run, chatbot processes the ToolMessage result
graph_builder.add_edge("evaluate_quiz", "generate_question") # After evaluating, generate the next question
graph_builder.add_edge("generate_question", "get_input") # After generating, get user input (answer or command)

# Compile the graph
try:
    app = graph_builder.compile()
except Exception as e:
    print(f"Error compiling graph: {e}")
    app = None # Set app to None if compilation fails


if app:
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