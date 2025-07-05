import json
from datetime import datetime
from dataclasses import dataclass

# Will be imported from .app (once app.py defines them and they are initialized in run_app)
# For now, these are placeholders for what this module needs.
# from .app import client, app_config
# To make this module self-contained for now, we can define dummy placeholders if not running full app
# However, the actual instances will come from app.py during runtime.

# To avoid import errors during isolated linting/type-checking before full app structure is ready:
# We expect IdiomaticState to be defined, typically in agent.py.
# from .agent import IdiomaticState # This will be the final import
# Placeholder for now:
# from typing import TYPE_CHECKING, TypedDict, Annotated # For IdiomaticState if defined locally for now
# if TYPE_CHECKING:
#     from .agent import IdiomaticState
# else:
#     # Temporary placeholder for IdiomaticState until agent.py is populated
#     # This helps avoid immediate import errors during stepwise refactoring.
#     class IdiomaticState(TypedDict):
#         messages: Annotated[list, lambda x, y: x + y] # dummy add_messages
#         name: str
#         user_level: str
#         category: str
#         score: int
#         history: list[str]
#         repetition_schedule: dict
#         finished: bool
#         last_question: dict | None

# Import IdiomaticState from agent.py where it's now defined
from .agent import IdiomaticState


from langchain_core.messages import AIMessage
from google.genai import types as genai_types # Assuming this alias is used in app.py for google.genai.types

# Imports from within the package
from .utils import display, Markdown # For displaying questions and results

# To be imported from app.py where client and app_config are initialized
# These will be set by app.py at runtime before these functions are called.
# This relies on app.py making them available, e.g. by them being module-level vars in app.py
# that run_app populates.
# For direct access:
# from .app import client as global_client, app_config as global_app_config
# Or, if we make them part of the state (which is complex for clients):
# For now, assume they will be importable from .app after run_app starts.

# Placeholder for the actual client and app_config that will be imported from app.py
# These will be populated by app.run_app()
client = None
app_config = None


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


def generate_idiom_question(state: "IdiomaticState") -> "IdiomaticState":
    """Use Gemini to dynamically generate an idiom and an associated Q/A"""
    # Access the client and app_config that should have been initialized in app.py's run_app
    from .app import client as runtime_client, app_config as runtime_app_config

    if not runtime_client or not runtime_app_config:
        raise RuntimeError("Client or app_config not initialized in app.py")

    print("--- Generating Question (from qna_generation.py) ---")
    q_config = runtime_app_config.quiz_generation_config

    gen_content_config = genai_types.GenerateContentConfig(
        max_output_tokens=q_config.max_output_tokens,
        temperature=q_config.temperature,
        top_p=q_config.top_p,
        response_mime_type=q_config.response_mime_type,
        response_schema=IdiomQuizItem
    )

    response = runtime_client.models.generate_content(
        model=q_config.model_name,
        contents=GENERATE_QnA_PROMPT,
        config=gen_content_config,
    )
    qna = json.loads(response.text)
    state["last_question"] = qna
    if qna.get("idiom"):
      state["history"].append(qna["idiom"])
    display(Markdown(qna["question"]))
    return state

def evaluate_quiz_answer(state: "IdiomaticState") -> "IdiomaticState":
    """Evaluates the user's answer to the last quiz question."""
    print("--- Evaluating Answer (from qna_generation.py) ---")
    user_input = state["messages"][-1].content.strip().lower()
    correct_answer = None
    if state.get("last_question") and isinstance(state["last_question"], dict):
        correct_answer = state["last_question"].get("answer", "").lower()

    result_message = ""
    success = False
    if correct_answer and user_input == correct_answer:
        result_message = "✅ Correct!"
        state["score"] += 1
        success = True
    elif correct_answer:
        result_message = f"❌ Incorrect! The correct answer was '{correct_answer}'."
    else:
        result_message = "❓Could not evaluate the answer. No last question found."

    display(Markdown(result_message))
    state['messages'].append(AIMessage(content=result_message))

    if state.get("last_question") and isinstance(state["last_question"], dict):
        idiom = state["last_question"].get("idiom")
        if idiom:
            # Ensure repetition_schedule exists and is a dict
            if "repetition_schedule" not in state or not isinstance(state["repetition_schedule"], dict):
                state["repetition_schedule"] = {}
            state["repetition_schedule"][idiom] = {
                "last_seen": datetime.utcnow().isoformat(),
                "success": success
            }
    return state
