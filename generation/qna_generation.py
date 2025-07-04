import json
from dataclasses import dataclass
from google.genai import types
from IPython.display import Markdown, display
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.schemas import IdiomaticState # For type hinting

# Assuming client is initialized elsewhere and passed or imported
# from ..idiomatic import client # This will need adjustment based on final structure
# For now, we'll define it as a placeholder if direct import isn't feasible
# or expect it to be passed to generate_idiom_question

# Placeholder for the main client, will be properly initialized in the main script
# and potentially passed to functions here or accessed via a global/singleton pattern
# For now, the function `generate_idiom_question` will need `client` to be in its scope.

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

# Note: `client` needs to be available in the scope of this function.
# It will be initialized in the main script and might need to be passed or imported.
def generate_idiom_question(state: 'IdiomaticState', client_instance) -> 'IdiomaticState':
    """Use Gemini to dynamically generate an idiom and an associated Q/A"""
    print("--- Generating Question ---")
    # difficulty = state["user_level"] # Not used in current prompt, but kept for potential future use
    response = client_instance.models.generate_content(
        model="gemini-2.0-flash", # Consider making model name configurable
        contents=GENERATE_QnA_PROMPT,
        config=GENERATE_QnA_CONFIG,
    )
    qna = json.loads(response.text)
    state["last_question"] = qna
    if "history" not in state: # Ensure history exists
        state["history"] = []
    state["history"].append(qna["idiom"])
    display(Markdown(qna["question"]))
    return state
