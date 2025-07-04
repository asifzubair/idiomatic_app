import json
from dataclasses import dataclass
import google.generativeai as genai
from IPython.display import Markdown, display
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.schemas import IdiomaticState # Adjusted for relative import

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

# Note: `client_instance` needs to be available in the scope of this function.
# It will be initialized in the main script and passed, along with other config values.
def generate_idiom_question(state: 'IdiomaticState',
                            client_instance,
                            model_name: str,
                            temperature: float,
                            top_p: float,
                            max_output_tokens: int) -> 'IdiomaticState':
    """Use Gemini to dynamically generate an idiom and an associated Q/A"""
    print("--- Generating Question ---")

    # The client_instance (genai.Client) passed from main has configured the API key.
    # We now create a GenerativeModel instance.
    model = genai.GenerativeModel(model_name)

    current_qna_config = genai.types.GenerationConfig( # Corrected to genai.types.GenerationConfig
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        response_mime_type="application/json", # Ensure this is valid for GenerationConfig
    )
    # For response_mime_type with genai.GenerativeModel, it's often set in model params or generate_content's request options
    # Forcing JSON output is typically done via specific instructions in the prompt or model settings if available,
    # or by setting response_format={"type": "json_object"} in generation_config for newer API versions/models.
    # Let's assume response_mime_type is handled correctly by the model or SDK for now,
    # but this might need adjustment if JSON isn't produced.
    # The prompt already asks for a dict, which is good.

    response = model.generate_content(
        contents=GENERATE_QnA_PROMPT,
        generation_config=current_qna_config,
    )
    # It's good practice to check response.parts and handle potential errors or empty responses
    # For now, assuming response.text is populated as expected.
    qna_text = response.text
    try:
        qna = json.loads(qna_text)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from LLM response: {qna_text}")
        # Handle error appropriately, e.g., by setting a default error question or re-raising
        state["last_question"] = {"idiom": "Error", "question": "Could not generate question.", "answer": ""}
        # Potentially add a message to the user/state about the error
        return state # Or raise an exception

    state["last_question"] = qna
    if "history" not in state:
        state["history"] = []
    state["history"].append(qna["idiom"])
    display(Markdown(qna["question"]))
    return state
