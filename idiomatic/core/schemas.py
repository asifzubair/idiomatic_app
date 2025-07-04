from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

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
