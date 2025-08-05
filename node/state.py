# graph/state.py
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]