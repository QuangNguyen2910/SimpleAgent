from typing import Annotated, TypedDict, Optional, List, Any
from langgraph.graph.message import add_messages
from ..model.llm import LLM
from langchain_core.tools import StructuredTool



# State TypedDict including new fields for the subquery workflow
class State(TypedDict):
    llm: Optional[LLM]
    tools: Optional[List[StructuredTool]]
    messages: Annotated[list, add_messages]
    answer: Optional[str]
    decision: Optional[str]  # "normal" hoặc "deep_research"
    parsed_action: Optional[List[Any]] # Use Any to avoid circular import, or define ToolCall here