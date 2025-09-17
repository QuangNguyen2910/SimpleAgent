from __future__ import nested_scopes
from langgraph.graph import END, StateGraph, START
from .state import State
from ..nodes.selector import select_node
from ..nodes.simple_answerer import simple_answerer
from ..nodes.deep_researcher import call_agent_and_parse, execute_tool
from langgraph.checkpoint.base import BaseCheckpointSaver # For type hinting

def should_answer(state: State) -> str:
    """Conditional edge to decide whether to answer simply or do deep research."""
    if state.get("decision") == "normal":
        return "normal"
    else:
        return "deep_research"

def should_continue(state: State) -> str:
    """Conditional edge to decide whether to continue the loop or end."""
    if state.get("parsed_action"):
        return "execute_tool"
    else:
        return END

def build_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    # Define the graph
    graph_builder = StateGraph(State)
    
    # Set the recursion limit
    # graph_builder.set_config({'recursion_limit': 50})
    
    # Add nodes
    graph_builder.add_node("select_node", select_node)
    graph_builder.add_node("simple_answerer", simple_answerer)
    graph_builder.add_node("agent_step", call_agent_and_parse)
    graph_builder.add_node("tool_executor", execute_tool)

    # Define edges
    graph_builder.add_edge(START, "select_node")
    graph_builder.add_conditional_edges(
        "select_node", 
        should_answer,
        {
            "normal": "simple_answerer",
            "deep_research": "agent_step"
        }
    )
    graph_builder.add_edge("simple_answerer", END)
    graph_builder.add_conditional_edges(
        "agent_step",
        should_continue,
        {
            "execute_tool": "tool_executor",
            END: END
        }
    )
    graph_builder.add_edge("tool_executor", "agent_step")

    return graph_builder.compile(checkpointer=checkpointer)
