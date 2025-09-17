# File: nodes/deep_researcher.py

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from langchain_core.runnables import RunnableConfig
from ..graph.state import State

# --- 1. Define the Action Schemas (mimicking bind_tools) ---

class ToolCall(BaseModel):
    """A structured request to call a specific tool with arguments."""
    name: str = Field(..., description="The name of the tool to call (e.g., 'search_web', 'evaluate_expression').")
    arguments: Dict[str, Any] = Field(..., description="A dictionary of arguments for the tool.")

class FinalAnswer(BaseModel):
    """The final answer to be returned to the user."""
    answer: str = Field(..., description="The comprehensive final answer for the user's question.")

# --- 2. Define the Main ReAct Schema ---
# This is the key. It forces the LLM to provide both reasoning and a structured action.

class ReActStep(BaseModel):
    """
    The model's output for a single step, containing its reasoning and the chosen action.
    The action must be either a list of tool calls or a final answer.
    """
    reasoning: str = Field(..., description="A detailed, step-by-step thought process. Explain your plan, what you know, and what you need to find out.")
    action: Union[List[ToolCall], FinalAnswer] = Field(..., description="The action to take. This will be EITHER a list of tool calls OR the final answer.")


# --- 3. The System Prompt to guide the LLM ---

REACT_HYBRID_PROMPT = """You are a highly intelligent research assistant. Your goal is to answer the user's question by breaking it down into a series of steps.

You operate in a loop:
1.  **Reasoning:** First, you think. Analyze the request, the available information, and your plan.
2.  **Action:** Based on your reasoning, you choose ONE of two possible actions:
    a. **Call Tools:** If you need more information, provide a list of one or more `ToolCall` objects.
    b. **Final Answer:** If you have all the information needed, provide the `FinalAnswer`.

**Available Tools:**
- `search_web(query: str)`: Use this to find information on the internet.
- `evaluate_expression(expression: str)`: Use this for mathematical calculations.

**Your output MUST ALWAYS be a single JSON object that validates against the required schema.**

Begin.
"""

# --- LangGraph Nodes and Edges ---

def call_agent_and_parse(state: State, config: RunnableConfig) -> State:
    """Node that calls the LLM, which is structured to output a ReActStep object."""
    print("--- Thực hiện Node: call_agent_and_parse (Hybrid) ---")
    
    # Configure the LLM to use our ReActStep schema
    llm = config["configurable"]["llm"]
    llm_with_structure = llm.with_structured_output(ReActStep)
    
    messages = state["messages"][-6:] if len(state["messages"]) >= 6 else state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages.insert(0, SystemMessage(content=REACT_HYBRID_PROMPT))

    # Invoke the LLM
    response: ReActStep = llm_with_structure.invoke(messages)
    
    print(f"\n🤔 REASONING: {response.reasoning}")
    
    # Add the full reasoning and action plan to the message history
    # This gives the agent memory of its previous steps.
    # We add it as a JSON string within an AIMessage.
    state["messages"].append(AIMessage(content=response.model_dump_json()))

    # Check the type of action and update the state
    if isinstance(response.action, FinalAnswer):
        print(f"✅ ACTION: Final Answer")
        state["answer"] = response.action.answer
        state["parsed_action"] = None # Signal to end the loop
    else: # It's a list of ToolCall objects
        print(f"🛠️ ACTION: Call Tools")
        for tool_call in response.action:
            print(f"- Tool: {tool_call.name}, Arguments: {tool_call.arguments}")
        state["parsed_action"] = response.action

    return state


def execute_tool(state: State) -> State:
    """Node that executes the tool calls parsed by the previous node."""
    print("--- Thực hiện Node: execute_tool (Hybrid) ---")
    tool_calls: List[ToolCall] = state["parsed_action"]
    known_tools = {tool.name: tool for tool in state["tools"]} if state.get("tools") else {}
    
    observations = []
    for tool_call in tool_calls:
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        
        tool_function = known_tools.get(tool_name)
        if not tool_function:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                # The .invoke() method of a StructuredTool can handle a dict of args
                observation = tool_function.invoke(tool_args)
            except Exception as e:
                observation = f"Error executing tool {tool_name}: {e}"
        
        observations.append(f"Observation from {tool_name}:\n{observation}")

    # Add all observations as a single HumanMessage for the next loop
    full_observation = "\n---\n".join(observations)
    print(f"👁️ OBSERVATIONS:\n{full_observation}")
    state["messages"].append(HumanMessage(content=full_observation))
    state["parsed_action"] = None # Clear the action after execution
    return state