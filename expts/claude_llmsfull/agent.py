from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import math
import json
import re

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    tool_calls: List[Dict[str, Any]]
    observations: List[str]
    steps: List[Dict]

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define math tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises an error if b is 0."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def square_root(a: float) -> float:
    """Calculate the square root of a number. Raises an error if a is negative."""
    if a < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return math.sqrt(a)

@tool
def power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return math.pow(base, exponent)

@tool
def sin(angle_radians: float) -> float:
    """Calculate the sine of an angle in radians."""
    return math.sin(angle_radians)

@tool
def cos(angle_radians: float) -> float:
    """Calculate the cosine of an angle in radians."""
    return math.cos(angle_radians)

@tool
def tan(angle_radians: float) -> float:
    """Calculate the tangent of an angle in radians."""
    return math.tan(angle_radians)

# Create a list of all math tools
math_tools = [
    add,
    subtract,
    multiply,
    divide,
    square_root,
    power,
    sin,
    cos,
    tan
]

# Define agent functions
def agent_executor(state: AgentState) -> Dict[str, Any]:
    """Execute the agent's decision-making process."""
    messages = state["messages"]
    
    # Model with tools bound
    model = llm.bind_tools(math_tools)
    
    # Get the model's response
    response = model.invoke(messages)
    
    # Extract tool calls if they exist
    tool_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls = response.tool_calls
    
    # Save updated state
    return {
        "messages": messages + [response],
        "tool_calls": tool_calls,
        "steps": state.get("steps", []) + [{"response": response.content}]
    }

def tool_executor(state: AgentState) -> Dict[str, Any]:
    """Execute tools based on the agent's decisions."""
    tool_calls = state["tool_calls"]
    observations = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        
        # Find the tool
        tool_fn = next((tool for tool in math_tools if tool.name == tool_name), None)
        
        if tool_fn:
            try:
                # Execute the tool
                result = tool_fn(**arguments)
                observation = f"Tool {tool_name} returned: {result}"
            except Exception as e:
                observation = f"Error executing {tool_name}: {str(e)}"
        else:
            observation = f"Tool {tool_name} not found"
            
        observations.append(observation)
    
    # Create tool response messages
    tool_response_messages = []
    for i, observation in enumerate(observations):
        # Create a tool response message
        tool_response_message = AIMessage(
            content=observation,
            name=tool_calls[i]["name"]
        )
        tool_response_messages.append(tool_response_message)
    
    # Return updated state with observations
    return {
        "messages": state["messages"] + tool_response_messages,
        "observations": observations,
        "steps": state.get("steps", []) + [{"tool_calls": tool_calls, "observations": observations}]
    }

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue or finish."""
    # Check if there are tool calls to execute
    if state.get("tool_calls"):
        return "tool_executor"
    return "end"

# Build the agent graph
def build_graph():
    # Create a new graph
    graph_builder = StateGraph(AgentState)
    
    # Add nodes
    graph_builder.add_node("agent_executor", agent_executor)
    graph_builder.add_node("tool_executor", tool_executor)
    
    # Add edges
    graph_builder.add_edge("agent_executor", should_continue)
    graph_builder.add_conditional_edges(
        "agent_executor",
        should_continue,
        {
            "tool_executor": "tool_executor",
            "end": END
        }
    )
    graph_builder.add_edge("tool_executor", "agent_executor")
    
    # Set entrypoint
    graph_builder.set_entry_point("agent_executor")
    
    # Compile the graph
    return graph_builder.compile()

# Create the compiled graph for imports
compiled_graph = build_graph()

if __name__ == "__main__":
    # Use the compiled graph
    graph = compiled_graph
    
    # Create the langgraph.json config file
    config = {
        "title": "Math Agent Workflow",
        "description": "A LangGraph agent that can perform math operations using tools",
        "schema": {
            "input": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "The conversation messages"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "The updated conversation with agent responses"
                    },
                    "steps": {
                        "type": "array",
                        "description": "A trace of steps the agent took"
                    }
                }
            }
        },
        "config": {
            "modules": ["agent"],
            "graph": "build_graph"
        }
    }
    
    # Save config file
    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Graph compiled and langgraph.json configuration created successfully")