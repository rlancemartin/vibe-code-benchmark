"""
LangGraph agent with math tools for arithmetic operations.
This agent can add, subtract, multiply, and divide numbers through tool use.
"""

import operator
from typing import Annotated, List, TypedDict, Union, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Define the tools for basic arithmetic operations
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return operator.add(a, b)

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return operator.sub(a, b)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return operator.mul(a, b)

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return operator.truediv(a, b)

# Define state for our agent
class State(TypedDict):
    """State for the calculator agent."""
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], add_messages]

# Define the function to determine if we should continue
def should_continue(state: State):
    """Determine if the agent should continue or finish."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, we should continue
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we're done
    return END

def create_agent():
    """Create and compile the calculator agent."""
    # Define the tools
    tools = [add, subtract, multiply, divide]
    
    # Create the model with tools
    model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    model_with_tools = model.bind_tools(tools)
    
    # Create the tool node
    tool_node = ToolNode(tools)
    
    # Create the agent node
    def agent(state: State):
        """Process the messages and return a response."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    
    # Define the edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()

def compile_to_json():
    """Compile the agent and serialize to JSON."""
    graph = create_agent()
    return graph.get_graph().to_json()

compiled_graph = create_agent()

if __name__ == "__main__":
    # This would be used for testing locally
    agent = create_agent()
    
    # Example interaction
    result = agent.invoke({
        "messages": [
            HumanMessage(content="What is 25 × 4?")
        ]
    })
    
    # Print the final result
    print(result["messages"][-1].content)