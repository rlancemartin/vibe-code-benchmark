from typing import Dict, List, Annotated, TypedDict
import operator
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, BaseTool


# Define math tools for arithmetic operations
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
        raise ValueError("Cannot divide by zero!")
    return operator.truediv(a, b)


# Collect all tools
math_tools = [add, subtract, multiply, divide]

# Create a tool node with the math tools
tool_node = ToolNode(math_tools)

# Create an LLM (Claude-3.5-sonnet) with tools
model = ChatAnthropic(
    model="claude-3-5-sonnet-latest",
    temperature=0
).bind_tools(math_tools)

# Define callback logic
def should_continue(state: MessagesState) -> str:
    """Determine if we should call tools or finish."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

def call_model(state: MessagesState) -> Dict:
    """Call the model and return the response."""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Build the graph
def build_graph():
    # Create state graph with message state
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Define the edges
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Set conditional edges for determining when to call tools vs. end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": None  # None means stop
        }
    )
    
    # Compile the graph
    return workflow.compile()

# Create the app
app = build_graph()

# If run as a script, test the app
if __name__ == "__main__":
    # Test with a simple arithmetic question
    result = app.invoke({
        "messages": [
            HumanMessage(content="What is 25 plus 15?")
        ]
    })
    
    # Print the conversation
    for message in result["messages"]:
        print(f"{type(message).__name__}: {message.content}") 