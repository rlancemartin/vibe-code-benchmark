from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_claude import ChatClaude
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# Define the model
llm = ChatClaude(model="claude-3-5-sonnet-latest")

# Define math tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: The first number
        b: The second number
    
    Returns:
        The sum of a and b
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.
    
    Args:
        a: The first number
        b: The second number to subtract from a
    
    Returns:
        The difference between a and b
    """
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: The first number
        b: The second number
    
    Returns:
        The product of a and b
    """
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    Args:
        a: The numerator
        b: The denominator (cannot be zero)
    
    Returns:
        The result of dividing a by b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Create a list of tools
tools = [add, subtract, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}

# Augment the LLM with tools
llm_with_tools = llm.bind_tools(tools)

# Node functions
def llm_with_prompt(state: MessagesState):
    """Process the input and decide on the next action."""
    messages = state["messages"]
    system_message = SystemMessage(content="""You are a helpful math assistant. 
You have access to math tools to perform calculations.
Always use the appropriate tool when calculations are needed.
Think carefully about which tool to use and the order of operations.""")
    
    return {
        "messages": [
            llm_with_tools.invoke([system_message] + messages)
        ]
    }

def process_tool_calls(state: MessagesState):
    """Execute any tool calls made by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return {"messages": []}
    
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_instance = tools_by_name[tool_name]
        
        try:
            observation = tool_instance.invoke(tool_args)
            results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            results.append(
                ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"])
            )
    
    return {"messages": results}

# Router function
def should_continue(state: MessagesState) -> Literal["call_tools", END]:
    """Determine if we should call tools or finish."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "call_tools"
    return END

# Create the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", llm_with_prompt)
builder.add_node("call_tools", process_tool_calls)

# Add edges
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tools": "call_tools",
        END: END
    }
)
builder.add_edge("call_tools", "agent")

# Compile graph
graph = builder.compile()

# For testing
if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="What is 25 * 32?")]}
    result = graph.invoke(inputs)
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}") 