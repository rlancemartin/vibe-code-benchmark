from typing import Literal, Dict, List
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END

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
        a: The number to subtract from
        b: The number to subtract
        
    Returns:
        The difference a - b
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
        The quotient a / b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent.
    
    Args:
        base: The base number
        exponent: The exponent
        
    Returns:
        base raised to the power of exponent
    """
    return base ** exponent

@tool
def square_root(number: float) -> float:
    """Calculate the square root of a number.
    
    Args:
        number: A non-negative number
        
    Returns:
        The square root of the number
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return number ** 0.5

# Setup Claude model with tools
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
tools = [add, subtract, multiply, divide, power, square_root]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Define agent nodes
def llm_node(state: MessagesState) -> Dict:
    """LLM decides whether to call a tool or respond directly"""
    
    # Add system message for agent behavior
    messages = [
        SystemMessage(content="""You are a helpful math assistant that can perform arithmetic operations.
        You have access to tools for addition, subtraction, multiplication, division, powers, and square roots.
        Use these tools to perform calculations and provide clear explanations of the process.
        For complex expressions, break them down into individual operations and use the appropriate tools.
        """)
    ] + state["messages"]
    
    # Call the model to decide next action
    response = model_with_tools.invoke(messages)
    
    return {"messages": [response]}

def tool_node(state: MessagesState) -> Dict:
    """Execute the tools called by the LLM"""
    
    # Initialize list for tool results
    results = []
    
    # Get the last message from the model (should contain tool calls)
    last_message = state["messages"][-1]
    
    # Process each tool call
    for tool_call in last_message.tool_calls:
        # Get the tool by name
        tool_name = tool_call["name"]
        tool_fn = tools_by_name[tool_name]
        
        # Execute the tool with the provided arguments
        tool_result = tool_fn.invoke(tool_call["args"])
        
        # Create a tool message with the result
        results.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": results}

# Decide whether to execute tools or end the interaction
def should_use_tools(state: MessagesState) -> Literal["tools", END]:
    """Determine whether to use tools or end based on LLM response"""
    
    # Get the last message
    last_message = state["messages"][-1]
    
    # If the model made tool calls, route to execute tools
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the interaction
    return END

# Build the agent graph
builder = StateGraph(MessagesState)

# Add nodes to the graph
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

# Add edges to connect the nodes
builder.add_edge(START, "llm")
builder.add_conditional_edges(
    "llm",
    should_use_tools,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_edge("tools", "llm")

# Compile the agent
math_agent = builder.compile()

# Standard interface for import
compiled_graph = math_agent

# What we'll run with langgraph dev
if __name__ == "__main__":
    # Test example
    result = math_agent.invoke({
        "messages": [
            HumanMessage(content="What is 23 * 47, then take the square root of that result?")
        ]
    })
    
    # Print the interaction
    print("------- Math Agent Interaction -------")
    for message in result["messages"]:
        if message.type == "human":
            print(f"\nHuman: {message.content}")
        elif message.type == "ai":
            print(f"\nAI: {message.content}")
        elif message.type == "tool":
            print(f"\nTool result: {message.content}")
    print("-------------------------------------")