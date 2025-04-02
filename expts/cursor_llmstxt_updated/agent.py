import os
from typing import Literal, TypedDict, Dict, Any, List, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# Set up LLM
model_name = "claude-3-5-sonnet-latest"
llm = ChatAnthropic(model=model_name)

# Define math tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers.
    
    Args:
        a: first number
        b: second number
        
    Returns:
        The sum of a and b
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.
    
    Args:
        a: first number
        b: second number
        
    Returns:
        The difference between a and b (a - b)
    """
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    
    Args:
        a: first number
        b: second number
        
    Returns:
        The product of a and b
    """
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    Args:
        a: first number (numerator)
        b: second number (denominator)
        
    Returns:
        The quotient of a divided by b (a / b)
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Create a list of tools
tools = [add, subtract, multiply, divide]

# Create a dictionary mapping tool names to tools
tools_by_name = {tool.name: tool for tool in tools}

# Augment the LLM with tools
llm_with_tools = llm.bind_tools(tools)

# Define nodes for the agent graph
def llm_node(state: MessagesState):
    """LLM decides whether to call a tool or respond directly"""
    
    system_message = """You are a math assistant that can perform arithmetic operations.
    You can use tools to perform addition, subtraction, multiplication, and division.
    When a user asks a math question, use the appropriate tool to calculate the answer.
    When you have the final answer, respond directly to the user with the result.
    Explain your reasoning step by step."""
    
    messages = [SystemMessage(content=system_message)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def tool_node(state: MessagesState):
    """Execute tool calls made by the LLM"""
    
    result = []
    last_message = state["messages"][-1]
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_to_call = tools_by_name[tool_name]
        
        # Execute the tool
        observation = tool_to_call.invoke(tool_args)
        
        # Create a tool message with the result
        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        )
    
    return {"messages": result}

# Conditional edge function
def should_continue(state: MessagesState) -> str:
    """Determine whether to call a tool or finish"""
    
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, continue to the tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    
    # Otherwise, we're done
    return "end"

# Build the agent graph
math_agent_builder = StateGraph(MessagesState)

# Add nodes
math_agent_builder.add_node("llm", llm_node)
math_agent_builder.add_node("tool", tool_node)

# Add edges
math_agent_builder.add_edge(START, "llm")
math_agent_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "call_tool": "tool",
        "end": END
    }
)
math_agent_builder.add_edge("tool", "llm")

# Compile the agent
compiled_graph = math_agent_builder.compile()

# Function to run the agent
def run_math_agent(question: str):
    """Run the math agent with a specific question"""
    
    # Create the initial state with the user's question
    initial_state = {"messages": [HumanMessage(content=question)]}
    
    # Run the agent
    result = compiled_graph.invoke(initial_state)
    
    return result

if __name__ == "__main__":
    # Example usage
    question = "What is 25 divided by 5, and then multiplied by 3?"
    result = run_math_agent(question)
    
    # Print the conversation
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool ({message.name}): {message.content}")
        else:
            print(f"AI: {message.content}") 