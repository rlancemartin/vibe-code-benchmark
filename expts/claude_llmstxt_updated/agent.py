from typing_extensions import Literal
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END


# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Define math tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The result of a - b
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    Args:
        a: First number (numerator)
        b: Second number (denominator)
        
    Returns:
        The result of a / b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent.
    
    Args:
        base: The base number
        exponent: The exponent
        
    Returns:
        base raised to the power of exponent
    """
    return base ** exponent


# Create a dictionary of tools by name for easy lookup
tools = [add, subtract, multiply, divide, power]
tools_by_name = {tool.name: tool for tool in tools}

# Augment the LLM with tools
llm_with_tools = llm.bind_tools(tools)


# Define agent nodes
def llm_node(state: MessagesState):
    """LLM node that decides whether to call a tool or respond directly."""
    
    # Add system message if it's the first message
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [
            SystemMessage(
                content="You are a helpful math assistant that can perform arithmetic operations. "
                "You can use tools to calculate results. Provide clear explanations with your answers."
            )
        ] + messages
    
    # Generate LLM response
    response = llm_with_tools.invoke(messages)
    
    # Return updated messages
    return {"messages": [response]}


def tool_node(state: MessagesState):
    """Execute tool calls made by the LLM."""
    
    # Get the most recent message which should contain tool calls
    last_message = state["messages"][-1]
    
    # Process each tool call
    tool_results = []
    for tool_call in last_message.tool_calls:
        # Get the tool by name
        tool_name = tool_call["name"]
        tool_function = tools_by_name[tool_name]
        
        # Call the tool with the provided arguments
        args = tool_call["args"]
        try:
            observation = tool_function.invoke(args)
            # Create a tool message with the result
            tool_results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            # Handle errors in tool execution
            tool_results.append(
                ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"])
            )
    
    # Return tool results as messages
    return {"messages": tool_results}


# Define conditional edge function
def should_continue(state: MessagesState) -> Literal["tool_execution", END]:
    """Determine if we should continue running the agent or terminate."""
    
    # Get the last message
    last_message = state["messages"][-1]
    
    # If the LLM has made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_execution"
    
    # Otherwise end the process
    return END


# Build the agent graph
def build_math_agent():
    """Build and compile the math agent graph."""
    
    # Create the graph with message state
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("llm", llm_node)
    workflow.add_node("tool_execution", tool_node)
    
    # Add edges
    workflow.add_edge(START, "llm")
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tool_execution": "tool_execution",
            END: END,
        }
    )
    workflow.add_edge("tool_execution", "llm")
    
    # Compile the graph
    return workflow.compile()


# Create the compiled agent
math_agent = build_math_agent()

# Standard interface for import
compiled_graph = math_agent


# Example use
if __name__ == "__main__":
    # Test with a simple math problem
    user_input = "What is the result of 25 divided by 5, then multiplied by 3, and finally raised to the power of 2?"
    messages = [HumanMessage(content=user_input)]
    
    # Invoke the agent
    result = math_agent.invoke({"messages": messages})
    
    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, "content"):
            role = getattr(message, "type", "unknown")
            print(f"{role}: {message.content}")
            print("-" * 50)