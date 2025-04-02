import operator
from typing import Annotated, Sequence, TypedDict, Union
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, create_agent_executor


# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation so far"]
    # This field is populated when the agent wants to use a tool
    next: Annotated[Union[str, None], "The name of the next node to call, or None if we're done"]


# Define math tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return operator.add(a, b)


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return operator.sub(a, b)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return operator.mul(a, b)


@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second. Returns an error if the second number is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return operator.truediv(a, b)


# Initialize the LLM
def build_math_agent():
    # Initialize the Claude 3.5 Sonnet model
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    
    # Define the tools we want to use
    tools = [add, subtract, multiply, divide]
    
    # Create the agent executor
    agent_executor = create_agent_executor(
        llm=llm,
        tools=tools,
        system_message="""You are a helpful assistant who is expert at arithmetic. 
You have access to tools to perform various math operations.
Always think step by step and show your reasoning.
""",
    )
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add the agent node that processes messages and decides to use tools
    workflow.add_node("agent", agent_executor)
    
    # Add the tool node that executes tools
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Configure conditional edges for when to finish
    def route_agent(state: AgentState) -> Union[str, Sequence[str]]:
        if state.get("next") is None:
            return END
        return state["next"]
    
    workflow.add_conditional_edges("agent", route_agent)
    
    # Compile the graph
    math_graph = workflow.compile()
    
    return math_graph


# Create the graph
math_graph = build_math_agent()

# This is the object that will be used by LangGraph server
agent_graph = math_graph 