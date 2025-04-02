from typing import TypedDict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# Define the state for our graph
class State(TypedDict):
    joke: str
    improved_joke: str

# Initialize Claude model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Define node functions for our graph
def create_joke(state: State) -> State:
    """The first LLM call that creates a joke."""
    response = model.invoke([
        SystemMessage(content="You are a professional comedian known for writing original jokes. Create a short, clever joke."),
        HumanMessage(content="Create a funny, original joke. Keep it short and clever.")
    ])
    return {"joke": response.content}

def improve_joke(state: State) -> State:
    """The second LLM call that improves the joke."""
    response = model.invoke([
        SystemMessage(content="You are a comedy expert who can improve jokes to make them funnier."),
        HumanMessage(content=f"Here's a joke: '{state['joke']}'. Improve this joke to make it funnier while keeping its essence and structure. Make it concise and witty.")
    ])
    return {"improved_joke": response.content}

# Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("create_joke", create_joke)
builder.add_node("improve_joke", improve_joke)

# Add edges to connect nodes
builder.add_edge(START, "create_joke")
builder.add_edge("create_joke", "improve_joke")
builder.add_edge("improve_joke", END)

# Compile the workflow
joke_workflow = builder.compile()

# Standard interface for import
compiled_graph = joke_workflow

# Export the graph for use with LangGraph CLI
if __name__ == "__main__":
    # Print the jokes for testing
    result = joke_workflow.invoke({})
    print("Original joke:", result["joke"])
    print("\nImproved joke:", result["improved_joke"]) 