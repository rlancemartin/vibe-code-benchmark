from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
import os

# Initialize the model
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define the graph state
class State(TypedDict):
    joke: str
    improved_joke: str

# Define the nodes
def create_joke(state: State):
    """Generate an initial joke"""
    msg = model.invoke("Write a short, original joke")
    return {"joke": msg.content}

def improve_joke(state: State):
    """Improve the joke to make it funnier"""
    msg = model.invoke(f"Make this joke funnier by adding wordplay and a surprising twist:\n{state['joke']}")
    return {"improved_joke": msg.content}

# Build the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("create_joke", create_joke)
workflow.add_node("improve_joke", improve_joke)

# Add edges to connect nodes
workflow.add_edge(START, "create_joke")
workflow.add_edge("create_joke", "improve_joke")
workflow.add_edge("improve_joke", END)

# Compile the workflow
app = workflow.compile()

# Standard interface for import
compiled_graph = app

# Save compiled graph for langgraph dev
if __name__ == "__main__":
    print("Joke creation workflow created and compiled!")
    print("Use 'langgraph dev' to run the workflow locally.") 