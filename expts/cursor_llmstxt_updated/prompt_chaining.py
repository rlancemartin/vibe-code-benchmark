from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# Define the state of our graph
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str

# Initialize the model
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define the nodes in our graph
def create_joke(state: State):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def improve_joke(state: State):
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay and an unexpected twist: {state['joke']}")
    return {"improved_joke": msg.content}

# Build the graph
def build_graph():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("create_joke", create_joke)
    workflow.add_node("improve_joke", improve_joke)
    
    # Add edges to connect nodes
    workflow.add_edge(START, "create_joke")
    workflow.add_edge("create_joke", "improve_joke")
    workflow.add_edge("improve_joke", END)
    
    # Compile the graph
    return workflow.compile()

# Create and export the compiled graph
joke_chain = build_graph()

# Standard interface for import
compiled_graph = joke_chain

# Example of how to invoke the graph
if __name__ == "__main__":
    result = joke_chain.invoke({"topic": "programming"})
    print("Original Joke:")
    print(result["joke"])
    print("\nImproved Joke:")
    print(result["improved_joke"]) 