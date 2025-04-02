from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
import json
import os

# Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]
    joke: str
    improved_joke: str

# Define the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define node functions
def joke_creator(state: State) -> State:
    """Create an initial joke"""
    prompt = HumanMessage(content="Create a short, funny joke on any topic.")
    response = llm.invoke([prompt])
    
    # Update state with the created joke
    return {
        "messages": [prompt, response],
        "joke": response.content,
        "improved_joke": ""
    }

def joke_improver(state: State) -> State:
    """Improve the initial joke"""
    # Create a prompt to improve the joke
    prompt = HumanMessage(content=f"Here's a joke: {state['joke']}. Please improve this joke to make it funnier, clearer, and more engaging.")
    response = llm.invoke([prompt])
    
    # Update state with the improved joke
    return {
        "messages": state["messages"] + [prompt, response],
        "joke": state["joke"],
        "improved_joke": response.content
    }

# Build the graph
def build_graph():
    # Create a new graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("create_joke", joke_creator)
    graph_builder.add_node("improve_joke", joke_improver)
    
    # Add edges
    graph_builder.add_edge("create_joke", "improve_joke")
    graph_builder.add_edge("improve_joke", END)
    
    # Set entry point
    graph_builder.set_entry_point("create_joke")
    
    # Compile the graph
    graph = graph_builder.compile()
    
    return graph

# Create the compiled graph for imports
compiled_graph = build_graph()

if __name__ == "__main__":
    # Use the compiled graph
    graph = compiled_graph
    
    # Create the langgraph.json file
    config = {
        "title": "Joke Creation and Improvement Workflow",
        "description": "A LangGraph workflow that creates a joke and then improves it",
        "schema": {
            "input": {
                "type": "object",
                "properties": {}
            },
            "output": {
                "type": "object",
                "properties": {
                    "joke": {
                        "type": "string",
                        "description": "The initial joke created"
                    },
                    "improved_joke": {
                        "type": "string",
                        "description": "The improved version of the joke"
                    }
                }
            }
        }
    }
    
    # Save config file
    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Graph compiled and langgraph.json created successfully")