from langgraph.graph import StateGraph, END
from langgraph.graph.nodes import LLMNode
from langgraph.llms.anthropic import Anthropic

# Define the state for the workflow
def joke_state():
    return {"joke": None, "improved_joke": None}

# Set up the LLM provider (Claude 3.5 Sonnet)
llm = Anthropic(model="claude-3-5-sonnet-latest")

# Node 1: Create a joke
def create_joke_node(state):
    prompt = "Write a short, original joke."
    response = llm.complete(prompt)
    return {"joke": response}

# Node 2: Improve the joke
def improve_joke_node(state):
    joke = state["joke"]
    prompt = f"Here is a joke: {joke}\n\nMake this joke even funnier and more clever. Return only the improved joke."
    response = llm.complete(prompt)
    return {"improved_joke": response}

# Build the workflow graph
graph = StateGraph(joke_state)
graph.add_node("create_joke", create_joke_node)
graph.add_node("improve_joke", improve_joke_node)

graph.add_edge("create_joke", "improve_joke")
graph.add_edge("improve_joke", END)

graph.set_entry_point("create_joke")

# Compile the graph for local execution
compiled_graph = graph.compile()

# Save compiled graph for langgraph dev
with open("langgraph.json", "w") as f:
    f.write(compiled_graph.to_json())
