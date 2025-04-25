from langgraph.graph import StateGraph, END
from langgraph.nodes import llm
from langgraph.routing import router

# Define the LLM node for each type
story_node = llm(
    model="claude-3-5-sonnet-latest",
    prompt="Write a short, original story based on the following input: {input}",
    output_key="story"
)

poem_node = llm(
    model="claude-3-5-sonnet-latest",
    prompt="Write a short, original poem based on the following input: {input}",
    output_key="poem"
)

joke_node = llm(
    model="claude-3-5-sonnet-latest",
    prompt="Write a short, original joke based on the following input: {input}",
    output_key="joke"
)

# Router function to pick the correct node
def route_input(state):
    user_input = state["input"].lower()
    if "poem" in user_input:
        return "poem"
    elif "joke" in user_input:
        return "joke"
    else:
        return "story"

# Build the graph
graph = StateGraph()
graph.add_node("story", story_node)
graph.add_node("poem", poem_node)
graph.add_node("joke", joke_node)
graph.add_router("route", route_input, outputs=["story", "poem", "joke"])

graph.add_edge("route", "story")
graph.add_edge("route", "poem")
graph.add_edge("route", "joke")
graph.add_edge("story", END)
graph.add_edge("poem", END)
graph.add_edge("joke", END)

compiled_graph = graph.compile()
