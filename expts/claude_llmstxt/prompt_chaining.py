from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Define the state for our workflow
class JokeState(TypedDict):
    # Our message history, which will be updated in an append-only way
    messages: Annotated[list, add_messages]
    # The original joke (string)
    joke: str
    # The improved joke (string)
    improved_joke: str


# Create a model instance using Claude 3.5 Sonnet
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Define our nodes that will handle the workflow steps

def create_joke(state: JokeState):
    """Node that creates an initial joke based on a topic."""
    # Extract the user's message to get the topic
    user_message = state["messages"][-1].content if state["messages"] else "general"
    
    # Create a system prompt to generate a joke
    system_prompt = "You are a witty comedian. Create a short, funny joke based on the topic provided."
    
    # Generate a joke using the LLM
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a joke about: {user_message}"}
        ]
    )
    
    # Extract the joke from the response
    joke = response.content
    
    # Return the updated state with the new joke
    return {
        "messages": [response],
        "joke": joke
    }


def improve_joke(state: JokeState):
    """Node that takes an existing joke and improves it."""
    # Get the original joke
    original_joke = state["joke"]
    
    # Create a system prompt to improve the joke
    system_prompt = """You are a master comedy writer. Take the given joke and improve it by:
    1. Making it more concise
    2. Improving the setup and punchline
    3. Adding an unexpected twist if possible
    4. Ensuring it's appropriate for general audiences
    
    Return ONLY the improved joke."""
    
    # Generate an improved joke using the LLM
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Improve this joke:\n\n{original_joke}"}
        ]
    )
    
    # Extract the improved joke from the response
    improved_joke = response.content
    
    # Return the updated state with the improved joke
    return {
        "messages": [response],
        "improved_joke": improved_joke
    }


def present_jokes(state: JokeState):
    """Final node that presents both the original and improved joke."""
    original_joke = state["joke"]
    improved_joke = state["improved_joke"]
    
    # Create a summary message
    summary = f"""Here are the jokes I created for you:

ORIGINAL JOKE:
{original_joke}

IMPROVED VERSION:
{improved_joke}
"""
    
    # Return a message that presents both jokes
    return {"messages": [{"role": "assistant", "content": summary}]}


# Build the graph
def build_graph():
    # Create a new graph
    graph_builder = StateGraph(JokeState)
    
    # Add nodes
    graph_builder.add_node("create_joke", create_joke)
    graph_builder.add_node("improve_joke", improve_joke)
    graph_builder.add_node("present_jokes", present_jokes)
    
    # Add edges to create a sequential workflow
    graph_builder.add_edge(START, "create_joke")
    graph_builder.add_edge("create_joke", "improve_joke")
    graph_builder.add_edge("improve_joke", "present_jokes")
    graph_builder.add_edge("present_jokes", END)
    
    # Compile the graph
    return graph_builder.compile()


# Build and compile the graph
compiled_graph = build_graph()

# Export the graph for use with langgraph server
if __name__ == "__main__":
    # Define the config for langgraph.json
    import json
    
    langgraph_config = {
        "title": "Joke Creation and Improvement",
        "description": "A workflow that creates a joke and then improves it",
        "graph": {
            "module": "prompt-chaining",
            "function": "build_graph"
        }
    }
    
    # Write the langgraph.json file
    with open("langgraph.json", "w") as f:
        json.dump(langgraph_config, f, indent=2)
    
    print("LangGraph workflow has been compiled successfully.")
    print("Run it locally with 'langgraph dev'")