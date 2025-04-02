from typing import Annotated, TypedDict
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# Define state with a messages field using the add_messages reducer
class JokeState(TypedDict):
    messages: Annotated[list, add_messages]
    joke: str
    improved_joke: str

# Initialize Claude 3.5 Sonnet
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Node 1: Create a joke
def create_joke(state: JokeState) -> JokeState:
    """Create a joke based on the input messages."""
    if not state.get("messages"):
        # Default message if none provided
        return {"messages": [HumanMessage(content="Tell me a joke about programming.")]}
    
    # Get the latest message content
    latest_message = state["messages"][-1]
    
    # Create a prompt for joke generation
    joke_prompt = f"Create a short, clever joke about: {latest_message.content}"
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=joke_prompt)])
    
    # Extract the joke
    joke = response.content
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=f"Here's a joke I created: {joke}")],
        "joke": joke
    }

# Node 2: Improve the joke
def improve_joke(state: JokeState) -> JokeState:
    """Improve the joke that was created."""
    # Get the original joke
    original_joke = state["joke"]
    
    # Create a prompt for joke improvement
    improvement_prompt = f"""
    Here's a joke: {original_joke}
    
    Please improve this joke to make it funnier, more clever, and more polished.
    Maintain the same general topic but enhance the delivery, punchline, or structure.
    """
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=improvement_prompt)])
    
    # Extract the improved joke
    improved_joke = response.content
    
    # Return the updated state with both jokes
    return {
        "messages": [AIMessage(content=f"I've improved the joke. Here's the better version: {improved_joke}")],
        "improved_joke": improved_joke
    }

# Build the graph
def build_joke_graph():
    # Create the graph
    builder = StateGraph(JokeState)
    
    # Add nodes
    builder.add_node("create_joke", create_joke)
    builder.add_node("improve_joke", improve_joke)
    
    # Add edges to create a sequential flow
    builder.add_edge(START, "create_joke")
    builder.add_edge("create_joke", "improve_joke")
    builder.add_edge("improve_joke", END)
    
    # Compile the graph
    return builder.compile()

# Create the graph
joke_graph = build_joke_graph()

# Standard interface for import
compiled_graph = joke_graph

# For testing the graph
if __name__ == "__main__":
    # Test with a default prompt
    result = joke_graph.invoke({
        "messages": [HumanMessage(content="Tell me a joke about artificial intelligence.")]
    })
    
    print("Original Joke:", result["joke"])
    print("\nImproved Joke:", result["improved_joke"]) 