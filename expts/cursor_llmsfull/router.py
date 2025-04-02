from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# Define state with a messages field using the add_messages reducer
class ContentState(TypedDict):
    messages: Annotated[list, add_messages]
    content_type: str
    generated_content: str

# Initialize Claude 3.5 Sonnet
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Router to classify the input
def classify_input(state: ContentState) -> Literal["story_generator", "poem_generator", "joke_generator"]:
    """Classify input as story, poem, or joke request."""
    # Get the latest message content
    latest_message = state["messages"][-1]
    
    # Create a prompt for classification
    classification_prompt = f"""
    Classify the following request as either 'story', 'poem', or 'joke':
    
    "{latest_message.content}"
    
    Return only one word: 'story', 'poem', or 'joke'.
    """
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=classification_prompt)])
    
    # Extract and normalize the content type
    content_type = response.content.strip().lower()
    
    # Update state with content type
    state["content_type"] = content_type
    
    # Route to the appropriate generator
    if "story" in content_type:
        return "story_generator"
    elif "poem" in content_type:
        return "poem_generator"
    else:
        return "joke_generator"

# Story generator
def generate_story(state: ContentState) -> ContentState:
    """Generate a story based on the input message."""
    # Get the latest message content
    latest_message = state["messages"][-1]
    
    # Create a prompt for story generation
    story_prompt = f"""
    Create an engaging short story based on this request: 
    "{latest_message.content}"
    
    Make it creative, with interesting characters and a clear narrative arc.
    Keep it concise but compelling.
    """
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=story_prompt)])
    
    # Extract the story
    generated_story = response.content
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=generated_story)],
        "content_type": "story",
        "generated_content": generated_story
    }

# Poem generator
def generate_poem(state: ContentState) -> ContentState:
    """Generate a poem based on the input message."""
    # Get the latest message content
    latest_message = state["messages"][-1]
    
    # Create a prompt for poem generation
    poem_prompt = f"""
    Create a beautiful poem based on this request:
    "{latest_message.content}"
    
    Use vivid imagery, thoughtful structure, and appropriate poetic devices.
    """
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=poem_prompt)])
    
    # Extract the poem
    generated_poem = response.content
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=generated_poem)],
        "content_type": "poem",
        "generated_content": generated_poem
    }

# Joke generator
def generate_joke(state: ContentState) -> ContentState:
    """Generate a joke based on the input message."""
    # Get the latest message content
    latest_message = state["messages"][-1]
    
    # Create a prompt for joke generation
    joke_prompt = f"""
    Create a clever, funny joke based on this request:
    "{latest_message.content}"
    
    Make it witty and concise with a good punchline.
    """
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=joke_prompt)])
    
    # Extract the joke
    generated_joke = response.content
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=generated_joke)],
        "content_type": "joke",
        "generated_content": generated_joke
    }

# Build the graph
def build_content_graph():
    # Create the graph
    builder = StateGraph(ContentState)
    
    # Add nodes
    builder.add_node("router", classify_input)
    builder.add_node("story_generator", generate_story)
    builder.add_node("poem_generator", generate_poem)
    builder.add_node("joke_generator", generate_joke)
    
    # Add edges
    builder.add_edge(START, "router")
    builder.add_edge("router", "story_generator")
    builder.add_edge("router", "poem_generator")
    builder.add_edge("router", "joke_generator")
    builder.add_edge("story_generator", END)
    builder.add_edge("poem_generator", END)
    builder.add_edge("joke_generator", END)
    
    # Compile the graph
    return builder.compile()

# Create the graph
compiled_graph = build_content_graph()

# For testing the graph
if __name__ == "__main__":
    # Test with a sample prompt
    result = compiled_graph.invoke({
        "messages": [HumanMessage(content="Tell me a funny joke about programming.")]
    })
    
    print(f"Content Type: {result['content_type']}")
    print(f"Generated Content: {result['generated_content']}") 