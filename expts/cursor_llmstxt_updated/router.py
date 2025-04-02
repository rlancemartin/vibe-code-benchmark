from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from enum import Enum

# Define content types
class ContentType(str, Enum):
    STORY = "story"
    POEM = "poem"
    JOKE = "joke"
    UNKNOWN = "unknown"

# Define the state of our graph
class State(TypedDict):
    content: str
    content_type: ContentType
    result: str

# Initialize the model
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define content type classifier
def classify_content(state: State):
    """Classify the content as story, poem, or joke"""
    prompt = f"""
    You are a content classifier. Given a piece of text, classify it as one of:
    - story
    - poem
    - joke
    
    Be specific in your answer, replying with ONLY the classification in lowercase.
    
    Text to classify: {state['content']}
    """
    
    response = llm.invoke(prompt)
    content_type = response.content.strip().lower()
    
    # Map the response to ContentType enum
    if content_type == "story":
        return {"content_type": ContentType.STORY}
    elif content_type == "poem":
        return {"content_type": ContentType.POEM}
    elif content_type == "joke":
        return {"content_type": ContentType.JOKE}
    else:
        return {"content_type": ContentType.UNKNOWN}

# Define the routing logic
def router(state: State):
    """Route to the appropriate node based on content type"""
    return state["content_type"]

# Define LLM processing functions
def generate_story(state: State):
    """Generate a story based on the input"""
    prompt = f"""
    Create an engaging short story based on this prompt: {state['content']}
    Make it descriptive with interesting characters and a clear plot.
    """
    response = llm.invoke(prompt)
    return {"result": response.content}

def generate_poem(state: State):
    """Generate a poem based on the input"""
    prompt = f"""
    Write a beautiful poem inspired by: {state['content']}
    Focus on imagery, rhythm, and emotion.
    """
    response = llm.invoke(prompt)
    return {"result": response.content}

def generate_joke(state: State):
    """Generate a joke based on the input"""
    prompt = f"""
    Create a funny joke related to: {state['content']}
    Make it clever with a good setup and punchline.
    """
    response = llm.invoke(prompt)
    return {"result": response.content}

def handle_unknown(state: State):
    """Handle content that couldn't be classified"""
    return {"result": "Sorry, I couldn't determine if your input is a story, poem, or joke. Please try again with clearer content."}

# Build the graph
def build_graph():
    # Initialize the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("classify", classify_content)
    workflow.add_node("generate_story", generate_story)
    workflow.add_node("generate_poem", generate_poem)
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("handle_unknown", handle_unknown)
    
    # Add edges
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        router,
        {
            ContentType.STORY: "generate_story",
            ContentType.POEM: "generate_poem",
            ContentType.JOKE: "generate_joke",
            ContentType.UNKNOWN: "handle_unknown"
        }
    )
    workflow.add_edge("generate_story", END)
    workflow.add_edge("generate_poem", END)
    workflow.add_edge("generate_joke", END)
    workflow.add_edge("handle_unknown", END)
    
    # Compile the graph
    return workflow.compile()

# Create and export the compiled graph
compiled_graph = build_graph()

# Example of how to invoke the graph
if __name__ == "__main__":
    # Example inputs
    story_input = "Tell me about a space explorer who discovers a new planet"
    poem_input = "The sunset over the ocean on a summer evening"
    joke_input = "A programmer walks into a bar"
    
    # Test with story input
    result = compiled_graph.invoke({"content": story_input})
    print(f"Input: {story_input}")
    print(f"Result: {result['result']}\n")
    
    # Test with poem input
    result = compiled_graph.invoke({"content": poem_input})
    print(f"Input: {poem_input}")
    print(f"Result: {result['result']}\n")
    
    # Test with joke input
    result = compiled_graph.invoke({"content": joke_input})
    print(f"Input: {joke_input}")
    print(f"Result: {result['result']}") 