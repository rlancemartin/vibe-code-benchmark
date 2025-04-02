from typing import TypedDict, Literal, Dict, Optional
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# Define the state of our graph
class RouterState(TypedDict):
    input: str
    result: Optional[str]
    content_type: Optional[str]

# Schema for structured output to use in content classification
class ContentClassifier(BaseModel):
    type: Literal["story", "poem", "joke"] = Field(
        description="The type of content requested in the input.",
    )
    reason: str = Field(
        description="Brief explanation of why this content type was chosen.",
    )

# Setup our Claude 3.5 Sonnet model
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# Augment the LLM with schema for structured content classification
classifier = llm.with_structured_output(ContentClassifier)

# Define our nodes
def classify_input(state: RouterState) -> Dict:
    """Classify the input as story, poem, or joke request"""
    
    classification = classifier.invoke(
        f"""Analyze the following input and determine if it's requesting a story, poem, or joke:
        
        Input: {state['input']}"""
    )
    
    return {"content_type": classification.type}

def generate_story(state: RouterState) -> Dict:
    """Generate a creative story based on the input"""
    
    story = llm.invoke(
        f"""Write a creative, engaging short story based on this input: {state['input']}
        Keep it concise but engaging, with a clear beginning, middle, and end."""
    )
    
    return {"result": story.content}

def generate_poem(state: RouterState) -> Dict:
    """Generate a poem based on the input"""
    
    poem = llm.invoke(
        f"""Write a beautiful poem based on this input: {state['input']}
        Make it expressive and evocative, with careful attention to imagery and rhythm."""
    )
    
    return {"result": poem.content}

def generate_joke(state: RouterState) -> Dict:
    """Generate a joke based on the input"""
    
    joke = llm.invoke(
        f"""Write a funny joke based on this input: {state['input']}
        Make it clever and humorous, with a good setup and punchline."""
    )
    
    return {"result": joke.content}

# Conditional edge function to route to the appropriate generator
def route_content(state: RouterState) -> str:
    """Route to the appropriate content generator based on classification"""
    
    content_type = state.get("content_type")
    if content_type == "story":
        return "story_generator"
    elif content_type == "poem":
        return "poem_generator"
    elif content_type == "joke":
        return "joke_generator"
    else:
        # Default to story if classification fails
        return "story_generator"

# Build our router workflow
router_builder = StateGraph(RouterState)

# Add nodes to the graph
router_builder.add_node("classifier", classify_input)
router_builder.add_node("story_generator", generate_story)
router_builder.add_node("poem_generator", generate_poem)
router_builder.add_node("joke_generator", generate_joke)

# Connect the nodes with edges
router_builder.add_edge(START, "classifier")
router_builder.add_conditional_edges(
    "classifier",
    route_content,
    {
        "story_generator": "story_generator",
        "poem_generator": "poem_generator",
        "joke_generator": "joke_generator",
    },
)
router_builder.add_edge("story_generator", END)
router_builder.add_edge("poem_generator", END)
router_builder.add_edge("joke_generator", END)

# Compile the workflow
compiled_graph = router_builder.compile()

# This is what we'll run with langgraph dev
if __name__ == "__main__":
    # Example usages
    story_request = "Tell me a story about a brave astronaut exploring a new planet"
    poem_request = "Write a poem about the changing seasons"
    joke_request = "Make me laugh with something about programmers"
    
    # Test with a story request
    story_result = compiled_graph.invoke({"input": story_request})
    print(f"Content Type: {story_result['content_type']}")
    print(f"Result:\n{story_result['result']}\n")
    
    # Test with a poem request
    poem_result = compiled_graph.invoke({"input": poem_request})
    print(f"Content Type: {poem_result['content_type']}")
    print(f"Result:\n{poem_result['result']}\n")
    
    # Test with a joke request
    joke_result = compiled_graph.invoke({"input": joke_request})
    print(f"Content Type: {joke_result['content_type']}")
    print(f"Result:\n{joke_result['result']}\n")