from typing import Literal, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Define state schema
class RouterState(TypedDict):
    input: str
    output: str
    input_type: str

# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define node functions
def router(state: RouterState):
    """Determines if input is a story, poem, or joke request."""
    system_message = SystemMessage(content="""
    You are a classifier that determines if a request is for a story, poem, or joke.
    Reply with only one word: 'story', 'poem', or 'joke'.
    """)
    
    human_message = HumanMessage(content=f"Input: {state['input']}")
    
    response = llm.invoke([system_message, human_message])
    input_type = response.content.strip().lower()
    
    # Ensure we only get valid types
    if input_type not in ["story", "poem", "joke"]:
        input_type = "story"  # Default to story if unclear
        
    return {"input_type": input_type}

def generate_story(state: RouterState):
    """Generate a short story based on the input."""
    system_message = SystemMessage(content="""
    You are a creative storyteller. Generate an engaging short story based on the input.
    """)
    
    human_message = HumanMessage(content=f"Input: {state['input']}")
    
    response = llm.invoke([system_message, human_message])
    return {"output": response.content}

def generate_poem(state: RouterState):
    """Generate a poem based on the input."""
    system_message = SystemMessage(content="""
    You are a talented poet. Create a beautiful poem with good rhythm and imagery based on the input.
    """)
    
    human_message = HumanMessage(content=f"Input: {state['input']}")
    
    response = llm.invoke([system_message, human_message])
    return {"output": response.content}

def generate_joke(state: RouterState):
    """Generate a joke based on the input."""
    system_message = SystemMessage(content="""
    You are a comedy expert. Create a funny and clever joke based on the input.
    """)
    
    human_message = HumanMessage(content=f"Input: {state['input']}")
    
    response = llm.invoke([system_message, human_message])
    return {"output": response.content}

# Define conditional routing function
def route_to_generator(state: RouterState) -> Literal["generate_story", "generate_poem", "generate_joke"]:
    """Routes to the appropriate generator based on input type."""
    if state["input_type"] == "poem":
        return "generate_poem"
    elif state["input_type"] == "joke":
        return "generate_joke"
    else:
        return "generate_story"

# Build workflow
builder = StateGraph(RouterState)

# Add nodes
builder.add_node("router", router)
builder.add_node("generate_story", generate_story)
builder.add_node("generate_poem", generate_poem)
builder.add_node("generate_joke", generate_joke)

# Add edges
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_to_generator,
    {
        "generate_story": "generate_story",
        "generate_poem": "generate_poem",
        "generate_joke": "generate_joke"
    }
)
builder.add_edge("generate_story", END)
builder.add_edge("generate_poem", END)
builder.add_edge("generate_joke", END)

# Compile the workflow
compiled_graph = builder.compile()

# For local testing/development
if __name__ == "__main__":
    inputs = [
        {"input": "Tell me a story about a dragon"},
        {"input": "Write a poem about the ocean"},
        {"input": "Tell me a joke about programming"}
    ]
    
    for input_data in inputs:
        result = compiled_graph.invoke(input_data)
        print(f"Input: {input_data['input']}")
        print(f"Type: {result['input_type']}")
        print(f"Output: {result['output']}\n") 