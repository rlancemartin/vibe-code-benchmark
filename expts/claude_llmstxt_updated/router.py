from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END


# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Schema for structured output to use as routing logic
class Route(BaseModel):
    content_type: Literal["poem", "story", "joke"] = Field(
        description="The type of content to generate based on the user's request"
    )


# Augment the LLM with schema for structured output
router_llm = llm.with_structured_output(Route)


# Define our state
class WorkflowState(TypedDict):
    input: str
    content_type: str
    output: str


# Define our node functions
def classify_input(state: WorkflowState):
    """Router that classifies the input and decides whether to generate a story, poem, or joke"""
    
    # Use structured output to determine content type
    decision = router_llm.invoke(
        [
            SystemMessage(
                content="Analyze the user request and determine if they want a story, poem, or joke. "
                "Choose 'story' if they want a narrative, 'poem' if they want poetry, or "
                "'joke' if they want something funny."
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    
    return {"content_type": decision.content_type}


def generate_story(state: WorkflowState):
    """Generate a short story based on the input"""
    
    response = llm.invoke(
        [
            SystemMessage(content="You are a creative storyteller. Create an engaging short story."),
            HumanMessage(content=f"Write a short story about: {state['input']}"),
        ]
    )
    
    return {"output": response.content}


def generate_poem(state: WorkflowState):
    """Generate a poem based on the input"""
    
    response = llm.invoke(
        [
            SystemMessage(content="You are a talented poet. Create a beautiful poem."),
            HumanMessage(content=f"Write a poem about: {state['input']}"),
        ]
    )
    
    return {"output": response.content}


def generate_joke(state: WorkflowState):
    """Generate a joke based on the input"""
    
    response = llm.invoke(
        [
            SystemMessage(content="You are a comedy writer. Create a funny joke."),
            HumanMessage(content=f"Write a joke about: {state['input']}"),
        ]
    )
    
    return {"output": response.content}


# Function to determine routing
def route_content(state: WorkflowState):
    """Route to the appropriate content generator based on classification"""
    
    if state["content_type"] == "story":
        return "story_generator"
    elif state["content_type"] == "poem":
        return "poem_generator"
    elif state["content_type"] == "joke":
        return "joke_generator"
    else:
        # Default to story if no match (shouldn't happen with structured output)
        return "story_generator"


# Build our workflow graph
def build_graph():
    # Create our graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("router", classify_input)
    workflow.add_node("story_generator", generate_story)
    workflow.add_node("poem_generator", generate_poem)
    workflow.add_node("joke_generator", generate_joke)
    
    # Add edges to connect nodes
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_content,
        {
            "story_generator": "story_generator",
            "poem_generator": "poem_generator",
            "joke_generator": "joke_generator",
        }
    )
    workflow.add_edge("story_generator", END)
    workflow.add_edge("poem_generator", END)
    workflow.add_edge("joke_generator", END)
    
    # Compile the graph
    return workflow.compile()

# Create and compile the graph
compiled_graph = build_graph()

# Example use
if __name__ == "__main__":
    # Test with different inputs
    inputs = [
        "Tell me a story about a brave astronaut",
        "Write a poem about autumn leaves",
        "Make me laugh with a joke about programming"
    ]
    
    for input_text in inputs:
        result = compiled_graph.invoke({"input": input_text})
        print(f"Input: {input_text}")
        print(f"Content Type: {result['content_type']}")
        print(f"Output:\n{result['output']}")
        print("\n" + "="*50 + "\n")