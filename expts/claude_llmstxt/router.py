from typing import Literal, TypedDict, Annotated, Dict, List
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END

# Define state
class InputType(TypedDict):
    """Input classifier schema."""
    input_type: Literal["story", "poem", "joke"]
    rationale: str

class State(TypedDict):
    """State of the graph."""
    input: str
    classification: InputType
    response: str

# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Initialize type classifier with structured output
classifier_llm = llm.with_structured_output(InputType)

# Create nodes
def classify_input(state: State) -> Dict:
    """Classify the input as a story, poem, or joke request."""
    result = classifier_llm.invoke([
        SystemMessage(content="You are a classifier that determines if a user is asking for a story, poem, or joke."),
        HumanMessage(content=state["input"])
    ])
    return {"classification": result}

def write_story(state: State) -> Dict:
    """Generate a story based on the input."""
    response = llm.invoke([
        SystemMessage(content="You are a creative storyteller. Create an engaging, vivid story based on the user's request."),
        HumanMessage(content=state["input"])
    ])
    return {"response": response.content}

def write_poem(state: State) -> Dict:
    """Generate a poem based on the input."""
    response = llm.invoke([
        SystemMessage(content="You are a poet. Create a beautiful poem based on the user's request."),
        HumanMessage(content=state["input"])
    ])
    return {"response": response.content}

def write_joke(state: State) -> Dict:
    """Generate a joke based on the input."""
    response = llm.invoke([
        SystemMessage(content="You are a comedian. Create a hilarious joke based on the user's request."),
        HumanMessage(content=state["input"])
    ])
    return {"response": response.content}

# Define the router function
def router(state: State) -> Literal["write_story", "write_poem", "write_joke"]:
    """Route the input to the appropriate node based on classification."""
    input_type = state["classification"]["input_type"]
    if input_type == "story":
        return "write_story"
    elif input_type == "poem":
        return "write_poem"
    elif input_type == "joke":
        return "write_joke"
    else:
        # Default to story if somehow we get an unrecognized type
        return "write_story"

# Create and compile the graph
def create_graph():
    """Create and return the compiled graph."""
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("classify_input", classify_input)
    workflow.add_node("write_story", write_story)
    workflow.add_node("write_poem", write_poem)
    workflow.add_node("write_joke", write_joke)
    
    # Add edges
    workflow.add_edge(START, "classify_input")
    workflow.add_conditional_edges(
        "classify_input",
        router,
        {
            "write_story": "write_story",
            "write_poem": "write_poem",
            "write_joke": "write_joke"
        }
    )
    workflow.add_edge("write_story", END)
    workflow.add_edge("write_poem", END)
    workflow.add_edge("write_joke", END)
    
    # Compile the graph
    return workflow.compile()

# Function to run the graph
def run_graph(user_input: str):
    """Run the graph with the given input."""
    graph = create_graph()
    result = graph.invoke({"input": user_input})
    return result["response"]

# Export the compiled graph
compiled_graph = create_graph()