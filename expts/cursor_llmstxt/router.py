import os
from typing import Annotated, Literal, TypedDict, Union
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from langgraph.graph import END, StateGraph
from langgraph.graph.state import inject

# Define the state structure with a TypedDict
class State(TypedDict):
    # Input will contain the original input
    input: str
    # Type will classify the input as "story", "poem", or "joke"
    input_type: Union[str, None]
    # Response will hold the generated content
    response: Union[str, None]

# Create a router function that determines the type of input
def router(state: State) -> Literal["story", "poem", "joke"]:
    """Route the input to the appropriate node based on its type."""
    # Simple classifier that categorizes input
    classifier = ChatAnthropic(model="claude-3-5-sonnet-latest")
    prompt = ChatPromptTemplate.from_template(
        """You are an expert classifier.
        Based on the following input, determine if it's requesting a STORY, POEM, or JOKE.
        Respond with exactly one word: 'story', 'poem', or 'joke'.
        
        Input: {input}
        """
    )
    chain = prompt | classifier | (lambda x: x.content.lower().strip())
    input_type = chain.invoke({"input": state["input"]})
    
    # Save the input type in state
    state["input_type"] = input_type
    return input_type

# Define the story node
def story_node(state: State) -> State:
    """Generate a story based on the input."""
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    prompt = ChatPromptTemplate.from_template(
        """You are a creative storyteller.
        Create an engaging short story based on the following input:
        
        Input: {input}
        
        Write a compelling story with characters, setting, and plot.
        """
    )
    chain = prompt | llm | (lambda x: x.content)
    response = chain.invoke({"input": state["input"]})
    return {"response": response}

# Define the poem node
def poem_node(state: State) -> State:
    """Generate a poem based on the input."""
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    prompt = ChatPromptTemplate.from_template(
        """You are a talented poet.
        Create a beautiful poem based on the following input:
        
        Input: {input}
        
        Write a poem with meaningful imagery and thoughtful structure.
        """
    )
    chain = prompt | llm | (lambda x: x.content)
    response = chain.invoke({"input": state["input"]})
    return {"response": response}

# Define the joke node
def joke_node(state: State) -> State:
    """Generate a joke based on the input."""
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    prompt = ChatPromptTemplate.from_template(
        """You are a hilarious comedian.
        Create a funny joke based on the following input:
        
        Input: {input}
        
        Write a joke that will make people laugh.
        """
    )
    chain = prompt | llm | (lambda x: x.content)
    response = chain.invoke({"input": state["input"]})
    return {"response": response}

# Build the graph with conditional routing
def build_graph() -> Runnable:
    """Build and compile the graph."""
    # Create a new graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("router", router)
    graph.add_node("story", story_node)
    graph.add_node("poem", poem_node)
    graph.add_node("joke", joke_node)
    
    # Define edges
    graph.add_conditional_edges(
        "router",
        lambda state: state["input_type"],
        {
            "story": "story",
            "poem": "poem",
            "joke": "joke",
        },
    )
    
    # Connect output nodes to END
    graph.add_edge("story", END)
    graph.add_edge("poem", END)
    graph.add_edge("joke", END)
    
    # Set the entry point
    graph.set_entry_point("router")
    
    # Compile the graph
    return graph.compile()

# Create the compiled graph as 'app' for the langgraph.json entrypoint
compiled_graph = build_graph()

# If this file is run directly, compile and save the graph
if __name__ == "__main__":
    # Print a success message
    print("Graph compiled successfully!") 