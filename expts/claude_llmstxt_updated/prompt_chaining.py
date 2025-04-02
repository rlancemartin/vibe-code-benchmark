from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END


# Define our state
class JokeState(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Define our node functions
def generate_joke(state: JokeState):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def check_joke_quality(state: JokeState):
    """Gate function to check if the joke is good enough or needs improvement"""
    # Simple check - if joke is too short, it might need improvement
    if len(state["joke"]) < 50:
        return "Needs Improvement"
    return "Good Enough"


def improve_joke(state: JokeState):
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def finalize_joke(state: JokeState):
    """Third LLM call for final polish"""
    if "improved_joke" in state:
        joke_to_improve = state["improved_joke"]
    else:
        joke_to_improve = state["joke"]
        
    msg = llm.invoke(f"Add a surprising twist to this joke: {joke_to_improve}")
    return {"final_joke": msg.content}


# Build our workflow graph
def build_graph():
    # Create our graph
    workflow = StateGraph(JokeState)
    
    # Add nodes
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("improve_joke", improve_joke)
    workflow.add_node("finalize_joke", finalize_joke)
    
    # Add edges to connect nodes
    workflow.add_edge(START, "generate_joke")
    workflow.add_conditional_edges(
        "generate_joke", 
        check_joke_quality, 
        {
            "Needs Improvement": "improve_joke",
            "Good Enough": "finalize_joke"
        }
    )
    workflow.add_edge("improve_joke", "finalize_joke")
    workflow.add_edge("finalize_joke", END)
    
    # Compile the graph
    return workflow.compile()


# Create and compile the graph
joke_chain = build_graph()

# Standard interface for import
compiled_graph = joke_chain


# Example use
if __name__ == "__main__":
    result = joke_chain.invoke({"topic": "programming"})
    print("Final joke:")
    print(result["final_joke"])