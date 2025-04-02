from typing import TypedDict, Literal, Dict
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# Define the state of our graph
class JokeState(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# Schema for structured output to use in joke evaluation
class JokeFeedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )

# Setup our Claude 3.5 Sonnet model
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(JokeFeedback)

# Define our nodes
def joke_generator(state: JokeState) -> Dict:
    """LLM generates a joke based on topic and optional feedback"""
    
    if state.get("feedback"):
        message = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        message = llm.invoke(f"Write a short, funny joke about {state['topic']}")
    
    return {"joke": message.content}

def joke_evaluator(state: JokeState) -> Dict:
    """LLM evaluates the joke for humor and provides feedback"""
    
    evaluation = evaluator.invoke(f"Grade the following joke for humor:\n{state['joke']}")
    return {"funny_or_not": evaluation.grade, "feedback": evaluation.feedback}

# Conditional edge function to route back to joke generator or end
def route_joke(state: JokeState) -> str:
    """Route back to joke generator or end based on feedback"""
    
    if state["funny_or_not"] == "funny":
        return "Accepted"
    else:
        return "Rejected + Feedback"

# Build our joke workflow
joke_workflow_builder = StateGraph(JokeState)

# Add nodes to the graph
joke_workflow_builder.add_node("joke_generator", joke_generator)
joke_workflow_builder.add_node("joke_evaluator", joke_evaluator)

# Connect the nodes with edges
joke_workflow_builder.add_edge(START, "joke_generator")
joke_workflow_builder.add_edge("joke_generator", "joke_evaluator")
joke_workflow_builder.add_conditional_edges(
    "joke_evaluator",
    route_joke,
    {
        "Accepted": END,
        "Rejected + Feedback": "joke_generator",
    },
)

# Compile the workflow
joke_workflow = joke_workflow_builder.compile()

# Standard interface for import
compiled_graph = joke_workflow

# This is what we'll run with langgraph dev
if __name__ == "__main__":
    # Example usage
    result = joke_workflow.invoke({"topic": "artificial intelligence"})
    print(f"Final joke: {result['joke']}")