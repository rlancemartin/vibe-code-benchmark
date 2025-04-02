"""
Evaluator-Optimizer Joke Workflow

This script implements a LangGraph workflow that:
1. Uses an LLM to generate a joke
2. Evaluates the quality of the joke with another LLM
3. If the joke isn't funny, provides feedback to improve it
4. Repeats the process until a high-quality joke is produced

The workflow is compiled and saved to a langgraph.json file for use with langgraph dev.
"""

from typing import Annotated, Literal, TypedDict, List, Dict, Any
from typing_extensions import TypedDict

import os
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph import persist


# Define our state
class State(TypedDict):
    # The original joke topic
    topic: str
    # The current joke text
    joke: str
    # Feedback on the joke
    feedback: str
    # Evaluation result: funny or not
    evaluation: str
    # Number of attempts made to improve the joke
    attempts: int


# Schema for structured output to use in evaluation
class JokeEvaluation(BaseModel):
    is_funny: Literal["yes", "no"] = Field(
        description="Is the joke funny or not?",
    )
    rating: int = Field(
        description="Rating from 1-10, where 10 is extremely funny",
    )
    feedback: str = Field(
        description="If the joke is not funny or could be improved, provide specific feedback on how to make it better.",
    )


# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Create LLM with structured output capability for evaluation
evaluator_llm = llm.with_structured_output(JokeEvaluation)


def joke_generator(state: State) -> Dict[str, Any]:
    """Generate a joke based on the given topic and optional feedback."""
    
    # If we have feedback, use it to improve the joke
    if state.get("feedback") and state.get("attempts", 0) > 0:
        prompt = f"""Create a joke about {state['topic']}. 

Previous joke: {state['joke']}

Feedback to improve the joke: {state['feedback']}

Make significant improvements based on the feedback, not just minor edits.
"""
    else:
        # First attempt, no feedback yet
        prompt = f"Create a joke about {state['topic']}. Make it clever and funny."
    
    # Generate the joke
    response = llm.invoke(prompt)
    
    # Update the state
    return {"joke": response.content, "attempts": state.get("attempts", 0) + 1}


def joke_evaluator(state: State) -> Dict[str, Any]:
    """Evaluate the joke and provide feedback."""
    
    prompt = f"""Evaluate this joke about {state['topic']}:

{state['joke']}

Be honest but constructive in your evaluation.
"""
    
    # Evaluate the joke
    evaluation = evaluator_llm.invoke(prompt)
    
    # Determine if we should continue or end
    is_funny = "funny" if evaluation.is_funny == "yes" and evaluation.rating >= 7 else "not_funny"
    
    return {
        "evaluation": is_funny,
        "feedback": evaluation.feedback
    }


def should_continue(state: State) -> Literal["continue", "end"]:
    """Determine if we should generate another joke or end."""
    
    # If joke is funny or we've made too many attempts, end
    if state["evaluation"] == "funny" or state.get("attempts", 0) >= 3:
        return "end"
    
    # Otherwise, continue improving
    return "continue"


# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_joke", joke_generator)
workflow.add_node("evaluate_joke", joke_evaluator)

# Add edges
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", "evaluate_joke")

# Add conditional edges based on evaluation
workflow.add_conditional_edges(
    "evaluate_joke",
    should_continue,
    {
        "continue": "generate_joke",
        "end": END
    }
)

# Compile the graph
app = workflow.compile()

# Save the compiled graph for use with langgraph dev
if __name__ == "__main__":
    # Create app config for langgraph dev
    config = persist.create_langgraph_config(
        app,
        project_name="joke-evaluator-optimizer",
        app_id="joke-evaluator-optimizer",
    )
    
    # Write to langgraph.json
    with open("langgraph.json", "w") as f:
        f.write(config.json(exclude_none=True))
    
    print("LangGraph workflow compiled and saved to langgraph.json")
    print("You can now run 'langgraph dev' to start the development server")

    # Sample run to demonstrate the workflow
    result = app.invoke({"topic": "artificial intelligence", "attempts": 0})
    
    print("\n--- Sample Joke Generation and Evaluation ---")
    print(f"Topic: {result['topic']}")
    print(f"Final joke: {result['joke']}")
    print(f"Funny? {result['evaluation']}")
    print(f"Attempts: {result['attempts']}")
    if result['evaluation'] == 'not_funny':
        print(f"Feedback: {result['feedback']}")