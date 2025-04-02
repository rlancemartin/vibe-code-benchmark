import os
from typing import TypedDict, Literal
from typing_extensions import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState


# Set up LLM
model_name = "claude-3-5-sonnet-latest"
llm = ChatAnthropic(model=model_name)

# Schema for structured output used in evaluation
class JokeEvaluation(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide detailed feedback on how to improve it.",
    )


# Augment the LLM with schema for structured output
evaluator_llm = llm.with_structured_output(JokeEvaluation)


# Graph state
class JokeState(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str
    iteration_count: int
    max_iterations: int


# Nodes
def generate_joke(state: JokeState):
    """LLM generates a joke based on topic and optional feedback"""
    iteration = state.get("iteration_count", 0)
    
    if state.get("feedback") and iteration > 0:
        system_message = """You are a professional comedian. Write a joke about the provided topic.
        Take into account the feedback to improve your joke.
        """
        msg = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"Topic: {state['topic']}\nFeedback: {state['feedback']}")
        ])
    else:
        system_message = """You are a professional comedian. Write a joke about the provided topic.
        Make it concise, clever, and funny.
        """
        msg = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"Topic: {state['topic']}")
        ])
    
    return {
        "joke": msg.content,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def evaluate_joke(state: JokeState):
    """LLM evaluates the joke's quality"""
    system_message = """You are a comedy critic with high standards.
    Evaluate the provided joke and determine if it's funny or not.
    Be honest but constructive in your feedback.
    """
    
    evaluation = evaluator_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Joke: {state['joke']}")
    ])
    
    return {
        "funny_or_not": evaluation.grade,
        "feedback": evaluation.feedback
    }


# Conditional edge function to route based on joke quality and iteration count
def route_joke(state: JokeState):
    """Route to end or back to generation based on evaluation and iteration count"""
    if state["funny_or_not"] == "funny":
        return "ACCEPTED"
    elif state["iteration_count"] >= state["max_iterations"]:
        return "MAX_ITERATIONS"
    else:
        return "REJECTED_WITH_FEEDBACK"


# Build the joke evaluator-optimizer workflow
workflow_builder = StateGraph(JokeState)

# Add nodes
workflow_builder.add_node("generate_joke", generate_joke)
workflow_builder.add_node("evaluate_joke", evaluate_joke)

# Add edges
workflow_builder.add_edge(START, "generate_joke")
workflow_builder.add_edge("generate_joke", "evaluate_joke")
workflow_builder.add_conditional_edges(
    "evaluate_joke",
    route_joke,
    {
        "ACCEPTED": END,
        "MAX_ITERATIONS": END,
        "REJECTED_WITH_FEEDBACK": "generate_joke",
    },
)

# Compile the workflow
compiled_graph = workflow_builder.compile()

# Function to run the workflow
def evaluate_and_optimize_joke(topic: str, max_iterations: int = 3):
    """Run the joke evaluator-optimizer workflow with a given topic"""
    initial_state = {
        "topic": topic,
        "max_iterations": max_iterations,
        "iteration_count": 0
    }
    
    final_state = compiled_graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Example usage
    result = evaluate_and_optimize_joke("artificial intelligence")
    print("Final joke:", result["joke"])
    print("Evaluation:", result["funny_or_not"])
    if result["funny_or_not"] == "not funny":
        print("Feedback:", result["feedback"])
    print(f"Took {result['iteration_count']} iterations") 