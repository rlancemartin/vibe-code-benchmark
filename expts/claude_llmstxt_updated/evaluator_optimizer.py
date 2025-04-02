from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END


# Initialize the LLM with claude-3-5-sonnet-latest
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Schema for structured output to use for joke evaluation
class JokeEvaluation(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Evaluate if the joke is funny or not."
    )
    reasoning: str = Field(
        description="Detailed reasoning for your evaluation."
    )
    feedback: str = Field(
        description="Specific feedback on how to improve the joke if it's not funny."
    )


# Augment the LLM with schema for structured joke evaluation
evaluator_llm = llm.with_structured_output(JokeEvaluation)


# Define our state
class JokeState(TypedDict):
    topic: str
    joke: str
    grade: str
    feedback: str
    iteration: int
    final_joke: str


# Define node functions
def generate_joke(state: JokeState):
    """Generate a joke about the given topic"""
    
    # First iteration or when feedback exists
    if state.get("iteration", 0) == 0:
        prompt = f"Write a joke about {state['topic']}."
    else:
        prompt = f"Improve this joke about {state['topic']} based on the following feedback: {state['feedback']}\n\nCurrent joke: {state['joke']}"
    
    response = llm.invoke([
        SystemMessage(content="You are a comedy writer. Create a joke based on the given topic."),
        HumanMessage(content=prompt)
    ])
    
    # Update the iteration counter
    iteration = state.get("iteration", 0) + 1
    
    return {
        "joke": response.content,
        "iteration": iteration
    }


def evaluate_joke(state: JokeState):
    """Evaluate the joke's quality and provide feedback"""
    
    evaluation = evaluator_llm.invoke([
        SystemMessage(content="You are a comedy critic. Evaluate the following joke and provide detailed feedback."),
        HumanMessage(content=f"Evaluate this joke: {state['joke']}")
    ])
    
    return {
        "grade": evaluation.grade,
        "feedback": evaluation.feedback
    }


def finalize_joke(state: JokeState):
    """Finalize the joke once it's been deemed funny"""
    
    return {
        "final_joke": state["joke"]
    }


# Function to determine routing based on joke evaluation
def should_continue(state: JokeState):
    """Determine if the joke needs improvement or if it's good enough"""
    
    # Limit iterations to prevent infinite loops (max 3 attempts)
    if state["iteration"] >= 3:
        return "END"
    
    # Route based on the grade
    if state["grade"] == "funny":
        return "GOOD"
    else:
        return "NEEDS_IMPROVEMENT"


# Build the graph
def build_graph():
    # Create the graph with our state
    workflow = StateGraph(JokeState)
    
    # Add nodes
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("evaluate_joke", evaluate_joke)
    workflow.add_node("finalize_joke", finalize_joke)
    
    # Add edges
    workflow.add_edge(START, "generate_joke")
    workflow.add_edge("generate_joke", "evaluate_joke")
    
    # Add conditional edges for the evaluation results
    workflow.add_conditional_edges(
        "evaluate_joke",
        should_continue,
        {
            "GOOD": "finalize_joke", 
            "NEEDS_IMPROVEMENT": "generate_joke",
            "END": "finalize_joke"  # Force end after max iterations
        }
    )
    
    workflow.add_edge("finalize_joke", END)
    
    # Compile the graph
    return workflow.compile()


# Create the compiled graph
joke_evaluator_optimizer = build_graph()

# Standard interface for import
compiled_graph = joke_evaluator_optimizer


# Example usage
if __name__ == "__main__":
    # Test with a topic
    result = joke_evaluator_optimizer.invoke({"topic": "programming"})
    
    print(f"Final joke (after {result['iteration']} iterations):")
    print(result["final_joke"])
    print("\nGrade:", result["grade"])
    
    if result["grade"] == "not funny" and result["iteration"] >= 3:
        print("\nMax iterations reached, but the joke still needs improvement.")
        print("Feedback:", result["feedback"])