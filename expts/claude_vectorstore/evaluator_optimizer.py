from typing import TypedDict, Literal, Dict, Optional
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# Define the state of our workflow graph
class JokeState(TypedDict):
    joke: str
    topic: Optional[str]
    feedback: Optional[str]
    quality_grade: Optional[str]
    iterations: Optional[int]

# Schema for structured output to use in joke evaluation
class JokeEvaluation(BaseModel):
    grade: Literal["high_quality", "low_quality"] = Field(
        description="Grade for the joke quality. 'high_quality' if the joke is funny and well-crafted, 'low_quality' if it needs improvement.",
    )
    feedback: str = Field(
        description="Specific, actionable feedback on how to improve the joke if it's low quality.",
    )
    strengths: str = Field(
        description="The strengths of the joke, what works well.",
    )
    weaknesses: str = Field(
        description="The weaknesses of the joke, what could be improved.",
    )

# Setup our Claude 3.5 Sonnet model
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(JokeEvaluation)

# Define our nodes
def joke_generator(state: JokeState) -> Dict:
    """Generate an initial joke or improve an existing joke based on feedback"""
    
    # Initialize iterations counter if not present
    iterations = state.get("iterations", 0)
    
    if iterations > 0 and state.get("feedback"):
        # Improve existing joke based on feedback
        prompt = f"""
        Create an improved version of this joke based on the feedback provided:
        
        Original Joke: {state['joke']}
        
        Feedback: {state['feedback']}
        
        Strengths to maintain: {state.get('strengths', 'No specific strengths mentioned')}
        
        Weaknesses to address: {state.get('weaknesses', 'No specific weaknesses mentioned')}
        
        Generate a new, improved version of the joke that addresses the feedback.
        """
    else:
        # Generate new joke based on topic
        topic = state.get("topic", "everyday life")
        prompt = f"""
        Create a funny, original joke about {topic}.
        
        The joke should be:
        - Clever and well-crafted
        - Appropriate for general audiences
        - Have a clear setup and punchline
        
        Just write the joke itself, with no additional explanation.
        """
    
    # Call the LLM to generate or improve the joke
    response = llm.invoke(prompt)
    
    # Return the joke and increment the iterations counter
    return {"joke": response.content, "iterations": iterations + 1}

def joke_evaluator(state: JokeState) -> Dict:
    """Evaluate the quality of the joke with detailed feedback"""
    
    # Prepare the evaluation prompt
    prompt = f"""
    Evaluate the following joke for quality, humor, and craftsmanship:
    
    {state['joke']}
    
    Provide a detailed evaluation including the joke's strengths and weaknesses.
    For low quality jokes, provide specific, actionable feedback for improvement.
    """
    
    # Call the augmented LLM with structured output for evaluation
    evaluation = evaluator.invoke(prompt)
    
    # Return the evaluation results
    return {
        "quality_grade": evaluation.grade,
        "feedback": evaluation.feedback,
        "strengths": evaluation.strengths,
        "weaknesses": evaluation.weaknesses
    }

# Conditional edge function for routing based on joke quality
def route_based_on_quality(state: JokeState) -> str:
    """Determine next steps based on joke quality and iteration count"""
    
    # Get the current evaluation and iteration count
    quality = state.get("quality_grade", "low_quality")
    iterations = state.get("iterations", 0)
    
    # Maximum number of improvement iterations
    max_iterations = 3
    
    # If joke is high quality or we've reached max iterations, end the workflow
    if quality == "high_quality" or iterations >= max_iterations:
        return "END"
    else:
        # Otherwise, route back to the generator for improvement
        return "IMPROVE"

# Build our evaluator-optimizer workflow
workflow_builder = StateGraph(JokeState)

# Add nodes to the graph
workflow_builder.add_node("joke_generator", joke_generator)
workflow_builder.add_node("joke_evaluator", joke_evaluator)

# Connect the nodes with edges
workflow_builder.add_edge(START, "joke_generator")
workflow_builder.add_edge("joke_generator", "joke_evaluator")
workflow_builder.add_conditional_edges(
    "joke_evaluator",
    route_based_on_quality,
    {
        "END": END,
        "IMPROVE": "joke_generator",
    },
)

# Compile the workflow
compiled_graph = workflow_builder.compile()

# This is what we'll run with langgraph dev
if __name__ == "__main__":
    # Example usage with a topic
    result = compiled_graph.invoke({"topic": "artificial intelligence"})
    
    # Print the final result
    print(f"Final joke after {result['iterations']} iterations:")
    print(f"\n{result['joke']}\n")
    
    print(f"Quality Grade: {result['quality_grade']}")
    print(f"Strengths: {result['strengths']}")
    
    if result['iterations'] > 1:
        print(f"\nImprovement Process:")
        print(f"Iterations: {result['iterations']}")
        print(f"Final feedback: {result['feedback']}")