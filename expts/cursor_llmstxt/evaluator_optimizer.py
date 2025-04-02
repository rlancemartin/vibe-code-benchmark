from typing import Annotated, List, TypedDict
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Define our state using Pydantic for validation
class JokeState(BaseModel):
    """State for the joke evaluation and optimization workflow."""
    joke: str = Field(description="The joke to be evaluated")
    evaluation: str = Field(default="", description="The evaluation of the joke quality")
    score: float = Field(default=0.0, description="The numerical score of the joke (0.0-10.0)")
    improved_joke: str = Field(default="", description="The improved version of the joke if score is low")
    messages: List[BaseMessage] = Field(default_factory=list, description="The conversation history")


# Node 1: Evaluate the joke
def evaluate_joke(state: JokeState):
    """Evaluate the quality and humor of a joke."""
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.2)
    
    prompt = HumanMessage(content=f"""You are a professional comedy evaluator with years of experience in stand-up comedy.
    
I need you to carefully evaluate the following joke and provide:
1. A thoughtful critique of the joke's quality, structure, punchline, and overall humor
2. A score from 0 to 10, where 0 is completely unfunny and 10 is exceptionally hilarious

The joke to evaluate is:
\"{state.joke}\"

First provide your critique, then on a new line write "Score:" followed by a single number between 0 and 10, with one decimal place of precision (e.g. 7.5).
""")
    
    # Get the evaluation from the LLM
    response = llm.invoke([prompt])
    
    # Parse the response to extract the score
    evaluation_text = response.content
    
    # Find the score in the evaluation text
    import re
    score_match = re.search(r"Score:\s*(\d+\.?\d*)", evaluation_text)
    if score_match:
        score = float(score_match.group(1))
    else:
        # Default to 5.0 if no score is found
        score = 5.0
    
    return {
        "evaluation": evaluation_text,
        "score": score,
        "messages": [prompt, response]
    }


# Node 2: Improve the joke if score is below threshold
def improve_joke(state: JokeState):
    """Improve jokes that received a low score."""
    # If the joke scored well (7.0 or higher), no improvement needed
    if state.score >= 7.0:
        return {
            "improved_joke": "",
            "messages": state.messages + [HumanMessage(content="The joke is good enough and doesn't need improvement.")]
        }
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.7)
    
    prompt = HumanMessage(content=f"""You are a professional comedy writer with years of experience in stand-up comedy.

I have a joke that was evaluated and received a score of {state.score}/10. Here's the joke:
\"{state.joke}\"

Here's the evaluation of the joke:
{state.evaluation}

Please rewrite and improve this joke based on the critique. Your goal is to make it funnier while addressing the weaknesses identified in the evaluation. Maintain the general subject matter but feel free to restructure the joke entirely if needed.

Only respond with the improved joke text - no explanations or other text.
""")
    
    # Get the improved joke from the LLM
    response = llm.invoke([prompt])
    
    return {
        "improved_joke": response.content.strip(),
        "messages": state.messages + [prompt, response]
    }


# Conditional router to determine if joke needs improvement
def should_improve(state: JokeState):
    """Determine if the joke needs improvement based on the score."""
    if state.score < 7.0:
        return "improve_joke"
    else:
        return END


# Build the graph
def build_graph():
    """Build and compile the workflow graph."""
    workflow = StateGraph(JokeState)
    
    # Add nodes
    workflow.add_node("evaluate_joke", evaluate_joke)
    workflow.add_node("improve_joke", improve_joke)
    
    # Add edges
    workflow.add_edge(START, "evaluate_joke")
    workflow.add_conditional_edges(
        "evaluate_joke",
        should_improve,
        {
            "improve_joke": "improve_joke",
            END: END
        }
    )
    workflow.add_edge("improve_joke", END)
    
    # Compile the graph
    return workflow.compile()

compiled_graph = build_graph()

if __name__ == "__main__":
    # Example joke to test the graph
    test_joke = "Why did the chicken cross the road? To get to the other side."
    
    # Build and run the graph
    graph = build_graph()
    result = graph.invoke({"joke": test_joke})
    
    # Print results
    print(f"Original Joke: {result['joke']}")
    print(f"Evaluation Score: {result['score']}/10")
    if result['improved_joke']:
        print(f"Improved Joke: {result['improved_joke']}")
    else:
        print("No improvement needed.") 