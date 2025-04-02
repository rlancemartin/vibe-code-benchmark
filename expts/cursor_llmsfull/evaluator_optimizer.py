from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# Define state with a messages field using the add_messages reducer
class JokeEvalState(TypedDict):
    messages: Annotated[list, add_messages]
    joke: str
    evaluation: str
    quality_score: int
    improved_joke: str

# Initialize Claude 3.5 Sonnet
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Node 1: Evaluate joke quality
def evaluate_joke(state: JokeEvalState) -> JokeEvalState:
    """Evaluate the quality of a joke."""
    # Get the joke from the state
    if "joke" not in state:
        # If no joke in state, extract from the message
        latest_message = state["messages"][-1]
        joke = latest_message.content
    else:
        joke = state["joke"]
    
    # Create a prompt for joke evaluation
    evaluation_prompt = f"""
    Evaluate the following joke:
    
    "{joke}"
    
    Provide a brief critique of the joke's humor, cleverness, and delivery.
    Then rate the joke on a scale from 1 to 10 (where 10 is the funniest).
    
    Format your response as follows:
    Evaluation: [your critique]
    Score: [number 1-10]
    """
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=evaluation_prompt)])
    
    # Parse the evaluation and score
    evaluation_text = response.content
    
    # Extract the score (assuming the format is followed)
    try:
        score_line = [line for line in evaluation_text.split('\n') if line.strip().startswith('Score:')][0]
        score_str = score_line.split(':')[1].strip()
        quality_score = int(score_str.split('/')[0] if '/' in score_str else score_str)
    except (IndexError, ValueError):
        # Default to middle score if parsing fails
        quality_score = 5
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=f"I've evaluated the joke: {evaluation_text}")],
        "joke": joke,
        "evaluation": evaluation_text,
        "quality_score": quality_score
    }

# Node decision function to determine if joke needs improvement
def should_improve(state: JokeEvalState) -> Literal["improve_joke", "end"]:
    """Decide whether to improve the joke based on quality score."""
    if state["quality_score"] < 7:  # Threshold for improvement
        return "improve_joke"
    else:
        return "end"

# Node 2: Improve the joke
def improve_joke(state: JokeEvalState) -> JokeEvalState:
    """Improve a low-quality joke."""
    # Get the original joke and evaluation
    original_joke = state["joke"]
    evaluation = state["evaluation"]
    
    # Create a prompt for joke improvement
    improvement_prompt = f"""
    Here's a joke: "{original_joke}"
    
    Evaluation: {evaluation}
    
    Based on this evaluation, please improve the joke to make it funnier, more clever, and better delivered.
    Focus on addressing the specific weaknesses mentioned in the evaluation.
    """
    
    # Get response from the LLM
    response = model.invoke([HumanMessage(content=improvement_prompt)])
    
    # Extract the improved joke
    improved_joke = response.content
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=f"I've improved the joke: {improved_joke}")],
        "improved_joke": improved_joke
    }

# Final node to summarize the process
def summarize_result(state: JokeEvalState) -> JokeEvalState:
    """Summarize the joke evaluation and improvement process."""
    # Prepare the summary based on whether improvement was done
    if "improved_joke" in state:
        summary = f"""
        Original joke: "{state['joke']}"
        
        Evaluation (Score: {state['quality_score']}/10):
        {state['evaluation']}
        
        Improved version:
        {state['improved_joke']}
        """
    else:
        summary = f"""
        Joke: "{state['joke']}"
        
        Evaluation (Score: {state['quality_score']}/10):
        {state['evaluation']}
        
        This joke was already good quality (score ≥ 7) and didn't need improvement.
        """
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=summary)]
    }

# Build the graph
def build_joke_eval_graph():
    # Create the graph
    builder = StateGraph(JokeEvalState)
    
    # Add nodes
    builder.add_node("evaluate_joke", evaluate_joke)
    builder.add_node("improve_joke", improve_joke)
    builder.add_node("summarize", summarize_result)
    
    # Add conditional edges
    builder.add_edge(START, "evaluate_joke")
    builder.add_conditional_edges(
        "evaluate_joke",
        should_improve,
        {
            "improve_joke": "improve_joke",
            "end": "summarize"
        }
    )
    builder.add_edge("improve_joke", "summarize")
    builder.add_edge("summarize", END)
    
    # Compile the graph
    return builder.compile()

# Create the graph
compiled_graph = build_joke_eval_graph()

# For testing the graph
if __name__ == "__main__":
    # Test with a sample joke
    result = compiled_graph.invoke({
        "messages": [HumanMessage(content="Why did the programmer quit his job? Because he didn't get arrays.")]
    })
    
    # Print the final message
    print(result["messages"][-1].content) 