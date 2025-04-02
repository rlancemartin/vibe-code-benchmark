from typing import Annotated, TypedDict, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]
    joke: str
    evaluation: Dict[str, Any]
    improved_joke: str
    quality: Literal["high", "low"]

# Joke evaluator function
def evaluate_joke(state: State) -> Dict[str, Any]:
    """Evaluates the quality and humor of a joke"""
    joke = state["joke"]
    
    # Create evaluation prompt
    evaluation_prompt = [
        SystemMessage(content="""You are an expert comedy critic. Your task is to evaluate the quality of jokes.
        Rate the joke on a scale of 1-10 for:
        - Originality (1-10)
        - Cleverness (1-10)
        - Humor (1-10)
        - Overall Score (1-10)
        
        Also provide a brief explanation of why you gave these scores.
        
        Format your response as a JSON object with these keys:
        {
            "originality": <score>,
            "cleverness": <score>,
            "humor": <score>,
            "overall": <score>,
            "explanation": "<explanation>"
        }
        """),
        HumanMessage(content=f"Evaluate this joke:\n\n{joke}")
    ]
    
    # Get evaluation
    response = llm.invoke(evaluation_prompt)
    
    # Extract the evaluation
    # Find the JSON part in the response
    try:
        # Try to find JSON-like content between curly braces
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group(0))
        else:
            # Fallback if no JSON found
            evaluation = {
                "originality": 5,
                "cleverness": 5,
                "humor": 5,
                "overall": 5,
                "explanation": "Could not parse evaluation"
            }
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        evaluation = {
            "originality": 5,
            "cleverness": 5,
            "humor": 5,
            "overall": 5,
            "explanation": "Could not parse evaluation"
        }
    
    # Determine if joke quality is high or low
    quality = "high" if evaluation["overall"] >= 7 else "low"
    
    return {
        "evaluation": evaluation,
        "quality": quality,
        "messages": state["messages"] + [response]
    }

# Joke improver function
def improve_joke(state: State) -> Dict[str, Any]:
    """Improves a joke based on evaluation"""
    joke = state["joke"]
    evaluation = state["evaluation"]
    
    # Create improvement prompt
    improvement_prompt = [
        SystemMessage(content="""You are a professional comedy writer. Your task is to improve jokes to make them funnier, 
        more original, and more clever. Use the evaluation feedback to guide your improvements."""),
        HumanMessage(content=f"""Here's a joke that needs improvement:

{joke}

Here's an evaluation of the joke:
Originality: {evaluation["originality"]}/10
Cleverness: {evaluation["cleverness"]}/10
Humor: {evaluation["humor"]}/10
Overall: {evaluation["overall"]}/10

Feedback: {evaluation["explanation"]}

Please improve this joke to make it funnier and more effective. Focus on addressing the weaknesses identified in the evaluation.""")
    ]
    
    # Generate improved joke
    response = llm.invoke(improvement_prompt)
    
    return {
        "improved_joke": response.content,
        "messages": state["messages"] + [response]
    }

# Router function to decide whether to improve joke
def quality_router(state: State) -> str:
    """Routes based on joke quality"""
    quality = state.get("quality", "low")
    
    if quality == "low":
        return "improve_joke"
    else:
        return "end"

# Build the graph
def build_graph():
    # Create a new graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("evaluate_joke", evaluate_joke)
    graph_builder.add_node("improve_joke", improve_joke)
    
    # Add edges
    graph_builder.add_edge(START, "evaluate_joke")
    
    # Add conditional routing based on joke quality
    graph_builder.add_conditional_edges(
        "evaluate_joke",
        quality_router,
        {
            "improve_joke": "improve_joke",
            "end": END
        }
    )
    
    # Connect joke improver to END
    graph_builder.add_edge("improve_joke", END)
    
    # Compile the graph
    return graph_builder.compile()

# Create the compiled graph for imports
compiled_graph = build_graph()

if __name__ == "__main__":
    # Use the compiled graph
    graph = compiled_graph
    
    # Create the langgraph.json config file
    config = {
        "title": "Joke Evaluator and Optimizer Workflow",
        "description": "A LangGraph workflow that evaluates joke quality and improves low-quality jokes",
        "schema": {
            "input": {
                "type": "object",
                "properties": {
                    "joke": {
                        "type": "string",
                        "description": "The joke to evaluate and potentially improve"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "joke": {
                        "type": "string",
                        "description": "The original joke"
                    },
                    "evaluation": {
                        "type": "object",
                        "description": "The evaluation scores and feedback"
                    },
                    "quality": {
                        "type": "string",
                        "description": "Whether the joke was deemed high or low quality"
                    },
                    "improved_joke": {
                        "type": "string",
                        "description": "The improved version of the joke (if low quality)"
                    }
                }
            }
        },
        "config": {
            "modules": ["evaluator-optimizer"],
            "graph": "build_graph"
        }
    }
    
    # Save config file
    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Graph compiled and langgraph.json configuration created successfully")