from typing import Annotated, Literal, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

# Define content types
ContentType = Literal["story", "poem", "joke", "unknown"]

# Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]
    content_type: ContentType
    response: str

# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Content classifier function
def classify_content(state: State) -> Dict[str, Any]:
    """Determines if the input is a story, poem, or joke"""
    messages = state["messages"]
    if not messages:
        return {"content_type": "unknown"}
    
    # Get the last user message
    last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
    
    # Create a classification prompt
    classification_prompt = [
        SystemMessage(content="You are a content classifier. Your task is to determine if the user's input is asking for a story, a poem, or a joke. Respond with exactly one word: 'story', 'poem', or 'joke'."),
        HumanMessage(content=f"Classify this as either a story, poem, or joke request:\n\n{last_message}")
    ]
    
    # Get classification
    response = llm.invoke(classification_prompt)
    content = response.content.lower().strip()
    
    # Extract the classification 
    if "story" in content:
        content_type = "story"
    elif "poem" in content:
        content_type = "poem"
    elif "joke" in content:
        content_type = "joke"
    else:
        content_type = "unknown"
        
    return {"content_type": content_type}

# Handler functions for each content type
def story_handler(state: State) -> Dict[str, Any]:
    """Generates or enhances a story"""
    messages = state["messages"]
    
    # Create story-specific prompt
    prompt = [
        SystemMessage(content="You are a master storyteller. Create a vivid, engaging short story with interesting characters and a satisfying plot."),
        *messages
    ]
    
    # Generate story
    response = llm.invoke(prompt)
    
    return {
        "response": response.content,
        "messages": state["messages"] + [response]
    }

def poem_handler(state: State) -> Dict[str, Any]:
    """Generates or enhances a poem"""
    messages = state["messages"]
    
    # Create poem-specific prompt
    prompt = [
        SystemMessage(content="You are a talented poet. Create a beautiful poem with vivid imagery, rhythm, and emotional resonance."),
        *messages
    ]
    
    # Generate poem
    response = llm.invoke(prompt)
    
    return {
        "response": response.content,
        "messages": state["messages"] + [response]
    }

def joke_handler(state: State) -> Dict[str, Any]:
    """Generates or enhances a joke"""
    messages = state["messages"]
    
    # Create joke-specific prompt
    prompt = [
        SystemMessage(content="You are a hilarious comedian. Create a clever joke with perfect timing and a surprising punchline."),
        *messages
    ]
    
    # Generate joke
    response = llm.invoke(prompt)
    
    return {
        "response": response.content,
        "messages": state["messages"] + [response]
    }

# Router function to direct flow based on content type
def router(state: State) -> str:
    """Routes to the appropriate handler based on content type"""
    content_type = state.get("content_type", "unknown")
    
    if content_type == "story":
        return "story_handler"
    elif content_type == "poem":
        return "poem_handler"
    elif content_type == "joke":
        return "joke_handler"
    else:
        # Default to story handler if unknown
        return "story_handler"

# Build the graph
def build_graph():
    # Create a new graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("classifier", classify_content)
    graph_builder.add_node("story_handler", story_handler)
    graph_builder.add_node("poem_handler", poem_handler)
    graph_builder.add_node("joke_handler", joke_handler)
    
    # Add edges
    graph_builder.add_edge(START, "classifier")
    
    # Add conditional routing
    graph_builder.add_conditional_edges(
        "classifier",
        router,
        {
            "story_handler": "story_handler",
            "poem_handler": "poem_handler",
            "joke_handler": "joke_handler"
        }
    )
    
    # Connect all handlers to END
    graph_builder.add_edge("story_handler", END)
    graph_builder.add_edge("poem_handler", END)
    graph_builder.add_edge("joke_handler", END)
    
    # Compile the graph
    return graph_builder.compile()

# Create the compiled graph for imports
compiled_graph = build_graph()

if __name__ == "__main__":
    # Use the compiled graph
    graph = compiled_graph
    
    # Create the langgraph.json config file
    config = {
        "title": "Content Router Workflow",
        "description": "A LangGraph workflow that routes content to appropriate handlers based on type (story, poem, or joke)",
        "schema": {
            "input": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "The conversation messages"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "The type of content (will be classified if not provided)"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "content_type": {
                        "type": "string",
                        "description": "The classified content type"
                    },
                    "response": {
                        "type": "string",
                        "description": "The generated content based on type"
                    }
                }
            }
        },
        "config": {
            "modules": ["router"],
            "graph": "build_graph"
        }
    }
    
    # Save config file
    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Graph compiled and langgraph.json configuration created successfully")