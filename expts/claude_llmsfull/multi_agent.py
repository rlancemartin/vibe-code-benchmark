from typing import Annotated, Dict, List, Any, Literal, TypedDict, Union
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.command import Command
import json

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define state schema
class SharedState(TypedDict):
    messages: Annotated[List, add_messages]
    travel_context: Dict[str, Any]
    hotel_context: Dict[str, Any]
    current_agent: str
    handoff_count: int

# Create specialized system prompts for each agent
TRAVEL_ADVISOR_PROMPT = """You are an expert travel advisor specializing in trip planning.
You help users plan their travel itineraries, recommend destinations, and provide travel tips.
If the user asks about specific hotel recommendations or detailed hotel information, use the HANDOFF_TO_HOTEL_ADVISOR command.

When a user mentions destinations, travel dates, or travel preferences, capture this information in your response.
"""

HOTEL_ADVISOR_PROMPT = """You are an expert hotel advisor specializing in accommodations.
You help users find the perfect hotel based on their preferences, budget, and location.
If the user asks about general travel planning, itineraries, or destinations that go beyond hotel selection, use the HANDOFF_TO_TRAVEL_ADVISOR command.

When a user mentions hotel preferences, budget, or accommodation needs, capture this information in your response.
"""

# Define agent functions
def travel_advisor(state: SharedState) -> Union[Dict[str, Any], Command[Literal["hotel_advisor", "end"]]]:
    """Travel advisor agent that handles trip planning queries."""
    messages = state["messages"]
    travel_context = state.get("travel_context", {})
    handoff_count = state.get("handoff_count", 0)
    
    # Check if we're in a loop with too many handoffs
    if handoff_count > 3:
        # If too many handoffs, just respond directly without handoff
        system_message = SystemMessage(content=TRAVEL_ADVISOR_PROMPT + "\nDo NOT hand off to the hotel advisor even if hotel information is requested. Answer with your best travel knowledge.")
    else:
        system_message = SystemMessage(content=TRAVEL_ADVISOR_PROMPT + "\nCommands you can use:\n- HANDOFF_TO_HOTEL_ADVISOR: Use this command when you need to hand off to the hotel specialist.")
    
    # Prepare agent-specific context
    agent_messages = [system_message] + messages
    
    # Get response from the model
    response = llm.invoke(agent_messages)
    response_content = response.content
    
    # Check for handoff command
    if "HANDOFF_TO_HOTEL_ADVISOR" in response_content:
        # Process the response to extract the clean content without the command
        clean_content = response_content.replace("HANDOFF_TO_HOTEL_ADVISOR", "").strip()
        
        # Create a handoff message
        handoff_message = FunctionMessage(
            content=f"Handing off to hotel advisor. Travel context: {json.dumps(travel_context)}",
            name="travel_advisor_handoff"
        )
        
        # Update the travel context with any new information
        updated_travel_context = travel_context.copy()
        
        # Create a command to route to the hotel advisor
        return Command(
            goto="hotel_advisor",
            update={
                "messages": messages + [AIMessage(content=clean_content)] + [handoff_message],
                "travel_context": updated_travel_context,
                "current_agent": "hotel_advisor",
                "handoff_count": handoff_count + 1
            }
        )
    
    # If no handoff, update state and continue
    return {
        "messages": messages + [response],
        "travel_context": travel_context,
        "current_agent": "travel_advisor"
    }

def hotel_advisor(state: SharedState) -> Union[Dict[str, Any], Command[Literal["travel_advisor", "end"]]]:
    """Hotel advisor agent that handles accommodation queries."""
    messages = state["messages"]
    hotel_context = state.get("hotel_context", {})
    handoff_count = state.get("handoff_count", 0)
    
    # Check if we're in a loop with too many handoffs
    if handoff_count > 3:
        # If too many handoffs, just respond directly without handoff
        system_message = SystemMessage(content=HOTEL_ADVISOR_PROMPT + "\nDo NOT hand off to the travel advisor even if travel planning is requested. Answer with your best hotel knowledge.")
    else:
        system_message = SystemMessage(content=HOTEL_ADVISOR_PROMPT + "\nCommands you can use:\n- HANDOFF_TO_TRAVEL_ADVISOR: Use this command when you need to hand off to the travel specialist.")
    
    # Prepare agent-specific context
    agent_messages = [system_message] + messages
    
    # Get response from the model
    response = llm.invoke(agent_messages)
    response_content = response.content
    
    # Check for handoff command
    if "HANDOFF_TO_TRAVEL_ADVISOR" in response_content:
        # Process the response to extract the clean content without the command
        clean_content = response_content.replace("HANDOFF_TO_TRAVEL_ADVISOR", "").strip()
        
        # Create a handoff message
        handoff_message = FunctionMessage(
            content=f"Handing off to travel advisor. Hotel context: {json.dumps(hotel_context)}",
            name="hotel_advisor_handoff"
        )
        
        # Update the hotel context with any new information
        updated_hotel_context = hotel_context.copy()
        
        # Create a command to route to the travel advisor
        return Command(
            goto="travel_advisor",
            update={
                "messages": messages + [AIMessage(content=clean_content)] + [handoff_message],
                "hotel_context": updated_hotel_context,
                "current_agent": "travel_advisor",
                "handoff_count": handoff_count + 1
            }
        )
    
    # If no handoff, update state and continue
    return {
        "messages": messages + [response],
        "hotel_context": hotel_context,
        "current_agent": "hotel_advisor"
    }

def route_based_on_query(state: SharedState) -> str:
    """Determines which agent should handle the initial query."""
    messages = state["messages"]
    if not messages:
        return "travel_advisor"  # Default to travel advisor
    
    last_message = messages[-1]
    content = last_message.content.lower() if hasattr(last_message, 'content') else str(last_message).lower()
    
    # Check for hotel-related keywords
    hotel_keywords = ["hotel", "accommodation", "stay", "room", "lodge", "motel", "resort", "book a room"]
    travel_keywords = ["itinerary", "destination", "travel", "trip", "visit", "flight", "tour", "vacation"]
    
    # Count matches for each category
    hotel_matches = sum(1 for keyword in hotel_keywords if keyword in content)
    travel_matches = sum(1 for keyword in travel_keywords if keyword in content)
    
    # Route based on keyword matches
    if hotel_matches > travel_matches:
        return "hotel_advisor"
    return "travel_advisor"

# Build the multi-agent graph
def build_graph():
    # Create a new graph
    graph_builder = StateGraph(SharedState)
    
    # Add nodes
    graph_builder.add_node("travel_advisor", travel_advisor)
    graph_builder.add_node("hotel_advisor", hotel_advisor)
    
    # Add edges
    # Initial routing
    graph_builder.add_conditional_edges(
        None,  # None refers to the entry point
        route_based_on_query,
        {
            "travel_advisor": "travel_advisor",
            "hotel_advisor": "hotel_advisor"
        }
    )
    
    # Add self-transitions and transitions between agents
    # Travel advisor can either continue or hand off to hotel advisor
    graph_builder.add_edge("travel_advisor", END)
    
    # Hotel advisor can either continue or hand off to travel advisor
    graph_builder.add_edge("hotel_advisor", END)
    
    # Compile the graph
    return graph_builder.compile()

# Create the compiled graph for imports
compiled_graph = build_graph()

if __name__ == "__main__":
    # Use the compiled graph
    graph = compiled_graph
    
    # Create the langgraph.json config file
    config = {
        "title": "Multi-Agent Travel Advisory System",
        "description": "A LangGraph workflow with specialized travel and hotel advisors that can hand off to each other",
        "schema": {
            "input": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "The conversation messages"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "The updated conversation with agent responses"
                    },
                    "current_agent": {
                        "type": "string",
                        "description": "The agent that provided the final response"
                    },
                    "travel_context": {
                        "type": "object",
                        "description": "Context information gathered by the travel advisor"
                    },
                    "hotel_context": {
                        "type": "object",
                        "description": "Context information gathered by the hotel advisor"
                    }
                }
            }
        },
        "config": {
            "modules": ["multi-agent"],
            "graph": "build_graph"
        }
    }
    
    # Save config file
    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Graph compiled and langgraph.json configuration created successfully")