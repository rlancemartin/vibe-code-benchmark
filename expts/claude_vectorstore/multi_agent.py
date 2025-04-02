from typing import TypedDict, Literal, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Define the state schema for our multi-agent system
class TravelPlanningState(TypedDict):
    messages: List[Dict]
    current_agent: Optional[str]
    travel_destination: Optional[str]
    hotel_recommendations: Optional[List[str]]
    travel_itinerary: Optional[Dict]

# Setup our Claude 3.5 Sonnet model
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Define our travel advisor agent
def travel_advisor(state: TravelPlanningState) -> Command[Literal["hotel_advisor", END]]:
    """
    Travel advisor agent that recommends destinations and can hand off to hotel advisor.
    """
    # Get the conversation history
    messages = state.get("messages", [])
    
    # Create a system message for the travel advisor
    system_message = """
    You are a knowledgeable travel advisor specialized in recommending travel destinations.
    
    Your role:
    - Recommend travel destinations based on user preferences
    - Provide information about attractions, best times to visit, and local culture
    - Create travel itineraries
    
    When a user asks about hotel recommendations or accommodations, you should hand off to the hotel_advisor agent.
    First provide a helpful response about the travel destination, and then politely indicate you're transferring
    to a hotel specialist. Do not make specific hotel recommendations yourself.
    
    Always identify yourself as the "Travel Advisor" in your responses.
    """
    
    # Convert messages to LangChain format for Claude
    formatted_messages = [SystemMessage(content=system_message)]
    for message in messages:
        if message["role"] == "user":
            formatted_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_messages.append(AIMessage(content=message["content"]))
    
    # Call the model
    response = llm.invoke(formatted_messages)
    
    # Extract any potential travel destination from the response
    travel_destination = state.get("travel_destination")
    if travel_destination is None:
        # Simple extraction logic - in a real system, you'd want something more robust
        lower_response = response.content.lower()
        for destination in ["hawaii", "paris", "tokyo", "bali", "new york", "london", "caribbean"]:
            if destination in lower_response:
                travel_destination = destination
                break
    
    # Create simple travel itinerary if none exists
    travel_itinerary = state.get("travel_itinerary")
    if travel_destination and not travel_itinerary:
        travel_itinerary = {
            "destination": travel_destination,
            "duration": "7 days",
            "activities": ["Sightseeing", "Local cuisine", "Cultural experiences"]
        }
    
    # Decide if we should hand off to hotel advisor or end the conversation
    goto_next = END
    if "hotel" in response.content.lower() or "accommodation" in response.content.lower() or "where to stay" in response.content.lower():
        goto_next = "hotel_advisor"
    
    # Add agent identifier to the response
    agent_response = f"[Travel Advisor] {response.content}"
    
    # Create the updated messages including the new response
    updated_messages = messages + [{"role": "assistant", "content": agent_response}]
    
    # Return a Command with the next agent to call and state updates
    return Command(
        goto=goto_next,
        update={
            "messages": updated_messages,
            "current_agent": "travel_advisor",
            "travel_destination": travel_destination,
            "travel_itinerary": travel_itinerary
        }
    )

# Define our hotel advisor agent
def hotel_advisor(state: TravelPlanningState) -> Command[Literal["travel_advisor", END]]:
    """
    Hotel advisor agent that recommends accommodations and can hand off back to travel advisor.
    """
    # Get the conversation history
    messages = state.get("messages", [])
    
    # Create a system message for the hotel advisor
    system_message = """
    You are a specialized hotel advisor with expertise in accommodations worldwide.
    
    Your role:
    - Recommend hotels and accommodations based on user preferences
    - Provide information about hotel amenities, pricing, and availability
    - Suggest accommodation options for specific destinations
    
    If the user asks about general travel planning, attractions, or activities
    that aren't accommodation-related, hand off to the travel_advisor agent.
    
    Always identify yourself as the "Hotel Advisor" in your responses.
    """
    
    # Convert messages to LangChain format for Claude
    formatted_messages = [SystemMessage(content=system_message)]
    for message in messages:
        if message["role"] == "user":
            formatted_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_messages.append(AIMessage(content=message["content"]))
    
    # Call the model
    response = llm.invoke(formatted_messages)
    
    # Get the travel destination from state
    travel_destination = state.get("travel_destination")
    
    # Generate hotel recommendations if needed
    hotel_recommendations = state.get("hotel_recommendations")
    if travel_destination and not hotel_recommendations:
        # In a real system, you might fetch these from an API
        hotel_recommendations_map = {
            "hawaii": ["Four Seasons Resort Maui", "Ritz-Carlton Kapalua", "Halekulani Hotel"],
            "paris": ["The Ritz Paris", "Hôtel Plaza Athénée", "Le Meurice"],
            "tokyo": ["Park Hyatt Tokyo", "Mandarin Oriental Tokyo", "The Ritz-Carlton Tokyo"],
            "bali": ["Four Seasons Resort Bali", "The St. Regis Bali Resort", "Amankila"],
            "new york": ["The Plaza", "The Ritz-Carlton New York", "Four Seasons Hotel New York"],
            "london": ["The Ritz London", "The Savoy", "Claridge's"],
            "caribbean": ["Jade Mountain Resort", "Sandy Lane Hotel", "Four Seasons Resort Nevis"]
        }
        
        hotel_recommendations = hotel_recommendations_map.get(travel_destination, ["Luxury Hotel 1", "Boutique Hotel 2", "Budget-friendly Option 3"])
    
    # Decide if we should hand off to travel advisor or end the conversation
    goto_next = END
    if any(term in response.content.lower() for term in ["activities", "attractions", "sightseeing", "itinerary", "things to do"]):
        goto_next = "travel_advisor"
    
    # Add agent identifier to the response
    agent_response = f"[Hotel Advisor] {response.content}"
    
    # Create the updated messages including the new response
    updated_messages = messages + [{"role": "assistant", "content": agent_response}]
    
    # Return a Command with the next agent to call and state updates
    return Command(
        goto=goto_next,
        update={
            "messages": updated_messages,
            "current_agent": "hotel_advisor",
            "hotel_recommendations": hotel_recommendations
        }
    )

# Build our multi-agent system
builder = StateGraph(TravelPlanningState)

# Add nodes to the graph
builder.add_node("travel_advisor", travel_advisor)
builder.add_node("hotel_advisor", hotel_advisor)

# Connect the nodes with edges - start with travel advisor
builder.add_edge(START, "travel_advisor")

# Compile the workflow
travel_planning_system = builder.compile()

# Standard interface for import
compiled_graph = travel_planning_system

# This is what we'll run with langgraph dev
if __name__ == "__main__":
    # Example conversation
    result = travel_planning_system.invoke({
        "messages": [
            {"role": "user", "content": "I'm planning a vacation to Hawaii next month. Can you help me?"}
        ]
    })
    
    # Continue the conversation asking about hotels
    result = travel_planning_system.invoke({
        "messages": result["messages"] + [
            {"role": "user", "content": "What hotels would you recommend in Hawaii?"}
        ]
    })
    
    # Continue asking about activities
    result = travel_planning_system.invoke({
        "messages": result["messages"] + [
            {"role": "user", "content": "What activities should I do while I'm there?"}
        ]
    })
    
    # Print the final conversation
    print("\n===== CONVERSATION HISTORY =====\n")
    for message in result["messages"]:
        if message["role"] == "user":
            print(f"User: {message['content']}\n")
        else:
            print(f"{message['content']}\n")