import operator
from typing import Dict, List, Literal, TypedDict, Annotated, Any
from typing_extensions import NotRequired

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage, HumanMessage
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.graph import StateGraph, START
from langgraph.types import Command

# Define the model
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define the messages state schema with a reducer that concatenates messages
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: NotRequired[Dict[str, Any]]

# Tools for the travel advisor
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights from origin to destination on the given date."""
    return f"Found several flights from {origin} to {destination} on {date}. Options include morning, afternoon, and evening departures."

def check_visa_requirements(destination: str, nationality: str) -> str:
    """Check visa requirements for the destination based on nationality."""
    return f"For travelers with {nationality} nationality visiting {destination}, visa requirements apply. Please check the embassy website for details."

def get_attractions(destination: str) -> str:
    """Get popular attractions at the destination."""
    return f"Popular attractions in {destination} include museums, parks, and historical sites. The city is known for its cultural landmarks."

def transfer_to_hotel_advisor():
    """Transfer the conversation to the hotel advisor for accommodation recommendations."""
    return Command(
        goto="hotel_advisor",
        graph=Command.PARENT,
        # No update needed as we're using shared messages state
    )

# Tools for the hotel advisor
def search_hotels(location: str, check_in: str, check_out: str, guests: int = 2) -> str:
    """Search for hotels in the specified location for the given dates and number of guests."""
    return f"Found several hotels in {location} for {guests} guests from {check_in} to {check_out}. Options range from budget to luxury accommodations."

def check_amenities(hotel_type: str) -> str:
    """Check typical amenities for the specified hotel type."""
    amenities = {
        "budget": "Basic amenities include free Wi-Fi, daily housekeeping, and shared facilities.",
        "mid-range": "Amenities include free Wi-Fi, daily housekeeping, room service, and an on-site restaurant.",
        "luxury": "Luxury amenities include free high-speed Wi-Fi, 24/7 room service, concierge, spa facilities, pool, and fine dining restaurants."
    }
    return amenities.get(hotel_type.lower(), "Please specify hotel type as 'budget', 'mid-range', or 'luxury'.")

def book_hotel(hotel_name: str, room_type: str, check_in: str, check_out: str, guests: int) -> str:
    """Book a hotel room (simulation)."""
    return f"Booking confirmed for {hotel_name}, {room_type} room for {guests} guests from {check_in} to {check_out}. A confirmation email will be sent shortly."

def transfer_to_travel_advisor():
    """Transfer the conversation to the travel advisor for travel recommendations."""
    return Command(
        goto="travel_advisor",
        graph=Command.PARENT,
        # No update needed as we're using shared messages state
    )

# Create the travel advisor agent
travel_advisor = create_react_agent(
    model,
    [search_flights, check_visa_requirements, get_attractions, transfer_to_hotel_advisor],
    prompt="""You are a travel advisor helping users plan their trips.
You can provide information about flights, visa requirements, and attractions.
If the user asks about hotels or accommodations, transfer to the hotel advisor.

Always provide helpful travel advice and suggestions based on the user's preferences.
"""
)

# Create the hotel advisor agent
hotel_advisor = create_react_agent(
    model,
    [search_hotels, check_amenities, book_hotel, transfer_to_travel_advisor],
    prompt="""You are a hotel advisor helping users find and book accommodations.
You can provide information about available hotels, amenities, and handle booking requests.
If the user asks about flights, visa requirements, or attractions, transfer to the travel advisor.

Always provide helpful accommodation advice and suggestions based on the user's preferences.
"""
)

# Create the multi-agent graph
workflow = StateGraph(MessagesState)
workflow.add_node("travel_advisor", travel_advisor)
workflow.add_node("hotel_advisor", hotel_advisor)
workflow.add_edge(START, "travel_advisor")

# Compile the graph
compiled_graph = workflow.compile()

def compile_to_json():
    """Compile the graph to a LangGraph JSON configuration."""
    return compiled_graph.to_json()

if __name__ == "__main__":
    print("Multi-agent travel planning workflow ready.")
    print("Use 'langgraph dev' to run this graph locally with the LangGraph server.")