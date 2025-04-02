from typing import Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Define our state
class TravelState(TypedDict):
    messages: list[BaseMessage]
    current_agent: str

# Initialize our LLM
llm = ChatOpenAI(model="claude-3-5-sonnet-latest")

# Define the travel advisor agent node
def travel_advisor(state: TravelState) -> Command[Literal["hotel_advisor", END]]:
    """Travel advisor agent that recommends travel destinations and can hand off to hotel advisor."""
    
    # Extract messages history
    messages = state["messages"]
    
    # Create system prompt for travel advisor
    system_message = SystemMessage(
        content=(
            "You are a travel advisor. You help users plan their trips by recommending destinations, "
            "activities, and the best times to visit different places. "
            "When the user asks about accommodations or hotels, transfer to the hotel advisor. "
            "Always be helpful, informative, and enthusiastic about travel."
        )
    )
    
    # Include all messages for context
    response = llm.invoke([system_message] + messages)
    
    # Check if we need to transfer to hotel advisor
    if any(keyword in response.content.lower() for keyword in ["hotel", "accommodation", "place to stay", "lodging"]):
        # Transfer to hotel advisor
        return Command(
            goto="hotel_advisor",
            update={
                "messages": messages + [AIMessage(content=response.content)],
                "current_agent": "hotel_advisor"
            }
        )
    
    # Otherwise stay with travel advisor for the next interaction
    return Command(
        goto=END,
        update={
            "messages": messages + [AIMessage(content=response.content)],
            "current_agent": "travel_advisor"
        }
    )

# Define the hotel advisor agent node
def hotel_advisor(state: TravelState) -> Command[Literal["travel_advisor", END]]:
    """Hotel advisor agent that recommends accommodations and can hand off back to travel advisor."""
    
    # Extract messages history
    messages = state["messages"]
    
    # Create system prompt for hotel advisor
    system_message = SystemMessage(
        content=(
            "You are a hotel advisor. You help users find the perfect accommodations for their trips. "
            "You can recommend hotels, resorts, vacation rentals, and other lodging options based on "
            "budget, preferences, and location. When the user asks about general travel advice or "
            "destinations, transfer back to the travel advisor. "
            "Always be helpful, informative, and knowledgeable about accommodations worldwide."
        )
    )
    
    # Include all messages for context
    response = llm.invoke([system_message] + messages)
    
    # Check if we need to transfer back to travel advisor
    if any(keyword in response.content.lower() for keyword in ["destination", "attraction", "activity", "tour", "sight"]):
        # Transfer to travel advisor
        return Command(
            goto="travel_advisor",
            update={
                "messages": messages + [AIMessage(content=response.content)],
                "current_agent": "travel_advisor"
            }
        )
    
    # Otherwise stay with hotel advisor for the next interaction
    return Command(
        goto=END,
        update={
            "messages": messages + [AIMessage(content=response.content)],
            "current_agent": "hotel_advisor"
        }
    )

# Create and compile the graph
def create_multi_agent_graph():
    # Build the graph
    builder = StateGraph(TravelState)
    
    # Add nodes
    builder.add_node("travel_advisor", travel_advisor)
    builder.add_node("hotel_advisor", hotel_advisor)
    
    # Add edges - start with travel advisor by default
    builder.add_edge(START, "travel_advisor")
    
    # The agent nodes handle transitions internally using Command
    
    # Compile the graph
    return builder.compile()

# Create and expose the compiled graph for langgraph dev
compiled_graph = create_multi_agent_graph()

# Function to invoke the graph with a user message
def invoke_graph(user_input: str, state=None):
    """Invoke the graph with a user message."""
    if state is None:
        # Initialize state if this is the first message
        state = {
            "messages": [HumanMessage(content=user_input)],
            "current_agent": "travel_advisor"  # Start with travel advisor
        }
    else:
        # Add the new user message to existing messages
        state["messages"].append(HumanMessage(content=user_input))
    
    # Run the graph
    result = compiled_graph.invoke(state)
    return result

# For debugging/testing
if __name__ == "__main__":
    # Example usage
    result = invoke_graph("I'm planning a trip to Japan in April. What should I see?")
    print(f"Current agent: {result['current_agent']}")
    print(f"Response: {result['messages'][-1].content}") 