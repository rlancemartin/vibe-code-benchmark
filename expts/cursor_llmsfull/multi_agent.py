from typing import Annotated, List, Dict, Tuple, Union, cast
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    FunctionMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolInvocation
from langgraph.graph.agentbox import Command

# Define state with a messages field using the add_messages reducer
class TravelState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # This tracks which agent is active
    current_agent: str
    # This contains any tool calls that have been made
    tool_calls: List[Dict]
    # Used to store travel plans and recommendations
    travel_info: Dict
    hotel_info: Dict

# Initialize Claude 3.5 Sonnet
model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)

# Create system messages for each agent
travel_advisor_system = SystemMessage(
    content="""You are an expert travel advisor specializing in helping customers plan their trips.
Your job is to:
1. Understand the travel destination request from the user
2. Provide expert recommendations on activities, attractions, and experiences
3. Suggest best times to visit and any local customs or tips to be aware of
4. When the user seems satisfied with travel advice, hand off to the hotel advisor using the Command "hotel_advisor"

You only provide travel information and recommendations. For hotel recommendations and bookings,
you hand off to the hotel advisor."""
)

hotel_advisor_system = SystemMessage(
    content="""You are an expert hotel advisor specializing in accommodation recommendations.
Your job is to:
1. Understand user preferences about hotel accommodations (budget, amenities, location, etc.)
2. Provide personalized hotel recommendations based on the travel destination and user preferences
3. Suggest specific hotels with key information about each (star rating, amenities, approximate price range)
4. If user wants to go back to discussing travel details, hand off to the travel advisor using the Command "travel_advisor"

You focus exclusively on hotel-related information and recommendations."""
)

# Define the travel advisor agent
def travel_advisor(state: TravelState) -> Dict:
    """The travel advisor agent provides travel recommendations."""
    # Extract the relevant messages for this agent
    agent_messages = [travel_advisor_system] + state["messages"]
    
    # Get response from the LLM
    response = model.invoke(agent_messages)
    
    # Check if the agent wants to hand off to hotel advisor
    if "hotel_advisor" in response.content.lower():
        # Create a Command to hand off to hotel advisor
        return {
            "messages": [
                AIMessage(
                    content=response.content.split("hotel_advisor")[0].strip(),
                    additional_kwargs={
                        "commands": [Command(name="hotel_advisor", input={})]
                    }
                )
            ],
            "current_agent": "hotel_advisor",
        }
    else:
        # Return regular travel advice
        return {
            "messages": [AIMessage(content=response.content)],
            "current_agent": "travel_advisor",
        }

# Define the hotel advisor agent
def hotel_advisor(state: TravelState) -> Dict:
    """The hotel advisor agent provides hotel recommendations."""
    # Extract the relevant messages for this agent
    agent_messages = [hotel_advisor_system] + state["messages"]
    
    # Get response from the LLM
    response = model.invoke(agent_messages)
    
    # Check if the agent wants to hand off to travel advisor
    if "travel_advisor" in response.content.lower():
        # Create a Command to hand off to travel advisor
        return {
            "messages": [
                AIMessage(
                    content=response.content.split("travel_advisor")[0].strip(),
                    additional_kwargs={
                        "commands": [Command(name="travel_advisor", input={})]
                    }
                )
            ],
            "current_agent": "travel_advisor",
        }
    else:
        # Return regular hotel advice
        return {
            "messages": [AIMessage(content=response.content)],
            "current_agent": "hotel_advisor",
        }

# Router function to determine which agent to use
def router(state: TravelState) -> str:
    """Route to the appropriate agent based on the current_agent field."""
    return state["current_agent"]

# Build the graph
def build_travel_advisor_graph():
    # Create the graph
    workflow = StateGraph(TravelState)
    
    # Add nodes
    workflow.add_node("travel_advisor", travel_advisor)
    workflow.add_node("hotel_advisor", hotel_advisor)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "",  # This is the conditional node (the router)
        router,
        {
            "travel_advisor": "travel_advisor",
            "hotel_advisor": "hotel_advisor"
        }
    )
    
    # Add edges from nodes back to the router
    workflow.add_edge("travel_advisor", "")
    workflow.add_edge("hotel_advisor", "")
    
    # Set the entry point - start with travel advisor by default
    workflow.set_entry_point("travel_advisor")
    
    # Compile the graph
    travel_graph = workflow.compile()
    
    return travel_graph

# Create the graph
travel_advisor_graph = build_travel_advisor_graph()

# For testing the graph
if __name__ == "__main__":
    # Test with a sample prompt
    result = travel_advisor_graph.invoke({
        "messages": [HumanMessage(content="I'm planning a trip to Tokyo, Japan. Can you help me?")],
        "current_agent": "travel_advisor",
        "tool_calls": [],
        "travel_info": {},
        "hotel_info": {}
    })
    
    # Print the result
    print(result["messages"][-1].content) 