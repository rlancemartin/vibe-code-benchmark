from typing import Annotated, Dict, List, Literal, Optional
import random
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import InjectedToolCallId

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt


# Initialize the LLM with Claude 3.5 Sonnet
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Define tools for travel and hotel advisors
@tool
def get_travel_destinations():
    """Get recommendations for tropical travel destinations."""
    return random.choice(["Aruba", "Turks and Caicos", "Jamaica", "Bahamas", "St. Lucia"])


@tool
def get_hotel_recommendations(location: str):
    """Get hotel recommendations for a given destination."""
    options = {
        "aruba": ["The Ritz-Carlton, Aruba", "Bucuti & Tara Beach Resort"],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
        "jamaica": ["Sandals Royal Caribbean", "Round Hill Hotel & Villas"],
        "bahamas": ["The Ocean Club", "Rosewood Baha Mar"],
        "st. lucia": ["Jade Mountain Resort", "Sugar Beach, A Viceroy Resort"]
    }
    return options.get(location.lower(), ["No specific hotels found for this location"])


@tool
def get_activities(location: str):
    """Get activity recommendations for a given destination."""
    activities = {
        "aruba": ["Snorkeling at Arashi Beach", "Visit Arikok National Park", "Sunset cruise"],
        "turks and caicos": ["Diving at Coral Gardens", "Visit Grace Bay", "Kayaking in Mangrove Forest"],
        "jamaica": ["Visit Dunn's River Falls", "Rafting on the Martha Brae River", "Rum tasting tour"],
        "bahamas": ["Swimming with pigs at Exuma", "Snorkeling at Thunderball Grotto", "Nassau food tour"],
        "st. lucia": ["Hiking the Pitons", "Sulfur Springs volcano tour", "Diamond Botanical Gardens"]
    }
    return activities.get(location.lower(), ["No specific activities found for this location"])


# Create handoff tools
def make_handoff_tool(*, agent_name: str):
    """Create a tool that enables handoff to another agent using Command."""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        
        # Return a Command to navigate to the other agent
        return Command(
            goto=agent_name,  # Navigate to the specified agent
            graph=Command.PARENT,  # In the parent graph
            update={"messages": state["messages"] + [tool_message]},  # Pass the conversation history
        )

    return handoff_to_agent


# Define travel advisor agent
def create_travel_advisor():
    """Creates and returns the travel advisor agent."""
    
    # Define travel advisor tools
    travel_advisor_tools = [
        get_travel_destinations,
        get_activities,
        make_handoff_tool(agent_name="hotel_advisor"),
    ]
    
    # Node function
    def travel_advisor_node(state: MessagesState):
        """Travel advisor node function."""
        
        # Combine system message with user messages
        messages = [
            SystemMessage(content=(
                "You are a travel advisor specializing in tropical destinations. "
                "You can recommend destinations and activities. "
                "If the user needs hotel recommendations, transfer to the hotel_advisor. "
                "Be enthusiastic and knowledgeable about travel destinations."
            ))
        ] + state["messages"]
        
        # Get LLM response with tools
        response = llm.bind_tools(travel_advisor_tools).invoke(messages)
        
        # Check if response contains a tool call for handoff
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"].startswith("transfer_to_"):
                    # Execute the handoff tool
                    tool = next((t for t in travel_advisor_tools if t.name == tool_call["name"]), None)
                    if tool:
                        # Return the Command from the handoff tool
                        return tool.invoke({
                            "state": state,
                            "tool_call_id": tool_call["id"]
                        })
        
        # Otherwise continue the conversation through the human node
        return Command(
            update={"messages": [response]},
            goto="human"
        )
    
    return travel_advisor_node


# Define hotel advisor agent
def create_hotel_advisor():
    """Creates and returns the hotel advisor agent."""
    
    # Define hotel advisor tools
    hotel_advisor_tools = [
        get_hotel_recommendations,
        make_handoff_tool(agent_name="travel_advisor"),
    ]
    
    # Node function
    def hotel_advisor_node(state: MessagesState):
        """Hotel advisor node function."""
        
        # Combine system message with user messages
        messages = [
            SystemMessage(content=(
                "You are a hotel advisor specializing in luxury accommodations. "
                "You can recommend hotels based on destinations. "
                "If the user needs general travel advice or activity recommendations, "
                "transfer to the travel_advisor. "
                "Be knowledgeable about hotel amenities, locations, and pricing."
            ))
        ] + state["messages"]
        
        # Get LLM response with tools
        response = llm.bind_tools(hotel_advisor_tools).invoke(messages)
        
        # Check if response contains a tool call for handoff
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"].startswith("transfer_to_"):
                    # Execute the handoff tool
                    tool = next((t for t in hotel_advisor_tools if t.name == tool_call["name"]), None)
                    if tool:
                        # Return the Command from the handoff tool
                        return tool.invoke({
                            "state": state,
                            "tool_call_id": tool_call["id"]
                        })
        
        # Otherwise continue the conversation through the human node
        return Command(
            update={"messages": [response]},
            goto="human"
        )
    
    return hotel_advisor_node


# Define human node for interactive input
def human_node(state: MessagesState, config) -> Command[Literal["travel_advisor", "hotel_advisor"]]:
    """A node that collects user input and routes back to the active agent."""
    
    # Wait for user input via interrupt
    user_input = interrupt(value="Ready for user input.")
    
    # Identify the last active agent (the node that triggered human_node)
    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")
    
    # The trigger format is "graph_name:node_name"
    active_agent = langgraph_triggers[0].split(":")[1]
    
    # Return a Command to go back to the active agent with the user's input
    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": user_input,
                }
            ]
        },
        goto=active_agent,
    )


# Build and compile the multi-agent workflow
def build_multi_agent_system():
    """Build and compile the multi-agent system with travel and hotel advisors."""
    
    # Create the graph with MessagesState
    builder = StateGraph(MessagesState)
    
    # Add nodes to the graph
    builder.add_node("travel_advisor", create_travel_advisor())
    builder.add_node("hotel_advisor", create_hotel_advisor())
    builder.add_node("human", human_node)
    
    # Define the starting node (travel_advisor)
    builder.add_edge(START, "travel_advisor")
    
    # Compile the graph
    travel_agents = builder.compile()
    
    return travel_agents

# Create the multi-agent system
compiled_graph = build_multi_agent_system()

# Example use
if __name__ == "__main__":
    # Start a conversation with travel agent
    initial_input = {
        "messages": [
            {"role": "human", "content": "I want to plan a tropical vacation. Can you suggest a destination?"}
        ]
    }
    
    # This would normally enter into an interactive loop with interrupts
    result = compiled_graph.invoke(initial_input)
    print(result)