from typing import Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.types import Command

# Define tools
@tool
def get_travel_recommendations():
    """Get recommendations for travel destinations"""
    return ["aruba", "turks and caicos", "bali", "santorini", "costa rica"]

@tool
def get_hotel_recommendations(location: str):
    """Get hotel recommendations for a given destination."""
    hotel_options = {
        "aruba": ["The Ritz-Carlton, Aruba (Palm Beach)", "Bucuti & Tara Beach Resort (Eagle Beach)"],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
        "bali": ["Four Seasons Resort Bali at Sayan", "The Hanging Gardens of Bali"],
        "santorini": ["Katikies Hotel", "Mystique, a Luxury Collection Hotel"],
        "costa rica": ["Four Seasons Resort Costa Rica", "Nayara Springs"]
    }
    return hotel_options.get(location.lower(), ["No hotel recommendations available for this location"])

def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
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
        return Command(
            # navigate to another agent node in the PARENT graph
            goto=agent_name,
            graph=Command.PARENT,
            # This is the state update that the agent `agent_name` will see when it is invoked.
            # We're passing agent's FULL internal message history AND adding a tool message to make sure
            # the resulting chat history is valid.
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent

# Initialize the model
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define travel advisor tools and ReAct agent
travel_advisor_tools = [
    get_travel_recommendations,
    make_handoff_tool(agent_name="hotel_advisor"),
]
travel_advisor = create_react_agent(
    model,
    travel_advisor_tools,
    prompt=(
        "You are a general travel expert that can recommend travel destinations. "
        "If you need hotel recommendations, ask 'hotel_advisor' for help. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)

# Define hotel advisor tools and ReAct agent
hotel_advisor_tools = [
    get_hotel_recommendations,
    make_handoff_tool(agent_name="travel_advisor"),
]
hotel_advisor = create_react_agent(
    model,
    hotel_advisor_tools,
    prompt=(
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help."
        "You MUST include human-readable response before transferring to another agent."
    ),
)

# Build the graph
def build_graph():
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("travel_advisor", travel_advisor)
    builder.add_node("hotel_advisor", hotel_advisor)
    
    # We'll always start with the travel advisor
    builder.add_edge(START, "travel_advisor")
    
    # Compile the graph
    return builder.compile()

# Create and export the compiled graph - this is the object that will be used by langgraph CLI
compiled_graph = build_graph()

# Example of how to invoke the graph locally
if __name__ == "__main__":
    result = compiled_graph.invoke({"messages": [{"role": "user", "content": "I'm looking for a tropical vacation destination with nice beaches."}]})
    
    # Print the final response
    if result and "messages" in result:
        last_message = result["messages"][-1]
        print(f"Final response: {last_message['content']}") 