from typing import Annotated, Dict, Literal, List
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId

from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command

# Define the model to use - claude-3-5-sonnet-latest
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define handoff tools for both agents
def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        # Optionally pass current graph state to the tool (will be ignored by the LLM)
        state: Annotated[dict, InjectedState],
        # Optionally pass the current tool call ID (will be ignored by the LLM)
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
            # Navigate to another agent node in the PARENT graph
            goto=agent_name,
            graph=Command.PARENT,
            # Pass agent's full internal message history AND adding a tool message
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent

# Travel advisor tools
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights from origin to destination on a specific date."""
    return f"Found several flights from {origin} to {destination} on {date}. The best options are: \n1. Morning flight (8:00 AM) - $350\n2. Afternoon flight (2:30 PM) - $310\n3. Evening flight (7:45 PM) - $280"

@tool
def search_attractions(location: str) -> str:
    """Search for popular attractions in a location."""
    attractions = {
        "paris": "Top attractions in Paris: Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Arc de Triomphe, Seine River Cruise",
        "new york": "Top attractions in New York: Statue of Liberty, Times Square, Central Park, Empire State Building, Metropolitan Museum of Art",
        "tokyo": "Top attractions in Tokyo: Tokyo Skytree, Senso-ji Temple, Meiji Shrine, Tokyo Disneyland, Shibuya Crossing",
        "london": "Top attractions in London: Tower of London, British Museum, Buckingham Palace, London Eye, Westminster Abbey"
    }
    
    location_lower = location.lower()
    if location_lower in attractions:
        return attractions[location_lower]
    else:
        return f"Here are some general points of interest in {location}: museums, parks, historical sites, local cuisine, and cultural experiences."

# Hotel advisor tools
@tool
def search_hotels(location: str, check_in: str, check_out: str) -> str:
    """Search for hotels in a location for specific dates."""
    return f"Found several hotels in {location} for {check_in} to {check_out}:\n1. Luxury Hotel - $250/night (4.8/5 stars)\n2. Comfort Inn - $150/night (4.2/5 stars)\n3. Budget Stay - $80/night (3.9/5 stars)"

@tool
def hotel_amenities(hotel_type: str) -> str:
    """Get information about typical amenities for different hotel types."""
    amenities = {
        "luxury": "Luxury hotels typically offer: spa services, fine dining restaurants, concierge service, premium bedding, room service, fitness center, swimming pool",
        "mid-range": "Mid-range hotels typically offer: restaurant, basic fitness room, business center, free wifi, breakfast options, room service",
        "budget": "Budget hotels typically offer: free wifi, basic breakfast, TV, clean rooms with essential amenities"
    }
    
    hotel_type_lower = hotel_type.lower()
    if hotel_type_lower in amenities:
        return amenities[hotel_type_lower]
    else:
        return f"For {hotel_type} accommodations, you can expect standard amenities like wifi, basic room service, and clean facilities."

# Create agents using our custom implementation
def make_agent(model, tools, system_prompt):
    """Create a custom agent with the specified model, tools, and system prompt."""
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: MessagesState) -> Command[Literal["call_tools", "__end__"]]:
        messages = state["messages"]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = model_with_tools.invoke(messages)
        if len(response.tool_calls) > 0:
            return Command(goto="call_tools", update={"messages": [response]})

        return {"messages": [response]}

    def call_tools(state: MessagesState) -> Command[Literal["call_model"]]:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_ = tools_by_name[tool_name]
            tool_input_fields = tool_.get_input_schema().model_json_schema()["properties"]

            # Inject state if tool accepts it
            if "state" in tool_input_fields:
                tool_call = {**tool_call, "args": {**tool_call["args"], "state": state}}

            tool_response = tool_.invoke(tool_call)
            if isinstance(tool_response, ToolMessage):
                results.append(Command(update={"messages": [tool_response]}))
            # Handle tools that return Command directly
            elif isinstance(tool_response, Command):
                results.append(tool_response)

        return results

    graph = StateGraph(MessagesState)
    graph.add_node(call_model)
    graph.add_node(call_tools)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_tools", "call_model")

    return graph.compile()

# Create the travel advisor agent
travel_advisor = make_agent(
    model,
    [search_flights, search_attractions, make_handoff_tool(agent_name="hotel_advisor")],
    system_prompt="You are a travel advisor expert. You help users plan their trips by providing information about flights and attractions. If users ask about hotels or accommodations, transfer to the hotel advisor."
)

# Create the hotel advisor agent
hotel_advisor = make_agent(
    model,
    [search_hotels, hotel_amenities, make_handoff_tool(agent_name="travel_advisor")],
    system_prompt="You are a hotel advisor expert. You help users find suitable accommodations and provide information about hotel amenities. If users ask about flights or attractions, transfer to the travel advisor."
)

# Create the multi-agent graph
def build_graph():
    """Build and compile the multi-agent workflow graph."""
    builder = StateGraph(MessagesState)
    
    # Add agent nodes
    builder.add_node("travel_advisor", travel_advisor)
    builder.add_node("hotel_advisor", hotel_advisor)
    
    # Start with the travel advisor
    builder.add_edge(START, "travel_advisor")
    
    # Compile the graph
    return builder.compile()

compiled_graph = build_graph()

# For local execution
if __name__ == "__main__":
    graph = build_graph()
    
    # Example query
    user_input = "I'm planning a trip to Paris next month. What are some attractions I should see?"
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Print result
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool: {message.content}") 