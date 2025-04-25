from langgraph.graph import StateGraph, Command, END
from langgraph.graph.message import add_messages, MessageGraph, message_func
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Any

# Define the state for the workflow
class TravelState(TypedDict):
    messages: List[Any]
    travel_plan: str
    hotel_recommendation: str

# Travel advisor agent node
def travel_advisor(state: TravelState) -> Command:
    messages = state.get("messages", [])
    # Simulate travel advice
    travel_plan = "You should visit Paris."
    messages.append(AIMessage(content=travel_plan))
    # Handoff to hotel_advisor with Command
    return Command(
        next="hotel_advisor",
        state={"messages": messages, "travel_plan": travel_plan}
    )

# Hotel advisor agent node
def hotel_advisor(state: TravelState) -> Command:
    messages = state.get("messages", [])
    travel_plan = state.get("travel_plan", "")
    # Simulate hotel recommendation based on travel_plan
    hotel = f"For {travel_plan}, I recommend Hotel Le Meurice."
    messages.append(AIMessage(content=hotel))
    return Command(
        next=END,
        state={"messages": messages, "travel_plan": travel_plan, "hotel_recommendation": hotel}
    )

# Build the LangGraph workflow
graph = StateGraph(TravelState)
graph.add_node("travel_advisor", travel_advisor)
graph.add_node("hotel_advisor", hotel_advisor)
graph.set_entry_point("travel_advisor")
graph.add_edge("travel_advisor", "hotel_advisor")
graph.add_edge("hotel_advisor", END)

compiled_graph = graph.compile()

if __name__ == "__main__":
    # Example run
    result = compiled_graph.invoke({"messages": [HumanMessage(content="I want to plan a trip.")]})
    for msg in result["messages"]:
        print(msg.content)
