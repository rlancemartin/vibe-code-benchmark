import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import ChatAnthropic

# 1. Define math tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        return float('inf')
    return a / b

# 2. Define agent node
from langgraph.prebuilt import ToolExecutor, ToolInvocation, ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import chat_agent_executor

TOOLS = [add, subtract, multiply, divide]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math agent. You can use tools to do arithmetic. Answer math questions by calling tools as needed."),
    ("human", "{input}")
])

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

agent_executor = chat_agent_executor(
    llm=llm,
    tools=TOOLS,
    prompt=prompt,
)

def math_agent_node(state):
    # state["messages"] is a list of HumanMessage/AIMessage
    result = agent_executor.invoke({"messages": state["messages"]})
    state = dict(state)
    state["messages"] = state["messages"] + [AIMessage(content=result["output"])]
    return state

# 3. Define state and graph
class AgentState(dict):
    pass

graph = StateGraph(AgentState)
graph.add_node("math_agent", math_agent_node)
graph.set_entry_point("math_agent")
graph.add_edge("math_agent", END)
graph.compile()

# 4. Save compiled graph to langgraph.json
import json
compiled_graph = graph.get_graph()
with open("langgraph.json", "w") as f:
    json.dump(compiled_graph, f, indent=2)
