import langgraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.llms import LLMNode

# Use Claude 3.5 Sonnet as the LLM
MODEL = "claude-3-5-sonnet-latest"

# State definition for the workflow
class JokeState:
    def __init__(self, joke: str, evaluation: str = None, improved_joke: str = None):
        self.joke = joke
        self.evaluation = evaluation
        self.improved_joke = improved_joke

# Evaluator node: Grades the joke
class JokeEvaluator(LLMNode):
    def prompt(self, state: JokeState) -> str:
        return f"""
You are a joke evaluator. Grade the following joke on whether it is funny or not. 
If the joke is funny, respond with 'funny'. If not, respond with 'not funny'.
Joke: {state.joke}
"""
    def output(self, llm_response: str, state: JokeState) -> JokeState:
        state.evaluation = llm_response.strip().lower()
        return state

# Optimizer node: Improves the joke
class JokeOptimizer(LLMNode):
    def prompt(self, state: JokeState) -> str:
        return f"""
You are a joke writer. The following joke was graded as 'not funny':
"{state.joke}"
Rewrite or improve it to make it funnier. Respond with only the improved joke.
"""
    def output(self, llm_response: str, state: JokeState) -> JokeState:
        state.improved_joke = llm_response.strip()
        return state

# Build the workflow graph
graph = StateGraph(JokeState)

graph.add_node("evaluate", JokeEvaluator(model=MODEL))
graph.add_node("optimize", JokeOptimizer(model=MODEL))

def route(state: JokeState):
    if state.evaluation == "funny":
        return END
    return "optimize"

graph.add_edge("evaluate", route)
graph.add_edge("optimize", END)

graph.set_entry_point("evaluate")

compiled_graph = graph.compile()

if __name__ == "__main__":
    # Example usage
    joke = "Why did the chicken cross the road? To get to the other side."
    state = JokeState(joke=joke)
    result = compiled_graph.invoke(state)
    print("Evaluation:", result.evaluation)
    if result.evaluation == "funny":
        print("Joke is funny:", result.joke)
    else:
        print("Improved joke:", result.improved_joke)
