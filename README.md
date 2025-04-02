# Vibe Coding Benchmark

## Overview

Code agents like [Cursor](https://www.cursor.com/) have transformed how many of us work. Protocols like [MCP (Model Context Protocol)](https://www.anthropic.com/news/model-context-protocol) can connect these agents with external data sources. This repo tests how different code agents compare and how best to connect them to external data.

## Experiment Scope

### Code Assistants Tested
* [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
* [Cursor](https://www.cursor.com/)

### Context Retrieval Methods Tested

1. **Context Stuffing**: Complete [LangGraph documentation](https://langchain-ai.github.io/langgraph/llms-full.txt) (~260k tokens).

2. **Standard llms.txt**: [`llms.txt`](https://llmstxt.org/) files provide background information, links, and page descriptions to LLMs. Test a human generated [LangGraph `llms.txt` file](https://github.com/langchain-ai/langgraph/pull/3987) with an [MCP server](https://github.com/langchain-ai/mcpdoc) to fetch pages.

3. **Optimized llms.txt**: Use an [LLM to re-write](https://github.com/rlancemartin/llmstxt_architect) the LangGraph `llms.txt` file with clearer, more consistent page descriptions specifically designed for LLMs to understand.

4. **Vector Database**: Build a [vector database](https://github.com/langchain-ai/vibe-code-benchmark/blob/main/context_and_mcp/build_langgraph_context.py) of the LangGraph documentation (8,000 token chunks, k=3 retrieval, using OpenAI `text-embedding-3-large`) with an [MCP server](https://github.com/langchain-ai/vibe-code-benchmark/blob/main/context_and_mcp/langgraph_vectorstore_mcp.py) for semantic search.

## LangGraph Challenges

The benchmark includes five progressively complex LangGraph implementation tasks, with input prompts shown below.

1. **Prompt Chaining**: Create a joke generation workflow that chains two LLM calls

```
Create a LangGraph workflow that chains together LLMs calls that (1) creates a joke and then (2) improves it. Implement and compile the workflow in a file named `prompt-chaining.py`. Create or update a `langgrah.json` config file (if it already exists) with the compiled graph from `prompt-chaining.py` so that it can be run locally using `langgraph dev`, but don't actually run `langgraph dev`. Use claude-3-5-sonnet-latest as your model. 
```

2. **Router**: Create a content router that directs inputs to appropriate handlers

```
Create a LangGraph workflow that routes an input that can either be a story, poem, or joke to the appropriate LLM call. Implement 3 different LLM calls, one for each type of input, that produces a story, poem, or joke. Compile the workflow in a file named `router.py`. Create or update a `langgrah.json` config file (if it already exists) with the compiled graph from `router.py` so that it can be run locally using `langgraph dev`, but don't actually run `langgraph dev`. Use claude-3-5-sonnet-latest as your model. 
```
3. **Evaluator-Optimizer**: Build a joke quality evaluator with improvement loop

```
Create a LangGraph workflow that uses an LLM to evaluate the quality of a joke and then uses another LLM to improve the joke if it graded to be of low quality / not funny. Implement and compile the workflow in a file named `evaluator-optimizer.py`. Create or update a `langgrah.json` config file (if it already exists) with the compiled graph from `evaluator-optimizer.py` so that it can be run locally using `langgraph dev`, but don't actually run `langgraph dev`. Use claude-3-5-sonnet-latest as your model. 
```

4. **Agent**: Create a LangGraph math agent with tool binding

```
  Create a LangGraph agent that binds a few math tools and can perform arithmetic. Implement and compile the workflow in a file named `agent.py`. Create or update a `langgrah.json` config file (if it already exists) with the compiled graph from `agent.py` so that it can be run locally using `langgraph dev`, but don't actually run `langgraph dev`. Use claude-3-5-sonnet-latest as your model. 
```

5. **Multi-Agent**: Implement a travel planning system with agent handoff via Command

```
Create a LangGraph multi-agent workflow that has a travel_advisor and a hotel_advisor that uses `Command` for handoff. Implement and compile the workflow in a file named `multi-agent.py`. Create or update a `langgrah.json` config file (if it already exists) with the compiled graph from `multi-agent.py` so that it can be run locally using `langgraph dev`, but don't actually run `langgraph dev`. Use claude-3-5-sonnet-latest as your model. 
```

Each implementation must create a `langgraph.json` config file to support local execution with `langgraph dev`.

## Setup Instructions

### Context and MCP Server Setup

#### Using `llms_full.txt` (Direct Context)
Access the full documentation [here](https://langchain-ai.github.io/langgraph/llms-full.txt).

#### Using `llms.txt` with MCP Server
Use the [`mcpdoc` MCP server](https://github.com/langchain-ai/mcpdoc) to connect LangGraph's [llms.txt file](https://langchain-ai.github.io/langgraph/llms.txt) to each code assistant.

#### Using Vector Retrieval with MCP Server

Create a virtual environment and install dependencies:
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate a local LangGraph vectorstore:
```shell
$ cd context_and_mcp
$ python build_langgraph_context.py
```

Update `langgraph_vectorstore_mcp.py` with your local path:
```python
PATH = "/path/to/vibe-code-benchmark/context_and_mcp/"
```

### Code Assistant Configuration

#### Cursor Setup

> Note: Used `Claude-3.7-sonnet-thinking` for all experiments

**For `llms_full.txt`:**
* Open `Cursor Settings` → `Features` → `Docs` and add https://langchain-ai.github.io/langgraph/llms-full.txt
* Access via `@docs` in Cursor agent chat

**For `llms.txt` with MCP server:**
* Open `Cursor Settings` → `MCP` tab (opens `~/.cursor/mcp.json`)
* Add configuration:
```json
{
  "mcpServers": {
    "langgraph-llms-txt-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "mcpdoc",
        "mcpdoc",
        "--urls",
        "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt",
        "--transport",
        "stdio",
        "--port",
        "8081",
        "--host",
        "localhost"
      ]
    }
  }
}
```

**For Vectorstore with MCP server:**
* Open `Cursor Settings` → `MCP` tab
* Add configuration with your repository path:
```json
{
  "mcpServers": {
      "langgraph-vectorstore-mcp": {
          "command": "/path/to/vibe-code-benchmark/.venv/bin/python",
          "args": [
              "/path/to/vibe-code-benchmark/context_and_mcp/langgraph_vectorstore_mcp.py"
          ]
      }
  }
}
```

#### Claude Code Setup

> Note: Used `claude-3-7-sonnet-20250219` for all experiments

**For `llms_full.txt`:**
Save  https://langchain-ai.github.io/langgraph/llms-full.txt locally and prompt Claude Code with to retrieve it.

**For `llms.txt` with MCP server:**
```bash
claude mcp add-json langgraph-llms-txt-mcp '{"type":"stdio","command":"uvx" ,"args":["--from", "mcpdoc", "mcpdoc", "--urls", "langgraph:https://langchain-ai.github.io/langgraph/llms.txt"]}' -s local
```

**For Vectorstore with MCP server:**
```bash
claude mcp add-json langgraph-vectorstore-mcp '{"type":"stdio","command":"/path/to/vibe-code-benchmark/.venv/bin/python" ,"args":["/path/to/vibe-code-benchmark/context_and_mcp/langgraph_vectorstore_mcp.py"]}' -s local
```

Verify tools with:
```bash
$ claude
$ /mcp 
```

## Experiment Structure

Each experiment is organized as a dedicated branch for each `Assistant × Context` combination. This approach keeps generated code isolated when it was being generated. All code was merged into the `main` branch after generation for final evaluation. For each context method:

* **Full Context**: No MCP servers connected, only direct documentation reference
* **Index/Vectorstore**: Only the relevant MCP server connected

### Prompting Templates per Context Method

**`llms_full.txt`:**
```
You have access to the full LangGraph documentation, `llms_full.txt`. 
+ carefully review this 
+ use it to answer any LangGraph questions
```

**`llms.txt`:**
```
Use the langgraph-llms-txt-mcp server to answer any LangGraph questions -- 
+ call list_doc_sources tool to get the available llms.txt file
+ call fetch_docs tool to read it
+ reflect on the urls in llms.txt 
+ reflect on the input question 
+ call fetch_docs on any urls relevant to the question
+ use these documents to answer any LangGraph questions
```

**Vectorstore:**
```
Use the langgraph-vectorstore-mcp server to answer any LangGraph questions -- 
+ call langgraph_query_tool tool to gather documents 
+ you can call this tool multiple times to gather more documents
+ use these documents to answer any LangGraph questions
```

## Evaluation Framework

Evaluation includes four metrics:

### 1. Import Success (0-1)
* Checks if modules can be imported without errors
* Validates code structure and dependencies
* Critical first step for functional code

### 2. Run Success (0-1)
* Verifies scripts run without crashing
* Tests that LangGraph functions can be invoked with test inputs
* Ensures runtime compatibility

### 3. LLM-Based Quality Assessment (0-1)
* Each implementation is evaluated by an LLM (OpenAI o3-mini)
* Task-specific evaluation prompts assess quality, correctness, and coherence
* Scores range from 0 (poor) to 1 (excellent)

### 4. Deployment Success (0-0.5) [Optional]
* Tests whether implementations can be deployed with `langgraph dev`
* Awards 0.5 points for successful deployment of each script
* Current tested manually and added to evaluation results .csv file

### Running Evaluations

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Evaluate all implementations and generate visualizations
python -m eval.run_and_visualize

# Run with a custom name for organization
python -m eval.run_and_visualize --run-name march_final_benchmark

# Evaluate a specific experiment
python -m eval.eval --experiment claude_vectorstore

# Evaluate a specific script type
python -m eval.eval --script agent.py

# Visualize the most recent evaluation results
python -m eval.visualize_results
```

Evaluation results are organized in run-specific folders:

```
eval/logs/eval_run_TIMESTAMP/
├── eval_report_TIMESTAMP.txt       # Detailed report
├── eval_results_TIMESTAMP.csv      # Summary CSV with all metrics
├── grouped_bar_chart.png           # Total score by experiment
├── component_grouped_bar_chart.png # Score for each component of the experiment (Import, Run, LLM judgement, Deployment)
└── aggregate_comparison.png        # Aggregate score for IDE and Context type 
```

### Using Named Evaluation Runs

For better organization, you can assign custom names to evaluation runs:

```bash
# Create a named evaluation run
python -m eval.run_and_visualize --run-name baseline_benchmark

# Compare with a different configuration
python -m eval.run_and_visualize --run-name optimized_benchmark --experiment claude_vectorstore

# Visualize a specific named run
python -m eval.visualize_results --run-folder eval_run_baseline_benchmark
```

### Testing and Visualizing Deployment Scores

To include deployment scores in your evaluation:

1. Run the normal evaluation first:
   ```bash
   python -m eval.run_and_visualize
   ```

2. Manually test deployment in each experiment folder:
   ```bash
   cd expts/claude_vectorstore
   langgraph dev
   # Press Ctrl+C to stop after confirming it works
   ```

3. Update the CSV file with deployment scores:
   - Open the latest CSV file in `eval/logs/eval_run_TIMESTAMP/eval_results_TIMESTAMP.csv`
   - Update the "Deployment Score" column with appropriate values (default is 0)
   - NOTE: The CSV parsing code depends on consistent formatting. After editing, ensure there are no trailing commas in the CSV file

4. Generate visualizations that include deployment scores:
   ```bash
   python -m eval.visualize_results --show-deployment
   ```

This allows you to assess deployment capabilities separately from the automated tests while still including deployment scores in the final evaluation reports.

For detailed information about the evaluation process and visualization options, see the [Evaluation Framework README](eval/README.md).

Example run: 
```shell
python -m eval.visualize_results --run-folder eval_run_20250402 --show-deployment
```

## License

MIT