import importlib
import os
import csv
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# Test with all inputs for each script type 
# TODO: Future iteration of this test should specify the input keys for each script to standardize the inputs
TEST_INPUTS = {
    "prompt_chaining.py": {"topic": "programming", "joke": "Why did the chicken cross the road? To get to the other side."},
    "agent.py": {"messages": [{"role": "human", "content": "What is 10 x 4 / 50?"}]},
    "router.py": {"input": "Tell me a poem about artificial intelligence", "messages": [{"role":"user", "content":"Tell me a funny joke about programming."}], "content": "Tell me a poem about artificial intelligence"},
    "multi_agent.py": {"messages": [{"role": "user", "content": "I want to plan a trip to Paris for 3 days. Can you help?"}]},
    "evaluator_optimizer.py": {"topic": "chickens", "joke": "Why did the chicken cross the road? To get to the other side.", "max_iterations": 3, "iteration_count": 0},
}

# Eval prompts for each script type 
CORRECTNESS_PROMPTS = {
    "prompt_chaining.py": """
You are evaluating a joke generated by an AI system. Please evaluate if the output is actually a joke.

Input: {inputs}
Generated joke: {outputs}

Evaluation criteria:
1. Is the output recognizably a joke? (It should have a setup and punchline structure)
2. Is the joke coherent and understandable?

Please provide your evaluation as a JSON object with the following fields:
- score: A score between 0 and 1, where 0 means completely fails criteria and 1 means perfectly meets all criteria
- explanation: A brief explanation of your score
    """,
    
    "agent.py": """
You are evaluating the output of an agent-based system that is designed to answer questions using math tools.
Please evaluate if the output provides a reasonable and coherent answer to the input question.

Input question: {inputs}
Generated answer: {outputs}

Evaluation criteria:
1. Is the output responsive to the input question?
2. Is the answer correct or reasonable?
3. Is the answer coherent and well-structured?

Please provide your evaluation as a JSON object with the following fields:
- score: A score between 0 and 1, where 0 means completely fails criteria and 1 means perfectly meets all criteria
- explanation: A brief explanation of your score
    """,
    
    "router.py": """
You are evaluating the output of a content router system that should produce different types of content based on input.
Please evaluate if the output correctly matches the requested content type and is of good quality.

Input request: {inputs}
Generated content: {outputs}

Evaluation criteria:
1. Does the output match the requested content type?
2. Is the content well-formed and coherent?
3. Is the content creative and engaging?

Please provide your evaluation as a JSON object with the following fields:
- score: A score between 0 and 1, where 0 means completely fails criteria and 1 means perfectly meets all criteria
- explanation: A brief explanation of your score
    """,
    
    "multi_agent.py": """
You are evaluating the output of a multi-agent system designed for travel planning.
Please evaluate if the output shows evidence of multiple agents collaborating to provide travel advice.

Input request: {inputs}
Generated output: {outputs}

Evaluation criteria:
1. Does the output show evidence of multiple perspectives or agents?
2. Is the output responsive to the travel-related input?
3. Is the information provided helpful and coherent?

Please provide your evaluation as a JSON object with the following fields:
- score: A score between 0 and 1, where 0 means completely fails criteria and 1 means perfectly meets all criteria
- explanation: A brief explanation of your score
    """,
    
    "evaluator_optimizer.py": """
You are evaluating the output of a system that iteratively improves a joke based on feedback.
Please evaluate if the output shows evidence of improvement over iterations.

Input: {inputs}
Final output after optimization: {outputs}

Evaluation criteria:
1. Is the final output of high quality relative to earlier iterations?
2. Does the output show evidence of refinement or improvement?
3. Is the content creative and engaging?

Please provide your evaluation as a JSON object with the following fields:
- score: A score between 0 and 1, where 0 means completely fails criteria and 1 means perfectly meets all criteria
- explanation: A brief explanation of your score
    """
}

# Dynamically get all experiment folders from the expts directory
import os
expts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "expts")
experiment_folders = []
if os.path.isdir(expts_path):
    for folder in os.listdir(expts_path):
        if os.path.isdir(os.path.join(expts_path, folder)) and not folder.startswith('.'):
            experiment_folders.append(folder)
    experiment_folders.sort()
else:
    # Fallback in case the path doesn't exist
    experiment_folders = [
        "claude_llmstxt",
        "claude_llmsfull",
        "claude_llmstxt_updated",
        "claude_vectorstore",
        "cursor_llmsfull",
        "cursor_llmstxt_updated",
        "cursor_llmstxt",
        "cursor_vectorstore"
    ]

script_names = [
  "agent.py",
  "evaluator_optimizer.py",
  "multi_agent.py",
  "prompt_chaining.py",
  "router.py"
]

class EvalSchema(BaseModel):
    score: float = Field(description="A score between 0 and 1, where 0 means completely fails criteria and 1 means perfectly meets all criteria")
    explanation: str = Field(description="A brief explanation of your score")

def setup_log_files(custom_name=None):
    """Set up the log files for the evaluation results
    
    Args:
        custom_name (str, optional): Custom name for the evaluation run folder.
            If not provided, a timestamp will be used.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a specific folder for this evaluation run
    if custom_name:
        # For consistency, ensure the custom folder name starts with eval_run_
        folder_name = custom_name if custom_name.startswith("eval_run_") else f"eval_run_{custom_name}"
    else:
        folder_name = f"eval_run_{timestamp}"
    
    run_dir = os.path.join(logs_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create detailed report file
    report_file = os.path.join(run_dir, f"eval_report_{timestamp}.txt")
    with open(report_file, "w") as f:
        f.write(f"Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run Name: {folder_name}\n")
        f.write("="*80 + "\n\n")
    
    # Create CSV results file
    csv_file = os.path.join(run_dir, f"eval_results_{timestamp}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Script", "Success", "Import Success", "LLM Score", "Deployment Score"])
    
    return report_file, csv_file, folder_name


def evaluate_script(experiment_folder: str, script_name: str, report_file: str, csv_file: str) -> Tuple[bool, float, str, int]:
    """Evaluate a single script and log the results, returns (success, score, error_msg, import_success)"""
    module_name = script_name.replace(".py", "")
    input_data = TEST_INPUTS[script_name]
    
    print(f"Evaluating {experiment_folder}/{script_name}...")
    
    try:
        # Get graph to test
        module = importlib.import_module(f"expts.{experiment_folder}.{module_name}")
        graph_to_test = module.compiled_graph
        
        # Run the graph
        output = graph_to_test.invoke(input_data)
        success = True
        error_msg = ""
        import_success = 1  # Imports passed
        
        # Evaluate the output
        eval_prompt = CORRECTNESS_PROMPTS[script_name]
        evaluator = init_chat_model("openai:o3-mini").with_structured_output(EvalSchema)
        eval_result = evaluator.invoke(eval_prompt.format(inputs=input_data, outputs=output))
        
        # Log to report file
        with open(report_file, "a") as f:
            f.write(f"Experiment: {experiment_folder}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"Input: {json.dumps(input_data, indent=2)}\n")
            f.write(f"Output: {str(output)}\n")
            f.write(f"Score: {eval_result.score}\n")
            f.write(f"Explanation: {eval_result.explanation}\n")
            f.write("-"*80 + "\n\n")
        
        # Log to CSV with default deployment score of 0
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([experiment_folder, script_name, int(success), import_success, eval_result.score, 0])
        
        return success, eval_result.score, "", import_success
    
    except Exception as e:
        success = False
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        
        # Check if the error is truly import-related
        import_success = 0
        true_import_errors = ["ImportError", "ModuleNotFoundError"]
        # Check if any of the true import error types are in the error message or stack trace
        if any(error_type in error_msg or error_type in stack_trace for error_type in true_import_errors):
            import_success = 0  # Failed import
        else:
            import_success = 1  # Passed import but failed for other reasons (like graph validation)
        
        # Log error to report file
        with open(report_file, "a") as f:
            f.write(f"Experiment: {experiment_folder}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"ERROR: {error_msg}\n")
            f.write(f"Import Success: {import_success}\n")
            f.write(f"Stack Trace:\n{stack_trace}\n")
            f.write("-"*80 + "\n\n")
        
        # Log to CSV with default deployment score of 0
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([experiment_folder, script_name, 0, import_success, 0.0, 0])
        
        return success, 0.0, error_msg, import_success

def run_evaluations(specific_experiment=None, specific_script=None, run_name=None, show_deployment=False):
    """Run evaluations on all experiment folders and scripts
    
    Args:
        specific_experiment (str, optional): If provided, only test this experiment folder
        specific_script (str, optional): If provided, only test this script
        run_name (str, optional): Custom name for the evaluation run folder
        show_deployment (bool, optional): Flag to indicate if deployment scores should be shown in visualization
    
    Returns:
        str: The name of the run folder that was created
    """
    report_file, csv_file, run_folder = setup_log_files(custom_name=run_name)
    
    print(f"Starting evaluations. Results will be logged to {report_file} and {csv_file}")
    
    # Store the run folder name for later use
    run_folder_path = os.path.dirname(report_file)
    
    # Filter experiment folders and script names if specific ones are provided
    expts_to_test = [specific_experiment] if specific_experiment else experiment_folders
    scripts_to_test = [specific_script] if specific_script else script_names
    
    # Track overall statistics
    total_scripts = len(expts_to_test) * len(scripts_to_test)
    successful_scripts = 0
    failed_scripts = 0
    import_successful_scripts = 0
    total_score = 0.0
    
    print(f"Will evaluate {total_scripts} combinations ({len(expts_to_test)} experiments × {len(scripts_to_test)} scripts)")
    
    # Now evaluate all scripts
    for experiment_folder in expts_to_test:
        print(f"\nEvaluating experiment: {experiment_folder}")
        
        for script_name in scripts_to_test:
            try:
                success, score, error, import_success = evaluate_script(experiment_folder, script_name, report_file, csv_file)
                
                # Now use the import_success value returned directly from evaluate_script
                if import_success == 1:
                    import_successful_scripts += 1
                
                if success:
                    successful_scripts += 1
                    total_score += score
                    print(f"  ✓ {script_name} - Score: {score:.2f}")
                else:
                    failed_scripts += 1
                    import_status = "Import Failed" if import_success == 0 else "Runtime Error"
                    print(f"  ✗ {script_name} - {import_status}: {error[:100]}...")
                
                # We no longer need to fix the CSV file for deployment score
                # since we're now directly adding it in the writerow calls
                
            except Exception as e:
                print(f"  ✗ {script_name} - Unexpected error: {str(e)[:100]}...")
                failed_scripts += 1
                
                # Also need to ensure the deployment score placeholder is added to this error row
                with open(csv_file, "r", newline="") as f:
                    rows = list(csv.reader(f))
                    
                # Find the error row and add placeholder deployment score
                last_row = rows[-1]
                if last_row[0] == experiment_folder and last_row[1] == script_name:
                    # Insert placeholder deployment score before the error column
                    last_row.insert(5, "0")  # Default to 0, will be updated manually
                    
                    # Rewrite the file
                    with open(csv_file, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
    
    # Log summary
    with open(report_file, "a") as f:
        f.write("\nSUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total scripts evaluated: {total_scripts}\n")
        f.write(f"Successful scripts: {successful_scripts}\n")
        f.write(f"Failed scripts: {failed_scripts}\n")
        f.write(f"Scripts with successful imports: {import_successful_scripts}\n")
        f.write(f"Note: Deployment scores need to be added manually\n")
        
        if successful_scripts > 0:
            avg_score = total_score / successful_scripts
            f.write(f"Average LLM evaluation score: {avg_score:.2f}\n")
    
    print(f"\nEvaluation complete. {successful_scripts}/{total_scripts} scripts ran successfully.")
    print(f"Note: For deployment scores, run langgraph dev manually in each experiment folder")
    print(f"and update the CSV file with scores (5 for success, 0 for failure).")
    print(f"Results saved to {report_file} and {csv_file}")
    
    return run_folder

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM experiment graphs")
    parser.add_argument("--experiment", "-e", help="Specific experiment folder to test")
    parser.add_argument("--script", "-s", help="Specific script to test")
    parser.add_argument("--run-name", "-n", help="Custom name for the evaluation run folder")
    parser.add_argument("--show-deployment", action="store_true", help="Show deployment scores in visualization")
    
    args = parser.parse_args()
    
    run_folder = run_evaluations(specific_experiment=args.experiment, specific_script=args.script, run_name=args.run_name, show_deployment=args.show_deployment)
    print(f"Results saved in folder: {run_folder}")