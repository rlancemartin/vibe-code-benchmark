import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
from typing import List, Dict, Any, Tuple

def load_results(specific_file=None, run_folder=None):
    """Load evaluation results CSV file."""
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    # Determine the file path to load
    if specific_file:
        if os.path.sep in specific_file:
            file_path = specific_file
        elif run_folder:
            file_path = os.path.join(logs_dir, run_folder, specific_file)
        else:
            direct_path = os.path.join(logs_dir, specific_file)
            if os.path.exists(direct_path):
                file_path = direct_path
            else:
                # Find file in run folders
                run_folders = [d for d in os.listdir(logs_dir) 
                             if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith('eval_run_')]
                run_folders.sort(reverse=True)
                
                file_path = None
                for folder in run_folders:
                    candidate = os.path.join(logs_dir, folder, specific_file)
                    if os.path.exists(candidate):
                        file_path = candidate
                        break
                
                if not file_path:
                    raise FileNotFoundError(f"Could not find {specific_file}")
    else:
        # Find most recent run folder and CSV file
        run_folders = [d for d in os.listdir(logs_dir) 
                      if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith('eval_run_')]
        
        if not run_folders:
            csv_files = glob.glob(os.path.join(logs_dir, "eval_results_*.csv"))
            if not csv_files:
                raise FileNotFoundError("No evaluation results found")
            file_path = max(csv_files, key=os.path.getmtime)
        else:
            run_folders.sort(reverse=True)
            latest_folder = run_folders[0]
            
            csv_files = glob.glob(os.path.join(logs_dir, latest_folder, "eval_results_*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files in latest run folder {latest_folder}")
            
            file_path = max(csv_files, key=os.path.getmtime)
    
    print(f"Loading results from: {file_path}")
    
    # Now we can use standard pandas read_csv since we've removed the Error column
    df = pd.read_csv(file_path)
    
    # Remove any unnamed or empty columns from trailing commas
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Print basic info for debugging
    print(f"CSV Headers: {df.columns.tolist()}")
    
    # Convert numeric columns to be safe
    numeric_cols = ['Success', 'Import Success', 'LLM Score', 'Deployment Score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Print basic info to confirm correct loading
    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head().to_string()}")
    
    return df, os.path.dirname(file_path)

def process_results(df, show_deployment=False):
    """Process the data for visualization.
    
    Args:
        df: DataFrame with the results
        show_deployment: Flag indicating if deployment scores should be shown in visualizations
    """
    # Ensure columns are present
    required_cols = ['Experiment', 'Script']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract code_agent and context_type from Experiment column
    # Example: "claude_llmstxt" -> code_agent="claude", context_type="llmstxt"
    df['code_agent'] = df['Experiment'].str.split('_').str[0]
    df['context_type'] = df['Experiment'].str.split('_').str[1:].apply(lambda x: '_'.join(x) if x else '')
    
    # Update llmstxt_updated to llmstxt_2 for readability on y-axis
    df['context_type'] = df['context_type'].str.replace('llmstxt_updated', 'llmstxt_2')
    
    # Print debugging info to understand the split
    print("\nDebug - Experiment column split:")
    print(df[['Experiment', 'code_agent', 'context_type']].head(10).to_string())
    
    # Fix code agent names if needed
    df['code_agent'] = df['code_agent'].str.replace('claude-', 'claude')
    
    # Convert numeric columns and ensure they exist
    numeric_cols = ['Success', 'Import Success']
    
    # Ensure LLM Score exists (older files might not have it)
    if 'LLM Score' not in df.columns:
        print("WARNING: 'LLM Score' not found in dataframe. Adding with default value 0.")
        df['LLM Score'] = 0
    
    numeric_cols.append('LLM Score')
    
    # Check for Deployment Score
    if 'Deployment Score' in df.columns:
        numeric_cols.append('Deployment Score')
    
    # Convert to numeric values
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Create score columns for visualization
    df['Import Score'] = df['Import Success']
    df['Run Score'] = df['Success']
    
    # Calculate total score based on show_deployment flag
    has_deployment_data = 'Deployment Score' in df.columns and not df['Deployment Score'].isna().all()
    
    # More detailed debug print before calculation
    print("\nDEBUG - In process_results before Total Score calculation:")
    for experiment in df['Experiment'].unique():
        sample_records = df[df['Experiment'] == experiment].head(1)
        for _, row in sample_records.iterrows():
            print(f"  {row['Experiment']} / {row['Script']}:")
            print(f"    Import: {row['Import Success']}, Run: {row['Success']}, LLM: {row['LLM Score']}, Deployment: {row['Deployment Score']}")
    
    # Calculate Total Score conditionally including Deployment based on the flag
    if has_deployment_data and show_deployment:
        df['Total Score'] = df['Import Score'] + df['Run Score'] + df['LLM Score'] + df['Deployment Score']
        print("Including Deployment Score in Total Score (show_deployment=True)")
    else:
        df['Total Score'] = df['Import Score'] + df['Run Score'] + df['LLM Score']
        print("Excluding Deployment Score from Total Score (show_deployment=False)")
    
    # Calculate and print experiment-level scores to verify
    print("\nAggregated scores by experiment:")
    experiment_scores = df.groupby('Experiment').agg({
        'Import Score': 'sum',
        'Run Score': 'sum',
        'LLM Score': 'sum',
        'Deployment Score': 'sum',
        'Total Score': 'sum'
    })
    
    # Calculate max possible scores (5 scripts per experiment)
    scripts_per_experiment = 5
    max_import = scripts_per_experiment
    max_run = scripts_per_experiment
    max_llm = scripts_per_experiment
    max_deployment = scripts_per_experiment * 0.5
    
    # Add normalized percentage columns
    experiment_scores['Import %'] = (experiment_scores['Import Score'] / max_import) * 100
    experiment_scores['Run %'] = (experiment_scores['Run Score'] / max_run) * 100
    experiment_scores['LLM %'] = (experiment_scores['LLM Score'] / max_llm) * 100
    experiment_scores['Deployment %'] = (experiment_scores['Deployment Score'] / max_deployment) * 100
    
    # Calculate max total
    if has_deployment_data and show_deployment:
        max_total = max_import + max_run + max_llm + max_deployment
    else:
        max_total = max_import + max_run + max_llm
    
    experiment_scores['Total %'] = (experiment_scores['Total Score'] / max_total) * 100
    
    # Print the table with percentages
    print(experiment_scores.round(2).to_string())
    
    # Debug print after calculation showing a sample from each experiment
    print("\nSample records after Total Score calculation:")
    for experiment in df['Experiment'].unique():
        sample_record = df[df['Experiment'] == experiment].head(1).iloc[0]
        print(f"  {sample_record['Experiment']} / {sample_record['Script']}:")
        print(f"    Total Score: {sample_record['Total Score']:.2f} (with show_deployment={show_deployment})")
    
    # Print debug info
    print("\nExtracted data:")
    print(f"Code Agents: {sorted(df['code_agent'].unique())}")
    print(f"Context Types: {sorted(df['context_type'].unique())}")
    print(df[['Experiment', 'code_agent', 'context_type']].head().to_string())
    
    return df

def generate_bar_charts(df, source_file=None, output_dir=None, color_scheme="navy", show_deployment=False):
    """Generate bar charts comparing code agents and context types.
    
    Note: This function will always show deployment scores if they are available in the data,
    regardless of the show_deployment flag.
    """
    
    # Create figure with four subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Ensure all required columns exist in the dataframe
    required_score_columns = ['Import Score', 'Run Score', 'Total Score']
    for col in required_score_columns:
        if col not in df.columns:
            print(f"WARNING: Required column '{col}' not found in dataframe. Adding with default value 0.")
            df[col] = 0
    
    # Check if we have deployment scores
    has_deployment = 'Deployment Score' in df.columns and not df['Deployment Score'].isna().all()
    
    # Ensure LLM Score exists
    if 'LLM Score' not in df.columns:
        print("WARNING: 'LLM Score' not found in dataframe. Adding with default value 0.")
        df['LLM Score'] = 0
    
    # Set title
    title = 'Aggregate Performance Comparison (% of Maximum Possible Score)'
    if has_deployment:
        title += ' (with Deployment)'
    fig.suptitle(title, fontsize=26)
    
    # Define metrics to include in preferred order with readable names for legend
    # Start with base metrics that must exist
    component_metrics = ['Import Score', 'Run Score']
    total_metric = ['Total Score']
    llm_metric = ['LLM Score']
    
    # Add deployment if it exists
    if has_deployment:
        component_metrics.append('Deployment Score')
    
    # Create mapping for legend labels
    metric_labels = {
        'Deployment Score': 'Deployment Works',  
        'LLM Score': 'Correct Output',
        'Run Score': 'Code Runs',
        'Import Score': 'Valid Imports',
        'Total Score': 'Total Score'
    }
    
    # Define all metrics for processing (components + llm + total)
    all_metrics = component_metrics + llm_metric + total_metric
    
    # Debug output to verify metrics
    print(f"\nComponent metrics: {component_metrics}")
    print(f"LLM metric: {llm_metric}")
    print(f"Total metric: {total_metric}")
    print(f"All metrics for processing: {all_metrics}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # 1. Component Metrics by Code Agent (top left)
    agent_data = df.groupby(['code_agent'])[all_metrics].sum().reset_index()
    
    # Calculate the number of scripts per code agent for normalization
    scripts_per_agent = df.groupby('code_agent')['Script'].nunique()
    num_context_types = df['context_type'].nunique()
    
    # Calculate percentage of total possible score
    for metric in all_metrics:
        # Skip metrics that aren't in the dataframe (should not happen with our checks, but just in case)
        if metric not in agent_data.columns:
            print(f"WARNING: Metric '{metric}' not found in grouped dataframe. Skipping.")
            continue
            
        # Calculate maximum possible score for this metric
        if metric == 'Import Score' or metric == 'Run Score' or metric == 'LLM Score':
            max_possible = 1.0 * num_context_types * 5  # 5 scripts per context type
        elif metric == 'Deployment Score':
            max_possible = 1.0 * num_context_types * 5  # 5 scripts per context type
        elif metric == 'Total Score':
            if 'Deployment Score' in df.columns and not df['Deployment Score'].isna().all():
                max_possible = 4.0 * num_context_types * 5  # 4 points max per script with deployment
            else:
                max_possible = 3.0 * num_context_types * 5  # 3 points max per script without deployment
        else:
            max_possible = 1.0 * num_context_types * 5  # Default for unknown metrics
            
        # Apply the normalization, convert to percentage
        for idx, row in agent_data.iterrows():
            agent = row['code_agent']
            scripts_count = scripts_per_agent[agent] * num_context_types
            max_for_agent = max_possible
            
            # Handle KeyError - ensure the metric exists in the row
            try:
                agent_data.at[idx, metric] = (row[metric] / max_for_agent) * 100
            except KeyError:
                print(f"WARNING: Could not access metric '{metric}' for agent '{agent}'. Setting to 0.")
                agent_data.at[idx, metric] = 0
    
    # Sort by Valid Imports, Code Runs, then Deployment (if available)
    sort_order = ['Import Score', 'Run Score']
    if 'Deployment Score' in component_metrics:
        sort_order.append('Deployment Score')
    agent_data = agent_data.sort_values(sort_order, ascending=False)
    
    # Plot component metrics in top left with custom colors
    ax_top_left = axes[0, 0]
    
    # Use custom colors - dark blue for Valid Imports, lighter colors for others
    # Create a manual color scheme with darkest to lightest
    if len(component_metrics) == 3:  # With deployment
        custom_colors = ['#003f5c', '#bc5090', '#ffa600']  # Dark blue to orange
    else:  # Without deployment
        custom_colors = ['#003f5c', '#ffa600']  # Dark blue and orange
    
    agent_data.plot(
        x='code_agent',
        y=component_metrics,  # Only component metrics, no total
        kind='bar',
        ax=ax_top_left,
        color=custom_colors,  # Use custom colors instead of colormap
        edgecolor='black',
        linewidth=1,
        width=0.8
    )
    
    ax_top_left.set_title('Component Scores by Code Agent', fontsize=24)
    ax_top_left.set_ylabel('Percentage of Maximum Possible', fontsize=18)
    ax_top_left.set_xlabel('Code Agent', fontsize=18)
    
    # Set y-axis limits to 0-100 for percentage
    ax_top_left.set_ylim(0, 100)
    
    # Remove grid lines
    ax_top_left.grid(False)
    
    # Update legend with our friendly labels
    handles, labels = ax_top_left.get_legend_handles_labels()
    friendly_labels = [metric_labels[m] for m in labels]
    ax_top_left.legend(handles, friendly_labels, fontsize=16)
    plt.setp(ax_top_left.get_xticklabels(), fontsize=18, rotation=0)
    plt.setp(ax_top_left.get_yticklabels(), fontsize=16)
    
    # 2. Total Score by Code Agent (bottom left)
    # Sort specifically for the total score plot - highest first
    agent_data_total = agent_data.sort_values('Total Score', ascending=False)
    
    ax_bottom_left = axes[1, 0]
    agent_data_total.plot(
        x='code_agent',
        y=total_metric,  # Only total score
        kind='bar',
        ax=ax_bottom_left,
        color='#003f5c',  # Dark blue
        edgecolor='black',
        linewidth=1,
        width=0.8
    )
    
    ax_bottom_left.set_title('Total Score by Code Agent', fontsize=24)
    ax_bottom_left.set_ylabel('Percentage of Maximum Possible', fontsize=18)
    ax_bottom_left.set_xlabel('Code Agent', fontsize=18)
    
    # Set y-axis limits to 0-100 for percentage
    ax_bottom_left.set_ylim(0, 100)
    
    # Remove grid lines
    ax_bottom_left.grid(False)
    
    # Add values on top of bars - use sorted data
    for i, v in enumerate(agent_data_total['Total Score']):
        ax_bottom_left.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=14, fontweight='bold')
    
    plt.setp(ax_bottom_left.get_xticklabels(), fontsize=18, rotation=0)
    plt.setp(ax_bottom_left.get_yticklabels(), fontsize=16)
    
    # 3. Component Metrics by Context Type (top right)
    context_data = df.groupby(['context_type'])[component_metrics + total_metric].sum().reset_index()
    
    # Calculate the number of scripts per context type for normalization
    scripts_per_context = df.groupby('context_type')['Script'].nunique()
    num_code_agents = df['code_agent'].nunique()
    
    # Calculate percentage of total possible score
    for metric in all_metrics:
        # Skip metrics that aren't in the dataframe (should not happen with our checks, but just in case)
        if metric not in context_data.columns:
            print(f"WARNING: Metric '{metric}' not found in grouped context dataframe. Skipping.")
            continue
            
        # Calculate maximum possible score for this metric
        if metric == 'Import Score' or metric == 'Run Score' or metric == 'LLM Score':
            max_possible = 1.0 * num_code_agents * 5  # 5 scripts per code agent
        elif metric == 'Deployment Score':
            max_possible = 1.0 * num_code_agents * 5  # 5 scripts per code agent
        elif metric == 'Total Score':
            if 'Deployment Score' in df.columns and not df['Deployment Score'].isna().all():
                max_possible = 4.0 * num_code_agents * 5  # 4 points max per script with deployment
            else:
                max_possible = 3.0 * num_code_agents * 5  # 3 points max per script without deployment
        else:
            max_possible = 1.0 * num_code_agents * 5  # Default for unknown metrics
            
        # Apply the normalization, convert to percentage
        for idx, row in context_data.iterrows():
            context = row['context_type']
            scripts_count = scripts_per_context[context] * num_code_agents
            max_for_context = max_possible
            
            # Handle KeyError - ensure the metric exists in the row
            try:
                context_data.at[idx, metric] = (row[metric] / max_for_context) * 100
            except KeyError:
                print(f"WARNING: Could not access metric '{metric}' for context '{context}'. Setting to 0.")
                context_data.at[idx, metric] = 0
    
    # Sort by the same metrics as agent data
    context_data = context_data.sort_values(sort_order, ascending=False)
    
    # Plot component metrics in top right with custom colors
    ax_top_right = axes[0, 1]
    
    # Use the same custom colors as the top left chart
    context_data.plot(
        x='context_type',
        y=component_metrics,  # Only component metrics, no total
        kind='bar',
        ax=ax_top_right,
        color=custom_colors,  # Use same custom colors as top left
        edgecolor='black',
        linewidth=1,
        width=0.8
    )
    
    ax_top_right.set_title('Component Scores by Context Type', fontsize=24)
    ax_top_right.set_ylabel('Percentage of Maximum Possible', fontsize=18)
    ax_top_right.set_xlabel('Context Type', fontsize=18)
    
    # Set y-axis limits to 0-100 for percentage
    ax_top_right.set_ylim(0, 100)
    
    # Remove grid lines
    ax_top_right.grid(False)
    
    # Update legend with our friendly labels
    handles, labels = ax_top_right.get_legend_handles_labels()
    friendly_labels = [metric_labels[m] for m in labels]
    ax_top_right.legend(handles, friendly_labels, fontsize=16)
    plt.setp(ax_top_right.get_xticklabels(), fontsize=18, rotation=0)
    plt.setp(ax_top_right.get_yticklabels(), fontsize=16)
    
    # 4. Total Score by Context Type (bottom right)
    # Sort specifically for the total score plot - highest first
    context_data_total = context_data.sort_values('Total Score', ascending=False)
    
    ax_bottom_right = axes[1, 1]
    context_data_total.plot(
        x='context_type',
        y=total_metric,  # Only total score
        kind='bar',
        ax=ax_bottom_right,
        color=custom_colors, 
        edgecolor='black',
        linewidth=1,
        width=0.8
    )
    
    ax_bottom_right.set_title('Total Score by Context Type', fontsize=24)
    ax_bottom_right.set_ylabel('Percentage of Maximum Possible', fontsize=18)
    ax_bottom_right.set_xlabel('Context Type', fontsize=18)
    
    # Set y-axis limits to 0-100 for percentage
    ax_bottom_right.set_ylim(0, 100)
    
    # Remove grid lines
    ax_bottom_right.grid(False)
    
    # Add values on top of bars - use sorted data
    for i, v in enumerate(context_data_total['Total Score']):
        ax_bottom_right.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=14, fontweight='bold')
    
    plt.setp(ax_bottom_right.get_xticklabels(), fontsize=18, rotation=0)
    plt.setp(ax_bottom_right.get_yticklabels(), fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    filename_base = 'aggregate_comparison'
    if source_file:
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        filename_base = f"{filename_base}_{base_name}"
    
    save_dir = output_dir if output_dir else os.path.join(os.path.dirname(__file__), "logs")
    output_path = os.path.join(save_dir, f'{filename_base}.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Aggregate comparison saved to {output_path}")
    plt.close()

def generate_grouped_bar_chart(df, source_file=None, output_dir=None, show_deployment=False):
    """Generate grouped bar chart with IDEs on x-axis and different colored bars for context types."""
    plt.figure(figsize=(16, 10))
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure all required columns exist in the dataframe
    required_score_columns = ['Import Score', 'Run Score', 'LLM Score', 'Total Score']
    for col in required_score_columns:
        if col not in df.columns:
            print(f"WARNING: Required column '{col}' not found in dataframe. Adding with default value 0.")
            df[col] = 0
    
    # Check if deployment scores exist in the data
    has_deployment_data = 'Deployment Score' in df.columns and not df['Deployment Score'].isna().all()
    
    # Determine if we should use deployment scores based on the show_deployment flag
    has_deployment = has_deployment_data and show_deployment
    
    # We don't need to recalculate Total Score because it's already correctly calculated
    # in process_results based on the show_deployment flag.
    # Just add some debugging to verify
    # if has_deployment_data:
        # print(f"\nDEBUG - Using Total Score as calculated in process_results (show_deployment={show_deployment})")
        # sample_records = df[df['Experiment'] == 'claude_llmstxt_updated'].head(3)
        # for _, row in sample_records.iterrows():
            #print(f"  {row['Experiment']} / {row['Script']}:")
            #print(f"    Import: {row['Import Score']}, Run: {row['Run Score']}, LLM: {row['LLM Score']}, " + 
                  #f"Deployment: {row['Deployment Score']}, Total: {row['Total Score']}")
    
    # Calculate total for each code_agent/context_type combination
    print("\nCalculating sums for each agent/context combination...")
    sums = {}
    # Create a mapping for nicer agent names
    agent_display_names = {
        'claude': 'Claude Code',
        'cursor': 'Cursor'
    }
    
    for agent in df['code_agent'].unique():
        nice_agent_name = agent_display_names.get(agent, agent)
        sums[nice_agent_name] = {}
        for context in df['context_type'].unique():
            mask_agent = df['code_agent'] == agent
            mask_context = df['context_type'] == context
            combo_df = df[mask_agent & mask_context]
            total = combo_df['Total Score'].sum()
            sums[nice_agent_name][context] = total
    
    # Create the pivot table manually from our sums
    pivot_data = {}
    for agent in sums:
        pivot_data[agent] = {}
        for context in sums[agent]:
            pivot_data[agent][context] = sums[agent][context]
    
    # Convert to DataFrame
    pivot_df = pd.DataFrame(pivot_data).T
    
    # Calculate the number of scripts per combination for normalization
    script_counts = df.groupby(['code_agent', 'context_type'])['Script'].nunique().unstack()
    
    # Calculate maximum possible score for each combination
    max_possible_per_script = 3.0  # Import (1) + Run (1) + LLM (1)
    if has_deployment and show_deployment:
        max_possible_per_script = 3.5  # Include deployment score (0.5) when show_deployment=True
        
    # Print a simple debug message showing the maximum score we're using
    print(f"\nMax possible per script: {max_possible_per_script} points (show_deployment={show_deployment})")
    
    # Debug code to see raw totals before normalization
    print("\nRaw totals before normalization:")
    display_df = pivot_df.copy()
    for agent in display_df.index:
        for context in display_df.columns:
            print(f"{agent} / {context}: {display_df.loc[agent, context]:.2f}")
    
    # Normalize to percentage of maximum possible score
    for agent in pivot_df.index:
        for context in pivot_df.columns:
            try:
                scripts = 5  # We know there are 5 scripts per combination
                max_score = scripts * max_possible_per_script
                
                # Get raw value for debugging
                raw_value = pivot_df.loc[agent, context]
                
                # Calculate expected percentage with full normalization:
                # For claude_llmstxt_updated with show_deployment=True, should be ~94.9% (16.6/17.5)
                percentage = (raw_value / max_score) * 100
                
                # Debug log extensively
                print(f"{agent}/{context}: Raw={raw_value:.2f}, Max={max_score:.2f}, Percentage={percentage:.1f}%")
                
                # Expected values check
                if agent == "Claude Code" and context == "llmstxt_2" and show_deployment:
                    print(f"CHECK: Claude Code / llmstxt_2 (with deployment): {percentage:.1f}%")
                    if abs(percentage - 95.4) > 2.0:
                        print(f"WARNING: Expected ~95.4%, got {percentage:.1f}%")
                
                # Calculate the percentage 
                pivot_df.loc[agent, context] = percentage
                
                # Ensure we never exceed 100%
                if pivot_df.loc[agent, context] > 100:
                    print(f"Warning: Capping {agent}/{context} Total Score from {pivot_df.loc[agent, context]:.1f}% to 100%")
                    pivot_df.loc[agent, context] = 100.0
            except (KeyError, ZeroDivisionError):
                print(f"Warning: Error normalizing data for {agent}/{context}")
    
    # Print a simple summary of normalization
    print("Normalized to percentages of maximum possible score")
    
    # Define preferred order for context types
    preferred_order = ['llmsfull', 'llmstxt', 'vectorstore', 'llmstxt_2']
    
    # Create a mapping for nicer context type names for display/legend
    context_display_names = {
        'llmsfull': 'Stuff all docs into context',
        'llmstxt': 'llms.txt + URL loading',
        'vectorstore': 'Vectorstore + semantic search', 
        'llmstxt_2': 'llms.txt optimized + URL loading'
    }
    
    # Reorder the columns of the pivot_df according to our preferred order
    # First, get the intersection of our preferred order and the actual columns
    ordered_contexts = [c for c in preferred_order if c in pivot_df.columns]
    # Add any columns that are in pivot_df but not in our preferred order
    ordered_contexts += [c for c in pivot_df.columns if c not in preferred_order]
    
    # Reorder the dataframe
    pivot_df = pivot_df[ordered_contexts]
    
    # Get the updated list of contexts
    contexts = pivot_df.columns.tolist()
    
    # Use custom colors
    color_map = {
        'llmstxt_2': '#003f5c',
        'vectorstore': '#bc5090', 
        'llmstxt': '#ff6361',
        'llmsfull': '#ffa600'
    }
    
    # Create color list in the same order as contexts
    colors = [color_map.get(context, '#333333') for context in contexts]
    
    # Plot the grouped bar chart
    ax = pivot_df.plot(
        kind='bar',
        figsize=(16, 10),
        color=colors,
        width=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    # Add title and labels
    title = 'A well designed llms.txt results in the best performance'
    plt.title(title, fontsize=26)
    plt.xlabel('IDE / Code Agent', fontsize=24)  # Much larger axis label
    plt.ylabel('Score', fontsize=24)  # Much larger axis label
    
    # Note: We already renamed the indices when creating the pivot table
    # so no need to rename them again here
    
    # Set y-axis limits to 0-100 for percentage
    plt.ylim(0, 100)
    
    # Ensure no grid lines are displayed
    plt.grid(False)
    
    # Format legend with improved display names
    handles, labels = ax.get_legend_handles_labels()
    updated_labels = [context_display_names.get(label, label) for label in labels]
    plt.legend(handles, updated_labels, title='Context Type', fontsize=14, title_fontsize=16)
    
    # Format text and labels
    plt.xticks(fontsize=24, rotation=0)  # Much larger tick labels
    plt.yticks(fontsize=20)  # Larger tick labels
    
    # No need to rename tick labels as we've already renamed the index
    
    # Add labels on top of the bars
    for i, agent in enumerate(pivot_df.index):
        for j, context in enumerate(pivot_df.columns):
            try:
                value = pivot_df.loc[agent, context]
                # Calculate the bar position
                x_pos = i + (j - len(contexts)/2 + 0.5) * (0.8 / len(contexts))
                plt.text(x_pos, value + 2, f"{value:.1f}%", 
                         ha='center', va='bottom', fontsize=18, fontweight='bold')
            except KeyError:
                pass  # Skip if combination doesn't exist
    
    # Save figure
    filename_base = 'grouped_bar_chart'
    if source_file:
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        filename_base = f"{filename_base}_{base_name}"
    
    save_dir = output_dir if output_dir else os.path.join(os.path.dirname(__file__), "logs")
    output_path = os.path.join(save_dir, f'{filename_base}.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grouped bar chart saved to {output_path}")
    plt.close()

def generate_component_grouped_bar_chart(df, source_file=None, output_dir=None, show_deployment=False):
    """Generate grouped bar chart with component scores broken down by IDE and context type."""
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure all required columns exist in the dataframe
    required_score_columns = ['Import Score', 'Run Score', 'LLM Score']
    metrics = required_score_columns.copy()
    
    for col in required_score_columns:
        if col not in df.columns:
            print(f"WARNING: Required column '{col}' not found in dataframe. Adding with default value 0.")
            df[col] = 0
    
    # Check if deployment scores exist in the data
    has_deployment_data = 'Deployment Score' in df.columns and not df['Deployment Score'].isna().all()
    
    # Determine if we should include deployment in metrics based on the show_deployment flag
    if has_deployment_data and show_deployment:
        metrics.append('Deployment Score')
        print("Including Deployment Score in component visualization")
    elif has_deployment_data:
        print("Excluding Deployment Score from component visualization")
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(18, 6 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    # Map metrics to friendly names for titles
    metric_titles = {
        'Import Score': 'Valid Imports',
        'Run Score': 'Code Runs',
        'LLM Score': 'Correct Output',
        'Deployment Score': 'Deployment Works'
    }
    
    # Process each metric
    for i, metric in enumerate(metrics):
        # Create pivot table for this metric
        pivot_df = pd.pivot_table(
            df, 
            index='code_agent',  # x-axis: IDEs
            columns='context_type',  # different colored bars
            values=metric,
            aggfunc='sum'
        )
        
        # Normalize to percentage of maximum possible score (always 5 scripts with max 1 point each)
        for agent in pivot_df.index:
            for context in pivot_df.columns:
                try:
                    scripts = 5  # We know there are 5 scripts per combination
                    
                    # Calculate maximum possible score per script (normally 1.0, but need to handle LLM scores)
                    if metric == 'LLM Score':
                        # Look at actual data to see the range of values entered
                        llm_values = df[df['code_agent'] == agent][df['context_type'] == context]['LLM Score']
                        # Use 1.0 as default max score, assuming LLM scores are normalized to 0-1
                        max_per_script = 1.0
                    elif metric == 'Deployment Score':
                        max_per_script = 0.5
                    else:
                        max_per_script = 1.0
                    
                    max_score = scripts * max_per_script
                    raw_value = pivot_df.loc[agent, context]
                    percentage = (raw_value / max_score) * 100
                    
                    # Debug print for component scores
                    print(f"{metric} - {agent}/{context}: Raw={raw_value:.2f}, Max={max_score:.2f}, Percentage={percentage:.1f}%")
                    
                    # Set the value as a percentage
                    pivot_df.loc[agent, context] = percentage
                    
                    # Ensure we never exceed 100%
                    if pivot_df.loc[agent, context] > 100:
                        print(f"Warning: Capping {agent}/{context} in {metric} from {pivot_df.loc[agent, context]:.1f}% to 100%")
                        pivot_df.loc[agent, context] = 100.0
                except (KeyError, ZeroDivisionError):
                    print(f"Warning: Error normalizing data for {agent}/{context} in {metric}")
        
        # Define preferred order for context types
        preferred_order = ['llmsfull', 'llmstxt', 'vectorstore', 'llmstxt_2']
        
        # Create a mapping for nicer context type names for display/legend
        context_display_names = {
            'llmsfull': 'Stuff all docs into context',
            'llmstxt': 'llms.txt + URL loading',
            'vectorstore': 'Vectorstore + semantic search', 
            'llmstxt_2': 'llms.txt optimized + URL loading'
        }
        
        # Reorder the columns of the pivot_df according to our preferred order
        # First, get the intersection of our preferred order and the actual columns
        ordered_contexts = [c for c in preferred_order if c in pivot_df.columns]
        # Add any columns that are in pivot_df but not in our preferred order
        ordered_contexts += [c for c in pivot_df.columns if c not in preferred_order]
        
        # Reorder the dataframe
        pivot_df = pivot_df[ordered_contexts]
        
        # Get the updated list of contexts
        contexts = pivot_df.columns.tolist()
        
        # Use custom colors
        color_map = {
            'llmstxt_2': '#003f5c',
            'vectorstore': '#bc5090', 
            'llmstxt': '#ff6361',
            'llmsfull': '#ffa600'
        }
        
        # Create color list in the same order as contexts
        colors = [color_map.get(context, '#333333') for context in contexts]
        
        # Create a mapping for agent names to display names
        agent_display_names = {
            'claude': 'Claude Code',
            'cursor': 'Cursor'
        }
        
        # Create a new index with display names
        new_index = [agent_display_names.get(idx, idx) for idx in pivot_df.index]
        pivot_df.index = new_index
        
        # Plot the grouped bar chart for this metric
        ax = axes[i]
        pivot_df.plot(
            kind='bar',
            ax=ax,
            color=colors,
            width=0.8,
            edgecolor='black',
            linewidth=1
        )
        
        # Add title and labels
        ax.set_title(f'{metric_titles.get(metric, metric)}', fontsize=24)
        ax.set_xlabel('IDE / Code Agent', fontsize=24)  # Much larger axis label
        ax.set_ylabel('Percentage of Maximum Possible', fontsize=24)  # Much larger axis label
        
        # Set y-axis limits to 0-100 for percentage
        ax.set_ylim(0, 100)
        
        # Remove grid lines
        ax.grid(False)
        
        # Format legend if this is the first subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            updated_labels = [context_display_names.get(label, label) for label in labels]
            ax.legend(handles, updated_labels, title='Context Type', fontsize=16, title_fontsize=18)
        else:
            ax.get_legend().remove()  # Remove redundant legends from other subplots
        
        # Format text and labels
        ax.tick_params(axis='x', labelsize=24, rotation=0)  # Much larger tick labels
        ax.tick_params(axis='y', labelsize=20)  # Larger tick labels
        
        # No need to rename tick labels as we've already renamed the index
        
        # Add labels on top of the bars
        for j, agent in enumerate(pivot_df.index):
            for k, context in enumerate(pivot_df.columns):
                try:
                    value = pivot_df.loc[agent, context]
                    # Calculate the bar position
                    x_pos = j + (k - len(contexts)/2 + 0.5) * (0.8 / len(contexts))
                    ax.text(x_pos, value + 2, f"{value:.1f}%", 
                             ha='center', va='bottom', fontsize=16, fontweight='bold')
                except KeyError:
                    pass  # Skip if combination doesn't exist
    
    # Add overall title
    fig.suptitle('Component Scores by IDE and Context Type', fontsize=26, y=0.98)
    
    # Save figure
    filename_base = 'component_grouped_bar_chart'
    if source_file:
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        filename_base = f"{filename_base}_{base_name}"
    
    save_dir = output_dir if output_dir else os.path.join(os.path.dirname(__file__), "logs")
    output_path = os.path.join(save_dir, f'{filename_base}.png')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add space for the suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Component grouped bar chart saved to {output_path}")
    plt.close()

def main(specific_file=None, run_folder=None, color_scheme="Blues", show_color_options=False, show_deployment=False):
    """Main function to run visualization pipeline."""
    try:
        # Load and process data
        df, output_dir = load_results(specific_file, run_folder)
        processed_df = process_results(df, show_deployment=show_deployment)
        
        # Generate visualizations        
        generate_bar_charts(processed_df, specific_file, output_dir, color_scheme, show_deployment)
        generate_grouped_bar_chart(processed_df, specific_file, output_dir, show_deployment)
        generate_component_grouped_bar_chart(processed_df, specific_file, output_dir, show_deployment)
        
        # Print completion message
        if show_deployment:
            print("Visualizations include deployment scores")
        else:
            print("Deployment scores not included in visualizations (use --show-deployment to include them)")
        
        print("Visualization complete!")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations from evaluation results")
    parser.add_argument("--file", "-f", help="Specific results CSV file to use")
    parser.add_argument("--run-folder", "-r", help="Specific run folder to use")
    parser.add_argument("--color-scheme", "-c", default="Blues", 
                       help="Color scheme to use for heatmaps (e.g., Blues, viridis, RdYlGn)")
    parser.add_argument("--show-deployment", "-d", action="store_true", 
                       help="Show deployment scores in visualizations")
    args = parser.parse_args()
    main(args.file, args.run_folder, args.color_scheme, False, args.show_deployment)