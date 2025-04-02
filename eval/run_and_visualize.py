#!/usr/bin/env python
"""
Script to run evaluations and then generate visualization heatmaps
"""
import os
import sys
import glob
import argparse
from eval.eval import run_evaluations
from eval.visualize_results import main as visualize_main
import re
import time

def main():
    """Run evaluations and generate visualizations"""
    parser = argparse.ArgumentParser(description="Run evaluations and generate visualizations")
    parser.add_argument("--experiment", "-e", help="Specific experiment folder to test")
    parser.add_argument("--script", "-s", help="Specific script to test")
    parser.add_argument("--visualize-only", "-v", action="store_true", help="Skip evaluation and only generate visualizations")
    parser.add_argument("--run-name", "-n", help="Custom name for the evaluation run folder (defaults to timestamp)")
    parser.add_argument("--color-scheme", "-c", default="Blues", 
                       help="Color scheme to use for heatmaps (e.g., Blues, viridis, RdYlGn)")
    parser.add_argument("--show-deployment", action="store_true", help="Show deployment scores in visualization")
    
    args = parser.parse_args()
    
    # Run evaluation if not visualize-only
    if not args.visualize_only:
        print("Running evaluations...")
        run_folder = run_evaluations(
            specific_experiment=args.experiment, 
            specific_script=args.script,
            run_name=args.run_name,
            show_deployment=args.show_deployment
        )
        
        # Give a short pause to ensure file system catches up
        time.sleep(1)
        
        # Get the folder where results were saved
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        
        # Find the CSV file in the run folder
        csv_files = glob.glob(os.path.join(logs_dir, run_folder, "eval_results_*.csv"))
        
        if csv_files:
            latest_csv = os.path.basename(max(csv_files, key=os.path.getmtime))
            
            print(f"\nGenerating visualizations for {latest_csv} in {run_folder}...")
            visualize_main(specific_file=latest_csv, run_folder=run_folder, 
                          color_scheme=args.color_scheme,
                          show_deployment=args.show_deployment)
        else:
            print(f"No CSV files found in the run folder {run_folder}. Skipping visualization.")
    else:
        # Visualize only mode - use most recent run by default
        print("\nGenerating visualizations for most recent evaluation...")
        visualize_main(color_scheme=args.color_scheme, 
                      show_deployment=args.show_deployment)
    
    print("\nDone!")
    
if __name__ == "__main__":
    main()