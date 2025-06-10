def run_ml_experiments_only(dataset_path='delaney-processed.csv', tasks=None):
    """Run only traditional ML experiments for all tasks."""
    print("="*60)
    print("MOLECULAR PROPERTY PREDICTION - ML EXPERIMENTS (ALL TASKS)")
    print("="*60)
    
    from config import DATASET_CONFIG
    
    # Prepare dataset
    print("\n1. Preparing dataset...")
    start_time = time.time()
    from data_preprocessing import prepare_complete_dataset
    dataset = prepare_complete_dataset(dataset_path)
    prep_time = time.time() - start_time
    print(f"Dataset preparation completed in {prep_time:.2f} seconds")
    print(f"Dataset shape: {dataset.shape}")
    
    # Get tasks to run
    if tasks is None:
        tasks = DATASET_CONFIG['target_properties']
    
    print(f"\n2. Running experiments for {len(tasks)} tasks:")
    for task in tasks:
        print(f"   - {task}")
    
    # Run ML experiments
    print("\n3. Running traditional ML experiments...")
    start_time = time.time()
    from ml_experiments import run_all_tasks_experiments
    
    mae_results = run_all_tasks_experiments(dataset, experiment_type='mae', tasks=tasks)
    r2_results = run_all_tasks_experiments(dataset, experiment_type='r2', tasks=tasks)
    
    ml_time"""
Main script to run all experiments for molecular property prediction comparison.

This script orchestrates the complete experimental pipeline including:
1. Data preprocessing and feature extraction
2. Traditional machine learning experiments
3. Large language model experiments (optional)
"""

import argparse
import os
import time
from data_preprocessing import prepare_complete_dataset
from ml_experiments import run_all_experiments


def run_ml_experiments_only(dataset_path='delaney-processed.csv'):
    """Run only traditional ML experiments."""
    print("="*60)
    print("MOLECULAR PROPERTY PREDICTION - ML EXPERIMENTS")
    print("="*60)
    
    # Prepare dataset
    print("\n1. Preparing dataset...")
    start_time = time.time()
    dataset = prepare_complete_dataset(dataset_path)
    prep_time = time.time() - start_time
    print(f"Dataset preparation completed in {prep_time:.2f} seconds")
    print(f"Dataset shape: {dataset.shape}")
    
    # Run ML experiments
    print("\n2. Running traditional ML experiments...")
    start_time = time.time()
    ml_results = run_all_experiments(dataset)
    ml_time = time.time() - start_time
    print(f"ML experiments completed in {ml_time:.2f} seconds")
    
    # Save detailed results
    print("\n3. Saving results...")
    for embedding_type, results in ml_results.items():
        filename = f'{embedding_type}_ml_results.csv'
        results.to_csv(filename)
        print(f"Saved {embedding_type} results to {filename}")
    
    return ml_results


def run_complete_experiments(dataset_path='delaney-processed.csv', 
                           openai_key=None, anthropic_key=None):
    """Run complete experimental pipeline including LLMs."""
    print("="*60)
    print("MOLECULAR PROPERTY PREDICTION - COMPLETE EXPERIMENTS")
    print("="*60)
    
    # Run ML experiments first
    ml_results = run_ml_experiments_only(dataset_path)
    
    # Run LLM experiments if API keys provided
    if openai_key or anthropic_key:
        print("\n" + "="*60)
        print("RUNNING LLM EXPERIMENTS")
        print("="*60)
        
        from llm_experiments import GPTPredictor, ClaudePredictor, run_llm_experiment
        from data_preprocessing import load_and_prepare_data
        
        # Load basic dataset for LLM experiments
        basic_dataset = load_and_prepare_data(dataset_path)
        
        if openai_key:
            print("\n4. Running GPT experiments...")
            start_time = time.time()
            try:
                gpt_predictor = GPTPredictor(openai_key)
                run_llm_experiment(gpt_predictor, basic_dataset, "GPT_Response/gpt_logp_results.txt", 'LogP')
                gpt_time = time.time() - start_time
                print(f"GPT experiments completed in {gpt_time:.2f} seconds")
            except Exception as e:
                print(f"Error running GPT experiments: {e}")
        
        if anthropic_key:
            print("\n5. Running Claude experiments...")
            start_time = time.time()
            try:
                claude_predictor = ClaudePredictor(anthropic_key)
                run_llm_experiment(claude_predictor, basic_dataset, "Claude_Response/claude_logp_results.txt", 'LogP')
                claude_time = time.time() - start_time
                print(f"Claude experiments completed in {claude_time:.2f} seconds")
            except Exception as e:
                print(f"Error running Claude experiments: {e}")
    else:
        print("\n4. Skipping LLM experiments (no API keys provided)")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        from visualization import load_and_visualize_results
        load_and_visualize_results()
        print("Visualizations completed!")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("You can run visualization.py separately to generate plots.")
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print("\nOutput files:")
    print("- Results/results_dict_logp.pkl (MAE results)")
    print("- Results/r2_results_logp.pkl (R² results)")
    print("- Results/mae_summary.csv (MAE summary)")
    print("- Results/r2_summary.csv (R² summary)")
    if openai_key:
        print("- GPT_Response/gpt_logp_results.txt (GPT predictions)")
    if anthropic_key:
        print("- Claude_Response/claude_logp_results.txt (Claude predictions)")
    
    return ml_results


def print_usage_instructions():
    """Print usage instructions for the experiments."""
    print("""
USAGE INSTRUCTIONS:

1. Basic ML experiments only:
   python run_experiments.py --ml-only

2. Complete experiments with LLMs:
   python run_experiments.py --openai-key YOUR_OPENAI_KEY --anthropic-key YOUR_ANTHROPIC_KEY

3. Use custom dataset:
   python run_experiments.py --dataset-path your_dataset.csv --ml-only

REQUIREMENTS:
- Place 'delaney-processed.csv' in the same directory
- For LLM experiments, provide valid API keys
- Ensure all dependencies are installed (pip install -r requirements.txt)

OUTPUT FILES:
- ecfp_ml_results.csv, rdkit_ml_results.csv, maccs_ml_results.csv, chemberta_ml_results.csv
- gpt_logp_results.txt (if GPT experiments run)
- claude_logp_results.txt (if Claude experiments run)
    """)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run molecular property prediction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset-path', 
        default='delaney-processed.csv',
        help='Path to the ESOL dataset CSV file (default: delaney-processed.csv)'
    )
    
    parser.add_argument(
        '--ml-only', 
        action='store_true',
        help='Run only traditional ML experiments (skip LLM experiments)'
    )
    
    parser.add_argument(
        '--openai-key',
        help='OpenAI API key for GPT experiments'
    )
    
    parser.add_argument(
        '--anthropic-key',
        help='Anthropic API key for Claude experiments'
    )
    
    parser.add_argument(
        '--help-usage',
        action='store_true',
        help='Show detailed usage instructions'
    )
    
    args = parser.parse_args()
    
    if args.help_usage:
        print_usage_instructions()
        return
    
    # Check if dataset file exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file '{args.dataset_path}' not found!")
        print("Please ensure the ESOL dataset is in the correct location.")
        return
    
    # Run experiments based on arguments
    if args.ml_only:
        run_ml_experiments_only(args.dataset_path)
    else:
        run_complete_experiments(
            dataset_path=args.dataset_path,
            openai_key=args.openai_key,
            anthropic_key=args.anthropic_key
        )


if __name__ == "__main__":
    main()
