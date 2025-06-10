"""
Visualization functions for molecular property prediction results.

This module provides functions to create various plots and heatmaps
for comparing model performance across different embedding types.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


def plot_model_comparison_mae(results_dict, embedding_type):
    """
    Plot MAE comparison for a specific embedding type.
    
    Args:
        results_dict (dict): Dictionary containing results for different embeddings
        embedding_type (str): Embedding type to plot
    """
    # Get data without Linear and Spline models
    results = results_dict[embedding_type]
    results_filtered = results[results.index != 'Linear']
    results_filtered = results_filtered[results_filtered.index != 'Spline']
    
    # Extract model names and metrics
    models = results_filtered.index.tolist()
    means = results_filtered['Mean_MAE'].values
    stds = results_filtered['Std_MAE'].values
    
    # Define colors for different model types
    embedding_colors = {'ecfp': 'C1', 'rdkit': 'C2', 'chemberta': 'C3', 'maccs': 'C4'}
    
    # Assign colors to bars
    colors = []
    for model in models:
        if ('GPT' in model) or ('Claude' in model):
            colors.append('C0')  # Color for LLM models
        elif model.endswith('Baseline'):
            colors.append('C7')  # Color for baseline models
        else:
            colors.append(embedding_colors.get(embedding_type, 'C4'))  # Color based on embedding type
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(models)), means, yerr=stds, capsize=5, color=colors)
    
    # Customize plot
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('MAE')
    plt.title(f'Model Comparison using {embedding_type.upper()} embeddings (MAE)')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_model_comparison_r2(results_dict, embedding_type):
    """
    Plot R² comparison for a specific embedding type.
    
    Args:
        results_dict (dict): Dictionary containing R² results for different embeddings
        embedding_type (str): Embedding type to plot
    """
    # Get data without Linear and Spline models
    results = results_dict[embedding_type]
    results_filtered = results[results.index != 'Linear']
    results_filtered = results_filtered[results_filtered.index != 'Spline']
    
    # Extract model names and metrics
    models = results_filtered.index.tolist()
    r2_values = results_filtered['R2'].values
    
    # Define colors for different model types
    embedding_colors = {'ecfp': 'C1', 'rdkit': 'C2', 'chemberta': 'C3', 'maccs': 'C4'}
    
    # Assign colors to bars
    colors = []
    for model in models:
        if ('GPT' in model) or ('Claude' in model):
            colors.append('C0')  # Color for LLM models
        elif model.endswith('Baseline'):
            colors.append('C7')  # Color for baseline models
        else:
            colors.append(embedding_colors.get(embedding_type, 'C4'))  # Color based on embedding type
    
    # Create plot (reverse order for better visualization)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(models)), r2_values[::-1], color=colors[::-1])
    
    # Customize plot
    plt.xticks(range(len(models)), models[::-1], rotation=45, ha='right')
    plt.ylabel(r'$R^2$')
    plt.title(f'Model Comparison using {embedding_type.upper()} embeddings (R²)')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def prepare_combined_rank_df(results_dict, model_order, metric='Mean_MAE'):
    """
    Prepare combined ranking DataFrame for heatmap visualization.
    
    Args:
        results_dict (dict): Dictionary containing results for different embeddings
        model_order (list): Order of models for columns
        metric (str): Metric to rank by ('Mean_MAE' or 'R2')
        
    Returns:
        pd.DataFrame: Combined ranking DataFrame
    """
    combined_rank_df = pd.DataFrame()
    
    for embedding_type in ['ecfp', 'rdkit', 'chemberta', 'maccs']:
        # Get results for this embedding type
        results = results_dict[embedding_type]
        
        # Check if metric exists
        if metric not in results.columns:
            print(f"{metric} column not found in data for {embedding_type}")
            continue
        
        # Fill NaN values with large values for ranking
        results[metric].fillna(results[metric].max() + 1, inplace=True)

        # Calculate ranks (ascending for MAE, descending for R²)
        ascending = True if metric == 'Mean_MAE' else False
        ranks = results[metric].rank(method='min', ascending=ascending).reindex(model_order).astype(float)
        
        # Add as new row
        combined_rank_df = pd.concat([
            combined_rank_df, 
            pd.DataFrame([ranks.values], index=[embedding_type.upper()], columns=model_order)
        ])
    
    return combined_rank_df


def plot_combined_rank_heatmap(combined_rank_df, model_order, metric='MAE'):
    """
    Plot combined ranking heatmap across embedding types.
    
    Args:
        combined_rank_df (pd.DataFrame): Combined ranking DataFrame
        model_order (list): Order of models for columns
        metric (str): Metric name for title
    """
    # Create heatmap
    plt.figure(figsize=(18, 6))
    ax = sns.heatmap(combined_rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r", 
                     cbar=False, linewidths=0.5, linecolor='black')
    
    # Calculate group boundaries
    llm_end = len([model for model in model_order if ('GPT' in model) or ('Claude' in model)])
    supervised_end = llm_end + len([model for model in model_order 
                                   if model not in [m for m in model_order if ('GPT' in m) or ('Claude' in m)] + 
                                   [m for m in model_order if 'Baseline' in m]])
    
    # Draw boundary lines
    if llm_end > 0:
        ax.axvline(x=llm_end, color='black', linewidth=3)
    if supervised_end > llm_end:
        ax.axvline(x=supervised_end, color='black', linewidth=3)

    # Add group titles
    ax.text(llm_end / 2, -0.5, 'LLM', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text((llm_end + supervised_end) / 2, -0.5, 'Traditional Supervised Methods', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text((supervised_end + len(model_order) - 1) / 2, -0.5, 'Unsupervised Methods', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Set labels and title
    ax.set_xticklabels(model_order, rotation=45, ha='right')
    ax.set_yticklabels(combined_rank_df.index, rotation=0)
    ax.set_xlabel("Method")
    ax.set_ylabel("Embedding Type")

    # Set overall title
    plt.suptitle(f"Model Ranking Heatmap Across Embedding Types ({metric})", 
                 fontsize=18, fontweight='bold', y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_all_embeddings_mae(results_dict):
    """
    Plot MAE comparison for all embedding types.
    
    Args:
        results_dict (dict): Dictionary containing MAE results for different embeddings
    """
    for embedding in ['ecfp', 'rdkit', 'chemberta', 'maccs']:
        plot_model_comparison_mae(results_dict, embedding)


def plot_all_embeddings_r2(results_dict):
    """
    Plot R² comparison for all embedding types.
    
    Args:
        results_dict (dict): Dictionary containing R² results for different embeddings
    """
    for embedding in ['ecfp', 'rdkit', 'chemberta', 'maccs']:
        plot_model_comparison_r2(results_dict, embedding)


def create_comprehensive_ranking_visualization(mae_results_dict, r2_results_dict):
    """
    Create comprehensive ranking visualizations for both MAE and R².
    
    Args:
        mae_results_dict (dict): MAE results dictionary
        r2_results_dict (dict): R² results dictionary
    """
    # Define model order for ranking
    model_order = [
        'GPT-4o w/ no hint', 
        'Claude 3.5 Sonnet w/ no hint',
        'Linear', 'Lasso', 'Ridge', 'ElasticNet', 'KernelRidge', 'Spline', 
        'MLP_1', 'MLP_2', 'MLP_3', 'MLP_4', 'MLP_5', 'MLP_6', 
        'RandomForest', 'Bagging', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'SVM', 'KNN',
        'Mean_Baseline', 'Last_Value_Baseline', 'Random_Baseline'
    ]
    
    # Create MAE ranking heatmap
    print("Creating MAE ranking heatmap...")
    mae_rank_df = prepare_combined_rank_df(mae_results_dict, model_order, 'Mean_MAE')
    plot_combined_rank_heatmap(mae_rank_df, model_order, 'MAE')
    
    # Create R² ranking heatmap (adjust model order for R² results)
    r2_model_order = [
        'GPT-4o w/o hint', 
        'Claude 3.5 Sonnet w/o hint',
        'Linear', 'Lasso', 'Ridge', 'ElasticNet', 'KernelRidge', 'Spline', 
        'MLP_1', 'MLP_2', 'MLP_3', 'MLP_4', 'MLP_5', 'MLP_6', 
        'RandomForest', 'Bagging', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'SVM', 'KNN',
        'Mean_Baseline', 'Last_Value_Baseline', 'Random_Baseline'
    ]
    
    print("Creating R² ranking heatmap...")
    r2_rank_df = prepare_combined_rank_df(r2_results_dict, r2_model_order, 'R2')
    plot_combined_rank_heatmap(r2_rank_df, r2_model_order, 'R²')


def load_and_visualize_all_tasks():
    """
    Load saved results for all tasks and create visualizations.
    """
    from config import TASK_NAME_MAPPING, TASK_OUTPUT_PATTERNS
    import os
    
    available_tasks = []
    
    # Check which task results are available
    for task_name, task_key in TASK_NAME_MAPPING.items():
        mae_file = TASK_OUTPUT_PATTERNS['mae_results'].format(task=task_key)
        r2_file = TASK_OUTPUT_PATTERNS['r2_results'].format(task=task_key)
        
        if os.path.exists(mae_file) and os.path.exists(r2_file):
            available_tasks.append((task_name, task_key))
    
    if not available_tasks:
        print("No task result files found. Please run experiments first.")
        return
    
    print(f"Found results for {len(available_tasks)} tasks:")
    for task_name, task_key in available_tasks:
        print(f"  - {task_name} ({task_key})")
    
    # Create visualizations for each available task
    for task_name, task_key in available_tasks:
        print(f"\n{'='*60}")
        print(f"CREATING VISUALIZATIONS FOR: {task_name}")
        print(f"{'='*60}")
        
        try:
            # Load results
            mae_file = TASK_OUTPUT_PATTERNS['mae_results'].format(task=task_key)
            r2_file = TASK_OUTPUT_PATTERNS['r2_results'].format(task=task_key)
            
            with open(mae_file, 'rb') as f:
                mae_results = pickle.load(f)
            
            with open(r2_file, 'rb') as f:
                r2_results = pickle.load(f)
            
            print(f"Creating MAE visualizations for {task_name}...")
            plot_all_embeddings_mae(mae_results)
            
            print(f"Creating R² visualizations for {task_name}...")
            plot_all_embeddings_r2(r2_results)
            
            print(f"Creating ranking heatmaps for {task_name}...")
            create_comprehensive_ranking_visualization(mae_results, r2_results)
            
            # Save summary statistics
            save_task_summary_statistics(mae_results, r2_results, task_key)
            
        except Exception as e:
            print(f"Error visualizing results for {task_name}: {e}")


def save_task_summary_statistics(mae_results_dict, r2_results_dict, task_key):
    """
    Save summary statistics for a specific task.
    
    Args:
        mae_results_dict (dict): MAE results dictionary
        r2_results_dict (dict): R² results dictionary
        task_key (str): Task key for file naming
    """
    from config import TASK_OUTPUT_PATTERNS
    import os
    
    os.makedirs('Results', exist_ok=True)
    
    # Save MAE summary
    mae_summary = pd.DataFrame()
    for embedding, results in mae_results_dict.items():
        results_copy = results.copy()
        results_copy['Embedding'] = embedding.upper()
        mae_summary = pd.concat([mae_summary, results_copy])
    
    mae_file = TASK_OUTPUT_PATTERNS['mae_summary'].format(task=task_key)
    mae_summary.to_csv(mae_file)
    
    # Save R² summary
    r2_summary = pd.DataFrame()
    for embedding, results in r2_results_dict.items():
        results_copy = results.copy()
        results_copy['Embedding'] = embedding.upper()
        r2_summary = pd.concat([r2_summary, results_copy])
    
    r2_file = TASK_OUTPUT_PATTERNS['r2_summary'].format(task=task_key)
    r2_summary.to_csv(r2_file)
    
    print(f"Summary statistics saved: {mae_file}, {r2_file}")


def visualize_single_task(task_key):
    """
    Visualize results for a single task.
    
    Args:
        task_key (str): Task key (e.g., 'logp', 'mw', etc.)
    """
    from config import TASK_OUTPUT_PATTERNS
    import os
    
    mae_file = TASK_OUTPUT_PATTERNS['mae_results'].format(task=task_key)
    r2_file = TASK_OUTPUT_PATTERNS['r2_results'].format(task=task_key)
    
    if not (os.path.exists(mae_file) and os.path.exists(r2_file)):
        print(f"Result files not found for task '{task_key}'")
        return
    
    try:
        # Load results
        with open(mae_file, 'rb') as f:
            mae_results = pickle.load(f)
        
        with open(r2_file, 'rb') as f:
            r2_results = pickle.load(f)
        
        print(f"Creating visualizations for task: {task_key}")
        
        # Create visualizations
        plot_all_embeddings_mae(mae_results)
        plot_all_embeddings_r2(r2_results)
        create_comprehensive_ranking_visualization(mae_results, r2_results)
        
        # Save summary
        save_task_summary_statistics(mae_results, r2_results, task_key)
        
    except Exception as e:
        print(f"Error visualizing results for {task_key}: {e}")


def compare_tasks_across_methods():
    """
    Create cross-task comparison visualizations.
    """
    from config import TASK_NAME_MAPPING, TASK_OUTPUT_PATTERNS
    import os
    
    # Collect results from all available tasks
    all_mae_results = {}
    all_r2_results = {}
    
    for task_name, task_key in TASK_NAME_MAPPING.items():
        mae_file = TASK_OUTPUT_PATTERNS['mae_results'].format(task=task_key)
        r2_file = TASK_OUTPUT_PATTERNS['r2_results'].format(task=task_key)
        
        if os.path.exists(mae_file) and os.path.exists(r2_file):
            with open(mae_file, 'rb') as f:
                all_mae_results[task_name] = pickle.load(f)
            
            with open(r2_file, 'rb') as f:
                all_r2_results[task_name] = pickle.load(f)
    
    if not all_mae_results:
        print("No task results found for comparison.")
        return
    
    print(f"Creating cross-task comparison for {len(all_mae_results)} tasks...")
    
    # Create summary comparison plots
    create_cross_task_summary_plot(all_mae_results, 'MAE')
    create_cross_task_summary_plot(all_r2_results, 'R²')


def create_cross_task_summary_plot(all_results, metric_name):
    """
    Create summary plot comparing methods across all tasks.
    
    Args:
        all_results (dict): Results for all tasks
        metric_name (str): 'MAE' or 'R²'
    """
    # Extract LLM and best traditional method for each task
    summary_data = []
    
    for task_name, task_results in all_results.items():
        # Get results for one embedding type (they should be similar across embeddings for LLMs)
        embedding_results = next(iter(task_results.values()))
        
        # Find LLM results
        llm_results = {}
        traditional_best = None
        traditional_best_score = float('inf') if metric_name == 'MAE' else float('-inf')
        
        for method, scores in embedding_results.iterrows():
            if 'GPT' in method or 'Claude' in method:
                if metric_name == 'MAE':
                    llm_results[method] = scores.get('Mean_MAE', np.nan)
                else:
                    llm_results[method] = scores.get('R2', np.nan)
            elif not method.endswith('Baseline'):
                if metric_name == 'MAE':
                    score = scores.get('Mean_MAE', np.nan)
                    if not np.isnan(score) and score < traditional_best_score:
                        traditional_best = method
                        traditional_best_score = score
                else:
                    score = scores.get('R2', np.nan)
                    if not np.isnan(score) and score > traditional_best_score:
                        traditional_best = method
                        traditional_best_score = score
        
        # Add to summary
        for llm_method, llm_score in llm_results.items():
            summary_data.append({
                'Task': task_name,
                'Method': llm_method,
                'Score': llm_score,
                'Type': 'LLM'
            })
        
        if traditional_best:
            summary_data.append({
                'Task': task_name,
                'Method': f'Best Traditional ({traditional_best})',
                'Score': traditional_best_score,
                'Type': 'Traditional ML'
            })
    
    if not summary_data:
        print(f"No data available for {metric_name} cross-task comparison.")
        return
    
    # Create plot
    df = pd.DataFrame(summary_data)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df, x='Task', y='Score', hue='Method')
    plt.title(f'Cross-Task Performance Comparison ({metric_name})')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load and visualize all available task results
    load_and_visualize_all_tasks()
    
    # Create cross-task comparisons
    compare_tasks_across_methods()
