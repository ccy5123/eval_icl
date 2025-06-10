"""
Utility functions for molecular property prediction experiments.

This module contains helper functions for data processing, evaluation,
and result analysis.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


def setup_environment():
    """Set up the experimental environment."""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Set tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set random seeds for reproducibility
    np.random.seed(42)


def validate_dataset(df, required_columns=None):
    """
    Validate that the dataset has required columns and structure.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        required_columns (list): List of required column names
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if required_columns is None:
        required_columns = ['smiles']
    
    # Check if DataFrame is not empty
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for SMILES validity (basic check)
    if 'smiles' in df.columns:
        empty_smiles = df['smiles'].isna().sum()
        if empty_smiles > 0:
            print(f"Warning: {empty_smiles} empty SMILES strings found")
    
    return True


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive prediction metrics.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary of metric values
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = np.array(y_true)[mask]
    y_pred_clean = np.array(y_pred)[mask]
    
    if len(y_true_clean) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
    
    metrics = {
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'r2': r2_score(y_true_clean, y_pred_clean)
    }
    
    return metrics


def extract_numeric_prediction(text_prediction):
    """
    Extract numeric value from LLM text prediction.
    
    Args:
        text_prediction (str): Raw text prediction from LLM
        
    Returns:
        float: Extracted numeric value or NaN if extraction fails
    """
    try:
        # Common patterns for numeric extraction
        patterns = [
            r'(-?\d+\.?\d*)',  # Basic number pattern
            r'approximately\s*(-?\d+\.?\d*)',  # "approximately X"
            r'around\s*(-?\d+\.?\d*)',  # "around X"
            r'roughly\s*(-?\d+\.?\d*)',  # "roughly X"
            r'about\s*(-?\d+\.?\d*)',  # "about X"
            r'value.*?(-?\d+\.?\d*)',  # "value ... X"
            r'prediction.*?(-?\d+\.?\d*)',  # "prediction ... X"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_prediction.lower())
            if match:
                return float(match.group(1))
        
        # If no pattern matches, try to find any number
        numbers = re.findall(r'-?\d+\.?\d*', text_prediction)
        if numbers:
            return float(numbers[0])
        
        return np.nan
        
    except (ValueError, AttributeError):
        return np.nan


def parse_llm_results(filename):
    """
    Parse LLM experiment results from text file.
    
    Args:
        filename (str): Path to results file
        
    Returns:
        pd.DataFrame: Parsed results with true and predicted values
    """
    results = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by iteration separators
        iterations = content.split('=' * 50)
        
        for iteration in iterations:
            if 'Iteration:' not in iteration:
                continue
                
            lines = iteration.strip().split('\n')
            
            iteration_data = {}
            in_prediction = False
            prediction_text = []
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('Iteration:'):
                    iteration_data['iteration'] = int(line.split(':')[1].strip())
                elif line.startswith('SMILES:'):
                    iteration_data['smiles'] = line.split(':', 1)[1].strip()
                elif line.startswith('True Property:'):
                    iteration_data['true_value'] = float(line.split(':')[1].strip())
                elif line.startswith('Predicted Property:'):
                    in_prediction = True
                    continue
                elif in_prediction and line and not line.startswith('='):
                    prediction_text.append(line)
            
            if prediction_text:
                full_prediction = ' '.join(prediction_text)
                iteration_data['predicted_text'] = full_prediction
                iteration_data['predicted_value'] = extract_numeric_prediction(full_prediction)
                
            if len(iteration_data) >= 4:  # Ensure we have all required fields
                results.append(iteration_data)
    
    except FileNotFoundError:
        print(f"File {filename} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def create_results_summary(ml_results, llm_results=None):
    """
    Create a comprehensive summary of all experimental results.
    
    Args:
        ml_results (dict): ML experiment results
        llm_results (dict): LLM experiment results (optional)
        
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = []
    
    # Process ML results
    for embedding_type, results_df in ml_results.items():
        for model_name in results_df.index:
            summary_data.append({
                'Method': f"{embedding_type.upper()}_{model_name}",
                'Type': 'Traditional ML',
                'Mean_MAE': results_df.loc[model_name, 'Mean_MAE'],
                'Std_MAE': results_df.loc[model_name, 'Std_MAE'],
                'Median_MAE': results_df.loc[model_name, 'Median_MAE']
            })
    
    # Process LLM results if provided
    if llm_results:
        for llm_name, results_df in llm_results.items():
            if not results_df.empty:
                mae_values = []
                for _, row in results_df.iterrows():
                    if not np.isnan(row['predicted_value']):
                        mae = abs(row['true_value'] - row['predicted_value'])
                        mae_values.append(mae)
                
                if mae_values:
                    summary_data.append({
                        'Method': llm_name.upper(),
                        'Type': 'Large Language Model',
                        'Mean_MAE': np.mean(mae_values),
                        'Std_MAE': np.std(mae_values),
                        'Median_MAE': np.median(mae_values)
                    })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values('Mean_MAE') if not summary_df.empty else summary_df


def plot_results_comparison(summary_df, save_path=None):
    """
    Create visualization comparing different methods.
    
    Args:
        summary_df (pd.DataFrame): Summary results DataFrame
        save_path (str): Path to save the plot (optional)
    """
    if summary_df.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Separate traditional ML and LLM results
    ml_data = summary_df[summary_df['Type'] == 'Traditional ML']
    llm_data = summary_df[summary_df['Type'] == 'Large Language Model']
    
    y_pos = np.arange(len(summary_df))
    
    # Create horizontal bar plot
    colors = ['skyblue' if t == 'Traditional ML' else 'lightcoral' 
             for t in summary_df['Type']]
    
    bars = plt.barh(y_pos, summary_df['Mean_MAE'], 
                   xerr=summary_df['Std_MAE'], 
                   color=colors, alpha=0.7)
    
    plt.yticks(y_pos, summary_df['Method'], fontsize=8)
    plt.xlabel('Mean Absolute Error')
    plt.title('Molecular Property Prediction Performance Comparison')
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Traditional ML'),
        Patch(facecolor='lightcoral', label='Large Language Model')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def save_comprehensive_results(ml_results, llm_results=None, output_dir='Results'):
    """
    Save all results in a comprehensive format.
    
    Args:
        ml_results (dict): ML experiment results
        llm_results (dict): LLM experiment results (optional)
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual ML results
    for embedding_type, results_df in ml_results.items():
        filepath = os.path.join(output_dir, f'{embedding_type}_detailed_results.csv')
        results_df.to_csv(filepath)
    
    # Save LLM results if available
    if llm_results:
        for llm_name, results_df in llm_results.items():
            if not results_df.empty:
                filepath = os.path.join(output_dir, f'{llm_name}_parsed_results.csv')
                results_df.to_csv(filepath, index=False)
    
    # Create and save summary
    summary_df = create_results_summary(ml_results, llm_results)
    summary_path = os.path.join(output_dir, 'comprehensive_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Create comparison plot
    plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plot_results_comparison(summary_df, plot_path)
    
    print(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    setup_environment()
    
    # Test numeric extraction
    test_predictions = [
        "The predicted molecular weight is approximately 156.3",
        "I estimate the value to be around 203.7",
        "Based on the examples, the property should be 89.45",
        "The prediction is roughly 245.1 daltons"
    ]
    
    for pred in test_predictions:
        extracted = extract_numeric_prediction(pred)
        print(f"'{pred}' -> {extracted}")
