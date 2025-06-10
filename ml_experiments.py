"""
Machine learning experiments for molecular property prediction.

This module implements various ML models and runs comparative experiments
using different molecular representations. Supports both MAE and R² evaluation.
"""

import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')


def create_mlp_models():
    """
    Create various MLP architectures for testing.
    
    Returns:
        list: List of MLPRegressor models with different architectures
    """
    return [
        MLPRegressor(hidden_layer_sizes=(256, 64), random_state=42),
        MLPRegressor(hidden_layer_sizes=(512, 128), random_state=42),
        MLPRegressor(hidden_layer_sizes=(1024, 256), random_state=42),
        MLPRegressor(hidden_layer_sizes=(512,), random_state=42),
        MLPRegressor(hidden_layer_sizes=(512, 512), random_state=42),
        MLPRegressor(hidden_layer_sizes=(512, 512, 512), random_state=42)
    ]


def get_all_models():
    """
    Get dictionary of all models to be tested.
    
    Returns:
        dict: Dictionary mapping model names to model instances
    """
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'Bagging': BaggingRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'SVM': SVR(),
        'KNN': KNeighborsRegressor(),
        'KernelRidge': KernelRidge(),
        'Spline': make_pipeline(SplineTransformer(), LinearRegression())
    }

    # Add MLP models
    for i, mlp in enumerate(create_mlp_models(), 1):
        models[f'MLP_{i}'] = mlp

    return models


def run_single_experiment_mae(data_df, embedding_type, target_name, seed):
    """
    Run a single experiment iteration for MAE calculation.
    
    Args:
        data_df (pd.DataFrame): Dataset with embeddings
        embedding_type (str): Type of molecular representation
        target_name (str): Target property name
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Results for all models in this iteration
    """
    np.random.seed(seed)
    
    # Sample test and train sets
    test_set = data_df.sample(n=1)
    train_set = data_df.drop(test_set.index).sample(n=50)

    # Prepare features and targets
    X_train = np.stack(train_set[embedding_type].values)
    X_test = np.stack(test_set[embedding_type].values)
    y_train = train_set[target_name].values
    y_test = test_set[target_name].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize results
    results = {}
    
    # Train and evaluate models
    models = get_all_models()
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = mae
        except Exception as e:
            print(f"Error with model {name}: {e}")
            results[name] = np.nan

    # Add baseline predictions
    mean_pred = np.mean(y_train)
    random_pred = np.random.choice(y_train)
    last_value_pred = y_train[-1]

    results['Mean_Baseline'] = mean_absolute_error(y_test, [mean_pred])
    results['Random_Baseline'] = mean_absolute_error(y_test, [random_pred])
    results['Last_Value_Baseline'] = mean_absolute_error(y_test, [last_value_pred])

    return results


def run_experiment_mae(data_df, embedding_type, target_name='LogP', n_iterations=100):
    """
    Run complete experiment with multiple iterations for MAE calculation.
    
    Args:
        data_df (pd.DataFrame): Dataset with embeddings
        embedding_type (str): Type of molecular representation
        target_name (str): Target property name
        n_iterations (int): Number of experiment iterations
        
    Returns:
        pd.DataFrame: Summary statistics for all models
    """
    print(f"\nRunning MAE experiment with {embedding_type}...")
    
    # Initialize results storage
    all_results = {model_name: [] for model_name in get_all_models().keys()}
    all_results.update({
        'Mean_Baseline': [],
        'Random_Baseline': [],
        'Last_Value_Baseline': []
    })

    # Run iterations
    for seed in range(1, n_iterations + 1):
        if seed % 10 == 0:
            print(f"Processing iteration {seed}/{n_iterations}")

        iteration_results = run_single_experiment_mae(data_df, embedding_type, target_name, seed)
        
        # Store results
        for model_name, mae_value in iteration_results.items():
            all_results[model_name].append(mae_value)

    # Calculate summary statistics
    summary = pd.DataFrame({
        'Mean_MAE': {k: np.nanmean(v) for k, v in all_results.items()},
        'Std_MAE': {k: np.nanstd(v) for k, v in all_results.items()},
        'Median_MAE': {k: np.nanmedian(v) for k, v in all_results.items()}
    })

    return summary.sort_values('Mean_MAE')


def run_experiment_r2(embedding_type, target_name, df, n_iterations=100, train_size=50):
    """
    Run experiment calculating R² over all 100 iterations combined.
    
    Args:
        embedding_type (str): Type of molecular representation
        target_name (str): Target property name
        df (pd.DataFrame): Dataset with embeddings
        n_iterations (int): Number of experiment iterations
        train_size (int): Training set size
        
    Returns:
        pd.DataFrame: R² results for all models
    """
    print(f"\nRunning R² experiment with {embedding_type} for target '{target_name}'...")

    # Storage for predictions and true values
    predictions_store = {model_name: [] for model_name in get_all_models().keys()}
    true_store = {model_name: [] for model_name in get_all_models().keys()}

    # Baseline storage
    for baseline_name in ['Mean_Baseline', 'Random_Baseline', 'Last_Value_Baseline']:
        predictions_store[baseline_name] = []
        true_store[baseline_name] = []

    # Run iterations
    for seed in range(1, n_iterations + 1):
        if seed % 10 == 0:
            print(f"  Iteration {seed}/{n_iterations}...")

        np.random.seed(seed)

        # Sampling
        test_set = df.sample(n=1)
        train_set = df.drop(test_set.index).sample(n=train_size)

        X_train = np.stack(train_set[embedding_type].values)
        X_test = np.stack(test_set[embedding_type].values)

        y_train = train_set[target_name].values
        y_test = test_set[target_name].values

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model training and prediction
        models = get_all_models()
        for model_name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                predictions_store[model_name].append(y_pred[0])
                true_store[model_name].append(y_test[0])

            except:
                predictions_store[model_name].append(np.nan)
                true_store[model_name].append(np.nan)

        # Baselines
        mean_pred = np.mean(y_train)
        random_pred = np.random.choice(y_train)
        last_pred = y_train[-1]

        predictions_store['Mean_Baseline'].append(mean_pred)
        predictions_store['Random_Baseline'].append(random_pred)
        predictions_store['Last_Value_Baseline'].append(last_pred)

        true_store['Mean_Baseline'].append(y_test[0])
        true_store['Random_Baseline'].append(y_test[0])
        true_store['Last_Value_Baseline'].append(y_test[0])

    # Calculate R² for all models
    r2_dict = {}
    for model_name in predictions_store.keys():
        y_true_all = np.array(true_store[model_name])
        y_pred_all = np.array(predictions_store[model_name])

        # Remove NaN values
        valid_idx = (~np.isnan(y_true_all)) & (~np.isnan(y_pred_all))
        if valid_idx.sum() > 1:
            r2_val = r2_score(y_true_all[valid_idx], y_pred_all[valid_idx])
        else:
            r2_val = np.nan

        r2_dict[model_name] = r2_val

    # Return sorted DataFrame
    r2_df = pd.DataFrame({'R2': r2_dict}).sort_values('R2', ascending=False)
    return r2_df


def run_all_experiments(data_df, target_name='LogP', experiment_type='mae'):
    """
    Run experiments for all embedding types.
    
    Args:
        data_df (pd.DataFrame): Dataset with all embeddings
        target_name (str): Target property name
        experiment_type (str): 'mae' or 'r2'
        
    Returns:
        dict: Results for each embedding type
    """
    results_dict = {}
    embedding_types = ['ecfp', 'rdkit', 'maccs', 'chemberta']
    
    for embedding in embedding_types:
        print(f"\n{embedding.upper()} Results:")
        
        if experiment_type == 'mae':
            results = run_experiment_mae(data_df, embedding, target_name)
        elif experiment_type == 'r2':
            results = run_experiment_r2(embedding, target_name, data_df)
        else:
            raise ValueError("experiment_type must be 'mae' or 'r2'")
            
        print(results)
        results_dict[embedding] = results
    
    return results_dict


def add_llm_results_to_dict(results_dict, llm_results, experiment_type='mae'):
    """
    Add LLM results to existing results dictionary.
    
    Args:
        results_dict (dict): Existing ML results
        llm_results (dict): LLM results to add
        experiment_type (str): 'mae' or 'r2'
    """
    for key in results_dict:
        if experiment_type == 'mae':
            # Add MAE results
            for llm_name, llm_data in llm_results.items():
                results_dict[key].loc[llm_name] = llm_data
        elif experiment_type == 'r2':
            # Add R² results
            for llm_name, r2_value in llm_results.items():
                results_dict[key].loc[llm_name] = {'R2': r2_value}
        
        # Re-sort results
        if experiment_type == 'mae':
            results_dict[key] = results_dict[key].sort_values('Mean_MAE')
        elif experiment_type == 'r2':
            results_dict[key] = results_dict[key].sort_values('R2', ascending=False)
    
    return results_dict


def run_all_tasks_experiments(data_df, experiment_type='mae', tasks=None):
    """
    Run experiments for all tasks and embedding types.
    
    Args:
        data_df (pd.DataFrame): Dataset with all embeddings
        experiment_type (str): 'mae' or 'r2'
        tasks (list): List of target properties to test. If None, uses all from config.
        
    Returns:
        dict: Results for each task and embedding type
    """
    from config import DATASET_CONFIG, TASK_NAME_MAPPING
    
    if tasks is None:
        tasks = DATASET_CONFIG['target_properties']
    
    all_task_results = {}
    
    for task in tasks:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS FOR TASK: {task}")
        print(f"{'='*80}")
        
        # Check if task exists in dataset
        if task not in data_df.columns:
            print(f"Warning: Task '{task}' not found in dataset. Skipping...")
            continue
        
        # Run experiments for this task
        task_results = run_all_experiments(data_df, target_name=task, experiment_type=experiment_type)
        
        # Store results
        task_key = TASK_NAME_MAPPING.get(task, task.lower().replace(' ', '_'))
        all_task_results[task_key] = {
            'task_name': task,
            'results': task_results
        }
        
        # Save individual task results
        save_task_results(task_results, task_key, experiment_type)
    
    return all_task_results


def save_task_results(task_results, task_key, experiment_type):
    """
    Save results for a specific task.
    
    Args:
        task_results (dict): Results for the task
        task_key (str): Task key for file naming
        experiment_type (str): 'mae' or 'r2'
    """
    import os
    from config import TASK_OUTPUT_PATTERNS
    
    os.makedirs('Results', exist_ok=True)
    
    if experiment_type == 'mae':
        filename = TASK_OUTPUT_PATTERNS['mae_results'].format(task=task_key)
    elif experiment_type == 'r2':
        filename = TASK_OUTPUT_PATTERNS['r2_results'].format(task=task_key)
    else:
        raise ValueError("experiment_type must be 'mae' or 'r2'")
    
    with open(filename, 'wb') as f:
        pickle.dump(task_results, f)
    
    print(f"Results saved to {filename}")


def add_llm_results_for_all_tasks(experiment_type='mae'):
    """
    Add LLM results to all task results.
    
    Args:
        experiment_type (str): 'mae' or 'r2'
    """
    from config import TASK_NAME_MAPPING, TASK_OUTPUT_PATTERNS
    import os
    
    # Define LLM results for each task
    if experiment_type == 'mae':
        # Example LLM results - you'll need to replace these with actual values
        llm_results_by_task = {
            'logp': {
                'GPT-4o w/ no hint': {'Mean_MAE': 0.8776471, 'Std_MAE': 0.8459694971, 'Median_MAE': 0.557},
                'Claude 3.5 Sonnet w/ no hint': {'Mean_MAE': 0.6074995, 'Std_MAE': 0.6460814817, 'Median_MAE': 0.41275}
            },
            'mw': {
                'GPT-4o w/ no hint': {'Mean_MAE': 20.91047, 'Std_MAE': 32.50428147, 'Median_MAE': 12.1805},
                'Claude 3.5 Sonnet w/ no hint': {'Mean_MAE': 6.76517, 'Std_MAE': 12.67346291, 'Median_MAE': 0.01}
            },
            # Add more tasks as needed...
        }
    elif experiment_type == 'r2':
        llm_results_by_task = {
            'logp': {
                'GPT-4o w/o hint': 0.6259825777,
                'Claude 3.5 Sonnet w/o hint': 0.8021366322
            },
            # Add more tasks as needed...
        }
    
    # Update each task's results
    for task_key in llm_results_by_task:
        if experiment_type == 'mae':
            filename = TASK_OUTPUT_PATTERNS['mae_results'].format(task=task_key)
        else:
            filename = TASK_OUTPUT_PATTERNS['r2_results'].format(task=task_key)
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                task_results = pickle.load(f)
            
            # Add LLM results
            task_results = add_llm_results_to_dict(
                task_results, 
                llm_results_by_task[task_key], 
                experiment_type
            )
            
            # Save updated results
            with open(filename, 'wb') as f:
                pickle.dump(task_results, f)
            
            print(f"Updated {filename} with LLM results")


if __name__ == "__main__":
    from data_preprocessing import prepare_complete_dataset
    from config import DATASET_CONFIG
    import os
    
    # Create Results directory
    os.makedirs('Results', exist_ok=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_complete_dataset()
    
    # Run experiments for all tasks
    print("Running MAE experiments for all tasks...")
    mae_results = run_all_tasks_experiments(
        dataset, 
        experiment_type='mae', 
        tasks=DATASET_CONFIG['target_properties']
    )
    
    print("\nRunning R² experiments for all tasks...")
    r2_results = run_all_tasks_experiments(
        dataset, 
        experiment_type='r2', 
        tasks=DATASET_CONFIG['target_properties']
    )
    
    # Add LLM results (you'll need to update the LLM results in the function above)
    print("\nAdding LLM results...")
    add_llm_results_for_all_tasks('mae')
    add_llm_results_for_all_tasks('r2')
    
    print("\nAll experiments completed for all tasks!")
