"""
Machine learning experiments for molecular property prediction.

This module implements various ML models and runs comparative experiments
using different molecular representations.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import mean_absolute_error
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


def run_single_experiment(data_df, embedding_type, target_name, seed):
    """
    Run a single experiment iteration.
    
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


def run_experiment(data_df, embedding_type, target_name='LogP', n_iterations=100):
    """
    Run complete experiment with multiple iterations.
    
    Args:
        data_df (pd.DataFrame): Dataset with embeddings
        embedding_type (str): Type of molecular representation
        target_name (str): Target property name
        n_iterations (int): Number of experiment iterations
        
    Returns:
        pd.DataFrame: Summary statistics for all models
    """
    print(f"\nRunning experiment with {embedding_type}...")
    
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

        iteration_results = run_single_experiment(data_df, embedding_type, target_name, seed)
        
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


def run_all_experiments(data_df, target_name='LogP'):
    """
    Run experiments for all embedding types.
    
    Args:
        data_df (pd.DataFrame): Dataset with all embeddings
        target_name (str): Target property name
        
    Returns:
        dict: Results for each embedding type
    """
    results_dict = {}
    embedding_types = ['ecfp', 'rdkit', 'maccs', 'chemberta']
    
    for embedding in embedding_types:
        print(f"\n{embedding.upper()} Results:")
        results = run_experiment(data_df, embedding, target_name)
        print(results)
        results_dict[embedding] = results
    
    return results_dict


if __name__ == "__main__":
    from data_preprocessing import prepare_complete_dataset
    import os
    
    # Create Results directory
    os.makedirs('Results', exist_ok=True)
    
    # Prepare dataset
    dataset = prepare_complete_dataset()
    
    # Run all experiments
    all_results = run_all_experiments(dataset)
    
    # Save results in Results/ directory
    for embedding_type, results in all_results.items():
        results.to_csv(f'Results/{embedding_type}_ml_results.csv')
    
    print("\nAll experiments completed!")
