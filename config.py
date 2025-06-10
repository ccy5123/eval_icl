"""
Configuration file for molecular property prediction experiments.

This file contains all configurable parameters and settings for the experiments.
"""

# Dataset configuration
DATASET_CONFIG = {
    'default_csv_path': 'delaney-processed.csv',
    'target_properties': [
        'aM_w+b',         # LTMW - Linear transform of molecular weight
        'LogP',           # logP - Octanol-water partition coefficient  
        'sp3',            # sp3 - Fraction of SP3 carbons
        'TPSA',           # TPSA - Topological polar surface area
        'MolMR',          # MR - Molecular refractivity
        'HKA',            # HKA - Hall-Kier alpha
        'BJ',             # BJ - Balaban J index
        'Chi',            # Chi1v - Chi1v connectivity index
        'Molecular Weight' # MW - Molecular weight
    ],
    'test_size': 1,
    'train_size': 50,
    'random_seed_range': (1, 101)  # 100 iterations
}

# Task-specific output file patterns
TASK_OUTPUT_PATTERNS = {
    'mae_results': 'Results/results_dict_{task}.pkl',
    'r2_results': 'Results/r2_results_{task}.pkl',
    'gpt_results': 'GPT_Response/gpt_{task}_results.txt',
    'claude_results': 'Claude_Response/claude_{task}_results.txt',
    'mae_summary': 'Results/{task}_mae_summary.csv',
    'r2_summary': 'Results/{task}_r2_summary.csv'
}

# Task name mapping for file naming (in new order)
TASK_NAME_MAPPING = {
    'aM_w+b': 'ltmw',           # LTMW - Linear transform of molecular weight
    'LogP': 'logp',             # logP - Octanol-water partition coefficient
    'sp3': 'sp3',               # sp3 - Fraction of SP3 carbons
    'TPSA': 'tpsa',             # TPSA - Topological polar surface area
    'MolMR': 'mr',              # MR - Molecular refractivity
    'HKA': 'hka',               # HKA - Hall-Kier alpha
    'BJ': 'bj',                 # BJ - Balaban J index
    'Chi': 'chi1v',             # Chi1v - Chi1v connectivity index
    'Molecular Weight': 'mw'     # MW - Molecular weight
}

# Synthetic property parameters (for validation)
SYNTHETIC_PARAMS = {
    'a': 6.462356980390821,
    'b': -162.75140504630065
}

# Molecular fingerprint configuration
FINGERPRINT_CONFIG = {
    'ecfp_radius': 2,
    'ecfp_size': 2048,
    'rdkit_size': 2048,
    'chemberta_model': "seyonec/ChemBERTa-zinc-base-v1"
}

# Machine learning model configuration
ML_CONFIG = {
    'random_state': 42,
    'n_iterations': 100,
    'mlp_architectures': [
        (256, 64),
        (512, 128),
        (1024, 256),
        (512,),
        (512, 512),
        (512, 512, 512)
    ]
}

# LLM configuration
LLM_CONFIG = {
    'gpt_model': "gpt-4o",
    'claude_model': "claude-3-5-sonnet-20241022",
    'temperature': 0,
    'top_p': 1,
    'max_tokens': 1000,
    'prompt_types': [
        'no_hint',
        'label_hint', 
        'smiles_hint',
        'function_hint',
        'linear_hint',
        'all_hint'
    ]
}

# Output configuration
OUTPUT_CONFIG = {
    'ml_results_pattern': 'Results/{embedding}_ml_results.csv',
    'gpt_results_file': 'GPT_Response/gpt_logp_results.txt',
    'claude_results_file': 'Claude_Response/claude_logp_results.txt',
    'encoding': 'utf-8',
    'results_dir': 'Results',
    'gpt_response_dir': 'GPT_Response',
    'claude_response_dir': 'Claude_Response'
}

# Environment configuration
ENV_CONFIG = {
    'disable_tokenizer_parallelism': True,
    'suppress_warnings': True,
    'random_seed': 42
}
