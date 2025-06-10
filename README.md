# Supporting Information - eval_icl

## Overview

This repository contains the supporting code and data for the paper:
**"Evaluating In-Context Learning in Large Language Models for Molecular Property Regression"**

The repository implements comprehensive experiments comparing various methods for predicting molecular properties from the ESOL dataset, including:
- Traditional machine learning approaches with different molecular representations
- Large language models using in-context learning
- Raw LLM outputs and aggregated performance results

## Repository Structure

```
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration parameters (9 tasks defined)
├── data_preprocessing.py       # Data loading and feature extraction
├── ml_experiments.py          # Traditional ML model experiments (MAE & R²)
├── llm_experiments.py         # GPT and Claude prediction experiments
├── visualization.py           # Plotting and visualization functions
├── utils.py                   # Utility functions and result analysis
├── run_experiments.py         # Main script to run all experiments
├── delaney-processed.csv      # ESOL dataset (not included, see Dataset section)
├── GPT_Response/              # GPT-4o outputs for all 9 molecular property tasks
├── Claude_Response/           # Claude 3.5 Sonnet outputs for all 9 molecular property tasks
└── Results/                   # Pickle files and CSV summaries for all 9 tasks
    ├── results_dict_{task}.pkl # MAE results for each task
    ├── r2_results_{task}.pkl   # R² results for each task
    ├── {task}_mae_summary.csv  # MAE summary statistics for each task
    └── {task}_r2_summary.csv   # R² summary statistics for each task
```

*Note: `{task}` represents: logp, mw, tpsa, sp3, molmr, bj, chi, hka, amb (9 tasks total)*

## Contents

### Code Files
- **Core Scripts**: Complete implementation for reproducing all experiments
- **Configuration**: Modular settings for easy customization
- **Analysis Tools**: Utilities for processing and visualizing results

### Data and Results
- **`GPT_Response/`**: Contains the full response logs from **GPT-4o** for each molecular property prediction task
  - Files: `gpt_logp_results.txt`, `gpt_mw_results.txt`, `gpt_tpsa_results.txt`, etc.
  - Includes in-context examples, model predictions, and reasoning text
  
- **`Claude_Response/`**: Same format as GPT_Response/, but for **Claude 3.5 Sonnet**
  - Files: `claude_logp_results.txt`, `claude_mw_results.txt`, `claude_tpsa_results.txt`, etc.
  - Useful for analyzing differences in reasoning strategies between LLMs
  
- **`Results/`**: Contains processed results across **100 trials per task**
  - Pickle files: `results_dict_{task}.pkl` (MAE), `r2_results_{task}.pkl` (R²)
  - Summary CSV files: `{task}_mae_summary.csv`, `{task}_r2_summary.csv`
  - Statistical analysis and performance comparisons for all 9 tasks

## Dataset

The experiments use the **ESOL (Estimated SOLubility) dataset** based on Delaney (2004). This dataset contains:
- SMILES representations of molecules
- Experimental solubility values and other molecular properties
- Molecular descriptors for various prediction tasks

The following **9 molecular properties** are predicted in our experiments:
1. **LogP** - Octanol-water partition coefficient
2. **Molecular Weight** - Molecular weight
3. **TPSA** - Topological polar surface area
4. **sp3** - Fraction of SP3 carbons
5. **MolMR** - Molecular refractivity
6. **BJ** - Balaban J index
7. **Chi** - Chi1v connectivity index
8. **HKA** - Hall-Kier alpha
9. **aM_w+b** - Synthetic property (linear function of molecular weight)

**Note**: The dataset file (`delaney-processed.csv`) is not included in this repository. Please obtain it from the original source and place it in the root directory.

## Experimental Design

- **Training Set Size**: 50 molecules per iteration
- **Test Set Size**: 1 molecule per iteration  
- **Number of Iterations**: 100 (for statistical significance)
- **Target Properties**: LogP, molecular weight, and other molecular descriptors
- **Evaluation Metrics**: Mean Absolute Error (MAE), R²
- **Reproducibility**: Fixed random seeds across all trials
- **LLM Consistency**: SMILES input and true target values kept consistent across models

## Methods Compared

### Traditional Machine Learning Models
- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-based**: Random Forest, Gradient Boosting, XGBoost, AdaBoost
- **Instance-based**: K-Nearest Neighbors, Support Vector Regression
- **Neural Networks**: Multi-layer Perceptrons with various architectures
- **Other**: Kernel Ridge Regression, Spline Regression
- **Baselines**: Mean, Random, Last-value predictors

### Molecular Representations
- **ECFP**: Extended Connectivity Fingerprints (Morgan fingerprints)
- **RDKit**: RDKit structural fingerprints
- **MACCS**: Molecular ACCess System keys
- **ChemBERTa**: Transformer-based molecular embeddings

### Large Language Models
- **GPT-4o**: OpenAI's latest GPT model
- **Claude-3.5-Sonnet**: Anthropic's Claude model

## Installation and Setup

1. **Clone the repository**:
```bash
git clone [repository-url]
cd molecular-property-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Obtain the dataset**:
   - Download `delaney-processed.csv` from the original source
   - Place it in the root directory of this repository

## Usage

## Usage Examples

### 1. Run ML experiments for all 9 tasks (recommended first run)
```bash
python run_experiments.py --ml-only
```
This will:
- Generate pickle files for all 9 tasks (`Results/results_dict_{task}.pkl`, `Results/r2_results_{task}.pkl`)
- Create summary CSV files for each task
- Show visualizations automatically for all tasks

### 2. Run experiments for specific tasks only
```bash
python run_experiments.py --ml-only --tasks LogP "Molecular Weight" TPSA
```

### 3. View visualizations from saved results
```bash
python visualization.py
```
This will load all available task results and create visualizations.

### 4. Visualize a single task
```bash
python -c "from visualization import visualize_single_task; visualize_single_task('logp')"
```

### 5. Complete experiments with LLMs for all tasks
```bash
python run_experiments.py --openai-key YOUR_OPENAI_KEY --anthropic-key YOUR_ANTHROPIC_KEY
```

### 6. LLM experiments for specific tasks only
```bash
python run_experiments.py --openai-key YOUR_KEY --tasks LogP TPSA
```

### 7. Use custom dataset
```bash
python run_experiments.py --dataset-path your_dataset.csv --ml-only
```

### Command Line Options
- `--ml-only`: Run only traditional ML experiments (skip LLM experiments)
- `--dataset-path`: Specify custom dataset file path
- `--openai-key`: OpenAI API key for GPT experiments
- `--anthropic-key`: Anthropic API key for Claude experiments
- `--help-usage`: Show detailed usage instructions

## API Keys for LLM Experiments

To run LLM experiments, you need valid API keys:

1. **OpenAI API Key**: 
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Generate an API key in your account settings

2. **Anthropic API Key**:
   - Sign up at [Anthropic Console](https://console.anthropic.com/)
   - Generate an API key in your account settings

**Security Note**: Never commit API keys to version control. Use environment variables or pass them as command-line arguments.

## Output Files

The experiments generate several output files:

### Traditional ML Results
- `ecfp_ml_results.csv`: ECFP fingerprint results
- `rdkit_ml_results.csv`: RDKit fingerprint results  
- `maccs_ml_results.csv`: MACCS keys results
- `chemberta_ml_results.csv`: ChemBERTa embedding results

### LLM Results
- `gpt_logp_results.txt`: Raw GPT-4o predictions
- `claude_logp_results.txt`: Raw Claude predictions

### Output Files Generated

The experiments generate several types of output files **for each of the 9 tasks**:

### Pickle Files (Main Results)
- `Results/results_dict_{task}.pkl`: Complete MAE results for all models and embeddings
- `Results/r2_results_{task}.pkl`: Complete R² results for all models and embeddings
- Where `{task}` = `logp`, `mw`, `tpsa`, `sp3`, `molmr`, `bj`, `chi`, `hka`, `amb`

### Summary CSV Files (Per Task)
- `Results/{task}_mae_summary.csv`: MAE summary statistics across all methods
- `Results/{task}_r2_summary.csv`: R² summary statistics across all methods

### LLM Raw Outputs (Per Task)
- `GPT_Response/gpt_{task}_results.txt`: Raw GPT-4o predictions (if API key provided)
- `Claude_Response/claude_{task}_results.txt`: Raw Claude predictions (if API key provided)

### Visualizations (Generated automatically for each task)
- Model comparison bar plots for each embedding type and task
- Ranking heatmaps across embedding types for each task
- Cross-task comparison plots showing performance across all 9 properties
- Both MAE and R² based visualizations

## Notes on LLM Data

- All molecular property predictions are based on the ESOL dataset (Delaney, 2004)
- The SMILES input and true target values used in prompts are kept consistent across models for fair comparison
- Random seed was fixed for all trials to ensure reproducibility
- LLM outputs were not manually altered or filtered, except for minor formatting where needed for readability
- Responses include both numerical predictions and optional reasoning/explanation text

## Customization

### Adding New Models
Edit `ml_experiments.py` and add your model to the `get_all_models()` function:

```python
def get_all_models():
    models = {
        # ... existing models
        'YourModel': YourModelClass()
    }
    return models
```

### Modifying LLM Prompts
Edit the `create_prompt()` method in `llm_experiments.py` to test different prompt strategies.

### Changing Configuration
Modify parameters in `config.py` to adjust experiment settings.

## Results Analysis

Use the utility functions to analyze results:

```python
from utils import parse_llm_results, create_results_summary, plot_results_comparison

# Parse LLM results
gpt_results = parse_llm_results('gpt_logp_results.txt')
claude_results = parse_llm_results('claude_logp_results.txt')

# Create comprehensive summary
summary = create_results_summary(ml_results, {'gpt': gpt_results, 'claude': claude_results})

# Generate comparison plot
plot_results_comparison(summary, 'comparison_plot.png')
```

## Citation

If you use this code or data, please cite our paper:

```bibtex
@article{joe2025eval_icl,
    title={Evaluating In-Context Learning in Large Language Models for Molecular Property Regression},
    author={Joe, C. Y. and Song, Kyungwoo and Chang, Rakwoo},
    journal={[Journal Name]},
    year={2025},
    doi={[DOI]}
}
```

## Contact

For questions, please contact:  
- **Kyungwoo Song**: kyungwoo.song@yonsei.ac.kr  
- **Rakwoo Chang**: rchang90@uos.ac.kr
