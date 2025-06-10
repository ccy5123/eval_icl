# eval_icl

# Supporting Information - README

## Overview

This directory contains supporting files for the paper:

**"Evaluating In-Context Learning in Large Language Models for Molecular Property Regression"**

It includes raw outputs from large language models (LLMs) and aggregated performance results from various experiments.

---

## Folder Structure

├── GPT_Response/ # GPT-4o model outputs for all property prediction tasks
├── Claude_Response/ # Claude 3.5 Sonnet outputs for all property prediction tasks
├── Results/ # Aggregated performance results (MAE, R², rankings, etc.)
└── README.md # This file


---

## Contents

### `GPT_Response/`
- Contains the full response logs from **GPT-4o** for each molecular property prediction task.
- Each file corresponds to a specific task (e.g., `gpt_mw.txt`, `gpt_logp.txt`, `gpt_amb.txt`, etc.).
- The responses include:
  - In-context examples used in the prompt
  - Model-generated predictions
  - Optional reasoning or explanation text

### `Claude_Response/`
- Same format and content structure as `GPT_Response/`, but for **Claude 3.5 Sonnet**.
- Useful for analyzing differences in reasoning strategies or error patterns between LLMs.

### `Results/`
- Contains processed results across **100 trials per task**.
- Includes:
  - Mean Absolute Error (MAE)
  - Coefficient of Determination (R²)

---

## Notes

- All molecular property predictions are based on the ESOL dataset (Delaney, 2004).
- The SMILES input and true target values used in prompts are kept consistent across models for fair comparison.
- Random seed was fixed for all trials to ensure reproducibility.
- LLM outputs were not manually altered or filtered, except for minor formatting where needed for readability.

---

## Citation

If you use this data, please cite:

> Joe, C. Y., Song, K., & Chang, R. (2025). *Evaluating In-Context Learning in Large Language Models for Molecular Property Regression*. [Journal Name], [DOI].

---

For questions, please contact:  
**Kyungwoo Song** (kyungwoo.song@yonsei.ac.kr)  
**Rakwoo Chang** (rchang90@uos.ac.kr)
