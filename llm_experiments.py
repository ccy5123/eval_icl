"""
Large Language Model experiments for molecular property prediction.

This module implements GPT and Claude experiments for predicting molecular
properties using in-context learning.
"""

import os
import pandas as pd
import random
from openai import OpenAI
from anthropic import Anthropic

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LLMPredictor:
    """Base class for LLM prediction experiments."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        
    def create_prompt(self, smiles, example_data, hint_type="no_hint"):
        """
        Create prompt for LLM prediction.
        
        Args:
            smiles (str): Target SMILES string
            example_data (list): List of [smiles, property_value] pairs
            hint_type (str): Type of hint to include in prompt
            
        Returns:
            str: Formatted prompt
        """
        # Format example data
        example_str = "\n".join([f"{s}, {round(value, 8)}" for s, value in example_data])
        
        # Define prompt templates
        prompts = {
            "no_hint": f"""You are an experienced chemist with expertise in molecular structures. Using only your knowledge and without employing any external tools or code, predict the property for the following molecules. Below are examples of molecules and known property value:\n\n{example_str}\n\nNow, based on these examples, predict the property for the following molecule:\n\n{smiles}\n\nPlease provide the predicted specific property value!""",
            
            "label_hint": f"""You are an experienced chemist with expertise in molecular structures. Using only your knowledge and without employing any external tools or code, predict the molecular weight for the following molecules. Below are examples of molecules and known molecular weights:\n\n{example_str}\n\nNow, based on these examples, predict the molecular weight for the following molecule:\n\n{smiles}\n\nPlease provide the predicted specific molecular weight value!""",
            
            "smiles_hint": f"""You are an experienced chemist with expertise in molecular structures. Using only your knowledge and without employing any external tools or code, predict the property for the following molecules. Below are examples of molecules in SMILES format and known property value:\n\n{example_str}\n\nNow, based on these examples, predict the property for the following molecule given in SMILES format:\n\n{smiles}\n\nPlease provide the predicted specific property value!""",
            
            "function_hint": f"""You are an experienced chemist with expertise in molecular structures. Using only your knowledge and without employing any external tools or code, predict the property for the following molecules. Below are examples of molecules in SMILES format and known property values. Note that the property is a function of molecular weight, f(M.W.):\n\n{example_str}\n\nNow, based on these examples, predict the property for the following molecule given in SMILES format:\n\n{smiles}\n\nPlease provide the predicted specific property value!""",
            
            "linear_hint": f"""You are an experienced chemist with expertise in molecular structures. Using only your knowledge and without employing any external tools or code, predict the property for the following molecules. Below are examples of molecules in SMILES format and known property values. Note that the property is represented as a linear function of molecular weight, specifically a * M.W. + b:\n\n{example_str}\n\nNow, based on these examples, predict the property for the following molecule given in SMILES format:\n\n{smiles}\n\nPlease provide the predicted specific property value!""",
            
            "all_hint": f"""You are an experienced chemist with expertise in molecular structures. Using only your knowledge and without employing any external tools or code, predict the molecular weight for the following molecules. Below are examples of molecules in SMILES format and their known molecular weights:\n\n{example_str}\n\nNow, based on these examples, predict the molecular weight for the following molecule given in SMILES format:\n\n{smiles}\n\nPlease provide the predicted specific molecular weight value!"""
        }
        
        return prompts.get(hint_type, prompts["no_hint"])

    def preview_prompts(self, data_df, property_name, n_previews=3):
        """
        Preview prompts without making API calls.
        
        Args:
            data_df (pd.DataFrame): Dataset
            property_name (str): Name of target property
            n_previews (int): Number of prompts to preview
        """
        for seed in range(1, n_previews + 1):
            test_set = data_df.sample(n=1, random_state=seed)
            train_set = data_df.drop(test_set.index).sample(n=50, random_state=seed, replace=False)
            
            smiles_test = test_set['smiles'].values[0]
            example_data = train_set[['smiles', property_name]].values.tolist()
            
            print(f"\n--- Prompt Preview {seed} ---")
            print(self.create_prompt(smiles_test, example_data))
            print(f"True value: {test_set[property_name].values[0]}")


class GPTPredictor(LLMPredictor):
    """GPT-based molecular property predictor."""
    
    def __init__(self, api_key):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
    
    def predict(self, smiles, example_data):
        """
        Make prediction using GPT.
        
        Args:
            smiles (str): Target SMILES string
            example_data (list): Training examples
            
        Returns:
            str: GPT prediction response
        """
        prompt = self.create_prompt(smiles, example_data)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        return response.choices[0].message.content.strip()


class ClaudePredictor(LLMPredictor):
    """Claude-based molecular property predictor."""
    
    def __init__(self, api_key):
        super().__init__(api_key)
        self.client = Anthropic(api_key=api_key)
    
    def predict(self, smiles, example_data):
        """
        Make prediction using Claude.
        
        Args:
            smiles (str): Target SMILES string
            example_data (list): Training examples
            
        Returns:
            str: Claude prediction response
        """
        prompt = self.create_prompt(smiles, example_data)
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            max_tokens=1000
        )
        
        return response.content[0].text.strip()


def run_llm_experiment(predictor, data_df, filename, property_name, n_iterations=100):
    """
    Run LLM prediction experiment.
    
    Args:
        predictor: LLM predictor instance (GPT or Claude)
        data_df (pd.DataFrame): Dataset
        filename (str): Output filename
        property_name (str): Target property name
        n_iterations (int): Number of iterations
    """
    # Create output directory if it doesn't exist
    import os
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for seed in range(1, n_iterations + 1):
            print(f"Processing iteration {seed}/{n_iterations}")
            
            # Sample data
            test_set = data_df.sample(n=1, random_state=seed)
            train_set = data_df.drop(test_set.index).sample(n=50, random_state=seed, replace=False)
            
            smiles_test = test_set['smiles'].values[0]
            true_value = test_set[property_name].values[0]
            example_data = train_set[['smiles', property_name]].values.tolist()
            
            # Make prediction
            try:
                predicted_property = predictor.predict(smiles_test, example_data)
                print('Prediction successful')
            except Exception as e:
                predicted_property = f"Error: {str(e)}"
                print(f'Error in prediction: {e}')
            
            # Save results
            f.write(f"Iteration: {seed}\n")
            f.write(f"SMILES: {smiles_test}\n")
            f.write(f"True Property: {true_value}\n")
            f.write("Predicted Property:\n")
            f.write(f"{predicted_property}\n")
            f.write("="*50 + "\n")
            f.flush()  # Ensure data is written immediately
            
    print(f"Experiment completed. Results saved to {filename}")


def main():
    """Main function to run LLM experiments."""
    # Load data
    from data_preprocessing import load_and_prepare_data
    
    print("Loading dataset...")
    dataset = load_and_prepare_data()
    
    # API Keys (replace with your actual keys)
    OPENAI_API_KEY = "your_openai_api_key_here"
    ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
    
    # Initialize predictors
    gpt_predictor = GPTPredictor(OPENAI_API_KEY)
    claude_predictor = ClaudePredictor(ANTHROPIC_API_KEY)
    
    # Preview prompts (optional)
    print("\nPreviewing GPT prompts...")
    gpt_predictor.preview_prompts(dataset, 'LogP', n_previews=2)
    
    # Run experiments
    print("\n" + "="*50)
    print("Starting GPT experiment...")
    run_llm_experiment(gpt_predictor, dataset, "GPT_Response/gpt_logp_results.txt", 'LogP')
    
    print("\n" + "="*50)
    print("Starting Claude experiment...")
    run_llm_experiment(claude_predictor, dataset, "Claude_Response/claude_logp_results.txt", 'LogP')
    
    print("\nAll LLM experiments completed!")


if __name__ == "__main__":
    main()
