"""
Data preprocessing and feature extraction for molecular property prediction.

This module handles loading the ESOL dataset and computing various molecular
descriptors and fingerprints for machine learning experiments.
"""

import pandas as pd
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from transformers import AutoTokenizer, TFAutoModel
import os

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_and_prepare_data(csv_path='delaney-processed.csv'):
    """
    Load ESOL dataset and compute basic molecular descriptors.
    
    Args:
        csv_path (str): Path to the ESOL dataset CSV file
        
    Returns:
        pd.DataFrame: DataFrame with original data and computed descriptors
    """
    # Random parameters for synthetic property generation
    a = 6.462356980390821
    b = -162.75140504630065
    
    # Load data
    esol = pd.read_csv(csv_path)
    
    # Add basic descriptors
    esol['smiles_len'] = esol['smiles'].apply(len)
    esol['aM_w+b'] = a * esol.loc[:, 'Molecular Weight'] + b
    
    # Compute RDKit descriptors
    esol['LogP'] = esol['smiles'].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)))
    esol['TPSA'] = esol['smiles'].apply(lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x)))
    esol['sp3'] = esol['smiles'].apply(lambda x: Descriptors.FractionCSP3(Chem.MolFromSmiles(x)))
    esol['MolMR'] = esol['smiles'].apply(lambda x: Descriptors.MolMR(Chem.MolFromSmiles(x)))
    esol['BJ'] = esol['smiles'].apply(lambda x: Descriptors.BalabanJ(Chem.MolFromSmiles(x)))
    esol['Chi'] = esol['smiles'].apply(lambda x: Descriptors.Chi1v(Chem.MolFromSmiles(x)))
    esol['HKA'] = esol['smiles'].apply(lambda x: Descriptors.HallKierAlpha(Chem.MolFromSmiles(x)))
    
    print(f"Parameters used: a={a}, b={b}")
    return esol


def compute_molecular_fingerprints(df):
    """
    Compute various molecular fingerprints for the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing SMILES strings
        
    Returns:
        pd.DataFrame: DataFrame with added fingerprint columns
    """
    print("Computing molecular fingerprints...")
    
    # ECFP fingerprints
    print("Computing ECFP fingerprints...")
    ecfp_list = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        ecfp_list.append(fp)
    df['ecfp'] = ecfp_list

    # RDKit fingerprints
    print("Computing RDKit fingerprints...")
    rdkit_list = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        fp = np.array(Chem.RDKFingerprint(mol, fpSize=2048))
        rdkit_list.append(fp)
    df['rdkit'] = rdkit_list

    # MACCS fingerprints
    print("Computing MACCS fingerprints...")
    maccs_list = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        fp = np.array(MACCSkeys.GenMACCSKeys(mol))
        maccs_list.append(fp)
    df['maccs'] = maccs_list

    return df


def compute_chemberta_embeddings(df):
    """
    Compute ChemBERTa embeddings for molecules.
    
    Args:
        df (pd.DataFrame): DataFrame containing SMILES strings
        
    Returns:
        pd.DataFrame: DataFrame with added ChemBERTa embeddings
    """
    print("Computing ChemBERTa embeddings...")
    
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = TFAutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", from_pt=True)

    chemberta_list = []
    for smiles in df['smiles']:
        inputs = tokenizer(smiles, return_tensors="tf", padding=True, truncation=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        chemberta_list.append(embedding)
    
    df['chemberta'] = chemberta_list
    return df


def prepare_complete_dataset(csv_path='delaney-processed.csv'):
    """
    Complete data preparation pipeline.
    
    Args:
        csv_path (str): Path to the ESOL dataset CSV file
        
    Returns:
        pd.DataFrame: Fully prepared dataset with all features
    """
    # Load and prepare basic data
    df = load_and_prepare_data(csv_path)
    
    # Compute fingerprints
    df = compute_molecular_fingerprints(df)
    
    # Compute ChemBERTa embeddings
    df = compute_chemberta_embeddings(df)
    
    print("Dataset preparation complete!")
    return df


if __name__ == "__main__":
    # Example usage
    dataset = prepare_complete_dataset()
    print(f"Dataset shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")
