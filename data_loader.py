"""
Data loading module for Iris analysis pipeline.

This module handles loading data from CSV files and generating sample datasets.
"""

import pandas as pd
from sklearn.datasets import load_iris

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")

def generate_sample_data(output_path: str) -> pd.DataFrame:
    """
    Generate the classic Iris dataset and save it as CSV.
    
    Args:
        output_path: Path where to save the generated CSV
        
    Returns:
        DataFrame containing the Iris dataset
    """
    # Load Iris dataset from scikit-learn
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Generated Iris dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Saved to: {output_path}")
    
    return df