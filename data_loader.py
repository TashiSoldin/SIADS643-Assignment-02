"""
Data loading functions for penguin analysis.
"""

import pandas as pd
import seaborn as sns


def load_data_from_seaborn(dataset_name: str) -> pd.DataFrame:
    """
    Load data from seaborn built-in datasets.
    
    Args:
        dataset_name (str): Name of the seaborn dataset (e.g., 'penguins')
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return sns.load_dataset(dataset_name)


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Output file path
    """
    df.to_csv(output_path, index=False)