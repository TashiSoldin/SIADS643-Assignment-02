"""
Data cleaning functions for penguin analysis.
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean penguin data by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Make a copy
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Drop rows with more than 50% missing data
    missing_per_row = cleaned_df.isnull().sum(axis=1)
    cleaned_df = cleaned_df[missing_per_row <= len(cleaned_df.columns) * 0.5]
    
    # Fill missing numeric values with median
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    # Fill missing categorical values with 'Missing'
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col].fillna('Missing', inplace=True)
    
    return cleaned_df


def encode_categorical(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Encode categorical variables (except target) to numbers.
    
    Args:
        df (pd.DataFrame): Input data
        target_col (str): Target column name to exclude from encoding
        
    Returns:
        pd.DataFrame: Data with encoded categoricals
    """
    encoded_df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != target_col:
            encoded_df[col] = pd.factorize(df[col])[0]
    
    return encoded_df