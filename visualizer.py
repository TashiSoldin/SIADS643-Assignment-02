"""
Visualization module for penguin analysis pipeline.

This module creates charts and plots from the analyzed data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List


def create_numerical_distribution_plots(df: pd.DataFrame, output_path: str) -> None:
    """
    Create distribution plots for numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        output_path (str): Path to save the plot
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("No numerical columns found")
        return
    
    n_numeric = len(numeric_cols)
    n_cols = 2
    n_rows = (n_numeric + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    # Handle different subplot configurations
    if n_numeric == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_numeric > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i] if n_numeric > 1 else axes[0]
        ax.hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel('Frequency')
    
    # Hide unused subplots
    if n_numeric < len(axes):
        for i in range(n_numeric, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    save_visualization(output_path)
    print(f"Created distribution plots for {n_numeric} numeric columns")


def create_categorical_distribution_plots(df: pd.DataFrame, output_path: str) -> None:
    """
    Create bar plots for categorical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        output_path (str): Path to save the plot
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found")
        return
    
    n_categorical = len(categorical_cols)
    n_cols = 2
    n_rows = (n_categorical + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    # Handle different subplot configurations
    if n_categorical == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_categorical > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        ax = axes[i] if n_categorical > 1 else axes[0]
        value_counts = df[col].value_counts()
        value_counts.plot(kind='bar', ax=ax, color='lightcoral', alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    if n_categorical < len(axes):
        for i in range(n_categorical, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    save_visualization(output_path)
    print(f"Created bar charts for {n_categorical} categorical columns")


def create_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Create correlation heatmap for numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        output_path (str): Path to save the plot
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numerical columns for correlation heatmap")
        return
    
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.2f')
    plt.title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()
    save_visualization(output_path)
    print("Created correlation heatmap")


def create_visualizations(df: pd.DataFrame, output_dir: str,
                         numeric_filename: str = "numeric_distributions.png",
                         categorical_filename: str = "categorical_distributions.png",
                         correlation_filename: str = "correlation_heatmap.png") -> None:
    """
    Create all visualizations by calling the separate plotting functions.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        output_dir (str): Directory to save all plots
        numeric_filename (str): Filename for numeric distribution plots
        categorical_filename (str): Filename for categorical distribution plots
        correlation_filename (str): Filename for correlation heatmap
    """
    print("Creating visualizations...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get column information
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all visualizations
    create_numerical_distribution_plots(df, f"{output_dir}/{numeric_filename}")
    create_categorical_distribution_plots(df, f"{output_dir}/{categorical_filename}")
    create_correlation_heatmap(df, f"{output_dir}/{correlation_filename}")
    
    print("All visualizations created and saved!")


def save_visualization(output_path: str) -> None:
    """
    Save the current matplotlib figure to file.
    
    Args:
        output_path (str): Path to save the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()