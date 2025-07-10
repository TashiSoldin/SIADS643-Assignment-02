"""
Main script for penguin data analysis.

Usage:
    python main.py <data_source> <output_directory>
"""

import sys
import os

from data_loader import load_data_from_seaborn, load_data_from_csv, save_data
from data_cleaner import clean_data, encode_categorical
from visualizer import create_visualizations
from model_trainer import train_model, save_results


def main():
    """Main function to run the penguin analysis pipeline."""
    
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python main.py <data_source> <output_directory> [exclude_columns]")
        print("Examples:")
        print("  python main.py seaborn output/")
        print("  python main.py data/penguins.csv results/ island")
        print("  python main.py data/other.csv results/ id,date")
        sys.exit(1)
    
    data_source = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Get columns to exclude (optional)
    exclude_columns = []
    if len(sys.argv) > 3:
        exclude_columns = [col.strip() for col in sys.argv[3].split(',')]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting data analysis...")
    
    # Load data
    if data_source == 'seaborn':
        df = load_data_from_seaborn('penguins')
        save_data(df, "penguin_data.csv")               # Once loaded, save to CSV
    else:
        df = load_data_from_csv(data_source)
    print(f"Loaded data: {df.shape}")
    
    # Clean data
    cleaned_df = clean_data(df)
    print(f"Cleaned data: {cleaned_df.shape}")
    
    # Create visualizations
    create_visualizations(cleaned_df, f"{output_dir}/plots")
    
    # Prepare data for modeling and train
    model_df = encode_categorical(cleaned_df, 'species')
    
    results = train_model(model_df, 'species', exclude_columns=exclude_columns)
    
    # Save everything: cleaned data, model results, and visualizations
    save_data(cleaned_df, f"{output_dir}/cleaned_data.csv")
    if results:
        save_results(results, f"{output_dir}/model_results.txt", 'species')
        print(f"Model accuracy: {results['accuracy']:.3f}")
    
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()