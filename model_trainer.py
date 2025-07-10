"""
Model training module for penguin classification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Optional, List
import os


def prepare_data(df: pd.DataFrame, target_column: str, 
                exclude_columns: Optional[List[str]] = None) -> tuple:
    """
    Prepare data for modeling by splitting features/target.
    
    Args:
        df: Input DataFrame (should already be cleaned and encoded)
        target_column: Target column name
        exclude_columns: Columns to exclude from features
        
    Returns:
        tuple: (X, y) features and target
    """
    # Create features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Remove excluded columns
    if exclude_columns:
        X = X.drop([col for col in exclude_columns if col in X.columns], axis=1)
    
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> dict:
    """
    Train Random Forest model and return evaluation results.
    
    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        
    Returns:
        dict: Model results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'report': report,
        'feature_importance': feature_importance,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


def save_results(results: dict, output_path: str, target_column: str) -> None:
    """
    Save model results to file.
    
    Args:
        results: Model results dictionary
        output_path: Path to save file
        target_column: Target column name
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("MODEL RESULTS\n")
        f.write("=" * 20 + "\n")
        f.write(f"Target: {target_column}\n")
        f.write(f"Accuracy: {results['accuracy']:.3f}\n")
        f.write(f"Train size: {results['train_size']}\n")
        f.write(f"Test size: {results['test_size']}\n\n")
        
        f.write("Feature Importance:\n")
        for feature, importance in sorted(results['feature_importance'].items(), 
                                        key=lambda x: x[1], reverse=True):
            f.write(f"  {feature}: {importance:.3f}\n")
        
        f.write(f"\n{results['report']}")


def train_model(df: pd.DataFrame, target_column: str, 
                exclude_columns: Optional[List[str]] = None) -> Optional[dict]:
    """
    Train a classification model on the data.
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        exclude_columns: Columns to exclude from features
        
    Returns:
        dict: Model results or None if failed
    """
    try:
        print(f"Training model to predict '{target_column}'...")
        
        # Prepare data
        X, y = prepare_data(df, target_column, exclude_columns)
        
        # Train and evaluate
        results = train_and_evaluate(X, y)
        
        # Print results
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Training samples: {results['train_size']}")
        print(f"Test samples: {results['test_size']}")
        
        return results
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None