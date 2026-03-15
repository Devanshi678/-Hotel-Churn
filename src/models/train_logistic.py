"""
Train Logistic Regression baseline model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import os
from src.models.evaluate import (
    evaluate_model,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve
)


def load_preprocessed_data(data_dir='data/preprocessed'):
    """
    Load preprocessed data.
    """
    print("\n" + "="*70)
    print("LOADING PREPROCESSED DATA")
    print("="*70 + "\n")
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')['is_churned']
    
    X_val = pd.read_csv(f'{data_dir}/X_val.csv')
    y_val = pd.read_csv(f'{data_dir}/y_val.csv')['is_churned']
    
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')['is_churned']
    
    print(f"✓ X_train: {X_train.shape}")
    print(f"✓ X_val:   {X_val.shape}")
    print(f"✓ X_test:  {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    """
    print("\n" + "="*70)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*70 + "\n")
    
    print("Hyperparameters:")
    print("  max_iter: 1000")
    print("  solver: lbfgs")
    print("  class_weight: balanced")
    print("  random_state: 42")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    print("✓ Training complete!")
    
    return model


def save_model(model, save_path='models/logistic_regression.pkl'):
    """
    Save trained model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved to {save_path}")


if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data()
    
    # Train model
    model = train_logistic_regression(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred, y_val_proba = evaluate_model(model, X_val, y_val, model_name="Logistic Regression")
    
    # Plot visualizations
    plot_roc_curve(y_val, y_val_proba, "Logistic Regression", "results/logistic_regression_roc.png")
    plot_confusion_matrix(y_val, y_val_pred, "Logistic Regression", "results/logistic_regression_cm.png")
    plot_precision_recall_curve(y_val, y_val_proba, "Logistic Regression", "results/logistic_regression_pr.png")
    
    # Save model
    save_model(model)
    
    print("\n" + "="*70)
    print("✓✓✓ LOGISTIC REGRESSION TRAINING COMPLETE ✓✓✓")
    print("="*70 + "\n")