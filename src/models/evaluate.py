"""
Shared evaluation functions for all models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, X_val, y_val, model_name="Model"):
    """
    Evaluate model on validation set.
    
    Returns:
        y_val_pred: Predictions (0/1)
        y_val_proba: Probabilities for class 1
    """
    print("\n" + "="*70)
    print(f"EVALUATING {model_name.upper()}")
    print("="*70 + "\n")
    
    # Predictions
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Active (0)', 'Churned (1)']))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Active  Churned")
    print(f"Actual Active   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"Actual Churned  {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_val_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Additional metrics
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    
    print(f"\nKey Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return y_val_pred, y_val_proba


def plot_roc_curve(y_val, y_val_proba, model_name, save_path):
    """
    Plot ROC curve for a single model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2, color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ ROC curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_val, y_val_pred, model_name, save_path):
    """
    Plot confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_val, y_val_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'],
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_precision_recall_curve(y_val, y_val_proba, model_name, save_path):
    """
    Plot Precision-Recall curve.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, color='green')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Precision-Recall curve saved to {save_path}")
    plt.close()