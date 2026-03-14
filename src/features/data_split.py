"""
Train/Validation/Test Split with Time-Aware Strategy
Ensures no data leakage by splitting based on booking dates.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import os


def load_master_dataset(filepath='data/master_dataset.csv'):
    """
    Load the master dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Master dataset not found at {filepath}. Run combine_features.py first.")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded master dataset: {len(df)} records, {len(df.columns)} features")
    return df


def time_aware_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    Split data using time-aware strategy to prevent data leakage.
    
    Strategy:
    - Sort customers by their last booking date
    - Split chronologically: earliest → train, middle → val, latest → test
    - Ensures model is tested on "future" customers
    
    Args:
        df: Master dataset
        train_ratio: Proportion for training (default 60%)
        val_ratio: Proportion for validation (default 20%)
        test_ratio: Proportion for test (default 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    print("\n" + "="*70)
    print("TIME-AWARE TRAIN/VAL/TEST SPLIT")
    print("="*70 + "\n")
    
    # Sort by recency (days since last booking) - descending
    # Lower recency = more recent bookings = should be in test set
    df_sorted = df.sort_values('recency_days', ascending=False).reset_index(drop=True)
    
    n = len(df_sorted)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Split chronologically
    train_df = df_sorted.iloc[:train_size].copy()
    val_df = df_sorted.iloc[train_size:train_size + val_size].copy()
    test_df = df_sorted.iloc[train_size + val_size:].copy()
    
    print(f"Split strategy: Time-aware (sorted by recency)")
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    # Check class distribution
    print(f"\nChurn rate distribution:")
    print(f"  Train: {train_df['is_churned'].mean()*100:.2f}%")
    print(f"  Val:   {val_df['is_churned'].mean()*100:.2f}%")
    print(f"  Test:  {test_df['is_churned'].mean()*100:.2f}%")
    
    # Check segment distribution
    print(f"\nSegment distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        segment_dist = split_df['job_category'].value_counts(normalize=True) * 100
        print(f"\n  {split_name}:")
        for segment, pct in segment_dist.items():
            print(f"    {segment}: {pct:.1f}%")
    
    # Recency statistics
    print(f"\nRecency (days since last booking) ranges:")
    print(f"  Train: {train_df['recency_days'].min():.0f} - {train_df['recency_days'].max():.0f} days")
    print(f"  Val:   {val_df['recency_days'].min():.0f} - {val_df['recency_days'].max():.0f} days")
    print(f"  Test:  {test_df['recency_days'].min():.0f} - {test_df['recency_days'].max():.0f} days")
    
    return train_df, val_df, test_df


def stratified_random_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    Alternative: Stratified random split (maintains class distribution).
    Use this if time-aware split causes issues.
    
    Args:
        df: Master dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    print("\n" + "="*70)
    print("STRATIFIED RANDOM SPLIT")
    print("="*70 + "\n")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['is_churned'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['is_churned'],
        random_state=random_state
    )
    
    n = len(df)
    print(f"Split strategy: Stratified Random")
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    print(f"\nChurn rate distribution:")
    print(f"  Train: {train_df['is_churned'].mean()*100:.2f}%")
    print(f"  Val:   {val_df['is_churned'].mean()*100:.2f}%")
    print(f"  Test:  {test_df['is_churned'].mean()*100:.2f}%")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir='data/splits'):
    """
    Save train/val/test splits to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    print(f"\n✓ Splits saved to {output_dir}/")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    df = load_master_dataset()
    
    # Use stratified split to maintain class balance
    train_df, val_df, test_df = stratified_random_split(df)
    
    save_splits(train_df, val_df, test_df)
    
    print("\n✓✓✓ Data split complete! Ready for model training.\n")