"""
Data preprocessing for ML models.
- Load train/val/test splits
- Encode categorical variables
- Apply SMOTE to balance training data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
import os


def load_splits(data_dir='data/splits'):
    """
    Load train/val/test splits.
    """
    print("\n" + "="*70)
    print("LOADING DATA SPLITS")
    print("="*70 + "\n")
    
    train = pd.read_csv(f'{data_dir}/train.csv')
    val = pd.read_csv(f'{data_dir}/val.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    
    print(f"✓ Train: {len(train)} samples")
    print(f"✓ Val:   {len(val)} samples")
    print(f"✓ Test:  {len(test)} samples")
    
    return train, val, test


def encode_categorical_features(train, val, test):
    """
    Encode categorical variables using LabelEncoder.
    Fit on train, transform all splits.
    """
    print("\n" + "="*70)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*70 + "\n")
    
    categorical_cols = ['job_category', 'value_tier']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Fit on train only
        train[col + '_encoded'] = le.fit_transform(train[col])
        val[col + '_encoded'] = le.transform(val[col])
        test[col + '_encoded'] = le.transform(test[col])
        
        # Save encoder for later use
        encoders[col] = le
        
        print(f"✓ Encoded {col}:")
        print(f"  Classes: {list(le.classes_)}")
    
    # Drop original categorical columns
    train = train.drop(categorical_cols, axis=1)
    val = val.drop(categorical_cols, axis=1)
    test = test.drop(categorical_cols, axis=1)
    
    return train, val, test, encoders


def prepare_features_and_target(train, val, test):
    """
    Separate features (X) and target (y).
    Drop non-feature columns.
    """
    print("\n" + "="*70)
    print("SEPARATING FEATURES AND TARGET")
    print("="*70 + "\n")
    
    # Columns to drop
    drop_cols = ['customer_id', 'is_churned']
    drop_cols = [col for col in drop_cols if col in train.columns]
    
    # Separate X and y
    X_train = train.drop(drop_cols, axis=1)
    y_train = train['is_churned']
    
    X_val = val.drop(drop_cols, axis=1)
    y_val = val['is_churned']
    
    X_test = test.drop(drop_cols, axis=1)
    y_test = test['is_churned']
    
    print(f"✓ Number of features: {X_train.shape[1]}")
    print(f"✓ Feature names: {list(X_train.columns)}")
    print(f"\n✓ Training set class distribution:")
    print(f"  Active (0): {(y_train == 0).sum()}")
    print(f"  Churned (1): {(y_train == 1).sum()}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to balance training data.
    Uses 0.6 sampling strategy - creates churned samples to be 60% of active samples.
    """
    print("\n" + "="*70)
    print("APPLYING SMOTE TO BALANCE TRAINING DATA")
    print("="*70 + "\n")
    
    print(f"Before SMOTE:")
    print(f"  Active (0): {(y_train == 0).sum()}")
    print(f"  Churned (1): {(y_train == 1).sum()}")
    print(f"  Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")
    
    # sampling_strategy=0.6 means churned will be 60% of active count
    # Example: 2075 active → 1245 churned (instead of full 2075)
    smote = SMOTE(sampling_strategy=0.6, random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE:")
    print(f"  Active (0): {(y_train_balanced == 0).sum()}")
    print(f"  Churned (1): {(y_train_balanced == 1).sum()}")
    print(f"  Ratio: {(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.2f}:1")
    
    return X_train_balanced, y_train_balanced


def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test, encoders, save_dir='data/preprocessed'):
    """
    Save preprocessed data and encoders.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save data
    pd.DataFrame(X_train).to_csv(f'{save_dir}/X_train.csv', index=False)
    pd.DataFrame(y_train, columns=['is_churned']).to_csv(f'{save_dir}/y_train.csv', index=False)
    
    pd.DataFrame(X_val).to_csv(f'{save_dir}/X_val.csv', index=False)
    pd.DataFrame(y_val, columns=['is_churned']).to_csv(f'{save_dir}/y_val.csv', index=False)
    
    pd.DataFrame(X_test).to_csv(f'{save_dir}/X_test.csv', index=False)
    pd.DataFrame(y_test, columns=['is_churned']).to_csv(f'{save_dir}/y_test.csv', index=False)
    
    # Save encoders
    with open(f'{save_dir}/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print(f"\n✓ Preprocessed data saved to {save_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Load splits
    train, val, test = load_splits()
    
    # Encode categorical features
    train, val, test, encoders = encode_categorical_features(train, val, test)
    
    # Prepare features and target
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_target(train, val, test)
    
    # Apply SMOTE to training data
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    # Save preprocessed data
    save_preprocessed_data(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test, encoders)
    
    print("✓✓✓ PREPROCESSING COMPLETE ✓✓✓\n")