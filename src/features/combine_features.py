"""
Combine all features into a single master dataset for ML training.
Merges churn labels + RFM features + customer demographics.
"""

import pandas as pd
from src.features.churn_labeling import calculate_churn_labels
from src.features.rfm_features import calculate_rfm_features


def create_master_dataset(as_of_date=None):
    """
    Create the final master dataset by combining all features.
    
    Args:
        as_of_date: Reference date for feature calculation
        
    Returns:
        DataFrame with all features ready for ML training
    """
    print("\n" + "="*70)
    print("CREATING MASTER ML DATASET")
    print("="*70 + "\n")
    
    # Step 1: Calculate churn labels
    print("Step 1: Calculating churn labels...")
    churn_df = calculate_churn_labels(as_of_date)
    print(f"✓ Churn labels: {len(churn_df)} customers")
    
    # Step 2: Calculate RFM features
    print("\nStep 2: Calculating RFM features...")
    rfm_df = calculate_rfm_features(as_of_date)
    print(f"✓ RFM features: {len(rfm_df)} customers")
    
    # Step 3: Merge datasets
    print("\nStep 3: Merging features...")
    master_df = pd.merge(
        churn_df,
        rfm_df,
        on='customer_id',
        how='inner',
        suffixes=('_churn', '_rfm')
    )
    
    # Handle duplicate columns - prefer churn version for job_category
    if 'job_category_churn' in master_df.columns:
        master_df['job_category'] = master_df['job_category_churn']
        master_df = master_df.drop(['job_category_churn', 'job_category_rfm'], axis=1, errors='ignore')
    
    if 'age_churn' in master_df.columns:
        master_df['age'] = master_df['age_churn']
        master_df = master_df.drop(['age_churn', 'age_rfm'], axis=1, errors='ignore')
    
    # Step 4: Feature engineering from combined data
    print("\nStep 4: Engineering derived features...")
    
    # Average spending per booking
    master_df['avg_spend_per_booking'] = (
        master_df['monetary_total_spent'] / 
        (master_df['frequency_total_bookings'] + 1)
    )
    
    # Recency to frequency ratio (engagement score)
    master_df['engagement_score'] = (
        master_df['frequency_total_bookings'] / 
        (master_df['recency_days'] + 1)
    )
    
    # Is customer at risk? (recency > 80% of churn threshold)
    master_df['at_risk_flag'] = (
        master_df['days_since_last_booking'] > 
        (master_df['churn_threshold'] * 0.8)
    ).astype(int)
    
    # Customer value tier (based on total spending)
    master_df['value_tier'] = pd.qcut(
        master_df['monetary_total_spent'],
        q=4,
        labels=['Low', 'Medium', 'High', 'VIP'],
        duplicates='drop'
    )
    
    # Step 5: Select final features for ML
    feature_columns = [
        # Target variable
        'is_churned',
        
        # Customer demographics
        'customer_id',
        'job_category',
        'age',
        
        # Churn features
        'days_since_last_booking',
        'churn_threshold',
        'at_risk_flag',
        
        # RFM features
        'recency_days',
        'frequency_total_bookings',
        'frequency_cancelled_bookings',
        'frequency_noshow_bookings',
        'cancellation_rate',
        'monetary_total_spent',
        'monetary_avg_per_booking',
        'monetary_max_booking',
        'monetary_avg_per_night',
        'total_nights_stayed',
        'avg_nights_per_stay',
        'frequency_bookings_per_year',
        
        # Derived features
        'avg_spend_per_booking',
        'engagement_score',
        'value_tier'
    ]
    
    master_df = master_df[feature_columns]
    
    # Step 6: Summary statistics
    print("\n" + "="*70)
    print("MASTER DATASET SUMMARY")
    print("="*70)
    print(f"\nTotal records: {len(master_df)}")
    print(f"Total features: {len(feature_columns)}")
    print(f"\nTarget distribution:")
    print(master_df['is_churned'].value_counts())
    print(f"\nChurn rate: {master_df['is_churned'].mean() * 100:.2f}%")
    
    print(f"\nValue tier distribution:")
    print(master_df['value_tier'].value_counts())
    
    print(f"\nSegment breakdown:")
    segment_churn = master_df.groupby('job_category')['is_churned'].agg(['count', 'sum', 'mean'])
    segment_churn.columns = ['Total', 'Churned', 'Churn_Rate']
    segment_churn['Churn_Rate'] = (segment_churn['Churn_Rate'] * 100).round(2)
    print(segment_churn)
    
    return master_df


def save_master_dataset(df, filename='data/master_dataset.csv'):
    """
    Save master dataset to CSV.
    """
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"\n✓ Master dataset saved to {filename}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Create and save master dataset
    master_df = create_master_dataset()
    save_master_dataset(master_df)
    
    # Display first few rows
    print("\nSample of master dataset:")
    print(master_df.head(10))