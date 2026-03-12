"""
Churn Labeling Logic
Labels customers as churned based on segment-specific thresholds.
"""

import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from src.database import get_session
from src.schema import Customer, Booking
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def calculate_churn_labels(as_of_date=None):
    """
    Calculate churn labels for all customers based on their segment.
    
    Args:
        as_of_date: Date to calculate churn from (default: today)
        
    Returns:
        DataFrame with customer_id, job_category, last_booking_date, 
        days_since_last_booking, churn_threshold, is_churned
    """
    if as_of_date is None:
        as_of_date = datetime.now()
    
    print(f"\nCalculating churn labels as of {as_of_date.date()}...")
    
    session = get_session()
    
    try:
        # Get all customers with their last booking date
        query = """
        SELECT 
            c.customer_id,
            c.job_category,
            MAX(b.check_in_date) as last_booking_date
        FROM customers c
        LEFT JOIN bookings b ON c.customer_id = b.customer_id
        WHERE b.status = 'Checked-Out'
        GROUP BY c.customer_id, c.job_category
        """
        
        df = pd.read_sql(query, session.bind)
        
        # Calculate days since last booking
        df['last_booking_date'] = pd.to_datetime(df['last_booking_date'])
        df['days_since_last_booking'] = (as_of_date - df['last_booking_date']).dt.days
        
        # Get churn threshold for each segment
        def get_churn_threshold(job_category):
            return config['customer_segments'][job_category]['churn_definition_days']
        
        df['churn_threshold'] = df['job_category'].apply(get_churn_threshold)
        
        # Label as churned if days since last booking > threshold
        df['is_churned'] = (df['days_since_last_booking'] > df['churn_threshold']).astype(int)
        
        # Summary statistics
        total_customers = len(df)
        churned_customers = df['is_churned'].sum()
        churn_rate = (churned_customers / total_customers) * 100
        
        print(f"\n✓ Churn Analysis Complete:")
        print(f"  Total customers: {total_customers}")
        print(f"  Churned customers: {churned_customers}")
        print(f"  Overall churn rate: {churn_rate:.2f}%")
        
        # Breakdown by segment
        print(f"\n  Churn by segment:")
        segment_summary = df.groupby('job_category').agg({
            'customer_id': 'count',
            'is_churned': 'sum'
        }).rename(columns={'customer_id': 'total', 'is_churned': 'churned'})
        segment_summary['churn_rate'] = (segment_summary['churned'] / segment_summary['total'] * 100).round(2)
        print(segment_summary)
        
        return df
        
    finally:
        session.close()


def save_churn_labels_to_csv(df, filename='data/churn_labels.csv'):
    """
    Save churn labels to CSV file for model training.
    """
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"\n✓ Churn labels saved to {filename}")


if __name__ == "__main__":
    # Test the churn labeling
    churn_df = calculate_churn_labels()
    save_churn_labels_to_csv(churn_df)