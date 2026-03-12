"""
RFM Feature Engineering
Calculates Recency, Frequency, Monetary features for each customer.
"""

import pandas as pd
from datetime import datetime
from src.database import get_session


def calculate_rfm_features(as_of_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) features for all customers.
    
    Args:
        as_of_date: Date to calculate features from (default: today)
        
    Returns:
        DataFrame with customer_id and RFM features
    """
    if as_of_date is None:
        as_of_date = datetime.now()
    
    print(f"\nCalculating RFM features as of {as_of_date.date()}...")
    
    session = get_session()
    
    try:
        # SQL query to calculate RFM metrics
        query = f"""
        SELECT 
            c.customer_id,
            c.job_category,
            c.age,
            
            -- RECENCY: Days since last booking
            COALESCE(
                DATE_PART('day', TIMESTAMP '{as_of_date}' - MAX(b.check_in_date)::timestamp),
                9999
            ) as recency_days,
            
            -- FREQUENCY: Total number of bookings
            COUNT(CASE WHEN b.status = 'Checked-Out' THEN 1 END) as frequency_total_bookings,
            
            -- Cancelled bookings
            COUNT(CASE WHEN b.status = 'Cancelled' THEN 1 END) as frequency_cancelled_bookings,
            
            -- No-show bookings
            COUNT(CASE WHEN b.status = 'No-Show' THEN 1 END) as frequency_noshow_bookings,
            
            -- MONETARY: Total spending
            COALESCE(SUM(CASE WHEN b.status = 'Checked-Out' THEN b.amount_spent ELSE 0 END), 0) as monetary_total_spent,
            
            -- Average spending per booking
            COALESCE(AVG(CASE WHEN b.status = 'Checked-Out' THEN b.amount_spent END), 0) as monetary_avg_per_booking,
            
            -- Maximum single booking amount
            COALESCE(MAX(CASE WHEN b.status = 'Checked-Out' THEN b.amount_spent END), 0) as monetary_max_booking,
            
            -- Total nights stayed
            COALESCE(SUM(CASE WHEN b.status = 'Checked-Out' THEN 
                (b.check_out_date - b.check_in_date)
            END), 0) as total_nights_stayed,
            
            -- Average nights per stay
            COALESCE(AVG(CASE WHEN b.status = 'Checked-Out' THEN 
                (b.check_out_date - b.check_in_date)
            END), 0) as avg_nights_per_stay
            
        FROM customers c
        LEFT JOIN bookings b ON c.customer_id = b.customer_id
        GROUP BY c.customer_id, c.job_category, c.age
        """
        
        df = pd.read_sql(query, session.bind)
        
        # Calculate derived features
        
        # Cancellation rate
        df['cancellation_rate'] = (
            df['frequency_cancelled_bookings'] / 
            (df['frequency_total_bookings'] + df['frequency_cancelled_bookings'] + df['frequency_noshow_bookings'] + 0.001)
        ).fillna(0)
        
        # Average spending per night
        df['monetary_avg_per_night'] = (
            df['monetary_total_spent'] / (df['total_nights_stayed'] + 0.001)
        ).fillna(0)
        
        # Booking frequency (bookings per year)
        # Assume customers have been active for at least 1 year
        df['frequency_bookings_per_year'] = df['frequency_total_bookings'] / 4  # 4 years of history
        
        print(f"\n✓ RFM Features Calculated:")
        print(f"  Total customers: {len(df)}")
        print(f"\n  Sample statistics:")
        print(df[['recency_days', 'frequency_total_bookings', 'monetary_total_spent']].describe())
        
        return df
        
    finally:
        session.close()


def save_rfm_features_to_csv(df, filename='data/rfm_features.csv'):
    """
    Save RFM features to CSV file.
    """
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"\n✓ RFM features saved to {filename}")


if __name__ == "__main__":
    # Test RFM feature generation
    rfm_df = calculate_rfm_features()
    save_rfm_features_to_csv(rfm_df)