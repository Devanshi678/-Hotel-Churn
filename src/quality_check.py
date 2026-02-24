"""
Data quality checks using Pandera.
Validates data before inserting into database to prevent bad data.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Define validation schema for Bookings
booking_schema = DataFrameSchema({
    "customer_id": Column(int, Check.greater_than(0)),
    "check_in_date": Column(pa.DateTime),
    "check_out_date": Column(pa.DateTime),
    "amount_spent": Column(float, Check.in_range(
        min_value=0, 
        max_value=config['quality_checks']['max_booking_amount']
    )),
    "room_type": Column(str, Check.isin(['Queen Bed', 'King Bed', 'Suite'])),
    "booking_channel": Column(str, Check.isin(['Direct Website', 'Agent', 'OTA (Booking.com)', 'Walk-in'])),
    "num_adults": Column(int, Check.in_range(min_value=1, max_value=10)),
    "num_children": Column(int, Check.in_range(min_value=0, max_value=10)),
    "status": Column(str, Check.isin(['Checked-Out', 'Cancelled', 'No-Show']))
})


# Define validation schema for Customers
customer_schema = DataFrameSchema({
    "join_date": Column(pa.DateTime),
    "age": Column(int, Check.in_range(min_value=18, max_value=100)),
    "job_category": Column(str, Check.isin(['business_traveler', 'vacation_traveler', 'occasional_visitor']))
})


def validate_bookings(bookings_df):
    """
    Validates booking data using Pandera schema.
    
    Args:
        bookings_df: Pandas DataFrame with booking data
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        # Check 1: Schema validation
        booking_schema.validate(bookings_df, lazy=True)
        
        # Check 2: Check-out date must be after check-in date
        invalid_dates = bookings_df[bookings_df['check_out_date'] <= bookings_df['check_in_date']]
        if not invalid_dates.empty:
            return False, f"Found {len(invalid_dates)} bookings where check_out <= check_in"
        
        # Check 3: No duplicate booking IDs (if booking_id exists in DataFrame)
        if 'booking_id' in bookings_df.columns:
            if bookings_df['booking_id'].duplicated().any():
                return False, "Duplicate booking_ids found"
        
        # Check 4: Nights stayed within reasonable range
        bookings_df['nights'] = (bookings_df['check_out_date'] - bookings_df['check_in_date']).dt.days
        min_nights = config['quality_checks']['min_nights_stay']
        max_nights = config['quality_checks']['max_nights_stay']
        
        invalid_nights = bookings_df[(bookings_df['nights'] < min_nights) | (bookings_df['nights'] > max_nights)]
        if not invalid_nights.empty:
            return False, f"Found {len(invalid_nights)} bookings with invalid night count (must be {min_nights}-{max_nights})"
        
        print(f"✓ All {len(bookings_df)} bookings passed quality checks")
        return True, None
        
    except pa.errors.SchemaErrors as e:
        return False, f"Schema validation failed: {e.failure_cases}"
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"


def validate_customers(customers_df):
    """
    Validates customer data using Pandera schema.
    
    Args:
        customers_df: Pandas DataFrame with customer data
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        # Check 1: Schema validation
        customer_schema.validate(customers_df, lazy=True)
        
        # Check 2: No duplicate customer IDs (if customer_id exists in DataFrame)
        if 'customer_id' in customers_df.columns:
            if customers_df['customer_id'].duplicated().any():
                return False, "Duplicate customer_ids found"
        
        print(f"✓ All {len(customers_df)} customers passed quality checks")
        return True, None
        
    except pa.errors.SchemaErrors as e:
        return False, f"Schema validation failed: {e.failure_cases}"
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"


def validate_support_logs(logs_df):
    """
    Validates support log data.
    
    Args:
        logs_df: Pandas DataFrame with support log data
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        # Check 1: Required columns exist
        required_cols = ['customer_id', 'date', 'log_text']
        if not all(col in logs_df.columns for col in required_cols):
            return False, f"Missing required columns: {required_cols}"
        
        # Check 2: No empty log text
        if logs_df['log_text'].isna().any() or (logs_df['log_text'] == '').any():
            return False, "Found empty log_text entries"
        
        # Check 3: Customer ID must be positive
        if (logs_df['customer_id'] <= 0).any():
            return False, "Found invalid customer_ids (must be > 0)"
        
        print(f"✓ All {len(logs_df)} support logs passed quality checks")
        return True, None
        
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"