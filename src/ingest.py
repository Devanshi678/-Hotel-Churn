"""
Orchestrates the data pipeline.
Coordinates: data generation → quality checks → database insertion.
This is the script that Prefect will run weekly.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.data_generator import generate_historical_data, generate_weekly_data
from src.quality_check import validate_bookings, validate_customers, validate_support_logs
from src.database import get_session, test_connection
from src.schema import Customer, Booking, SupportLog
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def cleanup_old_data():
    """
    Removes data older than 5 years to maintain a rolling window.
    Keeps database size manageable.
    """
    print("\n" + "="*60)
    print("CLEANING UP OLD DATA (5 YEAR ROLLING WINDOW)")
    print("="*60 + "\n")
    
    session = get_session()
    
    try:
        # Calculate cutoff date (5 years ago)
        cutoff_date = datetime.now() - timedelta(days=5*365)
        print(f"Removing data older than: {cutoff_date.date()}")
        
        # Delete old bookings
        old_bookings = session.query(Booking).filter(
            Booking.check_in_date < cutoff_date
        ).delete(synchronize_session=False)
        
        # Delete old support logs
        old_logs = session.query(SupportLog).filter(
            SupportLog.date < cutoff_date
        ).delete(synchronize_session=False)
        
        # Delete customers who have NO bookings left
        # (customers whose all bookings were deleted)
        customers_with_bookings = session.query(Booking.customer_id).distinct()
        customers_to_keep = [c[0] for c in customers_with_bookings]
        
        old_customers = session.query(Customer).filter(
            ~Customer.customer_id.in_(customers_to_keep)
        ).delete(synchronize_session=False)
        
        session.commit()
        
        print(f"\n✓ Deleted:")
        print(f"  - {old_bookings} old bookings")
        print(f"  - {old_logs} old support logs")
        print(f"  - {old_customers} customers with no recent bookings")
        
        # Show current data size
        customer_count = session.query(Customer).count()
        booking_count = session.query(Booking).count()
        log_count = session.query(SupportLog).count()
        
        print(f"\n✓ Current database size:")
        print(f"  - {customer_count} customers")
        print(f"  - {booking_count} bookings")
        print(f"  - {log_count} support logs")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Cleanup failed: {e}")
        raise
    finally:
        session.close()
    
    print("\n" + "="*60)
    print("✓✓✓ CLEANUP COMPLETED ✓✓✓")
    print("="*60 + "\n")


def run_historical_pipeline():
    """
    Runs the ONE-TIME historical data generation pipeline.
    Generates 4 years of data for initial model training.
    """
    print("\n" + "="*60)
    print("STARTING HISTORICAL DATA PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Test database connection
    print("Step 1: Testing database connection...")
    if not test_connection():
        print("✗ Cannot proceed - database connection failed")
        return False
    
    # Step 2: Generate historical data (calls data_generator.py)
    print("\nStep 2: Generating historical data...")
    try:
        generate_historical_data()
    except Exception as e:
        print(f"✗ Historical data generation failed: {e}")
        return False
    
    # Step 3: Verify data was inserted
    print("\nStep 3: Verifying data insertion...")
    session = get_session()
    try:
        customer_count = session.query(Customer).count()
        booking_count = session.query(Booking).count()
        log_count = session.query(SupportLog).count()
        
        print(f"✓ Database now contains:")
        print(f"  - {customer_count} customers")
        print(f"  - {booking_count} bookings")
        print(f"  - {log_count} support logs")
        
    finally:
        session.close()
    
    print("\n" + "="*60)
    print("✓✓✓ HISTORICAL PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
    print("="*60 + "\n")
    
    return True


def run_weekly_pipeline():
    """
    Runs the WEEKLY incremental data generation pipeline.
    This will be scheduled by Prefect to run every week.
    Also cleans up data older than 5 years.
    """
    print("\n" + "="*60)
    print("STARTING WEEKLY DATA PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Test database connection
    print("Step 1: Testing database connection...")
    if not test_connection():
        print("✗ Cannot proceed - database connection failed")
        return False
    
    # Step 2: Clean up old data (5 year rolling window)
    print("\nStep 2: Cleaning up old data...")
    try:
        cleanup_old_data()
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        # Continue anyway - cleanup failure shouldn't stop new data generation
    
    # Step 3: Generate weekly data
    print("\nStep 3: Generating weekly incremental data...")
    try:
        generate_weekly_data()
    except Exception as e:
        print(f"✗ Weekly data generation failed: {e}")
        return False
    
    # Step 4: Verify data was inserted
    print("\nStep 4: Verifying data insertion...")
    session = get_session()
    try:
        customer_count = session.query(Customer).count()
        booking_count = session.query(Booking).count()
        log_count = session.query(SupportLog).count()
        
        print(f"✓ Database now contains:")
        print(f"  - {customer_count} customers (total)")
        print(f"  - {booking_count} bookings (total)")
        print(f"  - {log_count} support logs (total)")
        
    finally:
        session.close()
    
    print("\n" + "="*60)
    print("✓✓✓ WEEKLY PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
    print("="*60 + "\n")
    
    return True


def run_quality_checks_on_existing_data():
    """
    Optional: Run quality checks on data already in the database.
    Useful for auditing existing data.
    """
    print("\n" + "="*60)
    print("RUNNING QUALITY CHECKS ON EXISTING DATA")
    print("="*60 + "\n")
    
    session = get_session()
    
    try:
        # Load bookings into DataFrame
        print("Loading bookings from database...")
        bookings = session.query(Booking).all()
        bookings_data = []
        for b in bookings:
            bookings_data.append({
                'booking_id': b.booking_id,
                'customer_id': b.customer_id,
                'check_in_date': b.check_in_date,
                'check_out_date': b.check_out_date,
                'amount_spent': b.amount_spent,
                'room_type': b.room_type,
                'booking_channel': b.booking_channel,
                'num_adults': b.num_adults,
                'num_children': b.num_children,
                'status': b.status
            })
        
        bookings_df = pd.DataFrame(bookings_data)
        
        # Validate bookings
        is_valid, error = validate_bookings(bookings_df)
        if not is_valid:
            print(f"✗ Booking validation failed: {error}")
        else:
            print("✓ All bookings are valid")
        
        # Load customers into DataFrame
        print("\nLoading customers from database...")
        customers = session.query(Customer).all()
        customers_data = []
        for c in customers:
            customers_data.append({
                'customer_id': c.customer_id,
                'join_date': c.join_date,
                'age': c.age,
                'job_category': c.job_category
            })
        
        customers_df = pd.DataFrame(customers_data)
        
        # Validate customers
        is_valid, error = validate_customers(customers_df)
        if not is_valid:
            print(f"✗ Customer validation failed: {error}")
        else:
            print("✓ All customers are valid")
        
        # Load support logs into DataFrame
        print("\nLoading support logs from database...")
        logs = session.query(SupportLog).all()
        logs_data = []
        for log in logs:
            logs_data.append({
                'interaction_id': log.interaction_id,
                'customer_id': log.customer_id,
                'date': log.date,
                'log_text': log.log_text
            })
        
        logs_df = pd.DataFrame(logs_data)
        
        # Validate support logs
        is_valid, error = validate_support_logs(logs_df)
        if not is_valid:
            print(f"✗ Support log validation failed: {error}")
        else:
            print("✓ All support logs are valid")
        
    finally:
        session.close()
    
    print("\n" + "="*60)
    print("✓✓✓ QUALITY CHECK AUDIT COMPLETED ✓✓✓")
    print("="*60 + "\n")