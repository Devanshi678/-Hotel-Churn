"""
Main entry point for the Hotel Churn Prediction data pipeline.
Run this file to execute the data generation process.
"""

import sys
from src.database import create_tables, test_connection
from src.ingest import run_historical_pipeline, run_weekly_pipeline


def main():
    """
    Main function that controls the pipeline execution.
    """
    print("\n" + "="*70)
    print(" HOTEL CHURN PREDICTION - DATA PIPELINE ")
    print("="*70 + "\n")
    
    # Check if database is accessible
    print("Checking database connection...")
    if not test_connection():
        print("\n✗ ERROR: Cannot connect to PostgreSQL database")
        print("Make sure Docker is running: docker-compose up -d")
        sys.exit(1)
    
    print("\n" + "-"*70)
    print("Select pipeline mode:")
    print("-"*70)
    print("1. Historical Data Generation (Run ONCE - generates 4 years of data)")
    print("2. Weekly Data Generation (Run WEEKLY - generates 7 days of data)")
    print("3. Create Database Tables Only")
    print("4. Exit")
    print("-"*70)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n→ Running HISTORICAL data pipeline...")
        
        # Create tables if they don't exist
        print("\nCreating database tables (if not exists)...")
        create_tables()
        
        # Run historical pipeline
        success = run_historical_pipeline()
        
        if success:
            print("\n✓✓✓ SUCCESS: Historical data loaded!")
            print("You can now proceed to feature engineering and model training.")
        else:
            print("\n✗✗✗ FAILED: Historical pipeline encountered errors.")
            sys.exit(1)
    
    elif choice == "2":
        print("\n→ Running WEEKLY data pipeline...")
        
        # Run weekly pipeline
        success = run_weekly_pipeline()
        
        if success:
            print("\n✓✓✓ SUCCESS: Weekly data added!")
        else:
            print("\n✗✗✗ FAILED: Weekly pipeline encountered errors.")
            sys.exit(1)
    
    elif choice == "3":
        print("\n→ Creating database tables...")
        create_tables()
        print("\n✓ Tables created successfully!")
    
    elif choice == "4":
        print("\nExiting...")
        sys.exit(0)
    
    else:
        print("\n✗ Invalid choice. Please enter 1, 2, 3, or 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()