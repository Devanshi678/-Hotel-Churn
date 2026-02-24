"""
Generates realistic fake hotel data using Faker library.
Handles both historical (4 years) and weekly incremental data generation.
Hill Station Hotel - Queen Bed, King Bed, Suite rooms
"""

from faker import Faker
import random
from datetime import datetime, timedelta
import yaml
from src.schema import Customer, Booking, SupportLog
from src.database import get_session

# Initialize Faker
fake = Faker()

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def generate_customers(num_customers):
    """
    Generate fake customer records.
    
    Args:
        num_customers: Number of customers to create
        
    Returns:
        List of Customer objects
    """
    customers = []
    job_categories = ['business_traveler', 'vacation_traveler', 'occasional_visitor']
    
    for _ in range(num_customers):
        customer = Customer(
            join_date=fake.date_between(start_date='-4y', end_date='today'),
            age=random.randint(21, 75),
            job_category=random.choice(job_categories)
        )
        customers.append(customer)
    
    print(f"✓ Generated {num_customers} customers")
    return customers


def generate_bookings_for_customer(customer, start_date, end_date):
    """
    Generate booking history for a single customer based on their segment.
    
    Args:
        customer: Customer object
        start_date: Start of date range
        end_date: End of date range
        
    Returns:
        List of Booking objects
    """
    bookings = []
    segment = config['customer_segments'][customer.job_category]
    
    # Determine number of bookings based on segment frequency
    days_in_range = (end_date - start_date).days
    years_in_range = days_in_range / 365
    
    if customer.job_category == 'business_traveler':
        num_bookings = int(years_in_range * 12)  # ~1 booking per month
    elif customer.job_category == 'vacation_traveler':
        num_bookings = int(years_in_range * 4)   # ~4 bookings per year
    else:  # occasional_visitor
        num_bookings = int(years_in_range * 1)   # ~1 booking per year
    
    # Add randomness
    num_bookings = max(1, int(num_bookings * random.uniform(0.5, 1.5)))
    
    # Generate bookings - Hill Station Hotel
    room_types = ['Queen Bed', 'King Bed', 'Suite']
    channels = ['Direct Website', 'Agent', 'OTA (Booking.com)', 'Walk-in']
    statuses = ['Checked-Out', 'Cancelled', 'No-Show']
    special_requests_options = ['None', 'Pet Friendly', 'Accessible Room', 'Crib', 'High Floor']
    
    for _ in range(num_bookings):
        # Random check-in date within range
        check_in = fake.date_between(start_date=start_date, end_date=end_date)
        
        # Stay duration based on segment
        avg_nights = segment['avg_nights_per_stay']
        nights = max(1, int(random.gauss(avg_nights, 2)))  # Normal distribution
        check_out = check_in + timedelta(days=nights)
        
        # Room pricing - Hill Station Hotel
        room_type = random.choice(room_types)
        base_price = {'Queen Bed': 120, 'King Bed': 150, 'Suite': 300}
        amount = base_price[room_type] * nights * random.uniform(0.8, 1.2)
        
        # Status (apply cancellation rate)
        if random.random() < config['business_rules']['cancellation_rate']:
            status = random.choice(['Cancelled', 'No-Show'])
        else:
            status = 'Checked-Out'
        
        booking = Booking(
            customer_id=customer.customer_id,
            check_in_date=check_in,
            check_out_date=check_out,
            amount_spent=round(amount, 2),
            room_type=room_type,
            booking_channel=random.choice(channels),
            num_adults=random.randint(1, 4),
            num_children=random.randint(0, 3),
            special_requests=random.choice(special_requests_options),
            status=status
        )
        bookings.append(booking)
    
    return bookings


def generate_support_logs_for_customer(customer, num_logs):
    """
    Generate fake customer support interaction logs.
    
    Args:
        customer: Customer object
        num_logs: Number of logs to create
        
    Returns:
        List of SupportLog objects
    """
    logs = []
    complaint_templates = [
        "Guest complained about slow Wi-Fi in room.",
        "Customer reported noisy neighbors on floor above.",
        "Request for late checkout was denied, guest unhappy.",
        "Room service took over 2 hours to deliver food.",
        "Air conditioning not working properly, maintenance called.",
        "Guest praised the friendly staff and mountain view.",
        "Complimentary upgrade to suite, customer very satisfied.",
        "Billing error resolved, guest appreciative of quick response.",
        "Guest requested extra blankets due to cold weather.",
        "Heating system malfunction reported, technician dispatched."
    ]
    
    for _ in range(num_logs):
        log = SupportLog(
            customer_id=customer.customer_id,
            date=fake.date_time_between(start_date='-2y', end_date='now'),
            log_text=random.choice(complaint_templates)
        )
        logs.append(log)
    
    return logs


def generate_historical_data():
    """
    Generate 4 years of historical data (ONE-TIME RUN).
    """
    print("\n=== Generating Historical Data ===")
    
    session = get_session()
    
    try:
        # Generate customers
        num_customers = config['data_generation']['historical']['num_customers']
        customers = generate_customers(num_customers)
        
        # Save customers to database first (to get customer_ids)
        session.add_all(customers)
        session.commit()
        print(f"✓ Saved {len(customers)} customers to database")
        
        # Generate bookings for each customer
        all_bookings = []
        start_date = datetime.now() - timedelta(days=4*365)
        end_date = datetime.now()
        
        for customer in customers:
            bookings = generate_bookings_for_customer(customer, start_date, end_date)
            all_bookings.extend(bookings)
        
        session.add_all(all_bookings)
        session.commit()
        print(f"✓ Saved {len(all_bookings)} bookings to database")
        
        # Generate support logs (30% of customers have at least 1 log)
        all_logs = []
        for customer in customers:
            if random.random() < 0.3:  # 30% chance
                num_logs = random.randint(1, 5)
                logs = generate_support_logs_for_customer(customer, num_logs)
                all_logs.extend(logs)
        
        session.add_all(all_logs)
        session.commit()
        print(f"✓ Saved {len(all_logs)} support logs to database")
        
        print("\n✓✓✓ Historical data generation complete! ✓✓✓\n")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error during data generation: {e}")
        raise
    finally:
        session.close()


def generate_weekly_data():
    """
    Generate 1 week of incremental data (WEEKLY RUN).
    """
    print("\n=== Generating Weekly Incremental Data ===")
    
    session = get_session()
    
    try:
        # Generate new customers
        num_new_customers = config['data_generation']['weekly']['new_customers_per_week']
        new_customers = generate_customers(num_new_customers)
        
        session.add_all(new_customers)
        session.commit()
        print(f"✓ Saved {len(new_customers)} new customers")
        
        # Generate bookings (mix of new and existing customers)
        num_bookings = config['data_generation']['weekly']['bookings_per_week']
        
        # Get existing customer IDs from database
        existing_customers = session.query(Customer).all()
        all_customers = existing_customers + new_customers
        
        all_bookings = []
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        for _ in range(num_bookings):
            customer = random.choice(all_customers)
            bookings = generate_bookings_for_customer(customer, start_date, end_date)
            if bookings:
                all_bookings.append(bookings[0])  # Take first booking
        
        session.add_all(all_bookings)
        session.commit()
        print(f"✓ Saved {len(all_bookings)} new bookings")
        
        print("\n✓✓✓ Weekly data generation complete! ✓✓✓\n")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error during weekly data generation: {e}")
        raise
    finally:
        session.close()