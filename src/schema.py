"""
Defines the database schema (tables) using SQLAlchemy ORM.
These Python classes will be converted to SQL CREATE TABLE commands.
"""

from sqlalchemy import Column, Integer, String, Date, Float, Text, ForeignKey, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class Customer(Base):
    """
    Stores static customer information.
    Updated rarely - only when new customers sign up.
    """
    __tablename__ = 'customers'
    
    customer_id = Column(Integer, primary_key=True, autoincrement=True)
    join_date = Column(Date, nullable=False)
    age = Column(Integer, nullable=False)
    job_category = Column(String(50), nullable=False)  # Business, Vacation, Occasional
    
    def __repr__(self):
        return f"<Customer(id={self.customer_id}, job={self.job_category})>"


class Booking(Base):
    """
    Stores all booking transactions.
    This is the main table for feature engineering.
    """
    __tablename__ = 'bookings'
    
    booking_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(Integer, ForeignKey('customers.customer_id'), nullable=False)
    check_in_date = Column(Date, nullable=False)
    check_out_date = Column(Date, nullable=False)
    amount_spent = Column(Float, nullable=False)
    room_type = Column(String(50), nullable=False)
    booking_channel = Column(String(50), nullable=False)
    num_adults = Column(Integer, nullable=False)
    num_children = Column(Integer, nullable=False)
    special_requests = Column(String(100))
    status = Column(String(20), nullable=False)  # Checked-Out, Cancelled, No-Show
    
    def __repr__(self):
        return f"<Booking(id={self.booking_id}, customer={self.customer_id}, status={self.status})>"


class SupportLog(Base):
    """
    Stores unstructured text from customer support interactions.
    Used for AI/NLP feature extraction.
    """
    __tablename__ = 'support_logs'
    
    interaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(Integer, ForeignKey('customers.customer_id'), nullable=False)
    date = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    log_text = Column(Text, nullable=False)
    
    def __repr__(self):
        return f"<SupportLog(id={self.interaction_id}, customer={self.customer_id})>"