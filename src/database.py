"""
Establishes connection to PostgreSQL database running in Docker.
Reads credentials from .env file for security.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from src.schema import Base

# Load environment variables from .env file
load_dotenv()

# Read database credentials
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

# Create database connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create the engine (connection to database)
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory (used to interact with database)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """
    Creates all tables defined in schema.py in the PostgreSQL database.
    Only needs to be run once at the start.
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created successfully!")


def get_session():
    """
    Returns a database session for inserting/querying data.
    Use this in data_generator.py and ingest.py
    """
    session = SessionLocal()
    try:
        return session
    except Exception as e:
        session.close()
        raise e


def test_connection():
    from sqlalchemy import text
    try:
        session = get_session()
        session.execute(text("SELECT 1"))
        print("✓ Database connection successful!")
        session.close()
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False