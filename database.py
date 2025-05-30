import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

# Load environment variables
load_dotenv()

# Define the base for models
Base = declarative_base()

# Define models (duplicate from streamlit_app.py)
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    budget = Column(Float, nullable=True)
    group_comp = Column(String, nullable=True)
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    
    # Relationships
    ratings = relationship("Rating", back_populates="user")
    
class Place(Base):
    __tablename__ = "places"
    
    id = Column(Integer, primary_key=True, index=True)
    place_id = Column(String, unique=True, index=True)
    place_name = Column(String)
    place_id_encoded = Column(Integer, nullable=True)
    
    # Relationships
    ratings = relationship("Rating", back_populates="place")
    
class Rating(Base):
    __tablename__ = "ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    place_id = Column(Integer, ForeignKey("places.id"))
    rating = Column(Float)
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    place = relationship("Place", back_populates="ratings")

def get_database_url():
    """
    Get database URL from environment variables or Streamlit secrets
    """
    # First try to get DATABASE_URL directly
    db_url = os.environ.get('DATABASE_URL')
    
    if db_url:
        return db_url
    
    # Try to get from Streamlit secrets (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'postgres' in st.secrets:
            postgres = st.secrets['postgres']
            db_url = f"postgresql://{postgres['username']}:{postgres['password']}@{postgres['host']}:{postgres['port']}/{postgres['database']}"
            return db_url
    except (ImportError, KeyError, AttributeError):
        pass
    
    # Fallback for local development
    try:
        import getpass
        current_user = getpass.getuser()
        return f'postgresql://{current_user}@localhost:5432/tourism_recommender'
    except:
        return 'postgresql://localhost:5432/tourism_recommender'

def setup_database():
    """
    Creates the PostgreSQL database and initializes tables
    """
    # Get database URL
    db_url = get_database_url()
    
    # For Streamlit Cloud, we assume the database already exists
    # Only try to create database for local development
    if 'localhost' in db_url:
        # Extract database name from the URL
        db_parts = db_url.split('/')
        db_name = db_parts[-1]
        
        # Construct connection string without the database name (to connect to postgres db)
        base_conn_string = '/'.join(db_parts[:-1]) + '/postgres'
        
        try:
            print("Connecting to PostgreSQL...")
            # Connect to default postgres database
            conn = psycopg2.connect(base_conn_string)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if our database exists
            cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
            exists = cursor.fetchone()
            
            # If database doesn't exist, create it
            if not exists:
                print(f"Creating database {db_name}...")
                cursor.execute(f'CREATE DATABASE {db_name}')
                print(f"Database {db_name} created successfully.")
            else:
                print(f"Database {db_name} already exists.")
            
            # Close connection to postgres
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Note: Could not create database (this is normal for cloud deployments): {e}")
    
    # Connect to our database and create tables
    try:
        print(f"Connecting to database...")
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
        print("Database tables created successfully.")
        print("Database setup complete!")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise

if __name__ == "__main__":
    setup_database() 