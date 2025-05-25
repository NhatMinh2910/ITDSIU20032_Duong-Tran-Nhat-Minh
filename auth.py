import os
import hashlib
import sqlalchemy
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
from contextlib import contextmanager
from database import Base, User, Place, Rating
import getpass

# Get current OS username
current_user = getpass.getuser()

# Database setup
DATABASE_URL = os.environ.get('DATABASE_URL', f'postgresql://{current_user}@localhost:5432/tourism_recommender')

# Initialize SQLAlchemy
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Context manager for database sessions
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password, hashed_password):
    return hash_password(plain_password) == hashed_password

# Authentication functions
def register_user(username, email, password, age=None, gender=None, budget=None, group_comp=None):
    with get_db() as db:
        # Check if user already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            return False, "Username already exists"
            
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            return False, "Email already exists"
        
        # Create new user
        hashed_password = hash_password(password)
        new_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            age=age,
            gender=gender,
            budget=budget,
            group_comp=group_comp
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return True, new_user.id

def login_user(username, password):
    with get_db() as db:
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return False, "User not found"
        
        if not verify_password(password, user.hashed_password):
            return False, "Incorrect password"
        
        return True, user.id

# User profile update
def update_user_profile(user_id, age=None, gender=None, budget=None, group_comp=None):
    with get_db() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "User not found"
        
        if age is not None:
            user.age = age
        if gender is not None:
            user.gender = gender
        if budget is not None:
            user.budget = budget
        if group_comp is not None:
            user.group_comp = group_comp
            
        db.commit()
        return True, "Profile updated"

# Get user by ID
def get_user_by_id(user_id):
    with get_db() as db:
        return db.query(User).filter(User.id == user_id).first()

# Check if database needs initialization
def check_and_init_db():
    try:
        with get_db() as db:
            # Try to query users table
            db.query(User).first()
    except Exception:
        # If exception, initialize the database
        init_db() 