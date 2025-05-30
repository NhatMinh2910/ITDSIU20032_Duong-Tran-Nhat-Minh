import os
import hashlib
import sqlalchemy
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
from contextlib import contextmanager
from database import Base, User, Place, Rating, get_database_url

# Database setup
DATABASE_URL = get_database_url()

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

# Save user ratings
def save_user_rating(user_id, place_id, rating_value):
    # Convert numpy types to Python native types
    if hasattr(place_id, 'item'):
        place_id = place_id.item()
    if hasattr(rating_value, 'item'):
        rating_value = rating_value.item()
    
    with get_db() as db:
        # Get place from database
        place = db.query(Place).filter(Place.place_id == str(place_id)).first()
        
        if not place:
            # If place doesn't exist, create it
            place = Place(place_id=str(place_id), place_name=str(place_id))
            db.add(place)
            db.commit()
            db.refresh(place)
        
        # Check if rating already exists
        existing_rating = db.query(Rating).filter(
            Rating.user_id == user_id, 
            Rating.place_id == place.id
        ).first()
        
        if existing_rating:
            # Update existing rating
            existing_rating.rating = rating_value
        else:
            # Create new rating
            new_rating = Rating(user_id=user_id, place_id=place.id, rating=rating_value)
            db.add(new_rating)
        
        db.commit()
        return True

# Get user ratings
def get_user_ratings(user_id):
    with get_db() as db:
        ratings = db.query(Rating, Place).join(Place).filter(Rating.user_id == user_id).all()
        return [(place.place_id, place.place_name, float(rating.rating)) for rating, place in ratings]

# Get user by ID
def get_user_by_id(user_id):
    with get_db() as db:
        return db.query(User).filter(User.id == user_id).first()

# Delete all user ratings
def delete_all_user_ratings(user_id):
    with get_db() as db:
        # Delete all ratings for this user
        deleted_count = db.query(Rating).filter(Rating.user_id == user_id).delete()
        db.commit()
        return deleted_count

# Check if database needs initialization
def check_and_init_db():
    try:
        with get_db() as db:
            # Try to query users table
            db.query(User).first()
    except Exception:
        # If exception, initialize the database
        init_db() 