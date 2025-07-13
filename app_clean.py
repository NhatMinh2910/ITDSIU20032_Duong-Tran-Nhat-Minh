import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import json
import os

# Import authentication and database functions
from auth import (
    get_db, init_db, hash_password, verify_password, register_user, 
    login_user, update_user_profile, save_user_rating, get_user_ratings, 
    get_user_by_id, check_and_init_db, delete_all_user_ratings
)
from database import get_database_url

# Set page configuration
st.set_page_config(
    page_title="TravelMate",
    page_icon="🌍",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Initialize database
check_and_init_db()

# Display header
st.title("TravelMate")
st.markdown("### Tourism Recommendation System  Using NCF Model")

# Authentication UI
if not st.session_state['authenticated']:
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")
            
            if submit_login:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state['authenticated'] = True
                        st.session_state['user_id'] = result
                        st.session_state['username'] = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result)
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Register")
            new_username = st.text_input("Username")
            email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Optional profile information
            st.subheader("Profile Information (Optional)")
            age = st.slider("Age", 10, 100, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            budget = st.number_input("Budget", 0, 10000, 500)
            group_comp = st.selectbox("Group Composition", 
                                     ["Solo", "Family", "Couple", "Friends", "Other"])
            
            submit_register = st.form_submit_button("Register")
            
            if submit_register:
                if not new_username or not email or not new_password:
                    st.error("Please enter username, email, and password")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                else:
                    success, result = register_user(
                        new_username, 
                        email, 
                        new_password,
                        age=age,
                        gender=gender,
                        budget=budget,
                        group_comp=group_comp
                    )
                    if success:
                        st.success("Registration successful! Please log in.")
                    else:
                        st.error(result)

else:
    # User is authenticated
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        # Clear all session state related to the current user
        keys_to_clear = [key for key in st.session_state.keys() if 'rating_seed' in key]
        for key in keys_to_clear:
            del st.session_state[key]
            
        st.session_state['authenticated'] = False
        st.session_state['user_id'] = None
        st.session_state['username'] = None
        st.rerun()

    # Load and preprocess data
    @st.cache_data
    def load_data():
        # Load datasets
        user_place_data = pd.read_csv('https://raw.githubusercontent.com/NhatMinh2910/Pre-thesis-Datasets/refs/heads/main/rats.csv')
        user_features = pd.read_csv('https://raw.githubusercontent.com/NhatMinh2910/Pre-thesis-Datasets/refs/heads/main/ufeat.csv')
        
        # Load place information from icat.csv
        try:
            place_info = pd.read_csv('icat.csv')
            # st.info(f"✅ Loaded {len(place_info)} places from icat.csv")
        except FileNotFoundError:
            # st.warning("⚠️ icat.csv not found. Using fallback place names.")
            place_info = None
        
        # Keep only needed columns
        user_features = user_features[['user_id', 'Age', 'Gender', 'Budget', 'GroupComp']]
        
        # Fill missing values
        user_features.fillna(0, inplace=True)
        
        # Map Gender to numeric
        user_features['Gender'] = user_features['Gender'].map({'Male': 1, 'Female': 0})
        
        # One-hot encode GroupComp
        user_features = pd.get_dummies(user_features, columns=['GroupComp'])
        
        # Encode user_id and place_id
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        user_place_data['user_id_encoded'] = user_encoder.fit_transform(user_place_data['user_id'])
        user_place_data['place_id_encoded'] = item_encoder.fit_transform(user_place_data['place_id'])
        
        # Normalize rating (same as in notebook)
        if user_place_data['rating'].max() > 1:
            user_place_data['rating'] = user_place_data['rating'] / user_place_data['rating'].max()
        
        # Create a mapping of place_id to place name
        places = pd.DataFrame({
            'place_id': user_place_data['place_id'].unique(),
            'place_id_encoded': item_encoder.transform(user_place_data['place_id'].unique())
        })
        
        # Map place names from icat.csv if available
        if place_info is not None:
            # st.info(f"🔍 Debug: User data has {len(user_place_data['place_id'].unique())} unique place_ids")
            # st.info(f"🔍 Debug: icat.csv has {len(place_info['place_id'].unique())} unique place_ids")
            # st.info(f"🔍 Debug: User place_id sample: {user_place_data['place_id'].unique()[:5].tolist()}")
            # st.info(f"🔍 Debug: icat place_id sample: {place_info['place_id'].unique()[:5].tolist()}")
            
            # Merge with place_info to get actual place names, categories, and quality
            places = places.merge(
                place_info[['place_id', 'ItemName', 'Category', 'Quality']], 
                on='place_id', 
                how='left'
            )
            
            successful_merges = places['ItemName'].notna().sum()
            # st.info(f"🔍 Debug: Successful merges: {successful_merges}/{len(places)}")
            if successful_merges == 0:
                # st.error("❌ No place names matched! There may be a data mismatch.")
                # st.info(f"🔍 Sample places after merge: {places[['place_id', 'ItemName']].head().to_dict()}")
                pass
            
            # Use ItemName as place_name, fallback to place_id if ItemName is missing
            # Create meaningful names for places not in icat.csv
            def create_place_name(place_id, item_name):
                if pd.notna(item_name):
                    return item_name
                else:
                    # Create a more meaningful name based on place_id
                    place_categories = [
                        "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                        "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                        "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                        "Historic Site", "Art Gallery", "Theme Park", "Marina",
                        "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                        "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                    ]
                    location_names = [
                        "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                        "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                        "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                    ]
                    
                    category_idx = int(place_id) % len(place_categories)
                    location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                    
                    return f"{location_names[location_idx]} {place_categories[category_idx]}"
            
            places['place_name'] = places.apply(lambda row: create_place_name(row['place_id'], row['ItemName']), axis=1)
            places['category'] = places['Category'].fillna('Tourism')
            places['quality'] = places['Quality'].fillna(3.5)  # Default to 3.5/5.0
            
            # Clean up category format (remove brackets and quotes)
            places['category'] = places['category'].astype(str).str.replace(r'[\[\]\']', '', regex=True)
            
            # st.success(f"✅ Successfully mapped {len(places[places['ItemName'].notna()])} place names from icat.csv")
        else:
            # Fallback: regenerate places from loaded data if mapping file missing
            # st.warning("Place mapping not found, regenerating...")
            places = pd.DataFrame({
                'place_id': user_place_data['place_id'].unique(),
                'place_id_encoded': item_encoder.transform(user_place_data['place_id'].unique())
            })
            
            # Load place information from icat.csv for fallback
            try:
                place_info = pd.read_csv('icat.csv')
                # Merge with place_info to get actual place names, categories, and quality
                places = places.merge(
                    place_info[['place_id', 'ItemName', 'Category', 'Quality']], 
                    on='place_id', 
                    how='left'
                )
                
                # Use ItemName as place_name, fallback to place_id if ItemName is missing
                # Create meaningful names for places not in icat.csv
                def create_place_name(place_id, item_name):
                    if pd.notna(item_name):
                        return item_name
                    else:
                        # Create a more meaningful name based on place_id
                        place_categories = [
                            "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                            "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                            "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                            "Historic Site", "Art Gallery", "Theme Park", "Marina",
                            "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                            "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                        ]
                        location_names = [
                            "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                            "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                            "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                        ]
                        
                        category_idx = int(place_id) % len(place_categories)
                        location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                        
                        return f"{location_names[location_idx]} {place_categories[category_idx]}"
                
                places['place_name'] = places.apply(lambda row: create_place_name(row['place_id'], row['ItemName']), axis=1)
                places['category'] = places['Category'].fillna('Tourism')
                places['quality'] = places['Quality'].fillna(3.5)  # Default to 3.5/5.0
                
                # Clean up category format (remove brackets and quotes)
                places['category'] = places['category'].astype(str).str.replace(r'[\[\]\']', '', regex=True)
                
                # st.success(f"✅ Successfully mapped {len(places[places['ItemName'].notna()])} place names from icat.csv (fallback)")
            except FileNotFoundError:
                # st.warning("⚠️ icat.csv not found. Using fallback place names.")
                place_info = None
                
                # Create meaningful names for all places
                def create_place_name(place_id):
                    # Create a more meaningful name based on place_id
                    place_categories = [
                        "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                        "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                        "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                        "Historic Site", "Art Gallery", "Theme Park", "Marina",
                        "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                        "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                    ]
                    location_names = [
                        "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                        "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                        "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                    ]
                    
                    category_idx = int(place_id) % len(place_categories)
                    location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                    
                    return f"{location_names[location_idx]} {place_categories[category_idx]}"
                
                places['place_name'] = places['place_id'].apply(create_place_name)
                places['category'] = 'Tourism'
                places['quality'] = 3.5
        
        return user_place_data, user_features, user_encoder, item_encoder, places

    # Build the NCF model
    def build_ncf_model(num_users, num_items, user_feat_dim, embedding_size=32):
        """
        Build an optimized Neural Collaborative Filtering model
        with reduced complexity and improved regularization.
        """
        # Use more efficient embedding size (32 instead of 50)
        user_input = Input(shape=(1,), name='user_input')
        place_input = Input(shape=(1,), name='place_input')
        user_features_input = Input(shape=(user_feat_dim,), name='user_features_input')
        
        # Add stronger regularization for embeddings
        user_embedding = Embedding(num_users, embedding_size,
                                  embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
                                  name='user_embedding')(user_input)
        place_embedding = Embedding(num_items, embedding_size,
                                   embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
                                   name='place_embedding')(place_input)
        
        user_flat = Flatten()(user_embedding)
        place_flat = Flatten()(place_embedding)
        
        # Add batch normalization for feature inputs to improve training
        user_features_normalized = tf.keras.layers.BatchNormalization()(user_features_input)
        
        # Concatenate embeddings and weighted user features
        combined = Concatenate()([user_flat, place_flat, user_features_normalized])
        
        # More efficient network architecture with batch normalization
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)  # Lower dropout for better efficiency
        
        x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[user_input, place_input, user_features_input], outputs=output)
        
        # Use Adam optimizer with a slightly higher learning rate and beta parameters
        optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model

    # Cache model to avoid reloading
    @st.cache_resource
    def load_cached_model(model_path, num_users, num_items, user_feat_dim):
        """Load and cache the model to improve performance"""
        model = build_ncf_model(
            num_users=num_users,
            num_items=num_items,
            user_feat_dim=user_feat_dim
        )
        try:
            model.load_weights(model_path)
            print("Model loaded from cache")
        except:
            print("No cached model found")
        return model

    # Load data
    user_place_data, user_features, user_encoder, item_encoder, places = load_data()

    # Detect if we're running on Streamlit Cloud (read-only filesystem)
    def is_streamlit_cloud():
        """Detect if running on Streamlit Cloud"""
        return (
            os.path.exists('/mount/src') or  # Streamlit Cloud specific path
            'STREAMLIT_SHARING' in os.environ or
            'STREAMLIT_CLOUD' in os.environ or
            not os.access('.', os.W_OK)  # Check if current directory is writable
        )

    # Model management for cloud vs local
    model_path = "ncf_model"
    model_status = "unknown"
    is_cloud = is_streamlit_cloud()
    
    if is_cloud:
        feature_cols = [c for c in user_features.columns if c != 'user_id']
        
        model = build_ncf_model(
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            user_feat_dim=len(feature_cols)
        )
        
    elif not os.path.exists(f"{model_path}.index"):
        # Local environment: Train and save model
        model_status = "training_new"
        # Check if we have enough columns in user_features
        feature_cols = [c for c in user_features.columns if c != 'user_id']
        
        # Build model
        model = build_ncf_model(
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            user_feat_dim=len(feature_cols)
        )
        
        st.warning("⚠️ Model not found. Training a new model (this may take a while)...")
        
        # Prepare data for training
        X_user = user_place_data['user_id_encoded'].values
        X_place = user_place_data['place_id_encoded'].values
        
        # Merge user features
        user_place_with_features = user_place_data.merge(user_features, on='user_id', how='left')
        
        # Apply weights to features
        age_idx = feature_cols.index('Age') 
        gender_idx = feature_cols.index('Gender')
        budget_idx = feature_cols.index('Budget') 
        
        # GroupComp columns (all columns that start with 'GroupComp_')
        groupcomp_indices = [i for i, c in enumerate(feature_cols) if c.startswith('GroupComp_')]
        
        # Define weights per feature
        weights = np.ones(len(feature_cols), dtype=np.float32) * 0.0
        weights[age_idx] = 0.3
        weights[gender_idx] = 0.3
        weights[budget_idx] = 0.2
        # Distribute GroupComp total weight 0.2 evenly
        for idx in groupcomp_indices:
            weights[idx] = 0.2 / len(groupcomp_indices)
        
        X_user_features = user_place_with_features[feature_cols].values.astype(np.float32)
        X_user_features_weighted = X_user_features * weights
        
        y = user_place_data['rating'].values
        
        # Show training data info
        st.info(f"Training model with {len(X_user)} ratings, {len(user_encoder.classes_)} users, {len(item_encoder.classes_)} places")
        
        # Add early stopping to prevent overfitting and save training time
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        with st.spinner("Training model..."):
            history = model.fit(
                [X_user, X_place, X_user_features_weighted], 
                y,
                epochs=15,  # Increased max epochs since we have early stopping
                batch_size=128,  # Larger batch size for faster training
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
        
        # Save the model (only on local)
        try:
            model.save_weights(model_path)
            model_status = "trained_new"
            st.success(f"Model trained successfully! Final loss: {history.history['loss'][-1]:.4f}")
            
            # Save mapping dictionaries
            with open("place_mapping.json", "w") as f:
                json.dump(places.to_dict(orient="records"), f)
        except Exception as e:
            st.warning(f"Could not save model (this is normal on cloud): {str(e)}")
            model_status = "trained_unsaved"
    
    else:
        # Local environment: Load existing model
        model_status = "loaded_existing"
        # Load the model using caching
        feature_cols = [c for c in user_features.columns if c != 'user_id']
        model = load_cached_model(
            model_path,
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            user_feat_dim=len(feature_cols)
        )
        
        # Load mapping dictionary
        if os.path.exists("place_mapping.json"):
            with open("place_mapping.json", "r") as f:
                places_list = json.load(f)
                places = pd.DataFrame(places_list)
        else:
            # Fallback: regenerate places from loaded data if mapping file missing
            places = pd.DataFrame({
                'place_id': user_place_data['place_id'].unique(),
                'place_id_encoded': item_encoder.transform(user_place_data['place_id'].unique())
            })
            
            # Load place information from icat.csv for fallback
            try:
                place_info = pd.read_csv('icat.csv')
                # Merge with place_info to get actual place names, categories, and quality
                places = places.merge(
                    place_info[['place_id', 'ItemName', 'Category', 'Quality']], 
                    on='place_id', 
                    how='left'
                )
                
                # Use ItemName as place_name, fallback to place_id if ItemName is missing
                # Create meaningful names for places not in icat.csv
                def create_place_name(place_id, item_name):
                    if pd.notna(item_name):
                        return item_name
                    else:
                        # Create a more meaningful name based on place_id
                        place_categories = [
                            "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                            "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                            "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                            "Historic Site", "Art Gallery", "Theme Park", "Marina",
                            "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                            "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                        ]
                        location_names = [
                            "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                            "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                            "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                        ]
                        
                        category_idx = int(place_id) % len(place_categories)
                        location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                        
                        return f"{location_names[location_idx]} {place_categories[category_idx]}"
                
                places['place_name'] = places.apply(lambda row: create_place_name(row['place_id'], row['ItemName']), axis=1)
                places['category'] = places['Category'].fillna('Tourism')
                places['quality'] = places['Quality'].fillna(3.5)  # Default to 3.5/5.0
                
                # Clean up category format (remove brackets and quotes)
                places['category'] = places['category'].astype(str).str.replace(r'[\[\]\']', '', regex=True)
                
            except FileNotFoundError:
                place_info = None
                
                # Create meaningful names for all places
                def create_place_name(place_id):
                    # Create a more meaningful name based on place_id
                    place_categories = [
                        "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                        "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                        "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                        "Historic Site", "Art Gallery", "Theme Park", "Marina",
                        "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                        "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                    ]
                    location_names = [
                        "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                        "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                        "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                    ]
                    
                    category_idx = int(place_id) % len(place_categories)
                    location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                    
                    return f"{location_names[location_idx]} {place_categories[category_idx]}"
                
                places['place_name'] = places['place_id'].apply(create_place_name)
                places['category'] = 'Tourism'
                places['quality'] = 3.5

    # Ensure places dataframe is always available
    if 'places' not in locals():
        st.error("Places data not loaded properly. Please refresh the page.")
        st.stop()
    
    with st.sidebar:
        st.sidebar.title("User Profile")

    # User profile section
    user = get_user_by_id(st.session_state['user_id'])
        
    with st.sidebar.expander("Update Profile"):
        with st.form("profile_update_form"):
            updated_age = st.slider("Age", 10, 100, int(user.age) if user.age else 30)
            updated_gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                          index=0 if user.gender == "Male" else 1 if user.gender == "Female" else 2)
            updated_budget = st.number_input("Budget", 0, 10000, int(user.budget) if user.budget else 500)
            updated_group_comp = st.selectbox("Group Composition", 
                                             ["Solo", "Family", "Couple", "Friends", "Other"],
                                             index=["Solo", "Family", "Couple", "Friends", "Other"].index(user.group_comp) if user.group_comp in ["Solo", "Family", "Couple", "Friends", "Other"] else 0)
            
            profile_submit = st.form_submit_button("Update Profile")
            
            if profile_submit:
                success, message = update_user_profile(
                    st.session_state['user_id'],
                    age=updated_age,
                    gender=updated_gender,
                    budget=updated_budget,
                    group_comp=updated_group_comp
                )
                if success:
                    st.success("Profile updated!")
                else:
                    st.error(message)

    # Reset Ratings section
    with st.sidebar.expander("🗑️ Reset Ratings"):
        # Show current rating count if any
        current_ratings_count = len(get_user_ratings(st.session_state['user_id']))
        if current_ratings_count > 0:
            st.write(f"You currently have **{current_ratings_count} ratings** saved.")
            
            if st.button("🗑️ Delete All Ratings", type="secondary"):
                # Delete all ratings
                deleted_count = delete_all_user_ratings(st.session_state['user_id'])
                if deleted_count > 0:
                    st.success(f"✅ Successfully deleted {deleted_count} ratings!")
                    st.balloons()
                    # Force page refresh to update the UI
                    st.rerun()
                else:
                    st.info("No ratings found to delete.")
        else:
            st.info("You have no ratings to delete.")

    # Rating system
    st.subheader("Rate Tourism Destinations")
    
    # Check if places data is available
    try:
        if len(places) == 0:
            st.error("No places data available. Please contact support.")
            st.stop()
    except Exception as e:
        st.error(f"Error accessing places data: {str(e)}")
        st.error("Please refresh the page. If the problem persists, contact support.")
        st.stop()
    
    # Get existing ratings to show what user has already rated
    existing_ratings = get_user_ratings(st.session_state['user_id'])
    
    # Fix data type mismatch: ensure place_ids are consistent between database and DataFrame
    # Database stores place_ids as strings, but DataFrame might have them as integers
    existing_ratings_fixed = []
    for place_id, place_name, rating in existing_ratings:
        # Convert place_id to the same type as in places DataFrame
        try:
            # Try to convert to int if places DataFrame uses integers
            if len(places) > 0 and isinstance(places['place_id'].iloc[0], (int, float)):
                place_id_fixed = int(place_id)
            else:
                place_id_fixed = str(place_id)
        except (ValueError, TypeError):
            place_id_fixed = place_id
        
        existing_ratings_fixed.append((place_id_fixed, place_name, rating))
    
    existing_ratings_dict = {place_id: rating for place_id, _, rating in existing_ratings_fixed}
    
    # Show rating progress - limited to 5 places
    REQUIRED_RATINGS = 5
    total_places = len(places)
    rated_count = len(existing_ratings_fixed)
    
    # Progress bar and status
    progress = min(rated_count / REQUIRED_RATINGS, 1.0)
    st.progress(progress)
    
    if rated_count < REQUIRED_RATINGS:
        remaining = REQUIRED_RATINGS - rated_count
        st.write(f"**Progress: {rated_count}/{REQUIRED_RATINGS} destinations rated** | {remaining} more needed for recommendations")
    else:
        st.write(f"**✅ Complete: {rated_count}/{REQUIRED_RATINGS} destinations rated** | Ready for recommendations!")
    
    # Show existing ratings first
    if existing_ratings_fixed:
        with st.expander(f"Your Ratings ({len(existing_ratings_fixed)} places rated)", expanded=False):
            # Show rating statistics and reset button
            col1, col2 = st.columns([2, 1])
            with col1:
                avg_rating = sum(rating for _, _, rating in existing_ratings_fixed) / len(existing_ratings_fixed)
                st.write(f"**Average Rating:** {avg_rating:.1f}/5.0 ⭐")
            with col2:
                if st.button("🗑️ Reset All", key="quick_reset", help="Delete all ratings"):
                    deleted_count = delete_all_user_ratings(st.session_state['user_id'])
                    if deleted_count > 0:
                        st.success(f"✅ Deleted {deleted_count} ratings!")
                        st.balloons()
                        st.rerun()
            
            # Show individual ratings
            for place_id, place_name, rating in existing_ratings_fixed:
                # Get enhanced place information from places DataFrame
                place_info = places[places['place_id'] == place_id]
                if len(place_info) > 0:
                    place_data = place_info.iloc[0]
                    display_name = str(place_data.get('place_name', f'Place {place_id}'))
                    category = place_data.get('category', 'Tourism')
                    quality = place_data.get('quality', 3.5)
                else:
                    # Fallback: generate name if not found in places DataFrame
                    def create_place_name_fallback(place_id):
                        place_categories = [
                            "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                            "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                            "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                            "Historic Site", "Art Gallery", "Theme Park", "Marina",
                            "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                            "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                        ]
                        location_names = [
                            "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                            "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                            "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                        ]
                        
                        category_idx = int(place_id) % len(place_categories)
                        location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                        
                        return f"{location_names[location_idx]} {place_categories[category_idx]}"
                    
                    display_name = create_place_name_fallback(place_id)
                    category = 'Tourism'
                    quality = 3.5
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{display_name}**")
                    if category and category != 'Unknown':
                        st.caption(f"📍 {category}")
                    if quality > 0:
                        quality_stars = "⭐" * min(int(quality), 5)
                        st.caption(f"Quality: {quality:.1f}/5.0 {quality_stars}")
                with col2:
                    st.write(f"⭐ {rating}/5")
    
    # Rating interface - only show if user hasn't rated 10 places yet
    if rated_count < REQUIRED_RATINGS:
        # Use consistent place selection based on user ID to avoid random sampling
        seed_key = f'rating_seed_{st.session_state["user_id"]}'
        if seed_key not in st.session_state:
            st.session_state[seed_key] = hash(str(st.session_state['user_id'])) % 1000
        
        user_seed = st.session_state[seed_key]
        
        # Select places that user hasn't rated yet, but in a consistent way
        try:
            unrated_places = places[~places['place_id'].isin(existing_ratings_dict.keys())]
        except Exception as e:
            st.error(f"Error filtering unrated places: {str(e)}")
            st.error("This might be caused by missing places data. Please refresh the page.")
            st.stop()
        
        if len(unrated_places) > 0:
            # Calculate how many more ratings are needed
            remaining_needed = REQUIRED_RATINGS - rated_count
            
            # Show places for rating - show up to 5 at a time, but limit total to what's needed
            num_to_show = min(5, remaining_needed, len(unrated_places))
            
            if len(unrated_places) > num_to_show:
                sample_places = unrated_places.sample(n=num_to_show, random_state=user_seed)
            else:
                sample_places = unrated_places.head(num_to_show)
            
            with st.form("rating_form"):
                st.write(f"**Rate {num_to_show} more destinations:**")
                
                # Create two columns for each place
                num_cols = 2
                ratings = {}
                
                # Display places in rows with 2 columns each
                for i in range(0, len(sample_places), num_cols):
                    cols = st.columns(num_cols)
                    
                    for j in range(num_cols):
                        if i+j < len(sample_places):
                            place = sample_places.iloc[i+j]
                            place_id = place['place_id']
                            place_name = place['place_name']
                            
                            # Format the place name to be more readable
                            display_name = str(place_name).replace("_", " ").title()
                            
                            with cols[j]:
                                # Display the destination name with prominent styling
                                st.markdown(f"### {display_name}")
                                
                                # Add category and quality information if available
                                if 'category' in place and place['category'] != 'Unknown':
                                    st.markdown(f"**Category:** {place['category']}")
                                
                                if 'quality' in place and place['quality'] > 0:
                                    quality_stars = "⭐" * min(int(place['quality']), 5)
                                    st.markdown(f"**Quality:** {place['quality']:.1f}/5.0 {quality_stars}")
                                
                                # Add a rating slider below the name
                                ratings[place_id] = st.slider(
                                    f"Rate {display_name}", 
                                    1, 5, 3, 
                                    key=f"rating_{place_id}"
                                )
                
                st.markdown("---")
                # Submit button for new ratings
                rating_submit = st.form_submit_button("Save Ratings")
                
                if rating_submit:
                    for place_id, rating in ratings.items():
                        save_user_rating(st.session_state['user_id'], place_id, rating)
                    st.success("Ratings saved!")
                    st.balloons()
                    st.rerun()  # Refresh to show updated ratings
        
        # Option to see different places if there are more available
        if len(unrated_places) > 5 and remaining_needed > 0:
            if st.button("Show Different Places to Rate"):
                # Change the seed to show different places
                new_seed = (user_seed + 1) % 1000
                st.session_state[seed_key] = new_seed
                st.rerun()
        
        # Add a way to modify existing ratings
        if existing_ratings_fixed:
            with st.expander("Modify Existing Ratings"):
                st.write("**Update your previous ratings:**")
                
                with st.form("modify_ratings_form"):
                    modified_ratings = {}
                    
                    # Show existing ratings in a form to modify
                    for place_id, place_name, current_rating in existing_ratings_fixed[:10]:  # Limit to 10 for UI
                        # Get the proper display name from places DataFrame
                        place_info = places[places['place_id'] == place_id]
                        if len(place_info) > 0:
                            display_name = str(place_info.iloc[0].get('place_name', f'Place {place_id}'))
                        else:
                            # Fallback: generate name if not found in places DataFrame
                            def create_place_name_fallback(place_id):
                                place_categories = [
                                    "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                                    "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                                    "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                                    "Historic Site", "Art Gallery", "Theme Park", "Marina",
                                    "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                                    "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                                ]
                                location_names = [
                                    "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                                    "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                                    "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                                ]
                                
                                category_idx = int(place_id) % len(place_categories)
                                location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                                
                                return f"{location_names[location_idx]} {place_categories[category_idx]}"
                            
                            display_name = create_place_name_fallback(place_id)
                        
                        modified_ratings[place_id] = st.slider(
                            f"{display_name}",
                            1, 5, int(current_rating),
                            key=f"modify_{place_id}"
                        )
                    
                    modify_submit = st.form_submit_button("Update Selected Ratings")
                    
                    if modify_submit:
                        for place_id, rating in modified_ratings.items():
                            save_user_rating(st.session_state['user_id'], place_id, rating)
                        st.success("Ratings updated!")
                        st.rerun()
    
    elif rated_count >= REQUIRED_RATINGS:
        # Recommendations section - automatic when 10 places are rated
        # Force refresh of cached computations when ratings change
        current_ratings_hash = hash(str(sorted(get_user_ratings(st.session_state['user_id']))))
        if 'last_ratings_hash' not in st.session_state:
            st.session_state['last_ratings_hash'] = current_ratings_hash
        elif st.session_state['last_ratings_hash'] != current_ratings_hash:
            st.session_state['last_ratings_hash'] = current_ratings_hash
            st.info("🔄 Detected rating changes, updating recommendations...")
        
        # Get user profile from database
        user = get_user_by_id(st.session_state['user_id'])
        
        # Get user ratings from database
        user_ratings = get_user_ratings(st.session_state['user_id'])
        
        # Fix data type mismatch for recommendations too
        user_ratings_fixed = []
        for place_id, place_name, rating in user_ratings:
            try:
                # Convert to same type as places DataFrame
                if len(places) > 0 and isinstance(places['place_id'].iloc[0], (int, float)):
                    place_id_fixed = int(place_id)
                else:
                    place_id_fixed = str(place_id)
            except (ValueError, TypeError):
                place_id_fixed = place_id
            user_ratings_fixed.append((place_id_fixed, place_name, rating))
        
        rated_place_ids = [place_id for place_id, _, _ in user_ratings_fixed]
        
        # Create user feature vector that matches training data format
        feature_cols = [c for c in user_features.columns if c != 'user_id']
        user_feature_vector = np.zeros(len(feature_cols), dtype=np.float32)
        
        # Map user profile to feature vector
        if 'Age' in feature_cols:
            age_idx = feature_cols.index('Age')
            user_feature_vector[age_idx] = float(user.age) if user.age else 30.0
            
        if 'Gender' in feature_cols:
            gender_idx = feature_cols.index('Gender')
            user_feature_vector[gender_idx] = 1.0 if user.gender == "Male" else 0.0
            
        if 'Budget' in feature_cols:
            budget_idx = feature_cols.index('Budget')
            user_feature_vector[budget_idx] = float(user.budget) if user.budget else 500.0
        
        # Handle GroupComp one-hot encoding
        group_mapping = {
            "Solo": "1Adlt",
            "Couple": "2Adlt", 
            "Family": "2Adlt+Child",
            "Friends": "GrpFriends",
            "Other": "1Adlt"
        }
        dataset_group = group_mapping.get(user.group_comp, "1Adlt")
        group_col = f"GroupComp_{dataset_group}"
        
        if group_col in feature_cols:
            group_idx = feature_cols.index(group_col)
            user_feature_vector[group_idx] = 1.0
        
        # Apply same weights as training
        weights = np.ones(len(feature_cols), dtype=np.float32) * 0.0
        if 'Age' in feature_cols:
            weights[feature_cols.index('Age')] = 0.3
        if 'Gender' in feature_cols:
            weights[feature_cols.index('Gender')] = 0.3
        if 'Budget' in feature_cols:
            weights[feature_cols.index('Budget')] = 0.2
        
        groupcomp_indices = [i for i, c in enumerate(feature_cols) if c.startswith('GroupComp_')]
        for idx in groupcomp_indices:
            weights[idx] = 0.2 / len(groupcomp_indices) if len(groupcomp_indices) > 0 else 0.0
        
        user_features_weighted = user_feature_vector * weights
        
        # For cold start: find most similar user in training data
        user_place_with_features = user_place_data.merge(user_features, on='user_id', how='left')
        training_features = user_place_with_features[feature_cols]
        training_features_weighted = training_features.values * weights
        
        # Find most similar user
        similarities = np.sum((training_features_weighted - user_features_weighted) ** 2, axis=1)
        most_similar_idx = np.argmin(similarities)
        
        # Use the actual encoded user ID, not the dataframe row index
        similar_user_encoded = user_place_with_features.iloc[most_similar_idx]['user_id_encoded']
        
        # Validate that the user ID is within the valid range
        max_user_id = len(user_encoder.classes_) - 1
        if similar_user_encoded > max_user_id:
            similar_user_encoded = max_user_id // 2  # Use middle user as fallback
        
        # Get unrated places
        all_places = places['place_id'].values
        unrated_places = [p for p in all_places if p not in rated_place_ids]
        
        st.info(f"🎯 **Analyzing your preferences** from {len(user_ratings_fixed)} ratings to recommend from {len(unrated_places)} remaining destinations...")
        
        predictions = []
        
        # Make predictions for all unrated places
        with st.spinner("Generating your top 3 recommendations..."):
            for place_id in unrated_places:
                try:
                    # Get place encoding
                    place_info = places[places['place_id'] == place_id]
                    if len(place_info) == 0:
                        continue
                        
                    place_encoded = place_info.iloc[0]['place_id_encoded']
                    if place_encoded >= len(item_encoder.classes_):
                        continue
                    
                    # Make prediction
                    pred = model.predict([
                        np.array([similar_user_encoded]), 
                        np.array([place_encoded]), 
                        user_features_weighted.reshape(1, -1)
                    ], verbose=0)
                    
                    raw_pred = float(pred[0][0])
                    
                    # Improved prediction normalization based on user's rating pattern
                    if len(user_ratings_fixed) > 0:
                        user_avg_rating = np.mean([rating for _, _, rating in user_ratings_fixed])
                        user_bias = user_avg_rating - 3.0  # Bias from neutral rating
                        
                        # Apply user bias to prediction
                        raw_pred += user_bias * 0.3  # Moderate influence
                    
                    # Normalize prediction to 1-5 scale
                    if raw_pred < 0:
                        # Handle negative predictions (common in this model)
                        if raw_pred <= -6:
                            normalized_pred = 0.0
                        else:
                            normalized_pred = (raw_pred + 6) / 6.0
                    elif raw_pred > 1:
                        normalized_pred = min(1.0, raw_pred / 5.0)
                    else:
                        normalized_pred = max(0.0, min(1.0, raw_pred))
                    
                    # Convert to 1-5 scale
                    pred_rating = 1.0 + (normalized_pred * 4.0)
                    pred_rating = max(1.0, min(5.0, pred_rating))
                    
                    # Get the proper place name - ensure we get the generated name, not place_id
                    place_name = place_info.iloc[0].get('place_name', str(place_id))
                    
                    # If place_name is still just the place_id, generate a proper name
                    if str(place_name) == str(place_id):
                        # Generate name using the same algorithm
                        place_categories = [
                            "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                            "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                            "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                            "Historic Site", "Art Gallery", "Theme Park", "Marina",
                            "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                            "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                        ]
                        location_names = [
                            "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                            "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                            "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                        ]
                        
                        category_idx = int(place_id) % len(place_categories)
                        location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                        
                        place_name = f"{location_names[location_idx]} {place_categories[category_idx]}"
                    
                    predictions.append((place_id, place_name, pred_rating, raw_pred))
                    
                except Exception as e:
                    continue
        
        # Sort by rating and get top 3
        predictions.sort(key=lambda x: x[2], reverse=True)
        top_3_predictions = predictions[:3]
        
        # Display top 3 recommendations
        if len(top_3_predictions) > 0:
            st.subheader("🏆 Your Top 3 Personalized Recommendations:")
            
            for i, (place_id, place_name, rating, raw_pred) in enumerate(top_3_predictions):
                # Get the proper display name from places DataFrame
                place_info = places[places['place_id'] == place_id]
                if len(place_info) > 0:
                    place_data = place_info.iloc[0]
                    display_name = str(place_data.get('place_name', f'Place {place_id}'))
                    category = place_data.get('category', 'Tourism')
                    quality = place_data.get('quality', 3.5)
                else:
                    # Fallback: generate name if not found in places DataFrame
                    def create_place_name_fallback(place_id):
                        place_categories = [
                            "Beach Resort", "Mountain Lodge", "City Hotel", "Cultural Center", 
                            "Adventure Park", "Shopping Mall", "Restaurant", "Museum",
                            "Nature Reserve", "Sports Complex", "Entertainment Venue", "Spa Resort",
                            "Historic Site", "Art Gallery", "Theme Park", "Marina",
                            "Golf Course", "Ski Resort", "Camping Ground", "Observatory",
                            "Botanical Garden", "Zoo", "Aquarium", "Concert Hall"
                        ]
                        location_names = [
                            "Sunset", "Golden", "Royal", "Grand", "Elite", "Premium", 
                            "Paradise", "Serene", "Majestic", "Emerald", "Crystal", "Azure",
                            "Harmony", "Tranquil", "Blissful", "Radiant", "Enchanted", "Pristine"
                        ]
                        
                        category_idx = int(place_id) % len(place_categories)
                        location_idx = (int(place_id) // len(place_categories)) % len(location_names)
                        
                        return f"{location_names[location_idx]} {place_categories[category_idx]}"
                    
                    display_name = create_place_name_fallback(place_id)
                    category = 'Tourism'
                    quality = 3.5
                
                # Create a nice card-like display for each recommendation
                with st.container():
                    st.markdown(f"### 🥇 **{i+1}. {display_name}**")
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Predicted Rating:** ⭐ **{rating:.1f}/5.0**")
                        if category and category != 'Unknown':
                            st.markdown(f"**Category:** 📍 {category}")
                        if quality > 0:
                            quality_stars = "⭐" * min(int(quality), 5)
                            st.markdown(f"**Quality Score:** 🏆 {quality:.1f}/5.0 {quality_stars}")
                    
                    with col2:
                        st.caption(f"Based on your ratings & profile")
                        if category and category != 'Unknown':
                            st.caption(f"📍 {category}")
                    
                    st.markdown("---")
            
            # Show user's preference summary
            with st.expander("📊 How we made these recommendations"):
                user_avg = np.mean([rating for _, _, rating in user_ratings_fixed])
                high_rated = len([r for _, _, r in user_ratings_fixed if r >= 4.0])
                low_rated = len([r for _, _, r in user_ratings_fixed if r <= 2.0])
                
                st.write("**Your Profile:**")
                st.write(f"- Age: {user.age}, Gender: {user.gender}")
                st.write(f"- Budget: ${user.budget}, Group: {user.group_comp}")
                
                st.write("**Your Rating Pattern:**")
                st.write(f"- Average rating: {user_avg:.1f}/5.0")
                st.write(f"- High-rated places (4-5⭐): {high_rated}")
                st.write(f"- Low-rated places (1-2⭐): {low_rated}")
                
                st.write("**Recommendation Method:**")
                st.write("- Neural Collaborative Filtering based on similar users")
                st.write("- Weighted by your demographic profile")
                st.write("- Adjusted for your personal rating bias")
        else:
            st.warning("Unable to generate recommendations. Please try rating some places first.")
    
    elif rated_count > 0:
        # Show encouragement if user has started but not finished rating
        remaining = REQUIRED_RATINGS - rated_count
        st.info(f"💡 **Almost there!** Rate {remaining} more destinations to unlock your personalized recommendations.")
