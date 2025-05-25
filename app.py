import streamlit as st
import pandas as pd
import numpy as np

# Import authentication and database functions
from auth import (
    get_db, init_db, hash_password, verify_password, register_user, 
    login_user, update_user_profile, get_user_by_id, check_and_init_db
)

# Set page configuration
st.set_page_config(
    page_title="Tourism Recommender System",
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
st.title("🌍 Tourism Destination Recommender")
st.markdown("### User Authentication System")

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
        st.session_state['authenticated'] = False
        st.session_state['user_id'] = None
        st.session_state['username'] = None
        st.rerun()

    # Main content for authenticated users
    st.subheader("Welcome to the Tourism Recommender System!")
    st.write("You are successfully logged in.")
    
    # User profile section
    st.sidebar.title("User Profile")
    user = get_user_by_id(st.session_state['user_id'])
    
    # Display current profile
    with st.sidebar.expander("Current Profile", expanded=True):
        st.write(f"**Username:** {st.session_state['username']}")
        st.write(f"**Email:** {user.email}")
        st.write(f"**Age:** {user.age}")
        st.write(f"**Gender:** {user.gender}")
        st.write(f"**Budget:** ${user.budget}")
        st.write(f"**Group Composition:** {user.group_comp}")
        
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
                    st.rerun()
                else:
                    st.error(message)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Tourism Recommender System - Authentication Demo</p>
</div>
""", unsafe_allow_html=True) 