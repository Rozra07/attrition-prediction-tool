import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load updated trained model
with open("new_attrition_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("new_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the Streamlit app
st.title("Employee Attrition Prediction Tool")
st.write("Enter employee details to predict attrition risk.")

# Input fields
years_at_company = st.slider("Years at Company", 1, 30, 5)
job_role = st.selectbox("Job Role", ["Sales", "HR", "Tech", "Operations", "Finance", "Marketing", "Consulting", "R&D", "Product Management"])
performance_rating = st.slider("Performance Rating", 1, 5, 3)
salary_change = st.slider("Salary Change in % (last year)", -10, 50, 5)
training_attended = st.slider("Training Sessions Attended (last 6 months)", 0, 20, 5)
estimated_attrition = st.slider("Estimated Attrition Probability (%)", 0, 100, 50, help="Choose based on company attrition trends and employee feedback.")
linkedin_activity = st.slider("LinkedIn Activity Score (0 = Low, 5 = Very High)", 0, 5, 2)
peer_relationship = st.slider("Peer Relationship Score (0 = Poor, 5 = Excellent)", 0, 5, 3)
college_tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
company_favorability = st.selectbox("Company's Favorability for College Tier", ["High", "Medium", "Low"])
previous_company = st.selectbox("Previous Company Type", ["Big MNC (Google, Apple, Diageo, etc.)", "Mid-sized Company", "Startup", "Other"])

# Convert categorical data
job_role_mapping = {"Sales": 0, "HR": 1, "Tech": 2, "Operations": 3, "Finance": 4, "Marketing": 5, "Consulting": 6, "R&D": 7, "Product Management": 8}
job_role_encoded = [0] * 9  # Ensure this matches the number of job roles
role_index = job_role_mapping[job_role]  # Correct index without -1 adjustment
job_role_encoded[role_index] = 1  # One-hot encoding

college_tier_mapping = {"Tier 1": 3, "Tier 2": 2, "Tier 3": 1}
company_favorability_mapping = {"High": 3, "Medium": 2, "Low": 1}
previous_company_mapping = {"Big MNC (Google, Apple, Diageo, etc.)": 3, "Mid-sized Company": 2, "Startup": 1, "Other": 0}

# Create feature array
features = np.array([
    years_at_company, performance_rating, salary_change,
    training_attended, linkedin_activity, estimated_attrition, peer_relationship,
    college_tier_mapping[college_tier], company_favorability_mapping[company_favorability], previous_company_mapping[previous_company]
] + job_role_encoded).reshape(1, -1)

# Debugging: Print feature shape
expected_features = scaler.n_features_in_
st.write(f"Feature shape: {features.shape[1]}")
st.write(f"Expected shape: {expected_features}")

# Ensure feature count matches scaler expectations
if features.shape[1] != expected_features:
    st.error(f"Feature mismatch error! Expected {expected_features} features but got {features.shape[1]}. Please retrain the model with updated features.")
else:
    # Scale the input data
    features_scaled = scaler.transform(features)

    # Predict attrition probability
    if st.button("Predict Attrition Risk"):
        attrition_prob = model.predict_proba(features_scaled)[:, 1][0] * 100
        st.write(f"### Estimated Attrition Probability: {attrition_prob:.2f}%")

        if attrition_prob > 70:
            st.error("High Attrition Risk! Consider retention strategies.")
        elif attrition_prob > 40:
            st.warning("Moderate Attrition Risk! Keep an eye on engagement.")
        else:
            st.success("Low Attrition Risk. Employee likely to stay.")
