import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load trained model (for later deployment, we would save and load it properly)
with open("attrition_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the Streamlit app
st.title("Employee Attrition Prediction Tool")
st.write("Enter employee details to predict attrition risk.")

# Input fields
years_at_company = st.slider("Years at Company", 1, 30, 5)
job_role = st.selectbox("Job Role", ["Sales", "HR", "Tech", "Operations", "Finance"])
performance_rating = st.slider("Performance Rating", 1, 5, 3)
salary_change = st.slider("Salary Change in % (last year)", -10, 50, 5)
training_attended = st.slider("Training Sessions Attended (last 6 months)", 0, 20, 5)
estimated_attrition = st.slider("Estimated Attrition Probability (%)", 0, 100, 50)
linkedin_activity = st.slider("LinkedIn Activity Score (0 = Low, 1 = High)", 0, 1, 0)

# Convert categorical data
job_role_mapping = {"Sales": 0, "HR": 1, "Tech": 2, "Operations": 3, "Finance": 4}
job_role_encoded = [0, 0, 0, 0]
role_index = job_role_mapping[job_role] - 1  # Adjust index for one-hot encoding
if role_index >= 0:
    job_role_encoded[role_index] = 1  # One-hot encoding

# Create feature array
features = np.array([
    years_at_company, performance_rating, salary_change,
    training_attended, linkedin_activity, estimated_attrition
] + job_role_encoded).reshape(1, -1)

# Debugging: Print the feature shape
st.write(f"Feature shape: {features.shape}")
st.write(f"Expected shape: {scaler.n_features_in_}")

# Ensure feature count matches scaler expectations
if features.shape[1] != scaler.n_features_in_:
    st.error("Feature mismatch error! Please check the input fields and retrain the model.")
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
