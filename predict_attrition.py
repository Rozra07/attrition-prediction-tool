import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
with open("attrition_model.pkl", "rb") as model_file:
    model = joblib.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = joblib.load(scaler_file)

# Define the Streamlit app
st.title("Employee Attrition Prediction Tool")
st.write("Enter employee details to predict attrition risk.")

# Input fields
years_at_company = st.slider("Years at Company", 1, 30, 5)
performance_rating = st.slider("Performance Rating", 1, 5, 3)
salary_change = st.slider("Salary Change in % (last year)", -10, 50, 5)
training_attended = st.slider("Training Sessions Attended (last 6 months)", 0, 20, 5)
estimated_attrition = st.slider("Estimated Attrition Probability (%)", 0, 100, 50, help="Choose based on company attrition trends and employee feedback.")
peer_relationship = st.slider("Peer Relationship Score (0 = Poor, 5 = Excellent)", 0, 5, 3)
college_tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
company_favorability = st.selectbox("Company's Favorability for College Tier", ["High", "Medium", "Low"])
previous_company = st.selectbox("Previous Company Type", ["MNC/Big Firm", "Mid-Size Firm", "Small Firm", "Startup"])
industry_background = st.selectbox("Industry Background", ["FMCG", "Tech", "Manufacturing", "Consulting", "BFSI", "Retail", "Alco-Bev", "Pharma"])

# Mapping categorical data
college_tier_mapping = {"Tier 1": 3, "Tier 2": 2, "Tier 3": 1}
company_favorability_mapping = {"High": 3, "Medium": 2, "Low": 1}
previous_company_mapping = {"MNC/Big Firm": 3, "Mid-Size Firm": 2, "Small Firm": 1, "Startup": 0}
industry_mapping = {"FMCG": 0, "Tech": 1, "Manufacturing": 2, "Consulting": 3, "BFSI": 4, "Retail": 5, "Alco-Bev": 6, "Pharma": 7}

# Create feature array
features = np.array([
    years_at_company, performance_rating, salary_change,
    training_attended, estimated_attrition, peer_relationship,
    college_tier_mapping[college_tier], company_favorability_mapping[company_favorability],
    previous_company_mapping[previous_company], industry_mapping[industry_background]
]).reshape(1, -1)

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
