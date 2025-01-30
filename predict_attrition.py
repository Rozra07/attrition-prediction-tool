
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_attrition(inputs):
    # Convert inputs into numpy array and reshape for model input
    input_array = np.array(inputs).reshape(1, -1)
    
    # Scale the input
    input_scaled = scaler.transform(input_array)
    
    # Predict probability
    probability = model.predict_proba(input_scaled)[:, 1][0]
    
    return probability

if __name__ == "__main__":
    # Example input: [age, designation_level, joining_ctc, total_experience, etc.]
    user_input = [30, 2, 800000, 5, 2, 1, 1, 1, 1, 2, 3, 1, 2, 1, 1, 0, 1, 2, 1]
    print("Probability of Resignation:", predict_attrition(user_input))
