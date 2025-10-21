import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Next-Gen Tumor Classifier")
st.write("Predict whether a tumor is malignant or benign.")

# Input features
def user_input_features():
    mean_radius = st.number_input('Mean Radius', 0.0, 30.0, 14.0)
    mean_texture = st.number_input('Mean Texture', 0.0, 40.0, 20.0)
    mean_perimeter = st.number_input('Mean Perimeter', 0.0, 200.0, 90.0)
    mean_area = st.number_input('Mean Area', 0.0, 2500.0, 600.0)
    mean_smoothness = st.number_input('Mean Smoothness', 0.0, 1.0, 0.1)
    # Add more features if needed
    features = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
    return np.array(features).reshape(1, -1)

input_data = user_input_features()
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    result = "Malignant (Cancerous)" if prediction[0] == 0 else "Benign (Non-Cancerous)"
    st.success(f"Prediction: {result}")
