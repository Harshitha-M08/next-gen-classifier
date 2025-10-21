import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Next-Gen Tumor Classifier")
st.write("Predict whether a tumor is malignant or benign.")

# Input features: all 30 features from the dataset
feature_names = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

# Create input form
st.sidebar.header("Input Tumor Features")
input_data = []
for feature in feature_names:
    val = st.sidebar.number_input(feature, 0.0, 1000.0, 0.0)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    result = "Malignant (Cancerous)" if prediction[0] == 0 else "Benign (Non-Cancerous)"
    st.success(f"Prediction: {result}")

    # SHAP Explainability
    explainer = shap.Explainer(model, scaler.transform(scaler.inverse_transform(np.identity(len(feature_names)))))
    shap_values = explainer(input_scaled)

    st.subheader("Feature Importance (SHAP)")
    fig, ax = plt.subplots(figsize=(8,6))
    shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(fig)
