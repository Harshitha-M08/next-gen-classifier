Next-Gen Tumor Classifier

Description:
This is a machine learning-powered web app that predicts whether a breast tumor is malignant (cancerous) or benign (non-cancerous) using 30+ tumor features. It is built with Python, Streamlit, and SHAP for interactive predictions and explainability.

Features

Uses SVM model with ~98% accuracy

Includes all 30+ features from the breast cancer dataset

SHAP explainability to show which features influence the prediction

Interactive Streamlit UI with sidebar input

Portfolio-ready and deployable on Hugging Face Spaces

Tech Stack
Python
scikit-learn
Streamlit
SHAP
Pandas & NumPy
Joblib
Usage
Clone or download the repo.

Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


Enter tumor features in the sidebar and click Predict.

View prediction and SHAP feature importance.

Repository Structure
next-gen-classifier/

├─ app.py              # Streamlit application

├─ svm_model.pkl       # Trained SVM model

├─ scaler.pkl          # Feature scaler

├─ requirements.txt    # Required Python packages

Future Enhancements

Add advanced ML models (XGBoost, LightGBM, CatBoost)

Improve UI/UX with sliders and better layout

Include prediction confidence scores

Expand to other cancer datasets
