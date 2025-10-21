"""
Script to train multiple ML models on breast cancer dataset
Run this to generate all model files needed for the app
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load breast cancer dataset
print("Loading breast cancer dataset...")
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Scaler saved")

# Dictionary to store models and their performance
models = {}
results = {}

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# 1. SVM Model
print("\n1. Training SVM Model...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
models['SVM'] = svm_model
results['SVM'] = svm_accuracy
joblib.dump(svm_model, 'svm_model.pkl')
print(f"   Accuracy: {svm_accuracy:.4f}")
print("   ✓ Model saved as svm_model.pkl")

# 2. Random Forest Model
print("\n2. Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
models['Random Forest'] = rf_model
results['Random Forest'] = rf_accuracy
joblib.dump(rf_model, 'rf_model.pkl')
print(f"   Accuracy: {rf_accuracy:.4f}")
print("   ✓ Model saved as rf_model.pkl")

# 3. XGBoost Model
print("\n3. Training XGBoost Model...")
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
models['XGBoost'] = xgb_model
results['XGBoost'] = xgb_accuracy
joblib.dump(xgb_model, 'xgb_model.pkl')
print(f"   Accuracy: {xgb_accuracy:.4f}")
print("   ✓ Model saved as xgb_model.pkl")

# 4. LightGBM Model
print("\n4. Training LightGBM Model...")
lgbm_model = LGBMClassifier(random_state=42, verbose=-1)
lgbm_model.fit(X_train_scaled, y_train)
lgbm_pred = lgbm_model.predict(X_test_scaled)
lgbm_accuracy = accuracy_score(y_test, lgbm_pred)
models['LightGBM'] = lgbm_model
results['LightGBM'] = lgbm_accuracy
joblib.dump(lgbm_model, 'lgbm_model.pkl')
print(f"   Accuracy: {lgbm_accuracy:.4f}")
print("   ✓ Model saved as lgbm_model.pkl")

# 5. CatBoost Model
print("\n5. Training CatBoost Model...")
catboost_model = CatBoostClassifier(random_state=42, verbose=0)
catboost_model.fit(X_train_scaled, y_train)
catboost_pred = catboost_model.predict(X_test_scaled)
catboost_accuracy = accuracy_score(y_test, catboost_pred)
models['CatBoost'] = catboost_model
results['CatBoost'] = catboost_accuracy
joblib.dump(catboost_model, 'catboost_model.pkl')
print(f"   Accuracy: {catboost_accuracy:.4f}")
print("   ✓ Model saved as catboost_model.pkl")

# Save feature names
feature_names = data.feature_names.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("\n✓ Feature names saved")

# Print summary
print("\n" + "="*60)
print("TRAINING COMPLETE - MODEL COMPARISON")
print("="*60)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:15} | Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "="*60)
print("All models trained and saved successfully!")
print("You can now run the Streamlit app: streamlit run app.py")
print("="*60)
