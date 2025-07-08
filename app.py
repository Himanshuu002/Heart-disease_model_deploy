import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("xgboost_best_model.pkl")

st.title("Heart Disease Prediction App")
st.write("Enter patient medical data to predict heart disease risk.")

# Input features based on SHAP importance
st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
age = st.number_input("Age", 20, 100, 50)
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
max_hr = st.number_input("Max Heart Rate", 60, 210, 150)
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
resting_bp = st.number_input("Resting BP (systolic)", 80, 200, 120)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
sex = st.selectbox("Sex", ["Female", "Male"])

# Map categorical to numerical as needed (dummy example below)
input_data = np.array([[
    ["Upsloping", "Flat", "Downsloping"].index(st_slope),
    ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(chest_pain),
    age,
    oldpeak,
    max_hr,
    ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(rest_ecg),
    cholesterol,
    resting_bp,
    1 if exercise_angina == "Yes" else 0,
    1 if fasting_blood_sugar == "Yes" else 0,
    1 if sex == "Male" else 0
]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    st.success(f"Prediction: {result}")
