import streamlit as st
import pandas as pd
import joblib
import os

# Dataset CSV file path (relative to app.py)
dataset_path = "Multiclass Diabetes Dataset.csv"

# Load existing dataset or create empty with columns if not present
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    columns = [
        "Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL",
        "LDL", "VLDL", "BMI", "Class"
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(dataset_path, index=False)

# Load trained model file
model_path = "best_diabetes_rf_model.pkl"
model = joblib.load(model_path)

st.title("Diabetes Prediction App")

# User inputs
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
urea = st.number_input("Urea (mg/dL)", min_value=0.0, value=10.0)
cr = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=1.0)
hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.0)
chol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, value=150.0)
tg = st.number_input("Triglycerides (mg/dL)", min_value=0.0, value=100.0)
hdl = st.number_input("HDL (mg/dL)", min_value=0.0, value=50.0)
ldl = st.number_input("LDL (mg/dL)", min_value=0.0, value=100.0)
vldl = st.number_input("VLDL (mg/dL)", min_value=0.0, value=15.0)
bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=25.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[
        gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi
    ]], columns=[
        "Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL",
        "LDL", "VLDL", "BMI"
    ])

    prediction = model.predict(input_data)[0]
    class_map = {0: "Non-Diabetic", 1: "Diabetic", 2: "Pre-Diabetic"}
    pred_label = class_map.get(prediction, "Unknown")

    st.success(f"Predicted Diabetes Status: **{pred_label}**")

    # Append new entry to dataframe & save
    new_entry = input_data.copy()
    new_entry["Class"] = prediction
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(dataset_path, index=False)
    st.info("New entry saved to dataset.")

if st.checkbox("Show dataset"):
    st.dataframe(df)

