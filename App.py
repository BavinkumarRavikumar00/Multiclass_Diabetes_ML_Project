import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model (make sure this path is correct)
model = joblib.load("best_diabetes_rf_model.pkl")

# Dataset path
dataset_path = "/Users/bavinravi/Desktop/ml-project/Multiclass Diabetes Dataset.csv"

st.title("Diabetes Prediction App")

# Define input features to collect from user
def user_input_features():
    Gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
    AGE = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    Urea = st.number_input("Urea (mg/dL)", min_value=0.0, value=10.0)
    Cr = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=1.0)
    HbA1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=15.0, value=5.0)
    Chol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, value=150.0)
    TG = st.number_input("Triglycerides (mg/dL)", min_value=0.0, value=100.0)
    HDL = st.number_input("HDL (mg/dL)", min_value=0.0, value=50.0)
    LDL = st.number_input("LDL (mg/dL)", min_value=0.0, value=100.0)
    VLDL = st.number_input("VLDL (mg/dL)", min_value=0.0, value=20.0)
    BMI = st.number_input("BMI (kg/m²)", min_value=0.0, value=25.0)

    data = {
        "Gender": Gender,
        "AGE": AGE,
        "Urea": Urea,
        "Cr": Cr,
        "HbA1c": HbA1c,
        "Chol": Chol,
        "TG": TG,
        "HDL": HDL,
        "LDL": LDL,
        "VLDL": VLDL,
        "BMI": BMI
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Show user input features
st.subheader("Input Features")
st.write(input_df)

# Prediction button
if st.button("Predict Diabetes Status"):

    prediction = model.predict(input_df)[0]

    # Map prediction to label if you want (example)
    label_map = {0: "Non-Diabetic", 1: "Diabetic", 2: "Pre-Diabetic"}
    pred_label = label_map.get(prediction, "Unknown")

    st.subheader("Prediction Result")
    st.write(f"The model predicts: **{pred_label}** (class {prediction})")

    # Save input + prediction to dataset CSV
    # Check if file exists
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        df = pd.DataFrame(columns=input_df.columns.tolist() + ["Class"])

    # Add Class column as prediction
    new_entry = input_df.copy()
    new_entry["Class"] = prediction

    # Append and save
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(dataset_path, index=False)

    st.success("New entry added to the dataset.")


