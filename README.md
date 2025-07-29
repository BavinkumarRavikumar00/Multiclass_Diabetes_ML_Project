# Multiclass_Diabetes_ML_Project

This project builds and deploys a machine learning model to predict diabetes status (Non-Diabetic, Diabetic, Pre-Diabetic) using clinical and biological features. The prediction model is served through a Streamlit web application.

---

## Dataset

- The dataset used is the **Multiclass Diabetes Dataset**.
- Features include: Gender, Age, Urea, Creatinine (Cr), HbA1c, Cholesterol, Triglycerides (TG), HDL, LDL, VLDL, BMI.
- Target labels: 0 = Non-Diabetic, 1 = Diabetic, 2 = Pre-Diabetic.

---

## Features

| Feature | Description |
|---------|-------------|
| Gender  | 0 = Female, 1 = Male |
| Age     | Age in years |
| Urea    | Blood urea level (mg/dL) |
| Cr      | Creatinine level (mg/dL) |
| HbA1c   | Glycated Hemoglobin (%) |
| Chol    | Total cholesterol (mg/dL) |
| TG      | Triglycerides (mg/dL) |
| HDL     | High-Density Lipoprotein (mg/dL) |
| LDL     | Low-Density Lipoprotein (mg/dL) |
| VLDL    | Very Low-Density Lipoprotein (mg/dL) |
| BMI     | Body Mass Index (kg/m²) |

---

## Model

- Used Random Forest Classifier with hyperparameter tuning.
- Model saved using `joblib` for deployment.

---
