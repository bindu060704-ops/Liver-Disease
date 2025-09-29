# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 17:57:39 2025

@author: hp
"""
# appy.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_excel("project-data.xlsx")

# Drop rows with missing target
df.dropna(subset=["category"], inplace=True)

# Encode categorical features
df["sex"] = LabelEncoder().fit_transform(df["sex"])

# Define features and target
X = df.drop("category", axis=1)
y = df["category"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="Liver Disease Predictor", layout="centered")
st.title("🧪 Liver Disease Classification App")
st.write("Enter patient details to predict liver disease category.")

# Input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=["m", "f"])
    albumin = st.number_input("Albumin", value=40.0)
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase", value=75.0)
    alanine_aminotransferase = st.number_input("Alanine Aminotransferase", value=30.0)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", value=28.0)
    bilirubin = st.number_input("Bilirubin", value=1.2)
    cholinesterase = st.number_input("Cholinesterase", value=8.5)
    cholesterol = st.number_input("Cholesterol", value=180.0)
    creatinina = st.number_input("Creatinina", value=1.0)
    gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", value=40.0)
    protein = st.number_input("Protein", value=70.0)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_data = {
        "age": age,
        "sex": 0 if sex == "f" else 1,
        "albumin": albumin,
        "alkaline_phosphatase": alkaline_phosphatase,
        "alanine_aminotransferase": alanine_aminotransferase,
        "aspartate_aminotransferase": aspartate_aminotransferase,
        "bilirubin": bilirubin,
        "cholinesterase": cholinesterase,
        "cholesterol": cholesterol,
        "creatinina": creatinina,
        "gamma_glutamyl_transferase": gamma_glutamyl_transferase,
        "protein": protein
    }

    input_df = pd.DataFrame([input_data])
    prediction = rf_model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🧬 Predicted Liver Disease Category: **{predicted_label}**")