import streamlit as st
import requests
import os

st.set_page_config(page_title="HealthGuard UAE - Diabetes AI", layout="centered")

st.title("HealthGuard UAE - Diabetes AI Agent")
st.markdown("Helping you stay healthy and informed 🇦🇪")

API_URL = "http://localhost:8000/analyze"

age = st.slider("Age", 18, 120, 30)
bmi = st.slider("BMI", 15.0, 45.0, 25.0)
glucose = st.slider("Glucose Level", 0, 200, 100)
hba1c = st.slider("HbA1c", 4.0, 14.0, 6.0)

if st.button("Predict"):
    data = {
        "age": age,
        "bmi": bmi,
        "glucose": glucose,
        "hba1c": hba1c
    }
    response = requests.post(API_URL, data=data)
    prediction = response.json()
    result = "No Diabetes Risk" if prediction["prediction"] == 0 else "Risk of Diabetes"
    st.success(f"Result: {result}")