import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("stroke_model.pkl")

st.title("Stroke Prediction Model")
st.write("Enter patient information to estimate stroke risk.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)

hypertension = st.selectbox(
    "Hypertension",
    ["No", "Yes"]
)

heart_disease = st.selectbox(
    "Heart Disease",
    ["No", "Yes"]
)

avg_glucose = st.number_input(
    "Average Glucose Level",
    min_value=0.0,
    max_value=500.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes", "Unknown"]
)

if st.button("Predict Stroke Risk"):

    input_data = pd.DataFrame({
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "avg_glucose_level": [avg_glucose],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Higher Stroke Risk")
    else:
        st.success("Lower Stroke Risk")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"Estimated probability: {probability:.2%}")

st.warning(
    "This application is an academic machine-learning project "
    "and should not be used as a medical diagnostic tool."
)