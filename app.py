import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("stroke_model.pkl")

# Page title
st.title("Stroke Prediction Model")

st.write(
    "Enter patient information below to estimate stroke risk."
)

# -----------------------------
# Patient information
# -----------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

hypertension = st.selectbox(
    "Hypertension",
    ["No", "Yes"]
)

heart_disease = st.selectbox(
    "Heart Disease",
    ["No", "Yes"]
)

ever_married = st.selectbox(
    "Ever Married",
    ["No", "Yes"]
)

work_type = st.selectbox(
    "Work Type",
    [
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]
)

residence_type = st.selectbox(
    "Residence Type",
    ["Urban", "Rural"]
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
    [
        "never smoked",
        "formerly smoked",
        "smokes",
        "Unknown"
    ]
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Stroke Risk"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [
            1 if hypertension == "Yes" else 0
        ],
        "heart_disease": [
            1 if heart_disease == "Yes" else 0
        ],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ Higher Stroke Risk")
    else:
        st.success("✅ Lower Stroke Risk")

    # Display probability if supported
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0][1]

        st.write(
            f"Estimated Stroke Probability: **{probability:.2%}**"
        )

# -----------------------------
# Disclaimer
# -----------------------------

st.warning(
    "This application is an academic machine-learning project "
    "and should not be used as a medical diagnostic tool."
)
