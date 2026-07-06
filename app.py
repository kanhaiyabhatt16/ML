import streamlit as st
import joblib

# Load model
model = joblib.load("insurancepredictionmodel.pkl")

st.title("Insurance Prediction Model")
with st.form("insurance_form"):

    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    gender= st.selectbox(
        "Gender",
        ["male", "female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

    submit = st.form_submit_button("Predict Insurance Cost")


if submit:

    # Encode categorical variables
    gender = 1 if gender == "male" else 0
    smoker = 1 if smoker == "yes" else 0

    region_map = {
        "northeast": 0,
        "northwest": 1,
        "southeast": 2,
        "southwest": 3
    }

    region = region_map[region]

    sample = [[
        age,
        gender,
        bmi,
        children,
        smoker,
        region
    ]]

    prediction = model.predict(sample)

    st.success(f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")