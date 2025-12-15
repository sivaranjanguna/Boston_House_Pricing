import pandas as pd
import streamlit as st
import joblib
import numpy as np

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# --------------------------------------------------
# Theme & UI Fixes
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f5f3ff;
}

/* Title */
h1 {
    color: #6d28d9;
    text-align: center;
    font-weight: 900;
    font-size: 52px;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #4b5563;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Labels */
label {
    color: black !important;
    font-weight: 600;
}

/* Input boxes */
.stNumberInput input {
    background-color: #6d28d9 !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-size: 15px;
}

/* + and - buttons */
.stNumberInput button {
    background-color: black !important;
    color: white !important;
    border-radius: 6px;
}

/* Predict button */
.stButton > button {
    width: 70%;
    height: 3.6em;
    display: block;
    margin: 35px auto;
    background-color: black;
    color: white;
    font-size: 22px;
    font-weight: 800;
    border-radius: 12px;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    background-color: #111;
}

/* Result box */
.result-box {
    text-align: center;
    font-size: 24px;
    font-weight: 800;
    color: #6d28d9;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
try:
    model = joblib.load("BHP_Model.pkl")
    scaler = joblib.load("BHP_Scaler.pkl")
except:
    st.error("Model or Scaler file not found.")
    st.stop()

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("Boston House Price Prediction")
st.markdown(
    "<div class='subtitle'>Enter property details to estimate house value</div>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Inputs (2 Columns)
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    LSTAT = st.number_input("LSTAT (Lower status %)", min_value=0.0, value=12.5)
    NOX = st.number_input("NOX (Nitric oxides)", min_value=0.0, value=0.5)
    RM = st.number_input("RM (Avg rooms)", min_value=0.0, value=6.0)
    DIS = st.number_input("DIS (Distance to employment)", min_value=0.0, value=5.0)

with col2:
    INDUS = st.number_input("INDUS (Non-retail business acres)", min_value=0.0, value=7.5)
    PTRATIO = st.number_input("PTRATIO (Pupil-teacher ratio)", min_value=0.0, value=18.0)
    TAX = st.number_input("TAX (Property tax)", min_value=0.0, value=300.0)
    AGE = st.number_input("AGE (Old houses %)", min_value=0.0, value=65.0)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict House Price"):

    input_data = pd.DataFrame([[ 
        LSTAT, INDUS, NOX, PTRATIO, RM, TAX, DIS, AGE
    ]], columns=[
        "LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE"
    ])

    scaled_data = scaler.transform(input_data)
    raw_prediction = model.predict(scaled_data)[0]

    # ✅ FIX NEGATIVE PREDICTION
    prediction = max(0, raw_prediction)

    st.divider()

    st.markdown(
        f"<div class='result-box'>Predicted Price: ${prediction * 1000:,.2f} USD</div>",
        unsafe_allow_html=True
    )

    if raw_prediction < 0:
        st.warning(
            "The model produced a negative value. "
            "Displayed price has been adjusted to $0. "
            "Try more realistic feature values."
        )
    elif prediction < 50:
        st.success("The predicted house price looks reasonable.")
    else:
        st.warning("The predicted house price is quite high.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption("Developed using Streamlit | Boston Housing Dataset")
