import pandas as pd
import streamit as st
import joblib
import numpy as np

model = joblib.load("BHP_Model.pkl")
scaler= joblib.load("BHP_Scaler.pkl")

st.title("BOSTON HOUSE PRICE PREDICTION APP")

st.markdown("plese enter the details given below and press predict button")

# ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
# ['MEDV']

st.divider()

LSTAT = st.number_input("LSTAT (Percentage of lower status of the population)", min_value=0.0, value=12.5)
INDUS = st.number_input("INDUS (Proportion of non-retail business acres per town)", min_value=0.0, value=7.5)
NOX = st.number_input("NOX (Nitric oxides concentration (parts per 10 million))", min_value=0.0, value=0.5)
PTRATIO = st.number_input("PTRATIO (Pupil-teacher ratio by town)", min  _value=0.0, value=18.0)
RM = st.number_input("RM (Average number of rooms per dwelling)", min_value=0.0, value=6.0)
TAX = st.number_input("TAX (Full-value property-tax rate per $10,000)", min_value=0.0, value=300.0)
DIS = st.number_input("DIS (Weighted distances to five Boston employment centres)", min_value=0.0, value=5.0)
AGE = st.number_input("AGE (Proportion of owner-occupied units built prior to 1940)", min_value=0.0, value=65.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "LSTAT": LSTAT,
        "INDUS": INDUS,
        "NOX": NOX,
        "PTRATIO": PTRATIO,
        "RM": RM,
        "TAX": TAX,
        "DIS": DIS,
        "AGE": AGE
    }])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.subheader(f"Prediction : '{np.expm1(prediction):.2f} (in $1000s)'")

    if prediction > 0 and prediction < 50:
        st.success("The predicted house price seems reasonable.")
    elif prediction >= 50:
        st.warning("The predicted house price is quite high, please verify the inputs.")
    else:
        st.error("The predicted house price is invalid, please check the input values.")