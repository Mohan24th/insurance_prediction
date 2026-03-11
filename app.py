import streamlit as st
from src.prediction import InsurancePremiumPredictor

st.title("Insurance Prediction")

Age = st.number_input("Enter Age")
Annual_Income_LPA = st.number_input("Enter Annual Income in LPA")
Policy_Term_Years = st.number_input("Enter Policy Term in Years")
Sum_Assured_Lakhs = st.number_input("Enter Sum Assured in Lakhs")

if st.button("Predict"):

    predictor = InsurancePremiumPredictor()

    prediction = predictor.predict(
        Age,
        Annual_Income_LPA,
        Policy_Term_Years,
        Sum_Assured_Lakhs
    )

    st.write(f"The predicted insurance premium is: {prediction}")