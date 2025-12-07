import streamlit as st
import random

st.title("Customer Churn Prediction App (Practice Version)")

st.sidebar.header("Enter Customer Details")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0,1])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

# Predict button
if st.button("Predict Churn"):
    # Random prediction for practice
    prediction = random.choice(["Churn", "No Churn"])
    st.success(f"The model predicts: {prediction}")

