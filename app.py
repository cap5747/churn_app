import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction App")

st.sidebar.header("Enter Customer Details")

# --- Input fields ---
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                       "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

# --- Load model ---
model = joblib.load("rf_churn_model.pkl")

# --- Encode categorical inputs ---
def encode_inputs(df):
    mapping_yes_no = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
    mapping_gender = {"Male": 1, "Female": 0}
    mapping_contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    mapping_payment = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }

    df["gender"] = df["gender"].map(mapping_gender)
    df["Partner"] = df["Partner"].map(mapping_yes_no)
    df["Dependents"] = df["Dependents"].map(mapping_yes_no)
    df["PhoneService"] = df["PhoneService"].map(mapping_yes_no)
    df["MultipleLines"] = df["MultipleLines"].map(mapping_yes_no)
    df["InternetService"] = df["InternetService"].map(mapping_yes_no)
    df["OnlineSecurity"] = df["OnlineSecurity"].map(mapping_yes_no)
    df["OnlineBackup"] = df["OnlineBackup"].map(mapping_yes_no)
    df["DeviceProtection"] = df["DeviceProtection"].map(mapping_yes_no)
    df["TechSupport"] = df["TechSupport"].map(mapping_yes_no)
    df["StreamingTV"] = df["StreamingTV"].map(mapping_yes_no)
    df["StreamingMovies"] = df["StreamingMovies"].map(mapping_yes_no)
    df["Contract"] = df["Contract"].map(mapping_contract)
    df["PaperlessBilling"] = df["PaperlessBilling"].map(mapping_yes_no)
    df["PaymentMethod"] = df["PaymentMethod"].map(mapping_payment)

    return df

# --- Predict button ---
if st.button("Predict Churn"):
    X_new = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    X_new_encoded = encode_inputs(X_new)
    prediction = model.predict(X_new_encoded)[0]
    result = "Churn" if prediction == 1 else "No Churn"
    
    st.success(f"The model predicts: {result}")

