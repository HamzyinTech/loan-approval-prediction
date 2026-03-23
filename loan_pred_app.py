import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

st.title("🏦 Loan Prediction App")

st.write("Fill in applicant details below:")

st.info("📌 All monetary values should be entered in Naira (₦).")

# 🔹 Human-friendly inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Numeric inputs
income = st.number_input(
    "Applicant Monthly Income (₦)",
    help="Enter your total monthly salary or earnings"
)
coincome = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.selectbox(
    "Loan Term (Months)",
    [12, 36, 60, 120, 180, 240, 360],
    help="Duration you will take to repay the loan"
)


# 🔥 Convert labels → numbers (VERY IMPORTANT)
gender = 1 if gender == "Female" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
income = st.number_input("Applicant Monthly Income (₦)")
coincome = st.number_input("Coapplicant Monthly Income (₦)")
loan_amount = st.number_input("Loan Amount Requested (₦)")

# Predict button
if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': income,
        'CoapplicantIncome': coincome,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    # scale numeric columns
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # prediction
    prediction = model.predict(input_data)

    # output
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")