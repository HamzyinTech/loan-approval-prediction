🏦 Loan Approval Prediction System Loan Approval Prediction - Machine Learning Project

This notebook presents an end-to-end machine learning workflow for predicting loan approval using applicant financial and demographic data.

The objective is to classify applicants as low-risk (approved) or high-risk (rejected) to support data-driven lending decisions. 📌 Overview

This project is an end-to-end machine learning system for credit risk classification, designed to predict whether a loan application will be approved based on applicant financial and demographic information.

The system simulates real-world banking decision processes by analyzing historical loan data and identifying high-risk and low-risk applicants, supporting more consistent and data-driven lending decisions.

🎯 Objective

Predict loan approval outcome (Approved / Rejected) Identify key factors influencing loan decisions Support financial risk assessment using machine learning Improve decision efficiency in lending processes

📊 Dataset Features Feature Description Gender Applicant gender Married Marital status Dependents Number of dependents Education Education level Self_Employed Employment status ApplicantIncome Applicant income CoapplicantIncome Co-applicant income LoanAmount Requested loan amount Loan_Amount_Term Loan repayment duration Credit_History Past credit record Property_Area Location category (Urban/Semiurban/Rural)

🧠 Machine Learning Pipeline

The project follows a structured end-to-end ML workflow:

Data Cleaning Missing values handled using median (numerical) and mode (categorical) Feature Engineering Label encoding for categorical variables Feature Scaling StandardScaler applied to numerical features Model Training Logistic Regression Random Forest Classifier Model Evaluation Accuracy Precision / Recall F1-score

📈 Model Performance Model Accuracy Notes Logistic Regression ~78% Stable baseline model Random Forest ~77% Better at capturing complex patterns

Random Forest performed better in identifying approved loans, making it more suitable for reducing missed approvals in financial decisions.

💼 Business Impact Helps financial institutions reduce default risk Improves consistency in loan approval decisions Identifies important financial factors affecting creditworthiness Supports data-driven lending strategies 🧩 Project Structure

Loan_Prediction_Project/

│

├── models/

│ ├── rf_model.pkl

│ ├── scaler.pkl

│

├── notebooks/

│ └── training.ipynb

│

├── data/

│

├── requirements.txt

└── README.md

🚀 How to Run This Project

Clone Repository git clone https://github.com/HamzyinTech/loan-approval-prediction.git cd loan-approval-prediction
Install Dependencies pip install -r requirements.txt
Load Model and Scaler import joblib import pandas as pd
model = joblib.load("models/rf_model.pkl") scaler = joblib.load("models/scaler.pkl") 4. Make Predictions new_applicant = pd.DataFrame([{ 'Gender': 1, 'Married': 0, 'Dependents': 0, 'Education': 1, 'Self_Employed': 0, 'ApplicantIncome': 5000, 'CoapplicantIncome': 2000, 'LoanAmount': 150, 'Loan_Amount_Term': 360, 'Credit_History': 1, 'Property_Area': 2 }])

numeric_cols = [ 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term' ]

new_applicant[numeric_cols] = scaler.transform(new_applicant[numeric_cols])

prediction = model.predict(new_applicant)

if prediction[0] == 1: print("✅ Loan Approved") else: print("❌ Loan Rejected")

🔮 Future Improvements Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) Advanced models (XGBoost, LightGBM) Model interpretability using SHAP values Deployment as a web application (Streamlit / Flask) API integration for real-time predictions

📌 Key Takeaways Feature engineering significantly improves model performance Class imbalance impacts recall and must be handled carefully Tree-based models capture complex financial patterns effectively Model interpretability is essential in financial applications 👨‍💻 Author

HamzyinTech
