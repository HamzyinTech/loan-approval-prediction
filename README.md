Loan Prediction Project
📌 Project Overview

This project demonstrates an end-to-end machine learning pipeline to predict loan approval using real-world financial data. The goal is to help lenders identify low- and high-risk applicants, reducing defaults and improving decision-making.

Key Objectives:

Predict whether a loan will be approved or rejected

Highlight financial risk associated with loan applications

Provide actionable insights for banking decision-making

🛠️ Tools & Technologies

Languages: Python (Pandas, NumPy)

Machine Learning: Scikit-learn (Logistic Regression, Random Forest)

Data Visualization: Matplotlib, Seaborn

Data Handling: Missing value imputation, Label Encoding, scaling, class imbalance handling

🧩 Data Preprocessing

Missing values handled using median for numeric features and mode for categorical features

Categorical variables encoded using Label Encoding

Features scaled using StandardScaler for models that require it

Target variable: Loan_Status (0 = Rejected, 1 = Approved)

📊 Results & Analysis

Model Performance on Test Set:

Model	Accuracy	Recall (Rejected)	Recall (Approved)
Logistic Regression	78%	Moderate	High
Random Forest	77%	42%	96%

Observations:

Random Forest captures non-linear relationships and performs well across classes.

High recall for approved loans ensures low-risk applicants are correctly identified.

Moderate recall for rejected loans highlights financial risk areas, guiding risk-aware decision-making.

💼 Business Impact

Financial Risk Assessment: Identifies risky applicants before granting loans, reducing defaults.

Decision Support: Predictions can inform loan approvals, prioritizing safety and profitability.

Feature Insights: Important factors influencing loan approval include Credit_History, ApplicantIncome, and LoanAmount.

🎯 Key Learnings

Accuracy alone is insufficient for imbalanced financial datasets

Handling class imbalance improves risk sensitivity at the cost of overall accuracy

Model evaluation must align with business objectives, not just statistical metrics

Tree-based models (Random Forest) outperform linear models in capturing complex patterns


✅ Conclusion

This project demonstrates a complete ML workflow from preprocessing to model evaluation and business interpretation.

Future improvements could include:

Advanced feature engineering

Ensemble or boosting methods (e.g., XGBoost, LightGBM)

Deployment as a decision-support system for lenders


📌 How to Run

Clone the repository

Install dependencies:

pip install -r requirements.txt


Open notebook.ipynb to explore preprocessing, modeling, and evaluation steps





📈📈 Predicting a New Applicant

import pandas as pd
import joblib

# Load trained model 
rf_model = joblib.load("models/rf_model.pkl")

new_applicant = pd.DataFrame([{
    'Gender': 1,
    'Married': 0,
    'Dependents': 0,
    'Education': 1,
    'Self_Employed': 0,
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 2
}])

# Make prediction
prediction = rf_model.predict(new_applicant)

# Show result
if prediction[0] == 1:
    print("✅ Loan likely to be APPROVED (low financial risk)")
else:
    print("⚠️ Loan likely to be REJECTED (high financial risk)")





This demonstrates how my trained model can be applied to real applicants to assess financial risk.













































