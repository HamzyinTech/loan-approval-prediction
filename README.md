🏦 Loan Approval Prediction System
📌 Overview

This project is an end-to-end machine learning system for credit risk classification, designed to predict whether a loan application will be approved based on applicant financial and demographic information.

The system simulates real-world banking decision processes by analyzing historical loan data to identify high-risk and low-risk applicants, enabling more consistent and data-driven lending decisions.

A Streamlit web application was also developed to allow real-time predictions through an interactive user interface.

🎯 Objective

Predict loan approval outcome (Approved / Rejected)
Identify key factors influencing loan decisions
Support financial risk assessment using machine learning
Improve decision efficiency in lending processes

📊 Dataset Features
Feature	Description
Gender	Applicant gender
Married	Marital status
Dependents	Number of dependents
Education	Education level
Self_Employed	Employment status
ApplicantIncome	Applicant income
CoapplicantIncome	Co-applicant income
LoanAmount	Requested loan amount
Loan_Amount_Term	Loan repayment duration
Credit_History	Past credit record
Property_Area	Location category (Urban/Semiurban/Rural)

🧠 Machine Learning Pipeline
Data Cleaning
Missing values handled using:
Median (numerical features)
Mode (categorical features)
Feature Engineering
Label encoding applied to categorical variables
Feature Scaling
StandardScaler applied to numerical features
Model Training
Logistic Regression (baseline model)
Random Forest Classifier (non-linear model)
Model Evaluation
Accuracy
Precision
Recall
F1-score

📈 Model Performance
Model	Accuracy	Notes
Logistic Regression	~78%	Stable baseline model
Random Forest	~77%	Better at capturing complex relationships

Random Forest performed better in identifying approved loans, making it more suitable for minimizing missed approvals in financial decision-making.


🌐 Streamlit Web Application

A Streamlit-based interactive application was built to allow users input applicant details and receive real-time loan approval predictions.

Features:
User-friendly interface
Real-time prediction
Automatic preprocessing (encoding + scaling)
Instant classification output (Approved / Rejected)
Run Application:
streamlit run app.py

💼 Business Impact
Helps financial institutions reduce default risk
Improves consistency in loan approval decisions
Identifies key factors influencing creditworthiness
Supports data-driven lending strategies
Enables real-time decision support through deployed application

🧩 Project Structure

Loan_Prediction_Project/

│

├── app.py

├── models/

│   ├── rf_model.pkl

│   ├── scaler.pkl

│

├── notebooks/

│   └── loan_pred.ipynb

│

├── images/

│   └── feature_importance.png

│

├── requirements.txt

└── README.md


🚀 How to Run This Project
1. Clone Repository
git clone https://github.com/HamzyinTech/loan-approval-prediction.git
cd loan-approval-prediction
2. Install Dependencies
pip install -r requirements.txt
3. Load Model and Scaler
import joblib
import pandas as pd

model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
4. Make Predictions
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

numeric_cols = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term'
]

new_applicant[numeric_cols] = scaler.transform(new_applicant[numeric_cols])

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("✅ Loan Approved")
else:
    print("❌ Loan Rejected")
    
🔮 Future Improvements
Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
Advanced models (XGBoost, LightGBM)
Model interpretability using SHAP values
API integration for real-time predictions
Cloud deployment of Streamlit app

📌 Key Takeaways
Feature engineering significantly improves model performance
Class imbalance affects recall and must be carefully handled
Tree-based models capture complex financial relationships effectively
Model interpretability is essential in financial applications
End-to-end deployment improves real-world usability

⚠️ Limitations
Performance depends on dataset quality and size
Does not account for external economic conditions
Potential bias due to class imbalance
Requires further tuning for production deployment


👨‍💻 Author

HamzyinTech💡
