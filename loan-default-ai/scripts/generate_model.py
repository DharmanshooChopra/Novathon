import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure directories exist
os.makedirs('../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)

# Generate dummy data
# Features needed for the dashboard:
# - Debt-to-Income (continuous, let's say 0.1 to 0.8)
# - Credit Utilization (continuous, let's say 0.1 to 0.9)
# - Late Payments (integer, let's say 0 to 10)
# Target: Default (0 = No, 1 = Yes)

np.random.seed(42)
n_samples = 1000

credit_score = np.random.randint(300, 900, n_samples)
credit_utilization = np.random.uniform(0.0, 1.0, n_samples)
late_payments = np.random.poisson(1, n_samples)
monthly_income = np.random.uniform(10000, 1000000, n_samples)
loan_amount = np.random.uniform(10000, 5000000, n_samples)
debt_to_income = np.random.uniform(0.0, 1.0, n_samples)

# Simple logic for default risk
risk_score = debt_to_income * 0.4 + credit_utilization * 0.4 + (late_payments / 10) * 0.2 - (credit_score / 900) * 0.2
risk_score += np.random.normal(0, 0.1, n_samples)

default = (risk_score > 0.45).astype(int)

df = pd.DataFrame({
    'credit_score': credit_score,
    'debt_to_income': debt_to_income,
    'credit_utilization': credit_utilization,
    'late_payments': late_payments,
    'loan_amount': loan_amount,
    'monthly_income': monthly_income,
    'default': default
})

# Use local path instead of parent folder path since we are executing from the root
try:
    df.to_csv('data/sample_data.csv', index=False)
except FileNotFoundError:
    df.to_csv('../data/sample_data.csv', index=False)

# Train dummy model
X = df[['credit_score', 'debt_to_income', 'credit_utilization', 'late_payments', 'loan_amount', 'monthly_income']]
y = df['default']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
try:
    joblib.dump(model, 'model/model.pkl')
    print("Saved model/model.pkl")
except FileNotFoundError:
    joblib.dump(model, '../model/model.pkl')
    print("Saved ../model/model.pkl")
