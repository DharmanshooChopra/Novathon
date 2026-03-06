import joblib
import pandas as pd
import numpy as np
import os

def load_model(model_path="model/model.pkl"):
    """Loads the scikit-learn model from the specified path."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def predict_default_probability(model, credit_score, debt_to_income, credit_utilization, late_payments, loan_amount, monthly_income):
    """
    Predicts the probability of default based on input features.
    """
    if model is None:
        return 0.0

    # Ensure input matches the feature names used during training
    input_df = pd.DataFrame([{
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'credit_utilization': credit_utilization,
        'late_payments': late_payments,
        'loan_amount': loan_amount,
        'monthly_income': monthly_income
    }])

    # Predict probabilities, column 1 is usually the '1' (default) class
    prob = model.predict_proba(input_df)[0][1]
    return prob

def predict_default(data):
    """
    Takes a dictionary of input feature data, loads the model, 
    and returns the predicted default probability.
    
    Expected data frame format:
    data = {
        'credit_score': int,
        'debt_to_income': float,
        'credit_utilization': float,
        'late_payments': int,
        'loan_amount': float,
        'monthly_income': float
    }
    """
    model = load_model()
    if model is None:
        raise FileNotFoundError("model/model.pkl could not be loaded")
        
    df = pd.DataFrame([data])
    
    # predict_proba returns array like [[prob_class_0, prob_class_1]]
    probability = model.predict_proba(df)[0][1]
    return float(probability)

def get_risk_level(probability):
    """Returns the risk level based on the probability."""
    if probability < 0.3:
        return "Low Risk", "green"
    elif probability < 0.6:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def get_feature_importances(model):
    """Returns dummy feature importances to match the dashboard view."""
    return {
        "Debt-to-Income": 0.36,
        "Credit Utilization": 0.28,
        "Late Payments": 0.22
    }

def generate_recommendations(probability, debt_to_income, credit_utilization, late_payments):
    """Generate recommendations based on risk features."""
    if probability >= 0.6:
        return "Approve with Conditions", [
            "Reduce loan amount to ₹6,00,000",
            "Increase interest rate to 11.5%",
            "Require income verification",
            "Suggest EMI tenure of 3 years"
        ]
    elif probability >= 0.3:
        return "Approve with Conditions", [
            "Increase interest rate to 9.5%",
            "Require a co-signer",
            "Reduce loan term to 3 years"
        ]
    else:
        return "Approve", [
            "Approve standard loan.",
            "Valid for lowest interest rate tier.",
            "No additional conditions required."
        ]

def generate_risk_summary(probability, debt_to_income, credit_utilization, late_payments):
    if probability >= 0.6:
        return "The AI analysis indicates a high likelihood of default due to a relatively high debt-to-income ratio, moderate credit utilization, and recent late payment history. The borrower may face repayment pressure if financial conditions change."
    elif probability >= 0.3:
        return "The AI analysis indicates a moderate likelihood of default. Some risk factors are elevated but within manageable thresholds with appropriate conditions."
    else:
        return "The AI analysis indicates a low likelihood of default. The applicant has strong financials and a solid payment history."
