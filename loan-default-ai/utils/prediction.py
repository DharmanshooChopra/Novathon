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

def get_profession_base_rate(profession):
    """Returns the base interest rate for a given profession."""
    rates = {
        "Government Employee": 9.5,
        "PSU Employee": 9.7,
        "Banking Professional": 10.0,
        "IT / Software Engineer": 10.2,
        "Doctor / Medical Professional": 10.4,
        "Chartered Accountant / Lawyer": 10.8,
        "Teacher / Professor": 11.0,
        "Business Owner": 12.5,
        "Freelancer / Self-Employed": 13.5,
        "Student / New Job": 14.5,
        "Other": 12.0
    }
    return rates.get(profession, 12.0)

def get_salary_adjustment(monthly_income):
    """Returns the interest rate adjustment based on monthly income."""
    if monthly_income >= 150000:
        return 0.0
    elif monthly_income >= 80000:
        return 0.5
    elif monthly_income >= 50000:
        return 1.0
    else:
        return 1.5

def get_risk_adjustment(probability):
    """Returns the interest rate adjustment based on AI risk score."""
    if probability < 0.3:
        return -0.5, "Approve"
    elif probability < 0.6:
        return 1.0, "Approve with Conditions"
    else:
        return 2.0, "High Risk"

def calculate_emi(principal, annual_interest_rate, tenure_years):
    """Calculates Equated Monthly Installment (EMI)."""
    if annual_interest_rate == 0:
        return principal / (tenure_years * 12) if tenure_years > 0 else 0
    
    r = (annual_interest_rate / 100) / 12
    n = tenure_years * 12
    
    if n == 0:
        return principal
        
    emi = (principal * r * (1 + r)**n) / ((1 + r)**n - 1)
    return emi

def calculate_dynamic_importance(credit_score, debt_to_income, credit_utilization, late_payments, loan_amount, monthly_income):
    """Calculates manual influence scores based on user input magnitude."""
    # Using the user's specific fallback logic
    importance = {
        "Credit Score": max(0, (750 - credit_score) / 750),
        "Debt-to-Income Ratio": min(1.0, debt_to_income),
        "Credit Utilization": min(1.0, credit_utilization),
        "Late Payments": min(1.0, late_payments / 10),
        "Loan Amount": min(1.0, loan_amount / 1000000),
        "Monthly Income": max(0, 1 - (monthly_income / 200000))
    }
    
    # Normalize to ensure sum = 1.0 for the bar chart
    total = sum(importance.values())
    if total == 0:
        return {k: 1.0/len(importance) for k in importance.keys()}
        
    return {k: v / total for k, v in importance.items()}

def get_feature_importances(model, data=None):
    """
    Returns feature importances. If data is provided, calculates dynamic 
    influence scores based on current inputs.
    """
    if data is not None:
        return calculate_dynamic_importance(
            data.get('credit_score', 680),
            data.get('debt_to_income', 0.4),
            data.get('credit_utilization', 0.5),
            data.get('late_payments', 0),
            data.get('loan_amount', 500000),
            data.get('monthly_income', 50000)
        )
        
    # Standard static importance if no specific data is provided
    return {
        "Debt-to-Income": 0.36,
        "Credit Utilization": 0.28,
        "Late Payments": 0.22,
        "Credit Score": 0.10,
        "Loan Amount": 0.03,
        "Monthly Income": 0.01
    }

def generate_loan_decision(
    risk_percent,
    credit_score,
    debt_to_income,
    credit_utilization,
    late_payments,
    monthly_income,
    loan_amount,
    profession,
    interest_rate
):
    """
    AI Loan Decision Engine: evaluates borrower profile and returns
    a structured decision with explanation and suggested actions.
    Returns: {"decision", "explanation", "actions", "color"}
    """

    # ── 1. REJECT LOAN ──────────────────────────────────────────────
    # Triggered by any one of: very high risk, poor CIBIL, excessive DTI,
    # or multiple (>1) late payments.
    if (
        risk_percent > 70
        or credit_score < 600
        or debt_to_income > 0.5
        or late_payments > 1
    ):
        return {
            "decision": "REJECT LOAN",
            "explanation": (
                "The applicant has a high debt burden and a weak credit score "
                "which significantly increases the risk of default."
            ),
            "actions": [
                "Improve CIBIL score to at least 650",
                "Clear existing overdue debts",
                "Reduce total monthly debt obligations",
                "Reapply after 6 months of clean payment history",
            ],
            "color": "#e74c3c",  # Red
        }

    # ── 2. APPROVE LOAN ─────────────────────────────────────────────
    # All green indicators: low risk, strong CIBIL, healthy DTI, zero late payments.
    if (
        risk_percent < 30
        and credit_score > 700
        and debt_to_income < 0.35
        and late_payments == 0
    ):
        return {
            "decision": "APPROVE LOAN",
            "explanation": (
                "The borrower has a strong credit history, manageable debt "
                "obligations, and stable income."
            ),
            "actions": [
                "Standard interest rate applies",
                "Normal loan terms and tenure",
                "No additional collateral required",
            ],
            "color": "#27ae60",  # Green
        }

    # ── 3. RENEGOTIATE LOAN TERMS ───────────────────────────────────
    # Elevated risk band (50-70%) OR loan amount > 4× annual income.
    if (50 <= risk_percent <= 70) or (loan_amount > monthly_income * 12 * 4):
        return {
            "decision": "RENEGOTIATE LOAN",
            "explanation": (
                "Loan amount is high relative to income or risk is moderately "
                "elevated. Structural adjustments to the loan are required."
            ),
            "actions": [
                "Reduce requested loan amount",
                "Extend loan tenure to lower monthly EMI",
                "Increase interest rate by 1.5%",
                "Provide additional income proof or co-applicant",
            ],
            "color": "#e67e22",  # Orange
        }

    # ── 4. APPROVE WITH CONDITIONS ──────────────────────────────────
    # Default for moderate-risk cases (risk 30-60, CIBIL 650-700, DTI 0.35-0.45).
    return {
        "decision": "APPROVE WITH CONDITIONS",
        "explanation": (
            "Borrower meets basic eligibility criteria but shows moderate risk "
            "indicators. Conditional approval with adjustments."
        ),
        "actions": [
            "Increase interest rate slightly (+0.5% – 1.0%)",
            "Require additional income / employment verification",
            "Reduce loan tenure to a maximum of 3 years",
        ],
        "color": "#f39c12",  # Amber/Yellow
    }

def generate_recommendations(probability, profession, monthly_income, loan_amount, tenure_years):
    """Generate recommendations based on risk features and profile."""
    base_rate = get_profession_base_rate(profession)
    salary_adj = get_salary_adjustment(monthly_income)
    risk_adj, decision = get_risk_adjustment(probability)
    
    final_rate = base_rate + salary_adj + risk_adj
    emi = calculate_emi(loan_amount, final_rate, tenure_years)
    
    if decision == "High Risk":
        return decision, [
            f"Likely Rejection due to High Risk ({probability*100:.1f}%)",
            "Consider reducing loan amount significantly",
            "Improve CIBIL score before reapplying",
            "Provide additional collateral"
        ], final_rate, emi
    elif decision == "Approve with Conditions":
        return decision, [
            f"Increase monitoring due to Moderate Risk",
            f"Interest adjusted to {final_rate:.2f}%",
            "Require co-signer or guarantor",
            f"Suggested EMI: ₹{emi:,.2f}"
        ], final_rate, emi
    else:
        return decision, [
            "Excellent profile for standard approval",
            f"Premium interest rate of {final_rate:.2f}% offered",
            "No additional collateral required",
            f"EMI set at ₹{emi:,.2f}"
        ], final_rate, emi

def generate_risk_summary(
    credit_score,
    debt_to_income,
    credit_utilization,
    late_payments,
    monthly_income,
    loan_amount,
    risk_percent
):
    """
    Dynamic AI Risk Analysis Engine.
    Analyzes borrower data and returns a structured risk summary.
    Returns: {risk_category, summary, positive_factors, risk_factors}
    """

    # ── Risk Category ───────────────────────────────────────────────
    if risk_percent < 30:
        risk_category = "Low Risk"
    elif risk_percent < 60:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"

    # ── Identify Risk Drivers ────────────────────────────────────────
    risk_factors = []
    if credit_score < 650:
        risk_factors.append(f"Low CIBIL score ({credit_score})")
    if debt_to_income > 0.4:
        risk_factors.append(f"High debt-to-income ratio ({debt_to_income*100:.0f}%)")
    if credit_utilization > 0.6:
        risk_factors.append(f"High credit utilization ({credit_utilization*100:.0f}%)")
    if late_payments >= 2:
        risk_factors.append(f"History of late payments ({late_payments} missed)")
    elif late_payments == 1:
        risk_factors.append("1 late payment on record")
    if loan_amount > monthly_income * 12:
        risk_factors.append("Loan amount exceeds annual income")
    if debt_to_income > 0.5:
        risk_factors.append("Debt obligations are unsustainably high")

    # ── Identify Positive Indicators ─────────────────────────────────
    positive_factors = []
    if credit_score > 720:
        positive_factors.append(f"Strong CIBIL score ({credit_score})")
    elif credit_score > 680:
        positive_factors.append(f"Good CIBIL score ({credit_score})")
    if debt_to_income < 0.3:
        positive_factors.append(f"Healthy debt-to-income ratio ({debt_to_income*100:.0f}%)")
    if credit_utilization < 0.3:
        positive_factors.append(f"Low credit utilization ({credit_utilization*100:.0f}%)")
    if late_payments == 0:
        positive_factors.append("Clean repayment history (0 late payments)")
    if monthly_income >= 100000:
        positive_factors.append(f"High monthly income (₹{monthly_income:,.0f})")
    elif monthly_income >= 60000:
        positive_factors.append(f"Stable monthly income (₹{monthly_income:,.0f})")

    # ── Build Narrative Summary ──────────────────────────────────────
    repayment_capacity = (
        "stable and well within acceptable thresholds"
        if risk_percent < 30 else
        "moderate — manageable with appropriate loan conditions"
        if risk_percent < 60 else
        "high-risk and requires significant improvement before loan sanction"
    )

    pos_text = (
        f"Positive indicators include: {'; '.join(positive_factors)}."
        if positive_factors else
        "No strong positive indicators were identified."
    )
    risk_text = (
        f"Risk factors identified: {'; '.join(risk_factors)}."
        if risk_factors else
        "No significant risk factors were detected."
    )

    summary = (
        f"The AI model predicts a **{risk_category}** profile with a "
        f"default probability of **{risk_percent}%**. "
        f"{pos_text} "
        f"{risk_text} "
        f"Based on these financial indicators, the borrower's repayment "
        f"capacity appears **{repayment_capacity}**."
    )

    return {
        "risk_category": risk_category,
        "summary": summary,
        "positive_factors": positive_factors,
        "risk_factors": risk_factors,
    }
