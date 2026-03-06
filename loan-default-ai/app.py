import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import importlib
import utils.prediction
importlib.reload(utils.prediction)

from utils.prediction import load_model, generate_risk_summary, generate_recommendations, get_risk_adjustment, predict_default_probability, get_feature_importances, generate_loan_decision

st.set_page_config(layout="wide", page_title="AI Loan Default Intelligence System")

# Load the model with caching
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Custom CSS for the dashboard style
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f4f7fb;
    }
    
    /* Top title styling */
    .title-header {
        text-align: center; 
        color: #1e3a8a; 
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    /* Customizing containers */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: none;
    }
    
    h3 {
        color: #1e3a8a !important;
        font-size: 1.1rem !important;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    .info-value {
        font-weight: 600;
        color: #111827;
    }
    
    .info-label {
        color: #4b5563;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Inputs for Borrower Data
with st.sidebar:
    st.markdown("<h3>Simulate Borrower Data</h3>", unsafe_allow_html=True)
    
    # Input Widgets
    name = st.text_input("Borrower Name", value="Rajesh Sharma")
    age = st.slider("Age", min_value=18, max_value=85, value=38)
    
    # Profession Dropdown
    profession = st.selectbox("Profession", [
        "IT / Software Engineer", "Government Employee", "PSU Employee", 
        "Banking Professional", "Doctor / Medical Professional", 
        "Chartered Accountant / Lawyer", "Teacher / Professor", 
        "Business Owner", "Freelancer / Self-Employed", 
        "Student / New Job", "Other"
    ], index=0)
    
    # Loan Type Dropdown
    loan_type = st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Car Loan", "Education Loan", "Business Loan"], index=0)
    
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000.0, max_value=5000000.0, value=800000.0, step=10000.0)
    loan_tenure = st.number_input("Loan Tenure (years)", min_value=1, max_value=30, value=3, step=1)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=10000.0, max_value=1000000.0, value=75000.0, step=5000.0)
    
    credit_score = st.slider("Credit Score (CIBIL)", min_value=300, max_value=900, value=680)
    credit_utilization = st.slider("Credit Utilization", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    late_payments = st.number_input("Late Payments", min_value=0, max_value=20, value=2, step=1)
    
    # Logic for dynamic interest rate
    from utils.prediction import get_profession_base_rate, get_salary_adjustment
    base_rate = get_profession_base_rate(profession)
    salary_adj = get_salary_adjustment(monthly_income)
    
    # We'll use a preliminary risk adjustment of 0 until prediction is run
    # or use the session state risk adjustment if available.
    risk_adj = 0.0
    if "risk_adjustment" in st.session_state:
         risk_adj = st.session_state.risk_adjustment
         
    final_interest_rate = base_rate + salary_adj + risk_adj
    
    # Calculate EMI and DTI automatically
    from utils.prediction import calculate_emi
    emi = calculate_emi(loan_amount, final_interest_rate, loan_tenure)
    debt_to_income = emi / monthly_income if monthly_income > 0 else 0.0
    
    with st.container(border=True):
        st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 0.5rem;'>Calculated Metrics</h4>", unsafe_allow_html=True)
        st.markdown(f"**EMI:** ₹{emi:,.2f}")
        st.markdown(f"**Debt-to-Income Ratio:** {debt_to_income:.2f} ({debt_to_income*100:.1f}%)")
        st.markdown(f"**Final Calculated Rate:** {final_interest_rate:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict Default Risk", type="primary", use_container_width=True)

# Application State Logic
if "risk_percent" not in st.session_state:
    st.session_state.risk_percent = 0
if "risk_label" not in st.session_state:
    st.session_state.risk_label = "🟡 MEDIUM RISK"
if "risk_color" not in st.session_state:
    st.session_state.risk_color = "#f1c40f"
if "risk_adjustment" not in st.session_state:
    st.session_state.risk_adjustment = 0.0
if "feature_importances" not in st.session_state:
    st.session_state.feature_importances = get_feature_importances(model)
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False
    
if predict_button:
    
    # Get raw probability
    raw_prob = predict_default_probability(model, credit_score, debt_to_income, credit_utilization, late_payments, loan_amount, monthly_income)
    
    # Convert to percentage
    prob_pct = min(100.0, max(0.0, raw_prob * 100))
    st.session_state.risk_percent = int(prob_pct)
    
    # Update risk adjustment in session state
    risk_adj, _ = get_risk_adjustment(st.session_state.risk_percent / 100)
    st.session_state.risk_adjustment = risk_adj
    
    # Update dynamic feature importance
    current_data = {
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'credit_utilization': credit_utilization,
        'late_payments': late_payments,
        'loan_amount': loan_amount,
        'monthly_income': monthly_income
    }
    st.session_state.feature_importances = get_feature_importances(model, data=current_data)
    
    # Determine risk label and color based on the gauge ranges
    if st.session_state.risk_percent < 30:
        st.session_state.risk_label = "🟢 LOW RISK"
        st.session_state.risk_color = "#2ecc71"
    elif st.session_state.risk_percent < 60:
        st.session_state.risk_label = "🟡 MEDIUM RISK"
        st.session_state.risk_color = "#f1c40f"
    else:
        st.session_state.risk_label = "🔴 HIGH RISK"
        st.session_state.risk_color = "#e74c3c"
        
    st.session_state.prediction_run = True

st.markdown("<h2 class='title-header'>🏦 AI Loan Default Intelligence System</h2>", unsafe_allow_html=True)

st.write("") # Spacer before main columns
st.write("") 

col1, col2, col3 = st.columns([1, 2.2, 1.2])

with col1:
    with st.container(border=True):
        st.markdown("<h3>👤 Borrower Info</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="https://ui-avatars.com/api/?name={''.join(name.split())}&background=random&color=fff" style="border-radius: 50%; width: 60px; height: 60px; margin-right: 15px;">
            <div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #1e3a8a;">{name}</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Age: {age} | {profession}</div>
            </div>
        </div>
        <div style="margin-top: 1.5rem;">
            <div class="info-row">
                <span class="info-label">Loan Details</span>
                <span class="info-value">{loan_type}</span>
            </div>
            <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #f3f4f6;">
            <div class="info-row">
                <span class="info-label">Loan Amount</span>
                <span class="info-value">₹{loan_amount:,.2f}</span>
            </div>
            <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #f3f4f6;">
            <div class="info-row">
                <span class="info-label">Monthly Income</span>
                <span class="info-value">₹{monthly_income:,.2f}</span>
            </div>
            <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #f3f4f6;">
            <div class="info-row">
                <span class="info-label">CIBIL Score</span>
                <span class="info-value">{credit_score}</span>
            </div>
            <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #f3f4f6;">
            <div class="info-row">
                <span class="info-label">Risk Status</span>
                <span class="info-value" style="color: {st.session_state.risk_color};">{st.session_state.risk_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown(f"<h3 style='text-align: center; border-bottom: none;'>{st.session_state.risk_label}</h3>", unsafe_allow_html=True)
        
        # Professional Fintech Gauge Chart
        risk_percent = st.session_state.risk_percent
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percent,
            delta = {'reference': 50, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#2ecc71"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#4b5563"},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},   # Low Risk (Green)
                    {'range': [30, 60], 'color': "#f1c40f"},  # Medium Risk (Yellow)
                    {'range': [60, 100], 'color': "#e74c3c"}  # High Risk (Red)
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 6},
                    'thickness': 0.8,
                    'value': risk_percent
                }
            },
            number = {'suffix': "%", 'font': {'size': 50, 'color': '#111827'}}
        ))
        
        fig_gauge.update_layout(
            height=300, 
            margin=dict(l=30, r=30, t=50, b=20),
            font=dict(family="Arial", color="#1e3a8a"),
            transition={'duration': 500}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown(f"""
        <div style="text-align: center; color: #4b5563; font-size: 0.9rem; margin-top: -10px;">
            AI Risk Score: {risk_percent}% | 
            <span style="color: {st.session_state.risk_color}; font-weight: 600;">{st.session_state.risk_label}</span>
        </div>
        """, unsafe_allow_html=True)

with col3:
    with st.container(border=True):
        st.markdown("<h3>📈 Feature Importance</h3>", unsafe_allow_html=True)
        
        # Use importance from session state
        importances = st.session_state.feature_importances
        
        features_df = pd.DataFrame({
            'Feature': list(importances.keys()),
            'Importance': list(importances.values())
        })
        
        # Sort values: highest importance at the top for better visualization
        features_df = features_df.sort_values(by='Importance', ascending=True)
        
        fig_bar = px.bar(
            features_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            text='Importance',
        )
        
        fig_bar.update_traces(
            marker_color='#3b82f6', 
            texttemplate='%{text:.2%}', # Display as percentage
            textposition='outside',
            cliponaxis=False
        )
        
        fig_bar.update_layout(
            height=300, # Increased height to fit all 6 features
            margin=dict(l=10, r=40, t=20, b=10),
            xaxis=dict(visible=False, range=[0, max(features_df['Importance']) * 1.2]),
            yaxis=dict(title=None, tickfont=dict(size=12, color='#4b5563')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

# Row 2: Document Verification Section
st.write("")
with st.container(border=True):
    st.markdown("<h3>📄 Document Verification Status</h3>", unsafe_allow_html=True)
    doc_col1, doc_col2, doc_col3, doc_col4 = st.columns(4)
    with doc_col1:
        st.checkbox("Aadhar Card Verified", value=True)
    with doc_col2:
        st.checkbox("PAN Card Verified", value=True)
    with doc_col3:
        st.checkbox("Salary Slips (3 months)", value=True)
    with doc_col4:
        st.checkbox("Bank Statement (6 months)", value=False)

st.write("") # Spacer between rows
st.write("") 

col_bottom1, col_bottom2 = st.columns([1.8, 1])

summary_text = generate_risk_summary(st.session_state.risk_percent / 100, debt_to_income, credit_utilization, late_payments)
rec_title, rec_bullets, final_rate, final_emi = generate_recommendations(st.session_state.risk_percent / 100, profession, monthly_income, loan_amount, loan_tenure)

# AI Loan Decision Engine Call
decision_obj = generate_loan_decision(
    st.session_state.risk_percent,
    credit_score,
    debt_to_income,
    credit_utilization,
    late_payments,
    monthly_income,
    loan_amount,
    profession,
    final_rate
)

with col_bottom1:
    with st.container(border=True):
        st.markdown("<h3>🤖 AI Risk Summary</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 1.2rem; border-radius: 6px; border-left: 4px solid #3b82f6;">
            <p style="color: #374151; line-height: 1.6; margin-bottom: 0;">
                {summary_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_bottom2:
    with st.container(border=True):
        st.markdown("<h3>📋 Recommendation</h3>", unsafe_allow_html=True)

        # Build bullet list HTML for suggested actions
        actions_html = "".join([f"<li>{a}</li>" for a in decision_obj["actions"]])

        # Decision badge background (light tint of decision color)
        badge_bg = decision_obj["color"] + "22"

        st.markdown(f"""
<div style="background-color:{decision_obj['color']}11;border-left:5px solid {decision_obj['color']};border-radius:8px;padding:1rem 1.2rem;">
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;">
<span style="background:{badge_bg};color:{decision_obj['color']};font-weight:700;font-size:1.05rem;padding:0.3rem 0.8rem;border-radius:20px;border:1.5px solid {decision_obj['color']};letter-spacing:0.5px;">
{decision_obj["decision"]}
</span>
</div>
<p style="color:#1e3a8a;font-weight:600;font-size:0.92rem;margin:0 0 0.9rem 0;">{decision_obj["explanation"]}</p>
<div style="background:white;border-radius:6px;border:1px solid #e2e8f0;padding:0.7rem 0.9rem;margin-bottom:0.9rem;font-size:0.88rem;">
<div style="display:flex;justify-content:space-between;padding:0.25rem 0;border-bottom:1px solid #f1f5f9;"><span style="color:#6b7280;">Loan Type</span><span style="font-weight:600;color:#111827;">{loan_type}</span></div>
<div style="display:flex;justify-content:space-between;padding:0.25rem 0;border-bottom:1px solid #f1f5f9;"><span style="color:#6b7280;">Interest Rate</span><span style="font-weight:700;color:{decision_obj['color']};">{final_rate:.2f}%</span></div>
<div style="display:flex;justify-content:space-between;padding:0.25rem 0;border-bottom:1px solid #f1f5f9;"><span style="color:#6b7280;">Monthly EMI</span><span style="font-weight:600;color:#111827;">₹{final_emi:,.2f}</span></div>
<div style="display:flex;justify-content:space-between;padding:0.25rem 0;"><span style="color:#6b7280;">Risk Score</span><span style="font-weight:600;color:{st.session_state.risk_color};">{st.session_state.risk_percent}%</span></div>
</div>
<div style="font-weight:600;font-size:0.87rem;color:#374151;margin-bottom:0.35rem;">📌 Suggested Actions</div>
<ul style="color:#374151;padding-left:18px;font-size:0.88rem;line-height:1.7;margin:0;">
{actions_html}
</ul>
</div>
""", unsafe_allow_html=True)

