import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import importlib
import utils.prediction
importlib.reload(utils.prediction)

from utils.prediction import load_model

st.set_page_config(layout="wide", page_title="AI Loan Default Intelligence System")

# Load the model
model = load_model()

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
    occupation = st.text_input("Occupation", value="Salaried Employee (IT Sector)")
    loan_type = st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Auto Loan", "Education Loan"], index=0)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000.0, max_value=5000000.0, value=800000.0, step=10000.0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=11.5, step=0.1)
    loan_tenure = st.number_input("Loan Tenure (years)", min_value=1, max_value=30, value=3, step=1)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=10000.0, max_value=1000000.0, value=75000.0, step=5000.0)
    credit_score = st.slider("Credit Score (CIBIL)", min_value=300, max_value=900, value=680)
    credit_utilization = st.slider("Credit Utilization", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    late_payments = st.number_input("Late Payments", min_value=0, max_value=20, value=2, step=1)
    
    # Calculate EMI and DTI automatically
    P = loan_amount
    r = (interest_rate / 100.0) / 12.0
    n = loan_tenure * 12
    if r > 0 and n > 0:
        emi = (P * r * (1 + r)**n) / ((1 + r)**n - 1)
    elif n > 0:
        emi = P / n
    else:
        emi = 0.0
        
    debt_to_income = emi / monthly_income if monthly_income > 0 else 0.0
    
    with st.container(border=True):
        st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 0.5rem;'>Calculated Metrics</h4>", unsafe_allow_html=True)
        st.markdown(f"**EMI:** ₹{emi:,.2f}")
        st.markdown(f"**Debt-to-Income Ratio:** {debt_to_income:.2f} ({debt_to_income*100:.1f}%)")
        st.markdown(f"**Estimated Monthly Burden:** ₹{emi:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict Default Risk", type="primary", use_container_width=True)

# Application State Logic
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.prob = 64.0 # Default fallback
    st.session_state.risk_label = "High Risk"
    st.session_state.risk_color = "#ef4444"
    
if predict_button:
    from utils.prediction import predict_default_probability
    
    # Get raw probability
    raw_prob = predict_default_probability(model, credit_score, debt_to_income, credit_utilization, late_payments, loan_amount, monthly_income)
    
    # Convert to percentage
    prob_pct = min(100.0, max(0.0, raw_prob * 100))
    st.session_state.prob = round(prob_pct, 1)
    
    # Determine risk label and color based on the gauge ranges
    if prob_pct < 30:
        st.session_state.risk_label = "Low Risk"
        st.session_state.risk_color = "#22c55e"
    elif prob_pct < 60:
        st.session_state.risk_label = "Medium Risk"
        st.session_state.risk_color = "#eab308"
    else:
        st.session_state.risk_label = "High Risk"
        st.session_state.risk_color = "#ef4444"
        
    st.session_state.prediction_run = True

st.markdown("<h2 class='title-header'>🏦 AI Loan Default Intelligence System</h2>", unsafe_allow_html=True)

st.write("") # Spacer before main columns
st.write("") 

col1, col2, col3 = st.columns([1, 1.8, 1.2])

with col1:
    with st.container(border=True):
        st.markdown("<h3>👤 Borrower Info</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="https://ui-avatars.com/api/?name={''.join(name.split())}&background=random&color=fff" style="border-radius: 50%; width: 60px; height: 60px; margin-right: 15px;">
            <div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #1e3a8a;">{name}</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Age: {age} | {occupation}</div>
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
        </div>
        """, unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center; border-bottom: none;'>📊 Default Probability</h3>", unsafe_allow_html=True)
        
        # Gauge Chart using session state for dynamics
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = st.session_state.prob,
            title = {'text': st.session_state.risk_label, 'font': {'size': 20, 'color': st.session_state.risk_color}},
            number = {'suffix': "%", 'font': {'size': 60, 'color': '#111827'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'visible': False},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': "#22c55e"}, # Green (Low Risk)
                    {'range': [30, 60], 'color': "#eab308"}, # Yellow (Medium Risk)
                    {'range': [60, 100], 'color': "#ef4444"}], # Red (High Risk)
                'threshold': {
                    'line': {'color': "#111827", 'width': 4},
                    'thickness': 0.75,
                    'value': 68}
            }
        ))
        
        fig_gauge.update_layout(
            height=300, 
            margin=dict(l=20, r=20, t=10, b=10),
            font=dict(family="sans-serif"),
        )
        
        
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

with col3:
    with st.container(border=True):
        st.markdown("<h3>📈 Feature Importance</h3>", unsafe_allow_html=True)
        
        from utils.prediction import get_feature_importances
        importances = get_feature_importances(model)
        
        features_df = pd.DataFrame({
            'Feature': list(importances.keys()),
            'Importance': list(importances.values())
        })
        
        fig_bar = px.bar(
            features_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            text='Importance',
        )
        
        fig_bar.update_traces(
            marker_color='#3b82f6', 
            texttemplate='%{text:.2f}', 
            textposition='outside',
            cliponaxis=False
        )
        
        fig_bar.update_layout(
            height=250,
            margin=dict(l=10, r=40, t=20, b=10),
            xaxis=dict(visible=False, range=[0, 0.45]),
            yaxis=dict(title=None, categoryorder='array', categoryarray=['Late Payments', 'Credit Utilization', 'Debt-to-Income'], tickfont=dict(size=12, color='#4b5563')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

st.write("") # Spacer between rows
st.write("") 

col_bottom1, col_bottom2 = st.columns([1.8, 1])

# Dynamic Risk Summary Logic
from utils.prediction import generate_risk_summary, generate_recommendations
summary_text = generate_risk_summary(st.session_state.prob / 100, debt_to_income, credit_utilization, late_payments)
rec_title, rec_bullets = generate_recommendations(st.session_state.prob / 100, debt_to_income, credit_utilization, late_payments)

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
        
        bullets_html = "".join([f"<li>{b}</li>" for b in rec_bullets])
        
        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 1rem; border-radius: 6px; border-left: 4px solid #3b82f6;">
            <div style="font-weight: 600; font-size: 1.1rem; color: #1e3a8a; margin-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem;">
                {rec_title}
            </div>
            <ul style="color: #374151; padding-left: 20px; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0;">
                {bullets_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
