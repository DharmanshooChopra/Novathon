# AI Loan Default Intelligence System 🏦

An interactive Streamlit-based banking dashboard designed for real-time loan default risk assessment using Machine Learning.

## 🚀 Features
- **Interactive Simulation**: Sidebar widgets to input borrower demographics and loan details.
- **Automated DTI Calculation**: Real-time EMI and Debt-to-Income (DTI) computation based on loan amount, interest rate, and tenure.
- **AI Risk Assessment**: Random Forest Classifier predicts default probability based on 6 key financial features.
- **Dynamic Risk Summary**: AI-generated explanation of the risk level and borrower profile.
- **Actionable Recommendations**: Banking-grade loan approval conditions based on predicted risk.
- **Professional UI**: Clean, responsive banking theme with Plotly visualizations.

## 📁 Project Structure
```
loan-default-ai/
├── app.py              # Main Streamlit application
├── model/
│   └── model.pkl       # Trained Random Forest model
├── data/
│   └── sample_data.csv # Synthetic training data
├── utils/
│   └── prediction.py   # Prediction logic and UI helper functions
├── scripts/
│   └── generate_model.py # Script to retrain/regenerate the model
└── requirements.txt    # Project dependencies
```

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DharmanshooChopra/Novathon.git
   cd Novathon/loan-default-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Model Information
The system uses a **Random Forest Classifier** trained on:
- Credit Score (CIBIL)
- Debt-to-Income Ratio (DTI)
- Credit Utilization
- Late Payments
- Loan Amount
- Monthly Income

The model is optimized for the Indian banking context, providing specific recommendations for Approval, Rejection, or Approval with Conditions.

## 🔍 Technical Analysis

### 1. Data Simulation Strategy
- Uses `np.random.poisson` for late payments to mirror real-world "skewed" credit behavior where most people have 0-1 late payments.
- Normal distribution noise is added to the `risk_score` to ensure the model isn't a simple linear function, making the AI behavior more realistic.

### 2. EMI & DTI Automation Logic
- **Formula**: `EMI = [P × r × (1+r)^n] / [(1+r)^n − 1]`
- The system automatically triggers a recalculation of the **Debt-to-Income Ratio** whenever the Monthly Income or Loan parameters change.
- This ensures the model always receives the most up-to-date financial burden metric.

### 3. Layout & UX Design
- **Card-based UI**: Uses `st.container(border=True)` to create distinct visual blocks, mimicking professional banking portals.
- **Visual Feedback**: The Plotly Gauge provides immediate emotional cues (Green/Yellow/Red) which are more intuitive for bank officers than raw probability numbers.
- **Sidebar isolation**: All inputs are kept in the sidebar to keep the main dashboard focused on "Results" and "Insights".

