import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Custom page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon=":money_with_wings:", layout="centered")

# Custom CSS for background and fonts
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #36d1c4 0%, #5b86e5 100%);
        border-radius: 8px;
        font-size: 18px;
        height: 3em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts loan approval using a machine learning model. "
    "Enter applicant details and click Predict to see the result."
)

# Title and description
st.title("🏦 Loan Approval Predictor")
st.write("#### Enter applicant details to predict loan approval:")

# Load model and scaler
lr_model = joblib.load('loan_approval_lr.pkl')
dt_model = joblib.load('loan_approval_dt.pkl')
scaler = joblib.load('loan_approval_scaler.pkl')


# Input fields in columns
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Applicant Income", min_value=1000, max_value=1000000, value=50000, step=1000, help="Monthly income in your local currency")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=10, help="Credit score (300-900)")
with col2:
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=10000000, value=200000, step=1000, help="Requested loan amount")
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

# Add validation
if loan_amount > income * 50:
    st.warning("⚠️ Warning: Loan amount is very high compared to income. This may affect approval chances.")
    
if credit_score < 500:
    st.warning("⚠️ Warning: Low credit score may reduce approval chances.")

# Encode categorical variables
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

# Predict button
if st.button("🔮 Predict"):
    try:
        # Compute IncomePerLoan feature
        income_per_loan = income / (loan_amount + 1)

        # Pre-validation: Basic requirements check before ML prediction
        def basic_requirements_check(income, loan_amount, credit_score):
            # Convert monthly income to annual
            annual_income = income * 12
            
            # Basic validation rules
            if loan_amount > annual_income:
                return False, f"❌ Loan amount (${loan_amount:,}) cannot exceed annual income (${annual_income:,})"
            
            if credit_score < 500:
                return False, f"❌ Credit score ({credit_score}) is too low. Minimum required: 500"
            
            # More realistic debt-to-income limits
            debt_to_income_ratio = loan_amount / annual_income
            
            if debt_to_income_ratio > 0.3:  # More than 30% of annual income
                return False, f"❌ Loan amount (${loan_amount:,}) is {debt_to_income_ratio:.1%} of annual income. Maximum allowed: 30%"
            
            # Additional checks for low-income applicants
            if annual_income < 24000:  # Less than $24K/year
                if debt_to_income_ratio > 0.2:  # More than 20% for low-income
                    return False, f"❌ Low income (${annual_income:,}/year) with high debt ratio ({debt_to_income_ratio:.1%}). Maximum allowed: 20%"
            
            return True, "✅ Passes basic requirements"

        # Check basic requirements first
        basic_check, basic_reason = basic_requirements_check(income, loan_amount, credit_score)
        
        if not basic_check:
            st.error(basic_reason)
            st.stop()  # Stop execution here, don't run ML models
        
        # If basic requirements pass, continue with business logic and ML
        st.success("✅ Passes basic requirements check")
        
        # Business logic validation for more realistic predictions
        def validate_loan_application(income, loan_amount, credit_score, education, self_employed):
            # Basic validation rules
            debt_to_income_ratio = loan_amount / (income * 12)  # Annual income
            
            # Conservative approval criteria
            if debt_to_income_ratio > 0.4:  # More than 40% of annual income
                return False, "Debt-to-income ratio too high"
            if credit_score < 500:
                return False, "Credit score too low"
            if education == "Not Graduate" and debt_to_income_ratio > 0.3:
                return False, "Non-graduate with high debt ratio"
            if self_employed == "Yes" and debt_to_income_ratio > 0.25:
                return False, "Self-employed with high debt ratio"
            
            return True, "Meets basic criteria"

        # Apply business logic
        is_valid, reason = validate_loan_application(income, loan_amount, credit_score, education, self_employed)
        
        if not is_valid:
            st.warning(f"⚠️ Business Logic Check: {reason}")
            # Still show ML predictions but with warning
        else:
            st.success("✅ Passes business logic checks")

        # Prepare input DataFrame with proper column names for the models
        input_df = pd.DataFrame({
            'ApplicantIncome': [income],
            'LoanAmount': [loan_amount], 
            'CreditScore': [credit_score],
            'Education': [education],  # Use original string values
            'SelfEmployed': [self_employed],  # Use original string values
            'IncomePerLoan': [income_per_loan]
        })
        
        # Try to use models directly with DataFrame
        try:
            # Use models with DataFrame input (they have built-in preprocessing)
            lr_pred = lr_model.predict(input_df)[0]
            lr_proba = lr_model.predict_proba(input_df)[0][1]
            
            dt_pred = dt_model.predict(input_df)[0]
            dt_proba = dt_model.predict_proba(input_df)[0][1]
            
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.write("Debug: Input DataFrame shape:", input_df.shape)
            st.write("Debug: Input DataFrame columns:", input_df.columns.tolist())
            st.stop()

        st.markdown("### Prediction Results")

        # Logistic Regression result
        st.subheader("Logistic Regression")
        if lr_pred == 1:
            st.success(f"🎉 Loan Approved! Probability: {lr_proba:.2%}")
        else:
            st.error(f"❌ Loan Not Approved. Probability: {lr_proba:.2%}")

        # Decision Tree result
        st.subheader("Decision Tree")
        if dt_pred == 1:
            st.success(f"🎉 Loan Approved! Probability: {dt_proba:.2%}")
        else:
            st.error(f"❌ Loan Not Approved. Probability: {dt_proba:.2%}")

        # Show input details
        with st.expander("See your input details"):
            st.write(f"**Income:** {income}")
            st.write(f"**Loan Amount:** {loan_amount}")
            st.write(f"**Credit Score:** {credit_score}")
            st.write(f"**Education:** {education}")
            st.write(f"**Self Employed:** {self_employed}")
            st.write(f"**Income per Loan:** {income_per_loan:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Debug info:", str(e))




# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
