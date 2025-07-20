import streamlit as st
import numpy as np
import joblib

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
model = joblib.load('loan_approval_lr.pkl')
scaler = joblib.load('loan_approval_scaler.pkl')

# Input fields in columns
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Applicant Income", min_value=0, step=1, help="Monthly income in your local currency")
    credit_score = st.number_input("Credit Score", min_value=0, max_value=900, step=1, help="Credit score (0-900)")
with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0, step=1, help="Requested loan amount")
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

# Encode categorical variables
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

# Predict button
if st.button("🔮 Predict"):
    try:
        input_data = np.array([[income, loan_amount, credit_score, education_val, self_employed_val]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        st.markdown("### Prediction Result")
        if prediction == 1:
            st.success("🎉 Loan Approved! Congratulations! ✅")
            st.balloons()
            st.info("Tip: Maintain your credit score for future loans.")
        else:
            st.error("❌ Loan Not Approved. Please check your details or contact support.")
            if credit_score < 600:
                st.warning("Your credit score is below average. Improving it may help.")
            if income < 3000:
                st.warning("A higher income can increase approval chances.")
            st.markdown("**Next Steps:** [Contact Support](mailto:loan.support@yourdomain.com?subject=Loan%20Approval%20Help&body=I%20need%20assistance%20with%20my%20loan%20application.) or [Learn how to improve your credit score](https://www.investopedia.com/how-to-improve-your-credit-score-4590099)")

        st.info(f"**Approval Probability:** {proba:.2%}")

        with st.expander("See your input details"):
            st.write(f"**Income:** {income}")
            st.write(f"**Loan Amount:** {loan_amount}")
            st.write(f"**Credit Score:** {credit_score}")
            st.write(f"**Education:** {education}")
            st.write(f"**Self Employed:** {self_employed}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)