import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the model and scaler with caching
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    # Load the classifier and scaler
    with open('classifier.pkl', 'rb') as model_file:
        classifier = joblib.load(model_file)
    scaler = joblib.load('scaler.pkl')
    return classifier, scaler

# Prediction function for single input
def prediction(classifier, scaler, Gender, Married, ApplicantIncome, LoanAmount, Credit_History):
    # Pre-process user input
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1
    LoanAmount = LoanAmount / 1000  # Scale loan amount if required

    # Fill missing features with placeholder values (e.g., 0)
    features = np.array([Gender, Married, ApplicantIncome, LoanAmount, Credit_History, 0, 0, 0, 0, 0, 0])
    scaled_features = scaler.transform([features])  # Wrap with an extra [] for 2D input

    # Predict with scaled features
    prediction = classifier.predict(scaled_features)
    return 'Approved' if prediction == 1 else 'Rejected'

# Streamlit Interface
def main():
    st.set_page_config(page_title="Loan Approval Pro", page_icon="💼", layout="centered")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .title { font-size: 2.4em; font-weight: bold; color: #2e3a45; }
        .subtitle { font-size: 1.2em; color: #6c757d; }
        .warning-text { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; font-size: 0.9em; }
        .success-text { font-size: 1.2em; color: #155724; }
        </style>
        """, unsafe_allow_html=True)

    # Branding and Title
    st.markdown('<p class="title">💼 Loan Approval Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Trusted Solution for Financial Decision Making</p>', unsafe_allow_html=True)

    # Additional Information Section with expandable explanations
    with st.expander("About This Tool"):
        st.write("""
            Loan Approval Pro is designed to help financial institutions make data-driven decisions on loan applications. By leveraging historical data, 
            this tool evaluates key factors and provides an approval prediction based on the applicant's profile.
        """)
    with st.expander("How the Prediction Works"):
        st.write("""
            This tool uses a machine learning model trained on historical loan data. It considers various factors, including gender, marital status, income,
            loan amount, and credit history, to provide a loan approval prediction.
        """)
    with st.expander("Why Was My Application Rejected?"):
        st.write("""
            Rejections may be due to insufficient income, a high loan amount relative to income, or unclear credit history. 
            Adjusting these factors may improve the likelihood of approval.
        """)

    # Load model and scaler
    classifier, scaler = load_model_and_scaler()

    # Input Form for single entry
    st.markdown("### Application Details")
    Gender = st.radio("Select your Gender:", ("Male", "Female"), help="Choose the gender of the applicant.")
    Married = st.radio("Marital Status:", ("Unmarried", "Married"), help="Choose the marital status of the applicant.")
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000, help="Enter the applicant's monthly income.")
    LoanAmount = st.slider("Loan Amount Requested (in USD)", min_value=0, max_value=500000, step=1000, value=150000, help="Enter the loan amount the applicant is requesting.")
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"), help="Select the applicant's credit history status.")

    # Validate if LoanAmount > ApplicantIncome * threshold
    income_threshold = 5  # Example threshold for affordability
    if LoanAmount > ApplicantIncome * income_threshold:
        st.markdown("<div class='warning-text'>⚠️ The requested loan amount seems high relative to the applicant's income. This could impact approval.</div>", unsafe_allow_html=True)

    # Prediction Button
    if st.button("Predict My Loan Status"):
        result = prediction(classifier, scaler, Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        
        # Summary Section
        st.write("---")
        st.subheader("Summary")
        st.write(f"""
            **Applicant Details**
            - **Gender**: {Gender}
            - **Marital Status**: {Married}
            - **Monthly Income**: ${ApplicantIncome}
            - **Loan Amount Requested**: ${LoanAmount}
            - **Credit History**: {Credit_History}

            **Decision**: The loan application was **{result}** based on the applicant's profile and historical approval criteria.
        """)

        # Explanation for approval or rejection based on inputs
        st.subheader("Explanation of the Decision")
        if result == "Approved":
            explanation = "The loan application was approved because it met the required criteria. Factors supporting approval include:\n"
            if Credit_History == 1:
                explanation += "- Positive credit history.\n"
            if LoanAmount <= ApplicantIncome * income_threshold:
                explanation += "- Loan amount is within an acceptable range based on the applicant's income.\n"
            if Married == 1 or ApplicantIncome >= 3000:
                explanation += "- Sufficient income level for marital status.\n"
        else:
            explanation = "The loan application was rejected due to the following reasons:\n"
            if Credit_History == 0:
                explanation += "- Poor credit history.\n"
            if LoanAmount > ApplicantIncome * income_threshold:
                explanation += "- The requested loan amount is high relative to income.\n"
            if Married == 0 and ApplicantIncome < 3000:
                explanation += "- Insufficient income for an unmarried applicant.\n"
        st.write(explanation)

        # Plot Graph 1: Income and Loan Requested Comparison
        st.write("#### Income vs. Loan Amount Requested")
        fig1, ax1 = plt.subplots()
        ax1.bar(["Applicant Income", "Loan Requested"], [ApplicantIncome, LoanAmount])
        ax1.set_ylabel("Amount (USD)")
        ax1.set_title("Comparison of Applicant's Monthly Income and Loan Requested Amount")
        st.pyplot(fig1)

        # Plot Graph 2: Approval Factors Impact
        st.write("#### Key Approval Factors")
        fig2, ax2 = plt.subplots()
        factors = ["Credit History", "Income vs Loan Threshold", "Marital & Income Status"]
        scores = [
            1 if Credit_History == 1 else 0,
            1 if LoanAmount <= ApplicantIncome * income_threshold else 0,
            1 if Married == 1 or ApplicantIncome >= 3000 else 0
        ]
        colors = ["green" if score == 1 else "red" for score in scores]
        ax2.barh(factors, scores, color=colors)
        ax2.set_xlim(0, 1.5)
        ax2.set_xlabel("Approval Indicator (1 = Meets Criteria, 0 = Does Not)")
        st.pyplot(fig2)

if __name__ == '__main__':
    main()

  
