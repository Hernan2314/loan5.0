import streamlit as st
import numpy as np
import joblib

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

    # Introductory Information Sections
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

        # Detailed Explanation of Decision
        st.subheader("Explanation of the Decision")
        explanation = ""
        if result == "Approved":
            explanation += "The loan application was approved because it met the required criteria. Here are the positive factors that supported your approval:\n"
            if Credit_History == 1:
                explanation += "- **Positive credit history**: Clear credit history often leads to higher approval chances.\n"
            if LoanAmount <= ApplicantIncome * income_threshold:
                explanation += f"- **Loan amount within a reasonable range**: The requested loan amount (${LoanAmount:,}) is proportionate to your monthly income (${ApplicantIncome:,}).\n"
            if Married == 1 or ApplicantIncome >= 3000:
                explanation += "- **Sufficient income and/or marital status**: Either your income or marital status is suitable, which supports the approval.\n"
        else:
            explanation += "The loan application was rejected due to the following reasons:\n"
            if Credit_History == 0:
                explanation += "- **Poor credit history**: Unclear debts can negatively impact loan approval.\n"
            if LoanAmount > ApplicantIncome * income_threshold:
                explanation += f"- **High loan amount relative to income**: The requested loan amount (${LoanAmount:,}) is high compared to your monthly income (${ApplicantIncome:,}), which could be seen as risky.\n"
            if Married == 0 and ApplicantIncome < 3000:
                explanation += "- **Low income for unmarried applicants**: Lower income levels may reduce approval chances, particularly for unmarried individuals.\n"
        st.write(explanation)

    # Additional Information Section at the end
    st.write("---")
    with st.expander("Why Was My Application Rejected?"):
        st.write("""
            Rejections may be due to insufficient income, a high loan amount relative to income, or unclear credit history. 
            Adjusting these factors may improve the likelihood of approval.
        """)
    with st.expander("Why Was My Application Approved?"):
        st.write("""
            Approval can be attributed to factors such as sufficient income, a loan amount within a reasonable range, and a good credit history.
            Meeting these criteria improves the chances of approval.
        """)
    with st.expander("Improving Your Approval Chances"):
        st.write("""
            To increase the likelihood of loan approval, consider building a stronger credit history, ensuring your requested loan amount is reasonable relative to income, 
            and maintaining a stable income level.
        """)

if __name__ == '__main__':
    main()


