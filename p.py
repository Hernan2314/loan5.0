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

    # Branding and Title
    st.title("💼 Loan Approval Pro")
    st.subheader("A Trusted Solution for Financial Decision Making")

    # Introductory Information Sections
    with st.expander("About This Tool"):
        st.write("Loan Approval Pro is designed to help financial institutions make data-driven decisions on loan applications. By leveraging historical data, this tool evaluates key factors and provides an approval prediction based on the applicant's profile.")
    with st.expander("How the Prediction Works"):
        st.write("This tool uses a machine learning model trained on historical loan data. It considers various factors, including gender, marital status, income, loan amount, and credit history, to provide a loan approval prediction.")

    # Load model and scaler
    classifier, scaler = load_model_and_scaler()

    # Input Form for single entry
    st.subheader("Application Details")
    Gender = st.radio("Select your Gender:", ("Male", "Female"))
    Married = st.radio("Marital Status:", ("Unmarried", "Married"))
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000)
    LoanAmount = st.slider("Loan Amount Requested (in USD)", min_value=0, max_value=500000, step=1000, value=150000)
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"))

    # Prediction Button
    if st.button("Predict My Loan Status"):
        result = prediction(classifier, scaler, Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        
        # Summary Section
        st.write("---")
        st.subheader("Summary")
        st.write(f"**Gender**: {Gender}")
        st.write(f"**Marital Status**: {Married}")
        st.write(f"**Monthly Income**: ${ApplicantIncome}")
        st.write(f"**Loan Amount Requested**: ${LoanAmount}")
        st.write(f"**Credit History**: {Credit_History}")
        st.write(f"**Decision**: The loan application was **{result}** based on the applicant's profile and historical approval criteria.")

        # Explanation of Decision
        st.subheader("Explanation of the Decision")
        if result == "Approved":
            st.write("The loan application was approved because it met the required criteria.")
            if Credit_History == 1:
                st.write("- Positive credit history.")
            if LoanAmount <= ApplicantIncome * 5:
                st.write(f"- The loan amount (${LoanAmount}) is within an acceptable range compared to your income (${ApplicantIncome}).")
            if Married == 1 or ApplicantIncome >= 3000:
                st.write("- Your income and/or marital status supported the approval.")
        else:
            st.write("The loan application was rejected due to the following reasons:")
            if Credit_History == 0:
                st.write("- Poor credit history.")
            if LoanAmount > ApplicantIncome * 5:
                st.write(f"- The loan amount (${LoanAmount}) is high compared to your income (${ApplicantIncome}).")
            if Married == 0 and ApplicantIncome < 3000:
                st.write("- Low income level for unmarried applicants.")

    # Additional Information Section at the end
    st.write("---")
    with st.expander("Why Was My Application Rejected?"):
        st.write("Rejections may be due to insufficient income, a high loan amount relative to income, or unclear credit history. Adjusting these factors may improve the likelihood of approval.")
    with st.expander("Why Was My Application Approved?"):
        st.write("Approval can be attributed to factors such as sufficient income, a loan amount within a reasonable range, and a good credit history. Meeting these criteria improves the chances of approval.")
    with st.expander("Improving Your Approval Chances"):
        st.write("To increase the likelihood of loan approval, consider building a stronger credit history, ensuring your requested loan amount is reasonable relative to income, and maintaining a stable income level.")

if __name__ == '__main__':
    main()
