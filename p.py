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
        st.write("Loan Approval Pro helps financial institutions make data-driven decisions on loan applications using key applicant details.")
    with st.expander("How the Prediction Works"):
        st.write("This tool uses a machine learning model trained on historical loan data, considering factors like gender, marital status, income, loan amount, and credit history.")

    # Load model and scaler
    classifier, scaler = load_model_and_scaler()

    # Input Form for single entry
    st.subheader("Application Details")
    Gender = st.radio("Select your Gender:", ("Male", "Female"))
    Married = st.radio("Marital Status:", ("Unmarried", "Married"))
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000)
    LoanAmount = st.slider("Loan Amount Requested (in USD)", min_value=0, max_value=500000, step=1000, value=150000)
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"))

    # Check if loan amount is too high relative to income
    income_threshold = 5  # Threshold for affordability
    if LoanAmount > ApplicantIncome * income_threshold:
        st.warning("⚠️ The requested loan amount is high relative to the applicant's income, which may impact approval.")

    # Prediction Button
    if st.button("Predict My Loan Status"):
        result = prediction(classifier, scaler, Gender, Married, ApplicantIncome, LoanAmount, Credit_History)

        # Display approval or rejection message
        if result == "Approved":
            st.success("✅ Your loan application status: Approved")
        else:
            st.error("❌ Your loan application status: Rejected")
        
        # Summary Section
        st.write("---")
        st.subheader("Summary")
        st.write(f"**Gender**: {Gender}")
        st.write(f"**Marital Status**: {Married}")
        st.write(f"**Monthly Income**: ${ApplicantIncome}")
        st.write(f"**Loan Amount Requested**: ${LoanAmount}")
        st.write(f"**Credit History**: {Credit_History}")
        st.write(f"**Decision**: The loan application was **{result}**.")

        # Explanation of Decision
        st.subheader("Explanation of the Decision")
        if result == "Approved":
            st.write("The loan application was approved because it met the following criteria:")
            if Credit_History == 1:
                st.write("- Positive credit history.")
            if LoanAmount <= ApplicantIncome * income_threshold:
                st.write("- The loan amount is within an acceptable range relative to the income.")
            if Married == 1 or ApplicantIncome >= 3000:
                st.write("- Suitable income level and/or marital status supported approval.")
        else:
            st.write("The loan application was rejected for the following reasons:")
            if Credit_History == 0:
                st.write("- Poor credit history.")
            if LoanAmount > ApplicantIncome * income_threshold:
                st.write("- The loan amount is high relative to the income.")
            if Married == 0 and ApplicantIncome < 3000:
                st.write("- Low income level for unmarried applicants.")

    # Additional Information Section at the end
    st.write("---")
    with st.expander("Why Was My Application Rejected?"):
        st.write("Rejections may be due to insufficient income, a high loan amount relative to income, or unclear credit history. Adjusting these factors may improve approval chances.")
    with st.expander("Why Was My Application Approved?"):
        st.write("Approval can result from sufficient income, a reasonable loan amount, and a good credit history. Meeting these criteria improves approval chances.")
    with st.expander("Improving Your Approval Chances"):
        st.write("To increase approval likelihood, consider building a stronger credit history, ensuring your loan amount is reasonable relative to income, and maintaining a stable income level.")

if __name__ == '__main__':
    main()
