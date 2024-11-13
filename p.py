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

# Impute missing values with predefined defaults, similar to notebook imputation logic
def impute_missing_values(features):
    # Define default values for imputation, following the notebook’s approach
    defaults = {
        'Gender': 0,  # Default to Male
        'Married': 0,  # Default to Unmarried
        'Dependents': 0,  # Default to 0 dependents
        'Education': 0,  # Default to Graduate
        'Self_Employed': 0,  # Default to No
        'ApplicantIncome': 5000,  # Default income
        'CoapplicantIncome': 0,  # Default to no coapplicant income
        'LoanAmount': 150,  # Default loan amount (median in thousands)
        'Loan_Amount_Term': 360,  # Default term in months
        'Credit_History': 1,  # Default to clear debts
        'Property_Area': 0  # Default to Urban
    }
    return [features.get(key, defaults[key]) for key in defaults]

# Prediction function for single input
def prediction(classifier, scaler, **kwargs):
    # Pre-process user input
    features = impute_missing_values(kwargs)
    scaled_features = scaler.transform([features])
    prediction = classifier.predict(scaled_features)
    return 'Approved' if prediction == 1 else 'Rejected'

# Streamlit Interface
def main():
    st.set_page_config(page_title="Loan Approval Pro", page_icon="💼", layout="centered")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .title { font-size: 2.5em; font-weight: bold; color: #2e3a45; }
        .subtitle { font-size: 1.2em; color: #6c757d; }
        .label { font-weight: bold; font-size: 1.1em; color: #333; }
        .info { color: #0066cc; font-style: italic; }
        </style>
        """, unsafe_allow_html=True)

    # Branding and Title
    st.markdown('<p class="title">💼 Loan Approval Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Trusted Solution for Financial Decision Making</p>', unsafe_allow_html=True)

    # Introductory Information Sections
    with st.expander("About This Tool"):
        st.write("Loan Approval Pro helps financial institutions make data-driven decisions on loan applications using key applicant details.")
    with st.expander("How the Prediction Works"):
        st.write("This tool uses a machine learning model trained on historical loan data, considering factors like gender, marital status, income, loan amount, and credit history.")

    # Load model and scaler
    classifier, scaler = load_model_and_scaler()

    # Input Form for single entry
    st.markdown('<p class="label">Application Details</p>', unsafe_allow_html=True)

    # Gender and Marital Status
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.radio("Select your Gender:", ("Male", "Female"), help="Select the applicant's gender.")
    with col2:
        Married = st.radio("Marital Status:", ("Unmarried", "Married"), help="Select the applicant's marital status.")

    # Dependents, Education, Self-Employed
    Dependents = st.selectbox("Number of Dependents:", ("0", "1", "2", "3+"), help="Select the number of dependents supported by the applicant.")
    Education = st.radio("Education Level:", ("Graduate", "Not Graduate"), help="Select the highest education level of the applicant.")
    Self_Employed = st.radio("Self Employed:", ("No", "Yes"), help="Specify if the applicant is self-employed.")

    # Income and Loan Information
    st.markdown('<p class="label">Income and Loan Information</p>', unsafe_allow_html=True)
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000,
                                help="Enter the monthly income of the applicant.")
    CoapplicantIncome = st.slider("Coapplicant's Monthly Income (in USD)", min_value=0, max_value=10000, step=500, value=0,
                                  help="Enter the monthly income of the coapplicant, if any.")
    LoanAmount = st.slider("Loan Amount Requested (in USD)", min_value=0, max_value=500000, step=1000, value=150000,
                           help="Enter the total loan amount requested by the applicant.")
    Loan_Amount_Term = st.slider("Loan Amount Term (in months)", min_value=12, max_value=480, step=12, value=360,
                                 help="Select the loan term in months.")

    # Credit History and Property Area
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"), help="Specify the applicant's credit history status.")
    Property_Area = st.selectbox("Property Area:", ("Urban", "Semiurban", "Rural"), help="Choose the area where the property is located.")

    # Convert inputs to match model expectations
    input_data = {
        'Gender': 0 if Gender == "Male" else 1,
        'Married': 0 if Married == "Unmarried" else 1,
        'Dependents': int(Dependents) if Dependents.isdigit() else 3,  # Convert "3+" to 3
        'Education': 0 if Education == "Graduate" else 1,
        'Self_Employed': 0 if Self_Employed == "No" else 1,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount / 1000,  # Assuming LoanAmount was scaled in thousands
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': 0 if Credit_History == "Unclear Debts" else 1,
        'Property_Area': {"Urban": 0, "Semiurban": 1, "Rural": 2}[Property_Area]
    }

    # Prediction Button
    if st.button("Predict My Loan Status", help="Click to predict the loan approval status based on the provided details"):
        result = prediction(classifier, scaler, **input_data)

        # Display approval or rejection message
        if result == "Approved":
            st.success("✅ Your loan application status: Approved")
        else:
            st.error("❌ Your loan application status: Rejected")

        # Summary Section
        st.write("---")
        st.markdown('<p class="label">Summary</p>', unsafe_allow_html=True)
        for key, value in input_data.items():
            st.write(f"**{key.replace('_', ' ').title()}**: {value}")

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
