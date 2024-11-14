import streamlit as st
import joblib

# Load the model and scaler with caching
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    # Load the classifier and scaler
    with open('classifier.pkl', 'rb') as model_file:
        classifier = joblib.load(model_file)
    scaler = joblib.load('scaler.pkl')
    return classifier, scaler

# Impute missing values with predefined defaults
def impute_missing_values(features):
    # Define default values for imputation, following the notebook’s approach
    defaults = {
        'Gender': 0,  # Default to Male
        'Married': 0,  # Default to Unmarried
        'ApplicantIncome': 5000,  # Default income
        'LoanAmount': 150,  # Default loan amount (median in thousands)
        'Credit_History': 1  # Default to clear debts
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

    # Branding and Title
    st.markdown('<h1>💼 Loan Approval Pro</h1>', unsafe_allow_html=True)
    st.write("Predict your loan approval status with key applicant details.")

    # Load model and scaler
    classifier, scaler = load_model_and_scaler()

    # Input Form for single entry
    st.write("### Application Details")

    # Gender and Marital Status
    Gender = st.radio("Select your Gender:", ("Male", "Female"))
    Married = st.radio("Marital Status:", ("Unmarried", "Married"))

    # Income and Loan Information
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000)
    LoanAmount = st.slider("Loan Amount Requested (in thousands)", min_value=0, max_value=500, step=1, value=150)
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"))

    # Convert inputs to match model expectations
    input_data = {
        'Gender': 0 if Gender == "Male" else 1,
        'Married': 0 if Married == "Unmarried" else 1,
        'ApplicantIncome': ApplicantIncome,
        'LoanAmount': LoanAmount,  # LoanAmount in thousands as per model
        'Credit_History': 0 if Credit_History == "Unclear Debts" else 1
    }

    # Prediction Button
    if st.button("Predict My Loan Status"):
        result = prediction(classifier, scaler, **input_data)

        # Display approval or rejection message
        if result == "Approved":
            st.success("✅ Your loan application status: Approved")
        else:
            st.error("❌ Your loan application status: Rejected")

        # Summary Section
        st.write("---")
        st.write("### Summary")
        for key, value in input_data.items():
            st.write(f"**{key.replace('_', ' ').title()}**: {value}")

if __name__ == '__main__':
    main()

