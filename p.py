import streamlit as st
import joblib
import pandas as pd

# Load the model and scaler with caching
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    # Load the classifier and scaler
    with open('classifier.pkl', 'rb') as model_file:
        classifier = joblib.load(model_file)
    scaler = joblib.load('scaler.pkl')
    return classifier, scaler

def impute_missing_values(features):
    # Use typical values for approved applications based on your analysis
    defaults = {
        'Gender': 0,               # Default to Male (since the median is 0.0 for Gender)
        'Married': 1,              # Default to Married (since the median is 1.0 for Married)
        'ApplicantIncome': 3800,   # Using the median for ApplicantIncome for more stability
        'LoanAmount': 128,         # Using the median LoanAmount (in thousands)
        'Credit_History': 1,       # Default to clear credit history, as the median is 1.0
        # Additional columns that may be expected by the model
        'CoapplicantIncome': 0,    # Default to no coapplicant income if it wasn't available
        'Dependents': 0,           # Default to 0 dependents, if this is common for approvals
        'Education': 0,            # Assume Graduate if common in approvals (if 0 represents Graduate)
        'Loan_Amount_Term': 360,   # Typical loan term for approved loans
        'Property_Area': 1,        # Default to Semiurban if it's common in approvals
        'Self_Employed': 0         # Default to No if most approved applicants are not self-employed
    }
    # Fill missing values with defaults
    filled_features = {key: features.get(key, defaults[key]) for key in defaults}
    return pd.DataFrame([filled_features], columns=defaults.keys())

# Prediction function for single input
def prediction(classifier, scaler, **kwargs):
    # Pre-process user input and ensure correct feature order
    features = impute_missing_values(kwargs)
    
    # Reorder columns to match the scaler's expected feature order
    features = features[scaler.feature_names_in_]
    
    # Scale the features
    scaled_features = scaler.transform(features)
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

    # Income and Loan Information
    st.markdown('<p class="label">Income and Loan Information</p>', unsafe_allow_html=True)
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000,
                                help="Enter the monthly income of the applicant.")
    LoanAmount = st.slider("Loan Amount Requested (in thousands)", min_value=0, max_value=500, step=1, value=150,
                           help="Enter the total loan amount requested by the applicant.")
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"), help="Specify the applicant's credit history status.")
    Self_Employed = st.radio("Self Employed:", ("No", "Yes"), help="Specify if the applicant is self-employed.")

    # Convert inputs to match model expectations
    input_data = {
        'Gender': 0 if Gender == "Male" else 1,
        'Married': 0 if Married == "Unmarried" else 1,
        'ApplicantIncome': ApplicantIncome,
        'LoanAmount': LoanAmount,  # LoanAmount in thousands as per model
        'Credit_History': 0 if Credit_History == "Unclear Debts" else 1,
        'Self_Employed': 0 if Self_Employed == "No" else 1
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
        st.markdown('<p class="label">Application Summary</p>', unsafe_allow_html=True)
        
        # Display summary of inputs
        gender_text = "Male" if input_data['Gender'] == 0 else "Female"
        marital_status = "Unmarried" if input_data['Married'] == 0 else "Married"
        credit_text = "No Unclear Debts" if input_data['Credit_History'] == 1 else "Unclear Debts"
        self_employed_text = "Yes" if input_data['Self_Employed'] == 1 else "No"
        
        summary_text = f"""
        - **Gender**: {gender_text}
        - **Marital Status**: {marital_status}
        - **Monthly Income**: ${input_data['ApplicantIncome']}
        - **Loan Amount Requested**: ${input_data['LoanAmount']}000
        - **Credit History**: {credit_text}
        - **Self Employed**: {self_employed_text}
        """
        
        st.markdown(summary_text)

        # Explanation Section
        st.write("---")
        if result == "Approved":
            st.write("### Explanation:")
            st.write("Your application was **Approved** based on factors such as sufficient monthly income, a manageable loan amount, and a positive credit history.")
        else:
            st.write("### Explanation:")
            st.write("Your application was **Rejected**. This could be due to insufficient monthly income, a high loan amount relative to your income, or an unclear credit history.")

        # Additional warning if the loan amount is high relative to income
        if LoanAmount > 200 and ApplicantIncome < 3000:
            st.warning("⚠️ The requested loan amount is relatively high compared to your income, which might increase the risk of rejection.")

if __name__ == '__main__':
    main()
