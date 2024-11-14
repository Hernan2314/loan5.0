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
    
    # Diagnostic print to verify expected features
    st.write("Scaler expected features:", scaler.feature_names_in_)
    st.write("Model expected features:", classifier.feature_names_in_)
    
    return classifier, scaler

# Impute missing values and create DataFrame with correct feature names
def impute_missing_values(features):
    # Define default values for imputation
    defaults = {
        'Gender': 0,  # Default to Male
        'Married': 0,  # Default to Unmarried
        'ApplicantIncome': 5000,  # Default income
        'LoanAmount': 150,  # Default loan amount (in thousands)
        'Credit_History': 1  # Default to clear debts
    }
    filled_features = {key: features.get(key, defaults[key]) for key in defaults}
    return pd.DataFrame([filled_features], columns=defaults.keys())  # Ensure correct column order

# Prediction function for single input
def prediction(classifier, scaler, **kwargs):
    # Pre-process user input and create DataFrame
    features = impute_missing_values(kwargs)
    
    # Reorder columns to match scaler's expected columns
    expected_columns = scaler.feature_names_in_
    try:
        features = features[expected_columns]
    except KeyError as e:
        st.error(f"Column mismatch detected: {e}")
        st.stop()  # Stop app execution if columns are mismatched

    # Scale and predict
    scaled_features = scaler.transform(features)
    prediction = classifier.predict(scaled_features)
    return 'Approved' if prediction == 1 else 'Rejected'

# Streamlit Interface
def main():
    st.set_page_config(page_title="Loan Approval Pro", page_icon="💼", layout="centered")

    # Title and description
    st.title("💼 Loan Approval Pro")
    st.subheader("Predict your loan approval status with key applicant details.")

    # Load model and scaler
    classifier, scaler = load_model_and_scaler()

    # Input Form for single entry
    Gender = st.radio("Gender:", ("Male", "Female"))
    Married = st.radio("Marital Status:", ("Unmarried", "Married"))
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", 0, 20000, 5000)
    LoanAmount = st.slider("Loan Amount Requested (in thousands)", 0, 500, 150)
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"))

    # Prepare input data
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

        # Display result and summary
        st.write("Result:", result)
        st.write("Summary:")
        st.write(f"Gender: {'Male' if input_data['Gender'] == 0 else 'Female'}")
        st.write(f"Marital Status: {'Unmarried' if input_data['Married'] == 0 else 'Married'}")
        st.write(f"Monthly Income: ${input_data['ApplicantIncome']}")
        st.write(f"Loan Amount: ${input_data['LoanAmount']}000")
        st.write(f"Credit History: {'No Unclear Debts' if input_data['Credit_History'] == 1 else 'Unclear Debts'}")

if __name__ == '__main__':
    main()

