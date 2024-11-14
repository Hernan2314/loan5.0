import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the model with caching
@st.cache(allow_output_mutation=True)
def load_model():
    with open('classifier.pkl', 'rb') as model_file:
        classifier = joblib.load(model_file)
    return classifier

# Impute missing values with only the required five features
def impute_missing_values(features):
    defaults = {
        'Gender': 0,
        'Married': 1,
        'ApplicantIncome': 3800,
        'LoanAmount': 128,
        'Credit_History': 1
    }
    filled_features = {key: features.get(key, defaults[key]) for key in defaults}
    return pd.DataFrame([filled_features])

# Prediction function for single input
def prediction(classifier, **kwargs):
    features = impute_missing_values(kwargs)
    prediction = classifier.predict(features)
    return 'Approved' if prediction == 1 else 'Rejected'

# Custom plot function for comparison chart
def plot_comparison_chart(applicant_income, loan_amount):
    avg_income = 3800  # Sample average for approved applicants
    avg_loan = 128     # Sample average loan amount for approved applicants
    fig, ax = plt.subplots()
    ax.bar(['Applicant Income', 'Avg Income'], [applicant_income, avg_income], color=['#0066cc', '#cccccc'])
    ax.bar(['Loan Amount', 'Avg Loan'], [loan_amount, avg_loan], color=['#0066cc', '#cccccc'])
    st.pyplot(fig)

# Email sending function
def send_email(recipient_email, summary_text, result):
    sender_email = "your_email@example.com"  # Replace with your email
    password = "your_password"               # Replace with your email password

    subject = "Your Loan Application Status"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    body = f"""\
    Dear Applicant,

    Here is the summary of your loan application:

    {summary_text}

    Status: {result}

    Thank you for using Loan Approval Pro.
    """
    message.attach(MIMEText(body, "plain"))

    try:
        # SMTP server setup - replace with the correct server if needed
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Enable security
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        st.error(f"An error occurred while sending the email: {e}")
        return False

# Main Streamlit app function
def main():
    st.set_page_config(page_title="Loan Approval Pro", page_icon="💼", layout="centered")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .title { font-size: 2.5em; font-weight: bold; color: #2e3a45; }
        .subtitle { font-size: 1.2em; color: #6c757d; }
        .name { font-size: 1em; font-weight: bold; color: #333333; }
        .label { font-weight: bold; font-size: 1.1em; color: #333; }
        .info { color: #0066cc; font-style: italic; }
        .stButton>button { background-color: #0066cc; color: white; padding: 10px 20px; font-size: 1.2em; }
        </style>
        """, unsafe_allow_html=True)

    # Display developer name above the title
    st.markdown('<p class="name">Developed by Hernan Andres Fermin</p>', unsafe_allow_html=True)
    st.markdown('<p class="title">💼 Loan Approval Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Trusted Solution for Financial Decision Making</p>', unsafe_allow_html=True)

    # About and explanation sections
    with st.expander("About This Tool"):
        st.write("Loan Approval Pro helps financial institutions make data-driven decisions on loan applications using key applicant details.")
    with st.expander("How the Prediction Works"):
        st.write("This tool uses a machine learning model trained on historical loan data, considering factors like gender, marital status, income, loan amount, and credit history.")

    # Load model
    classifier = load_model()

    # Step 1: Applicant Information
    st.write("### Step 1 of 2: Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.radio("👤 Select your Gender:", ("Male", "Female"), help="Select the applicant's gender.")
    with col2:
        Married = st.radio("💍 Marital Status:", ("Unmarried", "Married"), help="Select the applicant's marital status.")

    # Step 2: Financial Details
    st.write("### Step 2 of 2: Financial Details")
    ApplicantIncome = st.slider("💰 Applicant's Monthly Income (in USD)", min_value=0, max_value=50000, step=500, value=3800,
                                help="Enter the monthly income of the applicant.")
    LoanAmount = st.slider("💵 Loan Amount Requested (in thousands)", min_value=0, max_value=500, step=1, value=128,
                           help="Enter the total loan amount requested by the applicant.")
    Credit_History = st.selectbox("📈 Credit History Status:", ("Unclear Debts", "No Unclear Debts"), help="Specify the applicant's credit history status.")
    Self_Employed = st.radio("💼 Self Employed:", ("No", "Yes"), help="Specify if the applicant is self-employed.")

    # Email input field
    recipient_email = st.text_input("📧 Enter your email to receive the application summary (optional)")

    # Prepare input data for prediction
    input_data = {
        'Gender': 0 if Gender == "Male" else 1,
        'Married': 0 if Married == "Unmarried" else 1,
        'ApplicantIncome': ApplicantIncome,
        'LoanAmount': LoanAmount,
        'Credit_History': 0 if Credit_History == "Unclear Debts" else 1
    }

    # Show warning if loan amount is high relative to income
    if LoanAmount > 200 and ApplicantIncome < 3000:
        st.warning("⚠️ The requested loan amount is relatively high compared to your income, which might increase the risk of rejection.")

    # Show comparison chart for Applicant vs Average Approved
    st.write("### How You Compare to Typical Approved Applicants")
    plot_comparison_chart(ApplicantIncome, LoanAmount)

    # Prediction Button
    if st.button("Predict My Loan Status"):
        result = prediction(classifier, **input_data)
        if result == "Approved":
            st.success("✅ Your loan application status: Approved")
        else:
            st.error("❌ Your loan application status: Rejected")

        # Summary Section
        st.write("---")
        st.markdown('<p class="label">Application Summary</p>', unsafe_allow_html=True)
        
        gender_text = "Male" if input_data['Gender'] == 0 else "Female"
        marital_status = "Unmarried" if input_data['Married'] == 0 else "Married"
        credit_text = "No Unclear Debts" if input_data['Credit_History'] == 1 else "Unclear Debts"
        
        summary_text = f"""
        - **Gender**: {gender_text}
        - **Marital Status**: {marital_status}
        - **Monthly Income**: ${input_data['ApplicantIncome']}
        - **Loan Amount Requested**: ${input_data['LoanAmount']}000
        - **Credit History**: {credit_text}
        """
        
        st.markdown(summary_text)

        # Send email if recipient email is provided
        if recipient_email:
            if send_email(recipient_email, summary_text, result):
                st.success(f"📧 A summary has been sent to {recipient_email}")

    # Disclaimer Section
    st.write("---")
    with st.expander("Disclaimer"):
        st.write("This loan approval tool is for informational purposes only. Final loan decisions are subject to additional factors and the discretion of the lending institution.")

if __name__ == '__main__':
    main()
