import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the model and scaler with caching
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    # Load the classifier and scaler
    with open('classifier.pkl', 'rb') as model_file:
        classifier = joblib.load(model_file)
    scaler = joblib.load('scaler.pkl')
    return classifier, scaler

# Load a sample dataset for visualization (replace with actual data)
@st.cache(allow_output_mutation=True)
def load_data():
    # Placeholder data; replace with actual dataset for better context in visualizations
    data = pd.DataFrame({
        'Income': np.random.randint(1000, 20000, 100),
        'LoanAmount': np.random.randint(5000, 500000, 100) / 1000,
        'Credit_History': np.random.choice([0, 1], 100)
    })
    return data

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

    # Load model, scaler, and data for visualizations
    classifier, scaler = load_model_and_scaler()
    data = load_data()

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

        # Plot 1: Scatter plot of Income vs Loan Amount
        st.subheader("Income vs. Loan Amount Requested")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Income", y="LoanAmount", hue="Credit_History", data=data, ax=ax)
        ax.axvline(ApplicantIncome, color="red", linestyle="--", label="Your Income")
        ax.axhline(LoanAmount, color="blue", linestyle="--", label="Requested Loan Amount")
        plt.legend()
        st.pyplot(fig)

        # Explanation for scatter plot based on result
        if result == "Rejected":
            st.write("🔍 **Explanation**: The income vs. loan amount plot shows your position relative to other applicants. "
                     "If your requested loan amount is significantly higher than the average for your income level, this could contribute to a rejection.")
        else:
            st.write("🔍 **Explanation**: Your loan request amount aligns with the range of other applicants with similar income, which likely supported the approval decision.")

        # Plot 2: Histogram of Loan Amounts
        st.subheader("Distribution of Loan Amounts")
        fig, ax = plt.subplots()
        sns.histplot(data['LoanAmount'], bins=20, color="skyblue", ax=ax)
        ax.axvline(LoanAmount, color="blue", linestyle="--", label="Requested Loan Amount")
        plt.legend()
        st.pyplot(fig)

        # Explanation for histogram based on result
        if result == "Rejected":
            st.write("🔍 **Explanation**: The requested loan amount is shown on the histogram. If your amount is much higher than the average loan requests, "
                     "this could indicate a higher risk, leading to a rejection.")
        else:
            st.write("🔍 **Explanation**: Your requested loan amount is within the typical range, which may have positively impacted the approval decision.")

        # Plot 3: Approval Rate by Credit History
        st.subheader("Approval Rate by Credit History")
        approval_rate = data['Credit_History'].value_counts()
        fig, ax = plt.subplots()
        approval_rate.plot(kind="bar", color=["salmon", "lightgreen"], ax=ax)
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)

        # Explanation for approval rate by credit history based on result
        if result == "Rejected":
            st.write("🔍 **Explanation**: Approval rates by credit history indicate that applicants with unclear debts are less likely to be approved. "
                     "Improving your credit history could enhance approval chances.")
        else:
            st.write("🔍 **Explanation**: Having a clear credit history is often associated with higher approval rates, which may have contributed to your approval.")

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
            To increase the likelihood of loan approval, consider building a stronger credit

