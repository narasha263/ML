import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('model_predictions.pkl')

# Title for the web app
st.title("Business Analytics Predictive Model")

# Sidebar for user inputs
st.sidebar.header("Enter Data for Prediction")

# Collecting user input for all features
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1, value=30)
balance = st.sidebar.number_input("Balance", step=1, value=1000)
day = st.sidebar.number_input("Day of the Month", min_value=1, max_value=31, step=1)
duration = st.sidebar.number_input("Duration of Contact (seconds)", step=1, value=100)
campaign = st.sidebar.number_input("Number of Contacts", step=1, value=1)
pdays = st.sidebar.number_input("Days Passed Since Last Campaign (-1 if not applicable)", step=1, value=-1)
previous = st.sidebar.number_input("Number of Previous Contacts", step=1, value=0)

# Job types
job_blue_collar = st.sidebar.checkbox("Blue-collar Job", False)
job_entrepreneur = st.sidebar.checkbox("Entrepreneur", False)
job_housemaid = st.sidebar.checkbox("Housemaid", False)
job_management = st.sidebar.checkbox("Management", False)
# ... (Add other job types as checkboxes)

# Marital Status
marital_married = st.sidebar.checkbox("Married", False)
marital_single = st.sidebar.checkbox("Single", False)

# Education level
education_secondary = st.sidebar.checkbox("Secondary Education", False)
education_tertiary = st.sidebar.checkbox("Tertiary Education", False)

# Default, Housing, and Loan
default_yes = st.sidebar.checkbox("Default", False)
housing_yes = st.sidebar.checkbox("Has Housing Loan", False)
loan_yes = st.sidebar.checkbox("Has Personal Loan", False)

# Contact method
contact_telephone = st.sidebar.checkbox("Contact via Telephone", False)
contact_unknown = st.sidebar.checkbox("Contact Unknown", False)

# Month of Contact (use dropdowns for month selection)
month = st.sidebar.selectbox("Month of Contact",
                             ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"])

# Outcome
poutcome_success = st.sidebar.checkbox("Previous Outcome Success", False)

# Define the feature vector for prediction
input_data = np.array([[age, balance, day, duration, campaign, pdays, previous,
                        job_blue_collar, job_entrepreneur, job_housemaid, job_management,
                        marital_married, marital_single, education_secondary, education_tertiary,
                        default_yes, housing_yes, loan_yes, contact_telephone, contact_unknown,
                        month == "Aug", month == "Dec", month == "Feb", month == "Jan", month == "Jul",
                        poutcome_success]])

# Button to make predictions
if st.sidebar.button("Predict"):
    # Perform prediction using the loaded model
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == "yes":
        st.success("The model predicts a positive outcome!")
    else:
        st.warning("The model predicts a negative outcome.")

# Optional: Visualizations and dashboards (use Streamlit plotting functions)
