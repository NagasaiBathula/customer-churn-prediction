import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Churn Prediction",
    layout="centered"
)
# -----------------------------
# LOAD MODEL & COLUMNS
# -----------------------------

model = joblib.load("models/churn_model.pkl")

model_columns = joblib.load(
    "models/model_columns.pkl"
)

# -----------------------------
# APP TITLE
# -----------------------------

st.title("Customer Churn Prediction")

st.sidebar.header("About")

st.sidebar.write(
    """
    This ML app predicts customer churn
    using a Random Forest classifier.
    """
)

st.write("Enter customer details")

# -----------------------------
# USER INPUTS
# -----------------------------

tenure = st.slider(
    "Tenure",
    0,
    72,
    12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    value=500.0
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

avg_charges = (
    total_charges / (tenure + 1)
)

# -----------------------------
# CREATE EMPTY INPUT DATA
# -----------------------------

input_data = pd.DataFrame(
    columns=model_columns
)

input_data.loc[0] = 0

# -----------------------------
# FILL NUMERIC FEATURES
# -----------------------------

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges
input_data["AvgCharges"] = avg_charges

# -----------------------------
# FILL ENCODED FEATURES
# -----------------------------

# Contract
if contract == "One year":
    input_data["Contract_One year"] = 1

elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# Internet Service
if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1

elif internet_service == "No":
    input_data["InternetService_No"] = 1

# Online Security
if online_security == "Yes":
    input_data["OnlineSecurity_Yes"] = 1

# Tech Support
if tech_support == "Yes":
    input_data["TechSupport_Yes"] = 1

# Payment Method
if payment_method == "Electronic check":
    input_data[
        "PaymentMethod_Electronic check"
    ] = 1

elif payment_method == "Mailed check":
    input_data[
        "PaymentMethod_Mailed check"
    ] = 1

elif payment_method == "Credit card (automatic)":
    input_data[
        "PaymentMethod_Credit card (automatic)"
    ] = 1

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Churn"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    st.progress(float(probability))

    st.write(
        f"Churn Probability: "
        f"{probability:.2%}"
    )

    if prediction[0] == 1:

        st.error(
            "Customer is likely to churn"
        )

    else:

        st.success(
            "Customer is likely to stay"
        )