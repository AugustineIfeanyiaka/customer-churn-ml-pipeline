
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "optimized_churn_model.joblib"
THRESHOLD = 0.30


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("📊 Customer Churn Prediction")
st.write("Predict customer churn using a trained machine learning pipeline.")
st.caption("Portfolio demonstration using the IBM Telco Customer Churn dataset.")

st.divider()
st.subheader("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )
    contract = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0, step=1.0
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0, step=1.0
    )

if st.button("Predict Customer Churn", type="primary"):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    customer = customer[model.feature_names_in_]

    probability = model.predict_proba(customer)[0, 1]
    prediction = int(probability >= THRESHOLD)

    st.divider()
    st.subheader("Prediction Results")
    st.metric("Predicted Churn Probability", f"{probability:.1%}")
    st.progress(float(probability))

    if prediction == 1:
        st.warning("High Churn Risk — Customer is predicted to churn.")
    else:
        st.success("Low Churn Risk — Customer is predicted to stay.")

    st.caption(
        f"Decision threshold: {THRESHOLD:.2f}. "
        "Results are predictions, not guarantees."
    )
