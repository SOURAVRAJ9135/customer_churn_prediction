# gender female 1, male 0
# churn # 1 yes, 0 no
# scaler is 
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load your trained model
model = joblib.load("xgb_churn_model.pkl")

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details to predict churn risk:")

# ------------------ User Inputs ------------------
age = st.number_input("Age", min_value=18, max_value=65, value=18)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=1, max_value=60, value=1)
total_spend = st.number_input("Total Spend", min_value=100.0, max_value=1000.0, value = 100.0)
support_calls = st.number_input("Support Calls", min_value=0, max_value=10, value=0)
last_interaction = st.number_input("Days Since Last Interaction", min_value=1, max_value=30, value=1)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=30, value=0)
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
usage_frequency = st.number_input("Usage Frequency", min_value=1, max_value=30, value=10)

# ------------------ Derived Features ------------------
avg_monthly_spend = total_spend / max(tenure, 1)  # avoid division by zero
pay_delay_resolved = 0 if payment_delay >=20 else 1
usage_to_spend_ratio = usage_frequency / max(total_spend, 1)  # avoid division by zero

# ------------------ Encoding ------------------
gender_enc = 1 if gender == "Male" else 0
subscription_enc = {"Basic": 0, "Standard": 2, "Premium": 1}[subscription_type]
contract_enc = {"Monthly": 1, "Quarterly": 2, "Yearly": 0}[contract_length]

# ------------------ Prepare Input ------------------
input_data = np.array([[age, gender_enc, tenure, total_spend, support_calls,
                        last_interaction, subscription_enc, payment_delay,
                        contract_enc, avg_monthly_spend, pay_delay_resolved,
                        usage_frequency, usage_to_spend_ratio]])

# ------------------ Prediction ------------------
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of churn

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer Likely to Stay (Probability of Churn: {probability:.2f})")

