import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

@st.cache_resource
def load_model():
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(BASE_DIR, 'models', 'xgb_churn_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.title("📉 Customer Churn Prediction")
st.markdown("### Predict whether a customer will churn using Machine Learning")
st.markdown("---")

st.sidebar.header("📋 Enter Customer Details")

def user_inputs():
    gender            = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen    = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner           = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents        = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    tenure            = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service     = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines    = st.sidebar.selectbox("Multiple Lines",
                        ["No", "Yes", "No phone service"])
    internet_service  = st.sidebar.selectbox("Internet Service",
                        ["DSL", "Fiber optic", "No"])
    online_security   = st.sidebar.selectbox("Online Security",
                        ["No", "Yes", "No internet service"])
    online_backup     = st.sidebar.selectbox("Online Backup",
                        ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection",
                        ["No", "Yes", "No internet service"])
    tech_support      = st.sidebar.selectbox("Tech Support",
                        ["No", "Yes", "No internet service"])
    streaming_tv      = st.sidebar.selectbox("Streaming TV",
                        ["No", "Yes", "No internet service"])
    streaming_movies  = st.sidebar.selectbox("Streaming Movies",
                        ["No", "Yes", "No internet service"])
    contract          = st.sidebar.selectbox("Contract Type",
                        ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method    = st.sidebar.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"])
    monthly_charges   = st.sidebar.slider("Monthly Charges ($)", 0.0, 150.0, 65.0)
    total_charges     = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)

    data = {
        'gender'           : 1 if gender == "Male" else 0,
        'SeniorCitizen'    : 1 if senior_citizen == "Yes" else 0,
        'Partner'          : 1 if partner == "Yes" else 0,
        'Dependents'       : 1 if dependents == "Yes" else 0,
        'tenure'           : tenure,
        'PhoneService'     : 1 if phone_service == "Yes" else 0,
        'MultipleLines'    : 0 if multiple_lines == "No" else (
                             1 if multiple_lines == "Yes" else 2),
        'InternetService'  : 0 if internet_service == "DSL" else (
                             1 if internet_service == "Fiber optic" else 2),
        'OnlineSecurity'   : 0 if online_security == "No" else (
                             1 if online_security == "Yes" else 2),
        'OnlineBackup'     : 0 if online_backup == "No" else (
                             1 if online_backup == "Yes" else 2),
        'DeviceProtection' : 0 if device_protection == "No" else (
                             1 if device_protection == "Yes" else 2),
        'TechSupport'      : 0 if tech_support == "No" else (
                             1 if tech_support == "Yes" else 2),
        'StreamingTV'      : 0 if streaming_tv == "No" else (
                             1 if streaming_tv == "Yes" else 2),
        'StreamingMovies'  : 0 if streaming_movies == "No" else (
                             1 if streaming_movies == "Yes" else 2),
        'Contract'         : 0 if contract == "Month-to-month" else (
                             1 if contract == "One year" else 2),
        'PaperlessBilling' : 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod'    : ["Electronic check", "Mailed check",
                              "Bank transfer (automatic)",
                              "Credit card (automatic)"].index(payment_method),
        'MonthlyCharges'   : monthly_charges,
        'TotalCharges'     : total_charges
    }
    return pd.DataFrame([data])

input_df = user_inputs()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Profile")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}),
                 use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]
    churn_prob    = probability[1] * 100
    no_churn_prob = probability[0] * 100

    if prediction == 1:
        st.error("⚠️ HIGH CHURN RISK")
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
    else:
        st.success("✅ LOW CHURN RISK")
        st.metric("Retention Probability", f"{no_churn_prob:.1f}%")

    st.markdown("#### Probability Breakdown")
    st.progress(int(churn_prob))
    col_a, col_b = st.columns(2)
    col_a.metric("🔴 Will Churn", f"{churn_prob:.1f}%")
    col_b.metric("🟢 Will Stay",  f"{no_churn_prob:.1f}%")

st.markdown("---")
st.subheader("💡 Key Churn Risk Factors")

risk_factors = []
if input_df['Contract'].values[0] == 0:
    risk_factors.append("⚠️ Month-to-month contract — highest churn risk")
if input_df['tenure'].values[0] < 12:
    risk_factors.append("⚠️ Low tenure (< 12 months) — new customers churn more")
if input_df['MonthlyCharges'].values[0] > 70:
    risk_factors.append("⚠️ High monthly charges — increases churn likelihood")
if input_df['TechSupport'].values[0] == 0:
    risk_factors.append("⚠️ No tech support — customers without support churn more")
if input_df['InternetService'].values[0] == 1:
    risk_factors.append("⚠️ Fiber optic users churn more than DSL users")

if risk_factors:
    for factor in risk_factors:
        st.warning(factor)
else:
    st.success("✅ No major risk factors detected!")

st.markdown("---")
st.markdown(
    "Built by **Pravallika Chepuri** | "
    "Arizona State University | "
    "[GitHub](https://github.com/Chepuri25/customer-churn-prediction)"
)
