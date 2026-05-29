import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background: warm brown gradient light to dark ── */
    .stApp {
        background: linear-gradient(160deg,
            #f5e6d3 0%,
            #e8c9a0 25%,
            #c4956a 55%,
            #8b5e3c 80%,
            #4a2c0a 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,
            #3d1f00 0%,
            #5c3317 50%,
            #7a4a2a 100%);
        border-right: 2px solid rgba(255,200,130,0.3);
    }
    section[data-testid="stSidebar"] * {
        color: #f5e6d3 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #ffd9a0 !important;
        font-weight: 500 !important;
    }

    /* ── BIG Title ── */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem !important;
        font-weight: 700;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        line-height: 1.1;
        background: linear-gradient(90deg,
            #4a2c0a, #8b5e3c, #c4956a, #8b5e3c, #4a2c0a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
        letter-spacing: -1px;
    }

    .sub-title {
        font-size: 1.15rem;
        color: #5c3317;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: rgba(255, 240, 215, 0.6);
        border: 1.5px solid rgba(139, 94, 60, 0.4);
        border-radius: 16px;
        padding: 1rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 15px rgba(74, 44, 10, 0.15);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(74, 44, 10, 0.25);
        border-color: rgba(139, 94, 60, 0.7);
    }
    [data-testid="metric-container"] label {
        color: #7a4a2a !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #4a2c0a !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 240, 215, 0.5);
        border-radius: 14px;
        padding: 5px;
        gap: 5px;
        border: 1px solid rgba(139, 94, 60, 0.3);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #7a4a2a !important;
        font-weight: 500;
        padding: 8px 22px;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,
            #8b5e3c, #4a2c0a) !important;
        color: #f5e6d3 !important;
        font-weight: 600 !important;
        box-shadow: 0 3px 10px rgba(74, 44, 10, 0.3);
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg,
            #c4956a, #8b5e3c, #4a2c0a) !important;
        border-radius: 10px !important;
    }
    .stProgress > div {
        background: rgba(139, 94, 60, 0.2) !important;
        border-radius: 10px !important;
    }

    /* ── Alert Boxes ── */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        backdrop-filter: blur(8px);
    }
    .stSuccess {
        background: rgba(180, 230, 180, 0.4) !important;
        border: 1.5px solid rgba(60, 150, 60, 0.5) !important;
        border-radius: 12px !important;
        color: #1a5c1a !important;
    }
    .stError {
        background: rgba(230, 150, 130, 0.4) !important;
        border: 1.5px solid rgba(180, 60, 40, 0.5) !important;
        border-radius: 12px !important;
        color: #7a1a0a !important;
    }
    .stWarning {
        background: rgba(240, 200, 130, 0.5) !important;
        border: 1.5px solid rgba(180, 120, 30, 0.5) !important;
        border-radius: 12px !important;
        color: #6b4400 !important;
    }
    .stInfo {
        background: rgba(200, 220, 255, 0.4) !important;
        border: 1.5px solid rgba(80, 120, 200, 0.5) !important;
        border-radius: 12px !important;
        color: #1a3a7a !important;
    }

    /* ── Prediction Cards ── */
    .card-high {
        background: linear-gradient(135deg,
            rgba(200, 80, 50, 0.2), rgba(200, 80, 50, 0.08));
        border: 2px solid rgba(200, 80, 50, 0.5);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(200, 80, 50, 0.15);
    }
    .card-low {
        background: linear-gradient(135deg,
            rgba(60, 160, 80, 0.2), rgba(60, 160, 80, 0.08));
        border: 2px solid rgba(60, 160, 80, 0.5);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(60, 160, 80, 0.15);
    }
    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #4a2c0a;
        margin-bottom: 0.5rem;
    }
    .card-subtitle {
        font-size: 1.1rem;
        color: #7a4a2a;
        font-weight: 400;
    }

    /* ── Dataframes ── */
    .stDataFrame {
        background: rgba(255, 240, 215, 0.5) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(139, 94, 60, 0.3) !important;
    }

    /* ── Section headers ── */
    h2, h3, h4 {
        color: #4a2c0a !important;
        font-weight: 700 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(139, 94, 60, 0.3) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Selectbox & Slider ── */
    .stSelectbox [data-baseweb="select"] > div {
        background: rgba(255, 240, 215, 0.7) !important;
        border-color: rgba(139, 94, 60, 0.4) !important;
        border-radius: 10px !important;
        color: #4a2c0a !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #7a4a2a;
        font-size: 0.9rem;
    }
    .footer a {
        color: #4a2c0a !important;
        font-weight: 600;
        text-decoration: none;
        border-bottom: 1px solid rgba(74, 44, 10, 0.3);
    }
    .footer a:hover {
        border-bottom-color: #4a2c0a;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────
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

# ── Header ───────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">📉 Customer Churn Predictor</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered churn prediction using XGBoost · '
    'Built by Pravallika Chepuri · Arizona State University</div>',
    unsafe_allow_html=True)

# ── Stats Banner ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Accuracy",      "80%+")
c2.metric("📊 ROC-AUC",       "0.85+")
c3.metric("👥 Training Data", "7,043")
c4.metric("⚙️ Algorithm",     "XGBoost")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Customer Details")
st.sidebar.markdown("---")

def user_inputs():
    st.sidebar.markdown("### 👤 Demographics")
    gender           = st.sidebar.selectbox("Gender",
                       ["Male", "Female"])
    senior_citizen   = st.sidebar.selectbox("Senior Citizen",
                       ["No", "Yes"])
    partner          = st.sidebar.selectbox("Has Partner",
                       ["No", "Yes"])
    dependents       = st.sidebar.selectbox("Has Dependents",
                       ["No", "Yes"])

    st.sidebar.markdown("### 📱 Services")
    tenure           = st.sidebar.slider("Tenure (months)",
                       0, 72, 12)
    phone_service    = st.sidebar.selectbox("Phone Service",
                       ["No", "Yes"])
    multiple_lines   = st.sidebar.selectbox("Multiple Lines",
                       ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service",
                       ["DSL", "Fiber optic", "No"])
    online_security  = st.sidebar.selectbox("Online Security",
                       ["No", "Yes", "No internet service"])
    online_backup    = st.sidebar.selectbox("Online Backup",
                       ["No", "Yes", "No internet service"])
    device_protection= st.sidebar.selectbox("Device Protection",
                       ["No", "Yes", "No internet service"])
    tech_support     = st.sidebar.selectbox("Tech Support",
                       ["No", "Yes", "No internet service"])
    streaming_tv     = st.sidebar.selectbox("Streaming TV",
                       ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies",
                       ["No", "Yes", "No internet service"])

    st.sidebar.markdown("### 💳 Billing")
    contract         = st.sidebar.selectbox("Contract Type",
                       ["Month-to-month","One year","Two year"])
    paperless_billing= st.sidebar.selectbox("Paperless Billing",
                       ["No","Yes"])
    payment_method   = st.sidebar.selectbox("Payment Method", [
                       "Electronic check","Mailed check",
                       "Bank transfer (automatic)",
                       "Credit card (automatic)"])
    monthly_charges  = st.sidebar.slider("Monthly Charges ($)",
                       0.0, 150.0, 65.0)
    total_charges    = st.sidebar.slider("Total Charges ($)",
                       0.0, 9000.0, 1000.0)

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
                             1 if internet_service == "Fiber optic"
                             else 2),
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
        'PaymentMethod'    : ["Electronic check","Mailed check",
                              "Bank transfer (automatic)",
                              "Credit card (automatic)"].index(
                              payment_method),
        'MonthlyCharges'   : monthly_charges,
        'TotalCharges'     : total_charges
    }
    return pd.DataFrame([data]), {
        'tenure'          : tenure,
        'contract'        : contract,
        'monthly_charges' : monthly_charges,
        'tech_support'    : tech_support,
        'internet_service': internet_service
    }

input_df, raw = user_inputs()

# ── Predict ──────────────────────────────────────────────────
input_scaled  = scaler.transform(input_df)
prediction    = model.predict(input_scaled)[0]
probability   = model.predict_proba(input_scaled)[0]
churn_prob    = probability[1] * 100
no_churn_prob = probability[0] * 100

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯  Prediction Result",
    "📊  Customer Profile",
    "💡  Insights & Tips"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="card-high">
                <div class="card-title">⚠️ HIGH CHURN RISK</div>
                <div class="card-subtitle">
                    {churn_prob:.1f}% probability of churning
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-low">
                <div class="card-title">✅ LOW CHURN RISK</div>
                <div class="card-subtitle">
                    {no_churn_prob:.1f}% probability of staying
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Probability Breakdown")
        m1, m2 = st.columns(2)
        m1.metric("🔴 Churn Risk",  f"{churn_prob:.1f}%")
        m2.metric("🟢 Retention",   f"{no_churn_prob:.1f}%")

        st.markdown("**Risk Level:**")
        if churn_prob < 30:
            st.success(f"🟢 LOW RISK — {churn_prob:.1f}%")
        elif churn_prob < 60:
            st.warning(f"🟡 MEDIUM RISK — {churn_prob:.1f}%")
        else:
            st.error(f"🔴 HIGH RISK — {churn_prob:.1f}%")
        st.progress(int(churn_prob))

    with col2:
        st.markdown("#### 💡 Risk Analysis")
        risk_score = 0
        risk_factors = []
        good_factors = []

        if raw['contract'] == "Month-to-month":
            risk_factors.append("⚠️ **Month-to-month** contract")
            risk_score += 3
        else:
            good_factors.append("✅ Long-term contract")

        if raw['tenure'] < 12:
            risk_factors.append(
                f"⚠️ **New customer** ({raw['tenure']} months)")
            risk_score += 2
        else:
            good_factors.append(
                f"✅ Loyal customer ({raw['tenure']} months)")

        if raw['monthly_charges'] > 70:
            risk_factors.append(
                f"⚠️ **High charges** "
                f"(${raw['monthly_charges']:.0f}/mo)")
            risk_score += 2
        else:
            good_factors.append(
                f"✅ Fair pricing "
                f"(${raw['monthly_charges']:.0f}/mo)")

        if raw['tech_support'] == "No":
            risk_factors.append("⚠️ **No tech support**")
            risk_score += 1
        else:
            good_factors.append("✅ Has tech support")

        if raw['internet_service'] == "Fiber optic":
            risk_factors.append(
                "⚠️ **Fiber optic** — higher expectations")
            risk_score += 1

        if risk_factors:
            st.markdown("**🚨 Risk Factors:**")
            for f in risk_factors:
                st.warning(f)
        if good_factors:
            st.markdown("**💪 Strengths:**")
            for f in good_factors:
                st.success(f)

        st.markdown(f"**Overall Risk Score: {risk_score}/9**")
        st.progress(int((risk_score / 9) * 100))

with tab2:
    st.markdown("#### 👤 Demographics")
    d1, d2 = st.columns(2)
    with d1:
        st.dataframe(
            input_df[['gender','SeniorCitizen',
                      'Partner','Dependents']].T
            .rename(columns={0:'Value'}),
            use_container_width=True)
    with d2:
        st.dataframe(
            input_df[['tenure','MonthlyCharges',
                      'TotalCharges']].T
            .rename(columns={0:'Value'}),
            use_container_width=True)

    st.markdown("#### 📱 Services")
    st.dataframe(
        input_df[['PhoneService','MultipleLines',
                  'InternetService','OnlineSecurity',
                  'OnlineBackup','TechSupport',
                  'StreamingTV','StreamingMovies']].T
        .rename(columns={0:'Value'}),
        use_container_width=True)

    st.markdown("#### 💳 Billing")
    st.dataframe(
        input_df[['Contract','PaperlessBilling',
                  'PaymentMethod']].T
        .rename(columns={0:'Value'}),
        use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Top Churn Drivers")
        drivers = {
            "Contract Type"    : 85,
            "Tenure"           : 78,
            "Monthly Charges"  : 72,
            "Internet Service" : 65,
            "Tech Support"     : 58,
            "Payment Method"   : 45,
        }
        for driver, val in drivers.items():
            st.markdown(f"**{driver}** — {val}%")
            st.progress(val)

    with col2:
        st.markdown("#### 📊 Dataset Stats")
        stats = {
            "Total Customers"  : "7,043",
            "Churn Rate"       : "26.5%",
            "Avg Tenure"       : "32 months",
            "Avg Monthly Bill" : "$64.76",
            "Model Accuracy"   : "80%+",
            "ROC-AUC Score"    : "0.85+",
        }
        for k, v in stats.items():
            a, b = st.columns(2)
            a.markdown(f"**{k}**")
            b.markdown(f"`{v}`")

    st.markdown("---")
    st.markdown("#### 💼 Business Recommendations")
    st.info("📌 Offer **annual contracts** to month-to-month customers with >60% churn risk")
    st.info("📌 Provide **loyalty discounts** for customers with tenure < 12 months")
    st.info("📌 Offer **free tech support trial** for high-charge customers")
    st.info("📌 **Proactive outreach** when churn probability exceeds 50%")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    Built with ❤️ by <strong>Pravallika Chepuri</strong> &nbsp;|&nbsp;
    Arizona State University &nbsp;|&nbsp;
    <a href="https://github.com/Chepuri25/customer-churn-prediction"
       target="_blank">🐙 GitHub</a> &nbsp;|&nbsp;
    <a href="https://www.linkedin.com/in/pravallika-chepuri"
       target="_blank">💼 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
