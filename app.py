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
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ── Main Header ── */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        letter-spacing: -1px;
    }

    .sub-header {
        font-size: 1rem;
        color: rgba(255,255,255,0.6);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: rgba(0,210,255,0.4);
    }
    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.85rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d2ff !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: rgba(255,255,255,0.6) !important;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d2ff, #7b2ff7) !important;
        color: white !important;
    }

    /* ── Buttons ── */
    .stButton>button {
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: opacity 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.85;
    }

    /* ── Success / Error / Warning Boxes ── */
    .stSuccess, div[data-baseweb="notification"][kind="positive"] {
        background: rgba(0, 255, 136, 0.1) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
        border-radius: 12px !important;
        color: #00ff88 !important;
    }
    .stError, div[data-baseweb="notification"][kind="negative"] {
        background: rgba(255, 75, 75, 0.1) !important;
        border: 1px solid rgba(255, 75, 75, 0.3) !important;
        border-radius: 12px !important;
        color: #ff4b4b !important;
    }
    .stWarning, div[data-baseweb="notification"][kind="warning"] {
        background: rgba(255, 170, 0, 0.1) !important;
        border: 1px solid rgba(255, 170, 0, 0.3) !important;
        border-radius: 12px !important;
        color: #ffaa00 !important;
    }
    .stInfo {
        background: rgba(0, 210, 255, 0.1) !important;
        border: 1px solid rgba(0, 210, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #00d2ff !important;
    }

    /* ── Dataframes ── */
    .stDataFrame {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* ── Sliders ── */
    .stSlider [data-baseweb="slider"] {
        padding-top: 1rem;
    }

    /* ── Select boxes ── */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7) !important;
        border-radius: 10px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ── Subheaders ── */
    h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Prediction Card ── */
    .predict-card-high {
        background: linear-gradient(135deg,
            rgba(255,75,75,0.15), rgba(255,75,75,0.05));
        border: 1px solid rgba(255,75,75,0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .predict-card-low {
        background: linear-gradient(135deg,
            rgba(0,255,136,0.15), rgba(0,255,136,0.05));
        border: 1px solid rgba(0,255,136,0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .predict-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .predict-subtitle {
        font-size: 1rem;
        opacity: 0.8;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
    }
    .footer a {
        color: #00d2ff !important;
        text-decoration: none;
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
st.markdown('<p class="main-header">📉 Customer Churn Predictor</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered churn prediction using XGBoost · '
    'Built by Pravallika Chepuri · Arizona State University</p>',
    unsafe_allow_html=True)

# ── Stats Banner ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Accuracy",     "80%+")
c2.metric("📊 ROC-AUC",      "0.85+")
c3.metric("👥 Training Data","7,043")
c4.metric("⚙️ Algorithm",    "XGBoost")
st.markdown("---")

# ── Sidebar Inputs ───────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Customer Details")
st.sidebar.markdown("---")

def user_inputs():
    st.sidebar.markdown("### 👤 Demographics")
    gender          = st.sidebar.selectbox("Gender",
                      ["Male", "Female"])
    senior_citizen  = st.sidebar.selectbox("Senior Citizen",
                      ["No", "Yes"])
    partner         = st.sidebar.selectbox("Has Partner",
                      ["No", "Yes"])
    dependents      = st.sidebar.selectbox("Has Dependents",
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
                       ["No", "Yes"])
    payment_method   = st.sidebar.selectbox("Payment Method",[
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

# ────────────────────────────────────────────────────────────
# TAB 1
# ────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="predict-card-high">
                <div class="predict-title">⚠️ HIGH CHURN RISK</div>
                <div class="predict-subtitle">
                    {churn_prob:.1f}% probability of churning
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="predict-card-low">
                <div class="predict-title">✅ LOW CHURN RISK</div>
                <div class="predict-subtitle">
                    {no_churn_prob:.1f}% probability of staying
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Probability Breakdown")
        m1, m2 = st.columns(2)
        m1.metric("🔴 Churn",     f"{churn_prob:.1f}%")
        m2.metric("🟢 Retention", f"{no_churn_prob:.1f}%")

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
            risk_factors.append(f"⚠️ **New customer** ({raw['tenure']} months)")
            risk_score += 2
        else:
            good_factors.append(f"✅ Loyal ({raw['tenure']} months)")

        if raw['monthly_charges'] > 70:
            risk_factors.append(f"⚠️ **High charges** (${raw['monthly_charges']:.0f}/mo)")
            risk_score += 2
        else:
            good_factors.append(f"✅ Fair pricing (${raw['monthly_charges']:.0f}/mo)")

        if raw['tech_support'] == "No":
            risk_factors.append("⚠️ **No tech support**")
            risk_score += 1
        else:
            good_factors.append("✅ Has tech support")

        if raw['internet_service'] == "Fiber optic":
            risk_factors.append("⚠️ **Fiber optic** — higher expectations")
            risk_score += 1

        if risk_factors:
            st.markdown("**🚨 Risk Factors:**")
            for f in risk_factors:
                st.warning(f)

        if good_factors:
            st.markdown("**💪 Strengths:**")
            for f in good_factors:
                st.success(f)

        risk_pct = int((risk_score / 9) * 100)
        st.markdown(f"**Risk Score: {risk_score}/9**")
        st.progress(risk_pct)

# ────────────────────────────────────────────────────────────
# TAB 2
# ────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────
# TAB 3
# ────────────────────────────────────────────────────────────
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
    Built with ❤️ by <strong>Pravallika Chepuri</strong> |
    Arizona State University |
    <a href="https://github.com/Chepuri25/customer-churn-prediction"
       target="_blank">🐙 GitHub</a> |
    <a href="https://www.linkedin.com/in/pravallika-chepuri"
       target="_blank">💼 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
