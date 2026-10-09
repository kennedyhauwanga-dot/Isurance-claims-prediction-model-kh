"""
app.py
Insurance Claim Risk Prediction - XGBoost prototype
Three horizon cards (3M / 6M / 12M) fill in after clicking Predict.
Author: Kennedy N. Hauwanga (Student No. 218208399)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# COMPATIBILITY PATCH FOR SCIKIT-LEARN 1.5.0 MODELS
# ============================================================
import sklearn
from sklearn.compose import _column_transformer
from sklearn.impute import SimpleImputer

if not hasattr(_column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    _column_transformer._RemainderColsList = _RemainderColsList

if not hasattr(SimpleImputer, '_fill_dtype'):
    SimpleImputer._fill_dtype = property(lambda self: np.dtype('float64'))
# ============================================================

st.set_page_config(
    page_title="Insurance Claim Risk Prediction",
    layout="wide",
)

# ---------- Model configuration ----------
HORIZONS = {
    "3 months":  {"file": "xgb_3M.joblib",  "threshold": 0.30, "color": "#1976d2"},
    "6 months":  {"file": "xgb_6M.joblib",  "threshold": 0.22, "color": "#f57c00"},
    "12 months": {"file": "xgb_12M.joblib", "threshold": 0.12, "color": "#388e3c"},
}

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

def risk_level(prob):
    if prob < 0.30:
        return "Low Risk", "#2e7d32"
    if prob < 0.60:
        return "Medium Risk", "#f9a825"
    return "High Risk", "#c62828"

def render_card(horizon, cfg, prob=None):
    """Render one horizon card. If prob is None, show placeholder."""
    if prob is None:
        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border:2px dashed #c7d2e0;
                border-radius:14px;
                padding:26px 20px;
                text-align:center;
                min-height:250px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
            ">
              <div style="font-size:15px;color:#4a5568;font-weight:700;
                          letter-spacing:0.6px;text-transform:uppercase;">
                Within {horizon}
              </div>
              <div style="font-size:44px;font-weight:800;color:#c7d2e0;margin:14px 0;">
                --
              </div>
              <div style="font-size:13px;color:#94a3b8;">
                Click <b>Predict Claim Risk</b> to generate
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    risk, risk_color = risk_level(prob)
    decision = "Claim Likely" if prob >= cfg["threshold"] else "No Claim"
    decision_color = "#c62828" if prob >= cfg["threshold"] else "#2e7d32"
    
    # Calculate gauge position
    left_pct = min(max(prob * 100, 0), 100)

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e0e6ef;
            border-radius:14px;
            padding:24px 20px;
            text-align:center;
            box-shadow:0 3px 12px rgba(0,0,0,0.06);
        ">
          <div style="font-size:15px;color:#4a5568;font-weight:700;
                      letter-spacing:0.6px;text-transform:uppercase;">
            Within {horizon}
          </div>
          <div style="font-size:52px;font-weight:800;color:{cfg['color']};margin:10px 0 4px;">
            {prob:.1%}
          </div>
          <div style="font-size:13px;color:#666;margin-bottom:14px;">
            Likelihood of a claim
          </div>
          <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
            <span style="background:{risk_color};color:white;padding:4px 12px;
                         border-radius:20px;font-size:12px;font-weight:600;">
              {risk}
            </span>
            <span style="background:{decision_color};color:white;padding:4px 12px;
                         border-radius:20px;font-size:12px;font-weight:600;">
              {decision}
            </span>
          </div>
          <div style="margin-top:14px;">
            <div style="position:relative;height:20px;background:linear-gradient(to right,#4caf50,#ffc107,#f44336);
                        border-radius:10px;">
              <div style="position:absolute;left:{left_pct:.2f}%;top:-5px;transform:translateX(-50%);
                          width:12px;height:30px;background:white;border:3px solid #1a2a4a;
                          border-radius:6px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#888;margin-top:4px;">
              <span>0%</span><span>30%</span><span>60%</span><span>100%</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Sidebar ----------
st.sidebar.title("Configuration")

st.sidebar.divider()
st.sidebar.header("Policyholder")
age = st.sidebar.slider("Age", 18, 80, 35)
age_group = st.sidebar.selectbox("Age Group", ["Young", "Adult", "Middle-Aged", "Senior"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
employment_status = st.sidebar.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed", "Retired"])
income_level = st.sidebar.selectbox("Income Level", ["Low", "Medium", "High"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])
region = st.sidebar.selectbox("Region", ["Khomas", "Erongo", "Oshana", "Otjozondjupa", "Kavango", "Zambezi"])
urban_rural = st.sidebar.selectbox("Urban/Rural", ["Urban", "Rural"])

st.sidebar.divider()
st.sidebar.header("Vehicle")
vehicle_value = st.sidebar.number_input("Vehicle Value (NAD)", 20_000, 1_500_000, 250_000, step=10_000)
vehicle_age = st.sidebar.slider("Vehicle Age (yrs)", 0, 25, 5)
vehicle_age_group = st.sidebar.selectbox("Vehicle Age Group", ["New", "Recent", "Old"])
engine_capacity = st.sidebar.number_input("Engine Capacity (cc)", 800, 5000, 2000, step=100)
vehicle_weight = st.sidebar.number_input("Vehicle Weight (kg)", 700, 3500, 1500, step=50)
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Sedan", "SUV", "Pickup", "Hatchback", "Minivan"])
vehicle_brand = st.sidebar.selectbox("Vehicle Brand", ["Toyota", "Volkswagen", "Ford", "Nissan", "Hyundai", "BMW"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
vehicle_usage = st.sidebar.selectbox("Vehicle Usage", ["Private", "Commercial"])
safety_features = st.sidebar.selectbox("Safety Features", ["Yes", "No"])

st.sidebar.divider()
st.sidebar.header("Policy")
premium_amount = st.sidebar.number_input("Premium Amount (NAD)", 1_500, 80_000, 15_000, step=500)
deductible = st.sidebar.selectbox("Deductible (NAD)", [1000, 2500, 5000, 10000, 25000], index=2)
bonus_malus = st.sidebar.slider("Bonus-Malus", -5, 5, 0)
policy_type = st.sidebar.selectbox("Policy Type", ["Comprehensive", "Third-Party", "Third-Party Fire & Theft"])
payment_frequency = st.sidebar.selectbox("Payment Frequency", ["Monthly", "Quarterly", "Annual"])
policy_duration_years = st.sidebar.slider("Policy Duration (yrs)", 0, 20, 3)

st.sidebar.divider()
st.sidebar.header("History & Telematics")
years_driving = st.sidebar.slider("Years Driving", 0, 60, 15)
prior_claims = st.sidebar.number_input("Prior Claims", 0, 20, 0)
prior_claim_amount = st.sidebar.number_input("Prior Claim Amount (NAD)", 0, 500_000, 0, step=1000)
traffic_violations = st.sidebar.number_input("Traffic Violations", 0, 20, 0)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
annual_mileage = st.sidebar.number_input("Annual Mileage (km)", 1_000, 100_000, 20_000, step=1_000)
has_telematics = st.sidebar.selectbox("Has Telematics", ["Yes", "No"])
telematics_score = st.sidebar.slider("Telematics Score", 0, 100, 50)

# ---------- Build input DataFrame (with numerical encoding) ----------
input_df = pd.DataFrame([{
    "age": age,
    "age_group": age_group,
    "gender": gender,
    "marital_status": marital_status,
    "education_level": education_level,
    "employment_status": employment_status,
    "income_level": income_level,
    "home_ownership": home_ownership,
    "region": region,
    "urban_rural": 1 if urban_rural == "Urban" else 0,
    "vehicle_value": vehicle_value,
    "vehicle_age": vehicle_age,
    "vehicle_age_group": vehicle_age_group,
    "engine_capacity": engine_capacity,
    "vehicle_weight": vehicle_weight,
    "vehicle_type": vehicle_type,
    "vehicle_brand": vehicle_brand,
    "fuel_type": fuel_type,
    "transmission": transmission,
    "vehicle_usage": vehicle_usage,
    "safety_features": 1 if safety_features == "Yes" else 0,
    "premium_amount": premium_amount,
    "deductible": deductible,
    "bonus_malus": bonus_malus,
    "policy_type": policy_type,
    "payment_frequency": payment_frequency,
    "policy_duration_years": policy_duration_years,
    "years_driving": years_driving,
    "prior_claims": prior_claims,
    "prior_claim_amount": prior_claim_amount,
    "traffic_violations": traffic_violations,
    "credit_score": credit_score,
    "annual_mileage": annual_mileage,
    "has_telematics": 1 if has_telematics == "Yes" else 0,
    "telematics_score": telematics_score,
}])

# ---------- Header ----------
st.title("Insurance Claim Risk Prediction")
st.markdown(
    "<h4 style='color:#4a5568;margin-top:-12px;font-weight:500;'>"
    "by Kennedy N. Hauwanga"
    "</h4>",
    unsafe_allow_html=True,
)
st.caption(
    "XGBoost prototype - University of Namibia research thesis | "
    "Student No. 218208399"
)

h1, h2, h3 = st.columns(3)
h1.metric("Model", "XGBoost")
h2.metric("Horizons", "3 / 6 / 12 months")
h3.metric("Records Trained", "10,000")

st.divider()

# ---------- Prediction section ----------
st.subheader("Generate Prediction")
st.write(
    "Adjust the policyholder details in the sidebar, then click "
    "**Predict Claim Risk**. Likelihood percentages for each horizon "
    "will appear below."
)

predict_clicked = st.button("Predict Claim Risk", type="primary", use_container_width=True)

# ---------- Horizon cards ----------
st.markdown("### Predicted Claim Likelihood")

results = {}
error = None

if predict_clicked:
    for horizon, cfg in HORIZONS.items():
        path = cfg["file"]
        if not os.path.exists(path):
            error = f"Model file `{path}` not found in the repository root."
            break
        try:
            model = load_model(path)
        except Exception as e:
            error = (
                f"Could not load `{path}`: {e}\n\n"
                "Retrain the models with scikit-learn 1.5.0 to match Streamlit Cloud."
            )
            break
        try:
            results[horizon] = float(model.predict_proba(input_df)[0, 1])
        except Exception as e:
            error = f"Prediction failed for `{path}`: {e}"
            break

if error:
    st.error(error)

col1, col2, col3 = st.columns(3)
with col1:
    render_card("3 months", HORIZONS["3 months"], results.get("3 months"))
with col2:
    render_card("6 months", HORIZONS["6 months"], results.get("6 months"))
with col3:
    render_card("12 months", HORIZONS["12 months"], results.get("12 months"))

# ---------- Summary table ----------
if results:
    st.divider()
    st.subheader("Summary")
    summary = pd.DataFrame({
        "Horizon": list(results.keys()),
        "Probability": [f"{results[h]:.2%}" for h in results],
        "Risk Level": [risk_level(results[h])[0] for h in results],
        "Decision": [
            "Claim" if results[h] >= HORIZONS[h]["threshold"] else "No Claim"
            for h in results
        ],
        "Threshold": [f"{HORIZONS[h]['threshold']:.2f}" for h in results],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True, on_select="ignore")

# ---------- Reference table (XGBoost only) ----------
st.divider()
st.subheader("Reference Performance (Thesis, 12-Month Horizon)")
st.dataframe(pd.DataFrame({
    "Model": ["XGBoost"],
    "Accuracy": [0.7510],
    "Precision": [0.7500],
    "Recall": [1.0000],
    "F1": [0.8600],
    "AUC-ROC": [0.6776],
    "AUC-PR": [0.8585],
}), use_container_width=True, hide_index=True, on_select="ignore")

st.divider()
st.caption(
    "Prototype for research thesis - Kennedy N. Hauwanga, "
    "University of Namibia, BSc Data Science Honours."
)
