"""
app.py
Streamlit dashboard for the Insurance Claim Risk Prediction prototype.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Insurance Claim Risk Prediction",
    layout="wide",
)

NUMERICAL_COLS = [
    "age", "annual_premium", "deductible", "bonus_malus", "vehicle_value",
    "vehicle_age", "engine_capacity", "vehicle_weight", "years_driving",
    "no_claims_years", "policy_tenure", "credit_score", "mileage",
    "population_density",
]
CATEGORICAL_COLS = [
    "gender", "marital_status", "education_level", "employment_status",
    "region", "vehicle_type", "vehicle_brand", "fuel_type", "transmission",
    "usage_type", "policy_type", "payment_frequency", "safety_features",
]

HORIZON_FILES = {
    "3 months": "xgboost_3M.joblib",
    "6 months": "xgboost_6M.joblib",
    "12 months": "xgboost_12M.joblib",
}
DEFAULT_THRESHOLDS = {"3 months": 0.30, "6 months": 0.22, "12 months": 0.12}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def risk_label(prob):
    if prob < 0.30:
        return "Low Risk", "#2e7d32"
    if prob < 0.60:
        return "Medium Risk", "#f9a825"
    return "High Risk", "#c62828"


st.sidebar.title("Configuration")
horizon_label = st.sidebar.selectbox("Prediction Horizon", list(HORIZON_FILES.keys()), index=2)

st.sidebar.divider()
st.sidebar.header("Policyholder")
age = st.sidebar.slider("Age", 18, 80, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
employment_status = st.sidebar.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed", "Retired"])
region = st.sidebar.selectbox("Region", ["Khomas", "Erongo", "Oshana", "Otjozondjupa", "Kavango", "Zambezi"])

st.sidebar.divider()
st.sidebar.header("Vehicle")
vehicle_value = st.sidebar.number_input("Vehicle Value (NAD)", 20_000, 1_500_000, 250_000, step=10_000)
vehicle_age = st.sidebar.slider("Vehicle Age (yrs)", 0, 25, 5)
engine_capacity = st.sidebar.number_input("Engine Capacity (cc)", 800, 5000, 2000, step=100)
vehicle_weight = st.sidebar.number_input("Vehicle Weight (kg)", 700, 3500, 1500, step=50)
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Sedan", "SUV", "Pickup", "Hatchback", "Minivan"])
vehicle_brand = st.sidebar.selectbox("Vehicle Brand", ["Toyota", "Volkswagen", "Ford", "Nissan", "Hyundai", "BMW"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
usage_type = st.sidebar.selectbox("Usage", ["Private", "Commercial"])
safety_features = st.sidebar.selectbox("Safety Features", ["Yes", "No"])

st.sidebar.divider()
st.sidebar.header("Policy")
annual_premium = st.sidebar.number_input("Annual Premium (NAD)", 1_500, 80_000, 15_000, step=500)
deductible = st.sidebar.selectbox("Deductible (NAD)", [1000, 2500, 5000, 10000, 25000], index=2)
bonus_malus = st.sidebar.slider("Bonus-Malus", -5, 5, 0)
policy_type = st.sidebar.selectbox("Policy Type", ["Comprehensive", "Third-Party", "Third-Party Fire & Theft"])
payment_frequency = st.sidebar.selectbox("Payment Frequency", ["Monthly", "Quarterly", "Annual"])
policy_tenure = st.sidebar.slider("Policy Tenure (yrs)", 0, 20, 3)

st.sidebar.divider()
st.sidebar.header("History")
years_driving = st.sidebar.slider("Years Driving", 0, 60, 15)
no_claims_years = st.sidebar.slider("No-Claim Years", 0, 15, 3)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
mileage = st.sidebar.number_input("Annual Mileage (km)", 1_000, 100_000, 20_000, step=1_000)
population_density = st.sidebar.number_input("Pop. Density (per km2)", 10, 20_000, 1_000, step=100)

input_df = pd.DataFrame([{
    "age": age, "annual_premium": annual_premium, "deductible": deductible,
    "bonus_malus": bonus_malus, "vehicle_value": vehicle_value,
    "vehicle_age": vehicle_age, "engine_capacity": engine_capacity,
    "vehicle_weight": vehicle_weight, "years_driving": years_driving,
    "no_claims_years": no_claims_years, "policy_tenure": policy_tenure,
    "credit_score": credit_score, "mileage": mileage,
    "population_density": population_density,
    "gender": gender, "marital_status": marital_status,
    "education_level": education_level, "employment_status": employment_status,
    "region": region, "vehicle_type": vehicle_type,
    "vehicle_brand": vehicle_brand, "fuel_type": fuel_type,
    "transmission": transmission, "usage_type": usage_type,
    "policy_type": policy_type, "payment_frequency": payment_frequency,
    "safety_features": safety_features,
}])

st.title("Insurance Claim Risk Prediction")
st.caption("XGBoost prototype - University of Namibia research thesis")

c1, c2, c3 = st.columns(3)
c1.metric("Model", "XGBoost")
c2.metric("Horizon", horizon_label)
c3.metric("Records Trained", "10,000")
st.divider()

model_file = HORIZON_FILES[horizon_label]

if not os.path.exists(model_file):
    st.error(f"Model file `{model_file}` not found in the repository root.")
    st.info(f"Files present: {sorted(os.listdir('.'))}")
    st.stop()

try:
    model = load_model(model_file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Most likely a scikit-learn version mismatch between training and deployment.")
    st.stop()

try:
    proba = float(model.predict_proba(input_df)[0, 1])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

threshold = DEFAULT_THRESHOLDS[horizon_label]
pred = int(proba >= threshold)
label, color = risk_label(proba)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Claim Probability", f"{proba:.2%}")
r2.metric("Prediction", "Claim" if pred else "No Claim")
r3.metric("Threshold", f"{threshold:.2f}")
r4.metric("Risk Level", label)

st.markdown("### Risk Gauge")
st.markdown(f"""
<div style="background:#f0f4ff;border-radius:12px;padding:24px;margin-top:8px;">
  <div style="display:flex;justify-content:space-between;font-weight:600;margin-bottom:8px;font-size:13px;">
    <span>0%</span><span>30%</span><span>60%</span><span>100%</span>
  </div>
  <div style="position:relative;height:26px;background:linear-gradient(to right,#4caf50,#ffc107,#f44336);border-radius:13px;">
    <div style="position:absolute;left:{min(proba*100,100):.2f}%;top:-8px;transform:translateX(-50%);
                width:18px;height:42px;background:white;border:3px solid #1a2a4a;border-radius:9px;"></div>
  </div>
  <p style="text-align:center;margin-top:16px;font-size:16px;">
    Predicted claim probability: <b style="color:{color};">{proba:.2%}</b> - {label}
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()
st.subheader("Reference Performance (Thesis, 12-Month Horizon)")
st.dataframe(pd.DataFrame({
    "Model": ["XGBoost", "Random Forest", "Naive Bayes"],
    "Accuracy": [0.7510, 0.7495, 0.7500],
    "Precision": [0.7500, 0.7503, 0.7501],
    "Recall": [1.0000, 0.9960, 0.9973],
    "F1": [0.8600, 0.8558, 0.8562],
    "AUC-ROC": [0.6776, 0.6655, 0.6691],
    "AUC-PR": [0.8585, 0.8512, 0.8509],
}), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Prototype for research thesis - Kennedy N. Hauwanga, "
    "University of Namibia, BSc Data Science Honours."
)
