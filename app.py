"""
app.py
Insurance Claim Risk Prediction - XGBoost prototype
Three horizon cards (3M / 6M / 12M) fill in after clicking Predict.
Changing ANY risk factor silently resets the prototype to its initial state.
Bulk prediction: upload a policy list, predict risk per policyholder,
                 and generate suggested reserve amounts.

Author: Kennedy N. Hauwanga (Student No. 218208399)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sqlite3
import datetime
import io

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

# ---------- Database Setup ----------
DB_FILE = os.path.join(os.path.expanduser("~"), "predictions.db")


def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Predictions (
                PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
                Age INTEGER, Gender TEXT, VehicleValue REAL, AnnualPremium REAL,
                TimeHorizon TEXT, PredictedClass TEXT, Probability REAL,
                PredictionDate TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ModelMetadata (
                ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelName TEXT, Hyperparameters TEXT, Accuracy REAL,
                AUC_PR REAL, LogDate TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False


def log_prediction(age, gender, vehicle_val, premium, horizon, decision, prob):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO Predictions
            (Age, Gender, VehicleValue, AnnualPremium, TimeHorizon,
             PredictedClass, Probability, PredictionDate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (age, gender, vehicle_val, premium, horizon, decision, prob, now))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Could not log prediction: {e}")
        return False


def get_all_predictions():
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(
            "SELECT * FROM Predictions ORDER BY PredictionDate DESC", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def convert_df_to_excel(df, sheet_name="Predictions"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


init_db()

# ---------- Security & Authentication ----------
def check_credentials():
    def verify_login():
        if (st.session_state.get("username") == "khauwanga" and
                st.session_state.get("password") == "University@4201"):
            st.session_state["authentication_status"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["authentication_status"] = False

    def clear_login():
        st.session_state["username"] = ""
        st.session_state["password"] = ""
        st.session_state["authentication_status"] = None

    if ("authentication_status" not in st.session_state
            or st.session_state["authentication_status"] != True):
        st.title("System Login meant only for my Thesis: Kennedy N Hauwanga")
        st.write("Please enter your credentials to access the Claim Risk Prediction System.")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=verify_login)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("Login", type="primary", on_click=verify_login, use_container_width=True)
        with col2:
            st.button("Clear", on_click=clear_login, use_container_width=True)
        if st.session_state.get("authentication_status") == False:
            st.error("Incorrect username or password. Access denied.")
        return False
    return True


if not check_credentials():
    st.stop()

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


def calculate_business_risk_adjustment(age, gender, years_driving, prior_claims, violations,
                                       mileage, credit_score, urban_rural, vehicle_age,
                                       vehicle_value, vehicle_type):
    multiplier = 1.0
    if years_driving < 3:
        multiplier += 0.4
    elif years_driving < 8:
        multiplier += 0.15
    elif years_driving > 15:
        multiplier -= 0.25
    elif years_driving > 10:
        multiplier -= 0.1
    if urban_rural == "Rural":
        multiplier -= 0.15
    else:
        multiplier += 0.1
    if prior_claims == 1:
        multiplier += 0.3
    elif prior_claims == 2:
        multiplier += 0.6
    elif prior_claims >= 3:
        multiplier += 1.0
    if age < 25:
        multiplier += 0.25
    elif age > 65:
        multiplier += 0.1
    if violations > 0:
        multiplier += (0.15 * violations)
    if mileage > 30000:
        multiplier += 0.2
    if credit_score < 550:
        multiplier += 0.15
    if vehicle_age > 10:
        multiplier += 0.15
    if vehicle_value > 500000:
        multiplier += 0.1
    if vehicle_type in ["Pickup", "SUV"]:
        multiplier += 0.05
    return max(0.2, min(multiplier, 3.0))


def render_card(horizon, cfg, prob=None):
    if prob is None:
        st.markdown(
            f"""
            <div style="background:#ffffff;border:2px dashed #c7d2e0;border-radius:14px;
                padding:26px 20px;text-align:center;min-height:250px;
                display:flex;flex-direction:column;justify-content:center;align-items:center;">
              <div style="font-size:15px;color:#4a5568;font-weight:700;
                          letter-spacing:0.6px;text-transform:uppercase;">Within {horizon}</div>
              <div style="font-size:44px;font-weight:800;color:#c7d2e0;margin:14px 0;">--</div>
              <div style="font-size:13px;color:#94a3b8;">Click <b>Predict Claim Risk</b> to generate</div>
            </div>
            """, unsafe_allow_html=True)
        return
    risk, risk_color = risk_level(prob)
    decision = "Claim Likely" if prob >= cfg["threshold"] else "No Claim"
    decision_color = "#c62828" if prob >= cfg["threshold"] else "#2e7d32"
    left_pct = min(max(prob * 100, 0), 100)
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e0e6ef;border-radius:14px;
            padding:24px 20px;text-align:center;box-shadow:0 3px 12px rgba(0,0,0,0.06);">
          <div style="font-size:15px;color:#4a5568;font-weight:700;
                      letter-spacing:0.6px;text-transform:uppercase;">Within {horizon}</div>
          <div style="font-size:52px;font-weight:800;color:{cfg['color']};margin:10px 0 4px;">{prob:.1%}</div>
          <div style="font-size:13px;color:#666;margin-bottom:14px;">Likelihood of a claim</div>
          <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
            <span style="background:{risk_color};color:white;padding:4px 12px;
                border-radius:20px;font-size:12px;font-weight:600;">{risk}</span>
            <span style="background:{decision_color};color:white;padding:4px 12px;
                border-radius:20px;font-size:12px;font-weight:600;">{decision}</span>
          </div>
          <div style="margin-top:14px;">
            <div style="position:relative;height:20px;
                background:linear-gradient(to right,#4caf50,#ffc107,#f44336);border-radius:10px;">
              <div style="position:absolute;left:{left_pct:.2f}%;top:-5px;transform:translateX(-50%);
                  width:12px;height:30px;background:white;border:3px solid #1a2a4a;border-radius:6px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;
                        color:#888;margin-top:4px;">
              <span>0%</span><span>30%</span><span>60%</span><span>100%</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ---------- Bulk prediction utilities ----------
BULK_REQUIRED_COLUMNS = [
    "policy_id", "age", "gender", "marital_status", "education_level",
    "employment_status", "income_level", "home_ownership", "region",
    "urban_rural", "vehicle_value", "vehicle_age", "vehicle_age_group",
    "engine_capacity", "vehicle_weight", "vehicle_type", "vehicle_brand",
    "fuel_type", "transmission", "vehicle_usage", "safety_features",
    "premium_amount", "deductible", "bonus_malus", "policy_type",
    "payment_frequency", "policy_duration_years", "years_driving",
    "prior_claims", "prior_claim_amount", "traffic_violations",
    "credit_score", "annual_mileage", "has_telematics", "telematics_score",
]


def prepare_bulk_input(df):
    """
    Validate and convert an uploaded DataFrame into the exact schema
    used by the model. Returns (prepared_df, error_message).
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in BULK_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, (
            f"The uploaded file is missing these required columns: "
            f"**{', '.join(missing)}**. Please use the template."
        )

    # Normalize binary / text fields
    if "urban_rural" in df.columns:
        df["urban_rural"] = df["urban_rural"].astype(str).str.strip().str.title()
        df["urban_rural"] = df["urban_rural"].map({"Urban": 1, "Rural": 0}).fillna(0).astype(int)

    if "safety_features" in df.columns:
        df["safety_features"] = df["safety_features"].astype(str).str.strip().str.title()
        df["safety_features"] = df["safety_features"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    if "has_telematics" in df.columns:
        df["has_telematics"] = df["has_telematics"].astype(str).str.strip().str.title()
        df["has_telematics"] = df["has_telematics"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Numeric coercion
    numeric_cols = [
        "age", "vehicle_value", "vehicle_age", "engine_capacity", "vehicle_weight",
        "premium_amount", "deductible", "bonus_malus", "policy_duration_years",
        "years_driving", "prior_claims", "prior_claim_amount", "traffic_violations",
        "credit_score", "annual_mileage", "telematics_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, None


def compute_reserves_for_row(row, base_prob, threshold):
    """
    Compute a suggested reserve for a single policy/horizon.
    Expected Loss = Probability × Severity × Safety Loading
    """
    severity_rate = float(row["__severity_rate__"])
    safety_loading = float(row["__safety_loading__"])
    claim_severity = float(row["vehicle_value"]) * severity_rate
    expected_loss = base_prob * claim_severity
    reserve = expected_loss * safety_loading
    return reserve


def run_bulk_prediction(df, severity_rate, safety_loading):
    """
    For each policy row, run all three models and attach probability,
    decision, and reserve columns for each horizon.
    Returns an enriched DataFrame.
    """
    # Load all models once
    models = {}
    for horizon, cfg in HORIZONS.items():
        if not os.path.exists(cfg["file"]):
            raise FileNotFoundError(
                f"Model file `{cfg['file']}` not found for {horizon}."
            )
        models[horizon] = load_model(cfg["file"])

    out = df.copy()
    out["__severity_rate__"] = severity_rate
    out["__safety_loading__"] = safety_loading

    # Pre-compute business risk multiplier for each row
    multipliers = []
    for _, row in df.iterrows():
        multipliers.append(calculate_business_risk_adjustment(
            age=row["age"],
            gender=row["gender"],
            years_driving=row["years_driving"],
            prior_claims=row["prior_claims"],
            violations=row["traffic_violations"],
            mileage=row["annual_mileage"],
            credit_score=row["credit_score"],
            urban_rural="Urban" if int(row["urban_rural"]) == 1 else "Rural",
            vehicle_age=row["vehicle_age"],
            vehicle_value=row["vehicle_value"],
            vehicle_type=row["vehicle_type"],
        ))
    out["__multiplier__"] = multipliers

    # Columns to feed the model (everything except helper cols and policy_id)
    model_input_cols = [
        c for c in BULK_REQUIRED_COLUMNS if c != "policy_id"
    ]

    for horizon, cfg in HORIZONS.items():
        model = models[horizon]
        X = out[model_input_cols].copy()
        try:
            base_probs = model.predict_proba(X)[:, 1]
        except Exception as e:
            raise RuntimeError(f"Model prediction failed for {horizon}: {e}")

        adjusted = np.minimum(base_probs * out["__multiplier__"].values, 0.999)

        out[f"prob_{horizon.replace(' ', '_')}"] = adjusted
        out[f"decision_{horizon.replace(' ', '_')}"] = np.where(
            adjusted >= cfg["threshold"], "Claim", "No Claim"
        )
        out[f"risk_level_{horizon.replace(' ', '_')}"] = [
            risk_level(p)[0] for p in adjusted
        ]

        # Reserve per horizon
        severity = out["vehicle_value"].astype(float) * severity_rate
        reserves = adjusted * severity * safety_loading
        out[f"reserve_{horizon.replace(' ', '_')}"] = reserves.round(2)

    # Drop helper columns
    out = out.drop(columns=["__severity_rate__", "__safety_loading__", "__multiplier__"])
    return out


# ---------- Sidebar Callbacks ----------
def update_age_group():
    current_age = st.session_state.policyholder_age
    if current_age < 25:
        st.session_state.age_grp = "Young"
    elif current_age < 40:
        st.session_state.age_grp = "Adult"
    elif current_age < 60:
        st.session_state.age_grp = "Middle-Aged"
    else:
        st.session_state.age_grp = "Senior"


if 'age_grp' not in st.session_state:
    st.session_state.age_grp = "Adult"

# ---------- Sidebar ----------
st.sidebar.title("System Menu")

if st.sidebar.button("⏻ Logout", type="primary", use_container_width=True):
    del st.session_state["authentication_status"]
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Policyholder")

age = st.sidebar.slider("Age", 18, 80, 35, key="policyholder_age", on_change=update_age_group)
age_group = st.sidebar.selectbox(
    "Age Group (Auto-calculated)",
    ["Young", "Adult", "Middle-Aged", "Senior"],
    key="age_grp", disabled=True,
)
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
years_driving = st.sidebar.slider("Years Driving Experience", 0, 60, 15)
prior_claims = st.sidebar.number_input("Prior Claims (Last 3 Years)", 0, 20, 0)
prior_claim_amount = st.sidebar.number_input("Prior Claim Amount (NAD)", 0, 500_000, 0, step=1000)
traffic_violations = st.sidebar.number_input("Traffic Violations", 0, 20, 0)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
annual_mileage = st.sidebar.number_input("Annual Mileage (km)", 1_000, 100_000, 20_000, step=1_000)
has_telematics = st.sidebar.selectbox("Has Telematics", ["Yes", "No"])
telematics_score = st.sidebar.slider("Telematics Score", 0, 100, 50)

# ---------- Build input DataFrame ----------
input_df = pd.DataFrame([{
    "age": age, "age_group": age_group, "gender": gender,
    "marital_status": marital_status, "education_level": education_level,
    "employment_status": employment_status, "income_level": income_level,
    "home_ownership": home_ownership, "region": region,
    "urban_rural": 1 if urban_rural == "Urban" else 0,
    "vehicle_value": vehicle_value, "vehicle_age": vehicle_age,
    "vehicle_age_group": vehicle_age_group, "engine_capacity": engine_capacity,
    "vehicle_weight": vehicle_weight, "vehicle_type": vehicle_type,
    "vehicle_brand": vehicle_brand, "fuel_type": fuel_type,
    "transmission": transmission, "vehicle_usage": vehicle_usage,
    "safety_features": 1 if safety_features == "Yes" else 0,
    "premium_amount": premium_amount, "deductible": deductible,
    "bonus_malus": bonus_malus, "policy_type": policy_type,
    "payment_frequency": payment_frequency,
    "policy_duration_years": policy_duration_years,
    "years_driving": years_driving, "prior_claims": prior_claims,
    "prior_claim_amount": prior_claim_amount,
    "traffic_violations": traffic_violations, "credit_score": credit_score,
    "annual_mileage": annual_mileage,
    "has_telematics": 1 if has_telematics == "Yes" else 0,
    "telematics_score": telematics_score,
}])

# ---------- Silent reset on any change ----------
current_fingerprint = hash(tuple(input_df.iloc[0].astype(str).tolist()))
if "input_fingerprint" not in st.session_state:
    st.session_state["input_fingerprint"] = current_fingerprint
if "last_results" not in st.session_state:
    st.session_state["last_results"] = {}
if st.session_state["input_fingerprint"] != current_fingerprint:
    st.session_state["input_fingerprint"] = current_fingerprint
    st.session_state["last_results"] = {}

# ---------- Header ----------
st.title("Insurance Claim Risk Prediction")
st.markdown(
    "<h4 style='color:#4a5568;margin-top:-12px;font-weight:500;'>"
    "by Kennedy N. Hauwanga</h4>",
    unsafe_allow_html=True,
)
st.caption("University of Namibia research thesis | Student No. 218208399")
st.divider()

# ---------- Tabs ----------
tab_single, tab_bulk = st.tabs(["Single Policyholder", "Bulk Policy File"])

# ============================================================
# TAB 1: Single policyholder (unchanged behavior)
# ============================================================
with tab_single:
    st.subheader("Generate Prediction")
    st.write(
        "Adjust the policyholder details in the sidebar, then click "
        "**Predict Claim Risk**. Likelihood percentages for each horizon "
        "will appear below."
    )
    predict_clicked = st.button("Predict Claim Risk", type="primary", use_container_width=True)

    if predict_clicked:
        risk_multiplier = calculate_business_risk_adjustment(
            age=age, gender=gender, years_driving=years_driving,
            prior_claims=prior_claims, violations=traffic_violations,
            mileage=annual_mileage, credit_score=credit_score,
            urban_rural=urban_rural, vehicle_age=vehicle_age,
            vehicle_value=vehicle_value, vehicle_type=vehicle_type,
        )
        new_results = {}
        for horizon, cfg in HORIZONS.items():
            path = cfg["file"]
            if not os.path.exists(path):
                st.warning(f"Model file `{path}` not found. Skipping {horizon}.")
                continue
            try:
                model = load_model(path)
                base_prob = float(model.predict_proba(input_df)[0, 1])
                adjusted_prob = min(base_prob * risk_multiplier, 0.999)
                new_results[horizon] = adjusted_prob
                decision = "Claim" if adjusted_prob >= cfg["threshold"] else "No Claim"
                log_prediction(
                    age=age, gender=gender, vehicle_val=vehicle_value,
                    premium=premium_amount, horizon=horizon,
                    decision=decision, prob=adjusted_prob,
                )
            except Exception as e:
                st.warning(f"Prediction failed for `{horizon}`: {e}")
        if new_results:
            st.session_state["last_results"] = new_results
            st.success("Prediction results have been logged to the database.")

    results = st.session_state["last_results"]
    st.markdown("### Predicted Claim Likelihood")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_card("3 months", HORIZONS["3 months"], results.get("3 months"))
    with col2:
        render_card("6 months", HORIZONS["6 months"], results.get("6 months"))
    with col3:
        render_card("12 months", HORIZONS["12 months"], results.get("12 months"))

    if results:
        st.divider()
        st.subheader("Summary")
        summary = pd.DataFrame({
            "3 months": [
                f"{results.get('3 months', 0):.2%}",
                risk_level(results.get('3 months', 0))[0],
                "Claim" if results.get('3 months', 0) >= HORIZONS["3 months"]["threshold"] else "No Claim",
                f"{HORIZONS['3 months']['threshold']:.2f}",
            ],
            "6 months": [
                f"{results.get('6 months', 0):.2%}",
                risk_level(results.get('6 months', 0))[0],
                "Claim" if results.get('6 months', 0) >= HORIZONS["6 months"]["threshold"] else "No Claim",
                f"{HORIZONS['6 months']['threshold']:.2f}",
            ],
            "12 months": [
                f"{results.get('12 months', 0):.2%}",
                risk_level(results.get('12 months', 0))[0],
                "Claim" if results.get('12 months', 0) >= HORIZONS["12 months"]["threshold"] else "No Claim",
                f"{HORIZONS['12 months']['threshold']:.2f}",
            ],
        }, index=["Probability", "Risk Level", "Decision", "Threshold"])
        st.dataframe(summary, use_container_width=True)

# ============================================================
# TAB 2: Bulk policy file prediction
# ============================================================
with tab_bulk:
    st.subheader("Bulk Policy Risk & Reserve Recommendation")
    st.write(
        "Upload a CSV or Excel file containing multiple policies. "
        "The system will predict the claim risk for each policyholder across all "
        "three horizons and suggest a reserve amount to hold for each policy."
    )

    # ---- Template download ----
    template_df = pd.DataFrame([{
        "policy_id": "P0001",
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "education_level": "Bachelor",
        "employment_status": "Employed",
        "income_level": "Medium",
        "home_ownership": "Own",
        "region": "Khomas",
        "urban_rural": "Urban",
        "vehicle_value": 250000,
        "vehicle_age": 5,
        "vehicle_age_group": "Recent",
        "engine_capacity": 2000,
        "vehicle_weight": 1500,
        "vehicle_type": "Sedan",
        "vehicle_brand": "Toyota",
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "vehicle_usage": "Private",
        "safety_features": "Yes",
        "premium_amount": 15000,
        "deductible": 5000,
        "bonus_malus": 0,
        "policy_type": "Comprehensive",
        "payment_frequency": "Monthly",
        "policy_duration_years": 3,
        "years_driving": 15,
        "prior_claims": 0,
        "prior_claim_amount": 0,
        "traffic_violations": 0,
        "credit_score": 650,
        "annual_mileage": 20000,
        "has_telematics": "Yes",
        "telematics_score": 50,
    }])

    with st.expander("📄 Download template (required columns)"):
        st.write(
            "Use this template as the exact schema for your bulk upload file. "
            "All columns are required."
        )
        st.dataframe(template_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download template (CSV)",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="bulk_policy_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download template (Excel)",
            data=convert_df_to_excel(template_df, "Template"),
            file_name="bulk_policy_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()

    # ---- Upload ----
    uploaded = st.file_uploader(
        "Upload policy file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="bulk_upload",
    )

    # ---- Reserve assumptions ----
    st.markdown("#### Reserve Assumptions")
    col_a, col_b = st.columns(2)
    with col_a:
        severity_rate = st.slider(
            "Expected Claim Severity (% of vehicle value)",
            min_value=0.05, max_value=1.00, value=0.30, step=0.05,
            help="Portion of the vehicle's value that a claim is expected to cost on average.",
        )
    with col_b:
        safety_loading = st.slider(
            "Safety Loading Multiplier",
            min_value=1.00, max_value=2.00, value=1.25, step=0.05,
            help="Margin added to expected loss to cover uncertainty, expenses, and profit.",
        )

    st.caption(
        f"**Reserve formula:** "
        f"`Probability × (Vehicle Value × {severity_rate:.0%}) × {safety_loading:.2f}` "
        f"per horizon."
    )

    run_bulk = st.button(
        "Run Bulk Prediction", type="primary", use_container_width=True,
        disabled=uploaded is None,
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                raw = pd.read_csv(uploaded)
            else:
                raw = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            raw = None

        if raw is not None:
            st.write(f"**Rows detected:** {len(raw)}")
            with st.expander("Preview uploaded data (first 5 rows)"):
                st.dataframe(raw.head(5), use_container_width=True, hide_index=True)

            prepared, err = prepare_bulk_input(raw)
            if err:
                st.error(err)
            else:
                if run_bulk:
                    try:
                        with st.spinner("Running predictions for all policies..."):
                            enriched = run_bulk_prediction(
                                prepared, severity_rate, safety_loading
                            )
                    except FileNotFoundError as e:
                        st.error(str(e))
                        enriched = None
                    except Exception as e:
                        st.error(f"Bulk prediction failed: {e}")
                        enriched = None

                    if enriched is not None:
                        st.session_state["bulk_results"] = enriched
                        st.success(
                            f"Bulk prediction complete for **{len(enriched)}** policies."
                        )

    # ---- Display results ----
    if "bulk_results" in st.session_state:
        enriched = st.session_state["bulk_results"]

        st.divider()
        st.subheader("Portfolio Summary")

        total_policies = len(enriched)
        total_reserve_3m = enriched["reserve_3_months"].sum()
        total_reserve_6m = enriched["reserve_6_months"].sum()
        total_reserve_12m = enriched["reserve_12_months"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Policies Processed", f"{total_policies}")
        c2.metric("Reserve (3M)", f"NAD {total_reserve_3m:,.0f}")
        c3.metric("Reserve (6M)", f"NAD {total_reserve_6m:,.0f}")
        c4.metric("Reserve (12M)", f"NAD {total_reserve_12m:,.0f}")

        st.divider()
        st.subheader("Per-Policy Predictions & Recommended Reserves")

        display_cols = [
            "policy_id",
            "age", "gender", "vehicle_value",
            "prob_3_months", "decision_3_months", "reserve_3_months",
            "prob_6_months", "decision_6_months", "reserve_6_months",
            "prob_12_months", "decision_12_months", "reserve_12_months",
        ]
        display_cols = [c for c in display_cols if c in enriched.columns]

        show_df = enriched[display_cols].copy()
        for col in ["prob_3_months", "prob_6_months", "prob_12_months"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].map(lambda v: f"{v:.2%}")
        for col in ["reserve_3_months", "reserve_6_months", "reserve_12_months"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].map(lambda v: f"{v:,.2f}")

        st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.markdown("#### Download Results")

        # Full enriched output (all original + prediction columns)
        st.download_button(
            "⬇ Download full results (Excel)",
            data=convert_df_to_excel(enriched, "Bulk_Predictions"),
            file_name=f"bulk_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.download_button(
            "⬇ Download full results (CSV)",
            data=enriched.to_csv(index=False).encode("utf-8"),
            file_name=f"bulk_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        if st.button("Clear Bulk Results"):
            del st.session_state["bulk_results"]
            st.rerun()

# ---------- Reference table ----------
st.divider()
st.subheader("Reference Performance")

st.dataframe(pd.DataFrame({
    "Model": ["XGBoost"],
    "Accuracy": [0.7510],
    "Precision": [0.7500],
    "Recall": [1.0000],
    "F1": [0.8600],
    "AUC-ROC": [0.6776],
    "AUC-PR": [0.8585],
}), use_container_width=True, hide_index=True)

# ---------- Admin: Download Database (Excel) ----------
st.divider()
st.subheader("System Administration (Database Logs)")

df_logs = get_all_predictions()

if not df_logs.empty:
    st.write(f"Total logged predictions in database: **{len(df_logs)}**")
    st.dataframe(df_logs.head(50), use_container_width=True, hide_index=True)
    excel_data = convert_df_to_excel(df_logs)
    st.download_button(
        label="Click here to Download Prediction Logs (Excel)",
        data=excel_data,
        file_name=f"prediction_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
else:
    st.info(
        "No predictions have been logged to the database yet. "
        "Click **Predict Claim Risk** to generate the first entry."
    )
