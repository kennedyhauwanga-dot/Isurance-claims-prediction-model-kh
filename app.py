"""
app.py
Insurance Claim Risk Prediction - XGBoost prototype
Author: Kennedy N. Hauwanga (Student No. 218208399)

Tabs:
  1. Single Policyholder
  2. Bulk Policy File   (upload -> per-policy risk 3M/6M/12M + reserve + portfolio total)
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

st.set_page_config(page_title="Insurance Claim Risk Prediction", layout="wide")

# ---------- Model configuration ----------
HORIZONS = {
    "3 months":  {"file": "xgb_3M.joblib",  "threshold": 0.30, "color": "#1976d2", "short": "3M"},
    "6 months":  {"file": "xgb_6M.joblib",  "threshold": 0.22, "color": "#f57c00", "short": "6M"},
    "12 months": {"file": "xgb_12M.joblib", "threshold": 0.12, "color": "#388e3c", "short": "12M"},
}

# ---------- Database ----------
DB_FILE = os.path.join(os.path.expanduser("~"), "predictions.db")


def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS Predictions (
                PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
                Age INTEGER, Gender TEXT, VehicleValue REAL, AnnualPremium REAL,
                TimeHorizon TEXT, PredictedClass TEXT, Probability REAL,
                PredictionDate TIMESTAMP
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS ModelMetadata (
                ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelName TEXT, Hyperparameters TEXT, Accuracy REAL,
                AUC_PR REAL, LogDate TIMESTAMP
            )
        ''')
        conn.commit(); conn.close()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")


def log_prediction(age, gender, vehicle_val, premium, horizon, decision, prob):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute('''
            INSERT INTO Predictions
            (Age, Gender, VehicleValue, AnnualPremium, TimeHorizon,
             PredictedClass, Probability, PredictionDate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (age, gender, vehicle_val, premium, horizon, decision, prob, now))
        conn.commit(); conn.close()
        try:
            get_all_predictions.clear()
        except Exception:
            pass
        return True
    except Exception as e:
        st.warning(f"Could not log prediction: {e}")
        return False


@st.cache_data(ttl=5, show_spinner=False)
def get_all_predictions(limit=500):
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(
            f"SELECT * FROM Predictions ORDER BY PredictionDate DESC LIMIT {limit}",
            conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def convert_df_to_excel(df, sheet_name="Predictions"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name=sheet_name)
    return buf.getvalue()


init_db()

# ---------- Authentication ----------
def check_credentials():
    """
    Fixed login flow:
      - Uses st.form so the submit button reads the full, current widget values atomically.
      - No per-keystroke on_change callbacks interfering with session state.
      - Calls st.rerun() after a successful login so the app unlocks on the FIRST click.
    """
    def verify_login():
        username_ok = st.session_state.get("username", "") == "khauwanga"
        password_ok = st.session_state.get("password", "") == "University@4201"
        if username_ok and password_ok:
            st.session_state["authentication_status"] = True
        else:
            st.session_state["authentication_status"] = False

    def clear_login():
        st.session_state["username"] = ""
        st.session_state["password"] = ""
        st.session_state["authentication_status"] = None

    # Already authenticated? Let them through.
    if st.session_state.get("authentication_status") is True:
        return True

    st.title("System Login meant only for my Thesis: Kennedy N Hauwanga")
    st.write("Please enter your credentials to access the Claim Risk Prediction System.")

    with st.form("login_form", clear_on_submit=False):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        c1, c2 = st.columns([1, 1])
        with c1:
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
        with c2:
            cleared = st.form_submit_button("Clear", use_container_width=True)

    if submitted:
        verify_login()
        if st.session_state.get("authentication_status") is True:
            # Scrub the password from session state before rerunning
            st.session_state.pop("password", None)
            st.rerun()

    if cleared:
        clear_login()
        st.rerun()

    if st.session_state.get("authentication_status") is False:
        st.error("Incorrect username or password. Access denied.")

    return False


if not check_credentials():
    st.stop()

# ---------- Warm model cache once ----------
@st.cache_resource(show_spinner="Loading prediction models (first run only)...")
def warm_models():
    out = {}
    for h, cfg in HORIZONS.items():
        if os.path.exists(cfg["file"]):
            try:
                out[h] = joblib.load(cfg["file"])
            except Exception:
                out[h] = None
        else:
            out[h] = None
    return out


MODELS = warm_models()

# ---------- Helpers ----------
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
    m = 1.0
    if years_driving < 3: m += 0.4
    elif years_driving < 8: m += 0.15
    elif years_driving > 15: m -= 0.25
    elif years_driving > 10: m -= 0.1
    m += -0.15 if urban_rural == "Rural" else 0.1
    if prior_claims == 1: m += 0.3
    elif prior_claims == 2: m += 0.6
    elif prior_claims >= 3: m += 1.0
    if age < 25: m += 0.25
    elif age > 65: m += 0.1
    if violations > 0: m += (0.15 * violations)
    if mileage > 30000: m += 0.2
    if credit_score < 550: m += 0.15
    if vehicle_age > 10: m += 0.15
    if vehicle_value > 500000: m += 0.1
    if vehicle_type in ["Pickup", "SUV"]: m += 0.05
    return max(0.2, min(m, 3.0))


def calculate_business_risk_adjustment_vector(df):
    """Vectorised version for bulk prediction."""
    m = pd.Series(1.0, index=df.index)
    yd = df["years_driving"]
    m += np.where(yd < 3, 0.4,
         np.where(yd < 8, 0.15,
         np.where(yd > 15, -0.25,
         np.where(yd > 10, -0.1, 0.0))))
    m += np.where(df["urban_rural"] == 1, 0.1, -0.15)
    pc = df["prior_claims"]
    m += np.where(pc == 1, 0.3,
         np.where(pc == 2, 0.6,
         np.where(pc >= 3, 1.0, 0.0)))
    age = df["age"]
    m += np.where(age < 25, 0.25, np.where(age > 65, 0.1, 0.0))
    m += 0.15 * df["traffic_violations"]
    m += np.where(df["annual_mileage"] > 30000, 0.2, 0.0)
    m += np.where(df["credit_score"] < 550, 0.15, 0.0)
    m += np.where(df["vehicle_age"] > 10, 0.15, 0.0)
    m += np.where(df["vehicle_value"] > 500000, 0.1, 0.0)
    m += np.where(df["vehicle_type"].isin(["Pickup", "SUV"]), 0.05, 0.0)
    return m.clip(0.2, 3.0).values


def render_card(horizon, cfg, prob=None):
    if prob is None:
        st.markdown(f"""
            <div style="background:#fff;border:2px dashed #c7d2e0;border-radius:14px;
                padding:26px 20px;text-align:center;min-height:250px;
                display:flex;flex-direction:column;justify-content:center;align-items:center;">
              <div style="font-size:15px;color:#4a5568;font-weight:700;
                          letter-spacing:0.6px;text-transform:uppercase;">Within {horizon}</div>
              <div style="font-size:44px;font-weight:800;color:#c7d2e0;margin:14px 0;">--</div>
              <div style="font-size:13px;color:#94a3b8;">Click <b>Predict Claim Risk</b> to generate</div>
            </div>""", unsafe_allow_html=True)
        return
    risk, risk_color = risk_level(prob)
    decision = "Claim Likely" if prob >= cfg["threshold"] else "No Claim"
    decision_color = "#c62828" if prob >= cfg["threshold"] else "#2e7d32"
    left_pct = min(max(prob * 100, 0), 100)
    st.markdown(f"""
        <div style="background:#fff;border:1px solid #e0e6ef;border-radius:14px;
            padding:24px 20px;text-align:center;box-shadow:0 3px 12px rgba(0,0,0,.06);">
          <div style="font-size:15px;color:#4a5568;font-weight:700;
                      letter-spacing:0.6px;text-transform:uppercase;">Within {horizon}</div>
          <div style="font-size:52px;font-weight:800;color:{cfg['color']};margin:10px 0 4px;">{prob:.1%}</div>
          <div style="font-size:13px;color:#666;margin-bottom:14px;">Likelihood of a claim</div>
          <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
            <span style="background:{risk_color};color:#fff;padding:4px 12px;
                border-radius:20px;font-size:12px;font-weight:600;">{risk}</span>
            <span style="background:{decision_color};color:#fff;padding:4px 12px;
                border-radius:20px;font-size:12px;font-weight:600;">{decision}</span>
          </div>
          <div style="margin-top:14px;">
            <div style="position:relative;height:20px;
                background:linear-gradient(to right,#4caf50,#ffc107,#f44336);border-radius:10px;">
              <div style="position:absolute;left:{left_pct:.2f}%;top:-5px;transform:translateX(-50%);
                  width:12px;height:30px;background:#fff;border:3px solid #1a2a4a;border-radius:6px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;
                        color:#888;margin-top:4px;"><span>0%</span><span>30%</span><span>60%</span><span>100%</span></div>
          </div>
        </div>""", unsafe_allow_html=True)


# ---------- Bulk helpers ----------
BULK_REQUIRED_COLUMNS = [
    "policy_id", "age", "age_group", "gender", "marital_status", "education_level",
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
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in BULK_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing columns: **{', '.join(missing)}**. Use the template."

    for col, mapping in {
        "urban_rural": {"Urban": 1, "Rural": 0},
        "safety_features": {"Yes": 1, "No": 0},
        "has_telematics": {"Yes": 1, "No": 0},
    }.items():
        df[col] = df[col].astype(str).str.strip().str.title().map(mapping).fillna(0).astype(int)

    numeric_cols = [
        "age", "vehicle_value", "vehicle_age", "engine_capacity", "vehicle_weight",
        "premium_amount", "deductible", "bonus_malus", "policy_duration_years",
        "years_driving", "prior_claims", "prior_claim_amount", "traffic_violations",
        "credit_score", "annual_mileage", "telematics_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, None


def run_bulk_prediction(df, severity_rate, safety_loading):
    models = MODELS
    for h, cfg in HORIZONS.items():
        if models.get(h) is None:
            raise FileNotFoundError(f"Model file `{cfg['file']}` not found for {h}.")

    out = df.copy()
    out["risk_multiplier"] = calculate_business_risk_adjustment_vector(df)

    model_input_cols = [c for c in BULK_REQUIRED_COLUMNS if c != "policy_id"]
    X = out[model_input_cols]

    severity = out["vehicle_value"].astype(float) * severity_rate

    for h, cfg in HORIZONS.items():
        base = models[h].predict_proba(X)[:, 1]
        adj = np.minimum(base * out["risk_multiplier"].values, 0.999)
        short = cfg["short"]
        out[f"{short}_probability"] = adj
        out[f"{short}_risk_level"] = [risk_level(p)[0] for p in adj]
        out[f"{short}_decision"] = np.where(adj >= cfg["threshold"], "Claim", "No Claim")
        out[f"{short}_reserve"] = (adj * severity * safety_loading).round(2)

    out["total_reserve"] = (
        out["3M_reserve"] + out["6M_reserve"] + out["12M_reserve"]
    ).round(2)

    return out


def portfolio_summary(enriched):
    rows = []
    for h in HORIZONS:
        short = HORIZONS[h]["short"]
        total = enriched[f"{short}_reserve"].sum()
        avg = enriched[f"{short}_reserve"].mean()
        high_n = (enriched[f"{short}_risk_level"] == "High Risk").sum()
        med_n = (enriched[f"{short}_risk_level"] == "Medium Risk").sum()
        low_n = (enriched[f"{short}_risk_level"] == "Low Risk").sum()
        claim_n = (enriched[f"{short}_decision"] == "Claim").sum()
        rows.append({
            "Horizon": h,
            "Policies Flagged as Claim": int(claim_n),
            "High Risk": int(high_n),
            "Medium Risk": int(med_n),
            "Low Risk": int(low_n),
            "Total Reserve (NAD)": f"{total:,.2f}",
            "Average Reserve (NAD)": f"{avg:,.2f}",
        })
    return pd.DataFrame(rows)


# ---------- Sidebar Callbacks ----------
def update_age_group():
    a = st.session_state.policyholder_age
    if a < 25: st.session_state.age_grp = "Young"
    elif a < 40: st.session_state.age_grp = "Adult"
    elif a < 60: st.session_state.age_grp = "Middle-Aged"
    else: st.session_state.age_grp = "Senior"


if 'age_grp' not in st.session_state:
    st.session_state.age_grp = "Adult"

# ---------- Sidebar ----------
st.sidebar.title("System Menu")
if st.sidebar.button("⏻ Logout", type="primary", use_container_width=True):
    st.session_state["authentication_status"] = None
    st.session_state.pop("password", None)
    st.rerun()

st.sidebar.divider(); st.sidebar.header("Policyholder")
age = st.sidebar.slider("Age", 18, 80, 35, key="policyholder_age", on_change=update_age_group)
age_group = st.sidebar.selectbox("Age Group (Auto-calculated)",
    ["Young", "Adult", "Middle-Aged", "Senior"], key="age_grp", disabled=True)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
employment_status = st.sidebar.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed", "Retired"])
income_level = st.sidebar.selectbox("Income Level", ["Low", "Medium", "High"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])
region = st.sidebar.selectbox("Region", ["Khomas", "Erongo", "Oshana", "Otjozondjupa", "Kavango", "Zambezi"])
urban_rural = st.sidebar.selectbox("Urban/Rural", ["Urban", "Rural"])

st.sidebar.divider(); st.sidebar.header("Vehicle")
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

st.sidebar.divider(); st.sidebar.header("Policy")
premium_amount = st.sidebar.number_input("Premium Amount (NAD)", 1_500, 80_000, 15_000, step=500)
deductible = st.sidebar.selectbox("Deductible (NAD)", [1000, 2500, 5000, 10000, 25000], index=2)
bonus_malus = st.sidebar.slider("Bonus-Malus", -5, 5, 0)
policy_type = st.sidebar.selectbox("Policy Type", ["Comprehensive", "Third-Party", "Third-Party Fire & Theft"])
payment_frequency = st.sidebar.selectbox("Payment Frequency", ["Monthly", "Quarterly", "Annual"])
policy_duration_years = st.sidebar.slider("Policy Duration (yrs)", 0, 20, 3)

st.sidebar.divider(); st.sidebar.header("History & Telematics")
years_driving = st.sidebar.slider("Years Driving Experience", 0, 60, 15)
prior_claims = st.sidebar.number_input("Prior Claims (Last 3 Years)", 0, 20, 0)
prior_claim_amount = st.sidebar.number_input("Prior Claim Amount (NAD)", 0, 500_000, 0, step=1000)
traffic_violations = st.sidebar.number_input("Traffic Violations", 0, 20, 0)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
annual_mileage = st.sidebar.number_input("Annual Mileage (km)", 1_000, 100_000, 20_000, step=1_000)
has_telematics = st.sidebar.selectbox("Has Telematics", ["Yes", "No"])
telematics_score = st.sidebar.slider("Telematics Score", 0, 100, 50)

# ---------- Build single input ----------
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

# Silent reset on input change
current_fp = hash(tuple(input_df.iloc[0].astype(str).tolist()))
if "input_fingerprint" not in st.session_state:
    st.session_state["input_fingerprint"] = current_fp
if "last_results" not in st.session_state:
    st.session_state["last_results"] = {}
if st.session_state["input_fingerprint"] != current_fp:
    st.session_state["input_fingerprint"] = current_fp
    st.session_state["last_results"] = {}

# ---------- Header ----------
st.title("Insurance Claim Risk Prediction")
st.markdown("<h4 style='color:#4a5568;margin-top:-12px;font-weight:500;'>"
            "by Kennedy N. Hauwanga</h4>", unsafe_allow_html=True)
st.caption("University of Namibia research thesis | Student No. 218208399")
st.divider()

tab_single, tab_bulk = st.tabs(["Single Policyholder", "Bulk Policy File"])

# ============================================================
# TAB 1 - Single
# ============================================================
with tab_single:
    st.subheader("Generate Prediction")
    st.write("Adjust the policyholder details in the sidebar, then click "
             "**Predict Claim Risk**.")
    predict_clicked = st.button("Predict Claim Risk", type="primary", use_container_width=True)

    if predict_clicked:
        rmul = calculate_business_risk_adjustment(
            age=age, gender=gender, years_driving=years_driving,
            prior_claims=prior_claims, violations=traffic_violations,
            mileage=annual_mileage, credit_score=credit_score,
            urban_rural=urban_rural, vehicle_age=vehicle_age,
            vehicle_value=vehicle_value, vehicle_type=vehicle_type,
        )
        new = {}
        for h, cfg in HORIZONS.items():
            if MODELS.get(h) is None:
                st.warning(f"Model file `{cfg['file']}` not found. Skipping {h}."); continue
            try:
                base = float(MODELS[h].predict_proba(input_df)[0, 1])
                adj = min(base * rmul, 0.999)
                new[h] = adj
                decision = "Claim" if adj >= cfg["threshold"] else "No Claim"
                log_prediction(age=age, gender=gender, vehicle_val=vehicle_value,
                               premium=premium_amount, horizon=h,
                               decision=decision, prob=adj)
            except Exception as e:
                st.warning(f"Prediction failed for `{h}`: {e}")
        if new:
            st.session_state["last_results"] = new
            st.success("Prediction results have been logged to the database.")

    results = st.session_state["last_results"]
    st.markdown("### Predicted Claim Likelihood")
    c1, c2, c3 = st.columns(3)
    with c1: render_card("3 months", HORIZONS["3 months"], results.get("3 months"))
    with c2: render_card("6 months", HORIZONS["6 months"], results.get("6 months"))
    with c3: render_card("12 months", HORIZONS["12 months"], results.get("12 months"))

    if results:
        st.divider(); st.subheader("Summary")
        summary = pd.DataFrame({
            "3 months": [f"{results.get('3 months',0):.2%}",
                         risk_level(results.get('3 months',0))[0],
                         "Claim" if results.get('3 months',0)>=HORIZONS['3 months']['threshold'] else "No Claim",
                         f"{HORIZONS['3 months']['threshold']:.2f}"],
            "6 months": [f"{results.get('6 months',0):.2%}",
                         risk_level(results.get('6 months',0))[0],
                         "Claim" if results.get('6 months',0)>=HORIZONS['6 months']['threshold'] else "No Claim",
                         f"{HORIZONS['6 months']['threshold']:.2f}"],
            "12 months": [f"{results.get('12 months',0):.2%}",
                          risk_level(results.get('12 months',0))[0],
                          "Claim" if results.get('12 months',0)>=HORIZONS['12 months']['threshold'] else "No Claim",
                          f"{HORIZONS['12 months']['threshold']:.2f}"],
        }, index=["Probability","Risk Level","Decision","Threshold"])
        st.dataframe(summary, use_container_width=True)

# ============================================================
# TAB 2 - Bulk
# ============================================================
with tab_bulk:
    st.subheader("Bulk Policy Risk & Reserve Recommendation")

    template_df = pd.DataFrame([{
        "policy_id": "P0001", "age": 35, "age_group": "Adult", "gender": "Male",
        "marital_status": "Married", "education_level": "Bachelor",
        "employment_status": "Employed", "income_level": "Medium",
        "home_ownership": "Own", "region": "Khomas", "urban_rural": "Urban",
        "vehicle_value": 250000, "vehicle_age": 5, "vehicle_age_group": "Recent",
        "engine_capacity": 2000, "vehicle_weight": 1500,
        "vehicle_type": "Sedan", "vehicle_brand": "Toyota",
        "fuel_type": "Petrol", "transmission": "Manual",
        "vehicle_usage": "Private", "safety_features": "Yes",
        "premium_amount": 15000, "deductible": 5000, "bonus_malus": 0,
        "policy_type": "Comprehensive", "payment_frequency": "Monthly",
        "policy_duration_years": 3, "years_driving": 15,
        "prior_claims": 0, "prior_claim_amount": 0,
        "traffic_violations": 0, "credit_score": 650, "annual_mileage": 20000,
        "has_telematics": "Yes", "telematics_score": 50,
    }])
    with st.expander("📄 Download template (required columns)"):
        st.dataframe(template_df, use_container_width=True, hide_index=True)
        st.download_button("Download template (CSV)",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="bulk_policy_template.csv", mime="text/csv")
        st.download_button("Download template (Excel)",
            data=convert_df_to_excel(template_df, "Template"),
            file_name="bulk_policy_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    uploaded = st.file_uploader("Upload policy file (CSV or Excel)",
                                type=["csv","xlsx","xls"], key="bulk_upload")

    st.markdown("#### Reserve Assumptions")
    ca, cb = st.columns(2)
    with ca:
        severity_rate = st.slider("Expected Claim Severity (% of vehicle value)",
            min_value=0.05, max_value=1.00, value=0.30, step=0.05)
    with cb:
        safety_loading = st.slider("Safety Loading Multiplier",
            min_value=1.00, max_value=2.00, value=1.25, step=0.05)

    run_bulk = st.button("Run Bulk Prediction", type="primary",
                         use_container_width=True, disabled=uploaded is None)

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
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
            elif run_bulk:
                try:
                    with st.spinner("Running predictions for all policies..."):
                        enriched = run_bulk_prediction(prepared, severity_rate, safety_loading)
                    st.session_state["bulk_results"] = enriched
                    st.success(f"Bulk prediction complete for **{len(enriched)}** policies.")
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Bulk prediction failed: {e}")

    if "bulk_results" in st.session_state:
        enriched = st.session_state["bulk_results"]
        st.divider()

        st.subheader("🏛 Portfolio Reserve — Total Amount the Insurer Should Hold")
        total_3m = enriched["3M_reserve"].sum()
        total_6m = enriched["6M_reserve"].sum()
        total_12m = enriched["12M_reserve"].sum()
        grand_total = enriched["total_reserve"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Policies", f"{len(enriched):,}")
        m2.metric("Reserve @ 3M", f"NAD {total_3m:,.0f}")
        m3.metric("Reserve @ 6M", f"NAD {total_6m:,.0f}")
        m4.metric("Reserve @ 12M", f"NAD {total_12m:,.0f}")
        st.metric("💰 Grand Total Reserve (3M + 6M + 12M combined)",
                  f"NAD {grand_total:,.2f}")

        st.markdown("##### Breakdown by Horizon")
        st.dataframe(portfolio_summary(enriched), use_container_width=True, hide_index=True)

        st.markdown("##### Book Composition by Risk Level (12-Month Horizon)")
        comp = (
            enriched.groupby("12M_risk_level")
            .agg(policies=("policy_id", "count"),
                 total_reserve=("12M_reserve", "sum"))
            .reset_index()
            .rename(columns={"12M_risk_level": "Risk Level",
                             "policies": "Policies",
                             "total_reserve": "12M Reserve (NAD)"})
        )
        comp["12M Reserve (NAD)"] = comp["12M Reserve (NAD)"].map(lambda v: f"{v:,.2f}")
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📋 Per-Policy Risk and Reserve")

        show_cols = [
            "policy_id",
            "3M_probability", "3M_risk_level", "3M_reserve",
            "6M_probability", "6M_risk_level", "6M_reserve",
            "12M_probability", "12M_risk_level", "12M_reserve",
            "total_reserve",
        ]
        show_cols = [c for c in show_cols if c in enriched.columns]
        view = enriched[show_cols].copy()

        for c in ["3M_probability", "6M_probability", "12M_probability"]:
            if c in view.columns:
                view[c] = view[c].map(lambda v: f"{v:.2%}")
        for c in ["3M_reserve", "6M_reserve", "12M_reserve", "total_reserve"]:
            if c in view.columns:
                view[c] = view[c].map(lambda v: f"{v:,.2f}")

        view = view.rename(columns={
            "policy_id": "Policy",
            "3M_probability": "3M Prob", "3M_risk_level": "3M Risk", "3M_reserve": "3M Reserve",
            "6M_probability": "6M Prob", "6M_risk_level": "6M Risk", "6M_reserve": "6M Reserve",
            "12M_probability": "12M Prob", "12M_risk_level": "12M Risk", "12M_reserve": "12M Reserve",
            "total_reserve": "Total Reserve",
        })
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.markdown("#### ⬇ Downloads")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button(
                "Download full results (Excel)",
                data=convert_df_to_excel(enriched, "Bulk_Predictions"),
                file_name=f"bulk_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary", use_container_width=True)
        with dc2:
            st.download_button(
                "Download full results (CSV)",
                data=enriched.to_csv(index=False).encode("utf-8"),
                file_name=f"bulk_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True)

        if st.button("Clear Bulk Results"):
            del st.session_state["bulk_results"]
            st.rerun()

# ---------- Reference table ----------
st.divider()
st.subheader("Reference Performance")
st.dataframe(pd.DataFrame({
    "Model": ["XGBoost"], "Accuracy": [0.7510], "Precision": [0.7500],
    "Recall": [1.0000], "F1": [0.8600],
    "AUC-ROC": [0.6776], "AUC-PR": [0.8585],
}), use_container_width=True, hide_index=True)

# ---------- Admin: Logs (paginated) ----------
st.divider()
st.subheader("System Administration (Database Logs)")
df_logs = get_all_predictions()
if not df_logs.empty:
    total = len(df_logs)
    st.write(f"Total logged predictions in database: **{total}**")

    PAGE_SIZE = 50
    pages = max(1, (total - 1) // PAGE_SIZE + 1)
    page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    start = (page - 1) * PAGE_SIZE
    st.dataframe(df_logs.iloc[start:start + PAGE_SIZE],
                 use_container_width=True, hide_index=True)

    st.download_button("Click here to Download Prediction Logs (Excel)",
        data=convert_df_to_excel(df_logs),
        file_name=f"prediction_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary")
else:
    st.info("No predictions logged yet. Click **Predict Claim Risk** to generate the first entry.")
