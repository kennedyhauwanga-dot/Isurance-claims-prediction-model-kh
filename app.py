"""
app.py
Insurance Claim Risk Prediction - XGBoost prototype
Author: Kennedy N. Hauwanga (Student No. 218208399)

Tabs:
  1. Single Policyholder  (Predicts CLAIM PROBABILITY for 3M, 6M, 12M)
  2. Bulk Policy File     (Predicts CLAIM PROBABILITY + RESERVE amounts for the portfolio)

Risk adjustments:
  - Rural areas strongly reduce frequency and severity for both OD and TPL.
  - Urban areas increase frequency and severity, especially for TPL.
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

OD_CLAIM_SHARE  = 0.65
TPL_CLAIM_SHARE = 0.35

# ---------- Region & Town data (official) ----------
REGIONS = [
    "Erongo", "Hardap", "ǁKaras", "Kavango East", "Kavango West",
    "Khomas", "Kunene", "Ohangwena", "Omaheke", "Omusati",
    "Oshana", "Oshikoto", "Otjozondjupa", "Zambezi"
]

TOWNS_BY_REGION = {
    "Erongo": ["Arandis", "Henties Bay", "Karibib", "Omaruru", "Swakopmund", "Usakos", "Walvis Bay"],
    "Hardap": ["Aranos", "Mariental", "Rehoboth"],
    "ǁKaras": ["Karasburg", "Keetmanshoop", "Lüderitz", "Oranjemund"],
    "Kavango East": ["Rundu"],
    "Kavango West": ["Nkurenkuru"],
    "Khomas": ["Windhoek"],
    "Kunene": ["Khorixas", "Opuwo", "Outjo"],
    "Ohangwena": ["Eenhana", "Helao Nafidi"],
    "Omaheke": ["Gobabis"],
    "Omusati": ["Okahao", "Oshikuku", "Outapi", "Ruacana"],
    "Oshana": ["Ondangwa", "Ongwediva", "Oshakati"],
    "Oshikoto": ["Omuthiya", "Oniipa", "Tsumeb"],
    "Otjozondjupa": ["Grootfontein", "Okahandja", "Okakarara", "Otavi", "Otjiwarongo"],
    "Zambezi": ["Katima Mulilo"],
}

# ============================================================
# SCHEMA + DEFAULTS
# ============================================================
BULK_REQUIRED_COLUMNS = [
    "policy_id", "age", "age_group", "gender", "marital_status", "education_level",
    "employment_status", "income_level", "home_ownership", "region", "town",
    "urban_rural", "vehicle_value", "vehicle_age", "vehicle_age_group",
    "engine_capacity", "vehicle_weight", "vehicle_type", "vehicle_brand",
    "fuel_type", "transmission", "vehicle_usage", "safety_features",
    "premium_amount", "deductible", "bonus_malus", "policy_type",
    "payment_frequency", "policy_duration_years", "years_driving",
    "prior_claims", "prior_claim_amount", "traffic_violations",
    "credit_score", "annual_mileage", "has_telematics", "telematics_score",
]

MODEL_FEATURES = [c for c in BULK_REQUIRED_COLUMNS if c not in ("policy_id", "town")]

MODEL_DEFAULTS = {
    "age": 35, "age_group": "Adult", "gender": "Male",
    "marital_status": "Married", "education_level": "Bachelor",
    "employment_status": "Employed", "income_level": "Medium",
    "home_ownership": "Own", "region": "Khomas",
    "urban_rural": "Urban",
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
}

BINARY_MAPS = {
    "urban_rural": {"Urban": 1, "Rural": 0},
    "safety_features": {"Yes": 1, "No": 0},
    "has_telematics": {"Yes": 1, "No": 0},
}

NUMERIC_COLUMNS = [
    "age", "vehicle_value", "vehicle_age", "engine_capacity", "vehicle_weight",
    "premium_amount", "deductible", "bonus_malus", "policy_duration_years",
    "years_driving", "prior_claims", "prior_claim_amount", "traffic_violations",
    "credit_score", "annual_mileage", "telematics_score",
]

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
def risk_level(prob, threshold):
    if prob < threshold:
        return "Low Risk", "#2e7d32"
    if prob < 2 * threshold:
        return "Medium Risk", "#f9a825"
    return "High Risk", "#c62828"


# ============================================================
# CROWDING / DENSITY
# ============================================================
def compute_population_density(area_km2, population):
    try:
        a = float(area_km2)
        p = float(population)
        if a <= 0 or p <= 0:
            return None
        return p / a
    except (TypeError, ValueError):
        return None


def density_band(density):
    if density is None:
        return "Unknown"
    if density < 500:    return "Sparse / Rural"
    if density < 1500:   return "Suburban"
    if density < 3000:   return "Urban"
    if density < 6000:   return "Dense Urban"
    if density < 12000:  return "Very Dense"
    return "Hyper-Dense"


def density_od_delta(density):
    if density is None:
        return None
    if density < 500:    return -0.15
    if density < 1500:   return -0.05
    if density < 3000:   return  0.05
    if density < 6000:   return  0.15
    if density < 12000:  return  0.25
    return 0.35


def density_tpl_delta(density):
    if density is None:
        return None
    if density < 500:    return -0.30
    if density < 1500:   return -0.10
    if density < 3000:   return  0.10
    if density < 6000:   return  0.25
    if density < 12000:  return  0.40
    return 0.60


def density_delta_vector(series, kind="tpl"):
    dens = pd.to_numeric(series, errors="coerce")
    if kind == "od":
        breaks = [-np.inf, 500, 1500, 3000, 6000, 12000, np.inf]
        values = [-0.15, -0.05, 0.05, 0.15, 0.25, 0.35]
    else:
        breaks = [-np.inf, 500, 1500, 3000, 6000, 12000, np.inf]
        values = [-0.30, -0.10, 0.10, 0.25, 0.40, 0.60]
    return np.select(
        [(dens >= breaks[i]) & (dens < breaks[i + 1]) for i in range(len(values))],
        values,
        default=np.nan,
    )


# ============================================================
# OWN DAMAGE — frequency & severity
# ============================================================
def calculate_od_frequency_multiplier(age, gender, years_driving, prior_claims, violations,
                                      mileage, credit_score, urban_rural, vehicle_age,
                                      safety_features, density=None):
    m = 1.0
    if str(gender).strip().title() == "Female":
        m -= 0.10

    # Strong Urban vs Rural split
    if density is not None:
        m += density_od_delta(density)
    else:
        m += 0.15 if urban_rural == "Urban" else -0.20

    if vehicle_age > 10:      m += 0.12
    elif vehicle_age > 5:     m += 0.05
    if safety_features == 0:  m += 0.10

    if prior_claims == 1:   m += 0.25
    elif prior_claims == 2: m += 0.45
    elif prior_claims >= 3: m += 0.70

    if age < 25:   m += 0.15
    elif age > 65: m += 0.10

    if years_driving < 3:    m += 0.15
    elif years_driving < 8:  m += 0.05
    elif years_driving > 15: m -= 0.10

    if mileage > 30000:    m += 0.12
    if credit_score < 550: m += 0.08
    if violations > 0:     m += 0.05 * violations
    return max(0.50, min(m, 1.90))


def calculate_od_frequency_multiplier_vector(df):
    m = pd.Series(1.0, index=df.index)
    is_female = df["gender"].astype(str).str.strip().str.title() == "Female"
    m -= np.where(is_female, 0.10, 0.0)

    urban_fallback_od = np.where(df["urban_rural"] == 1, 0.15, -0.20)
    if "population_density" in df.columns:
        dens_delta = density_delta_vector(df["population_density"], kind="od")
        dens_delta = np.where(pd.isna(dens_delta), urban_fallback_od, dens_delta)
    else:
        dens_delta = urban_fallback_od
    m += dens_delta

    va = df["vehicle_age"]
    m += np.where(va > 10, 0.12, np.where(va > 5, 0.05, 0.0))
    m += np.where(df["safety_features"] == 0, 0.10, 0.0)

    pc = df["prior_claims"]
    m += np.where(pc == 1, 0.25,
         np.where(pc == 2, 0.45,
         np.where(pc >= 3, 0.70, 0.0)))

    age = df["age"]
    m += np.where(age < 25, 0.15, np.where(age > 65, 0.10, 0.0))
    yd = df["years_driving"]
    m += np.where(yd < 3, 0.15,
         np.where(yd < 8, 0.05,
         np.where(yd > 15, -0.10, 0.0)))

    m += np.where(df["annual_mileage"] > 30000, 0.12, 0.0)
    m += np.where(df["credit_score"] < 550, 0.08, 0.0)
    m += 0.05 * df["traffic_violations"]
    return m.clip(0.50, 1.90).values


def calculate_od_severity_loading(vehicle_value, vehicle_type, vehicle_age, gender):
    s = 1.0
    if str(gender).strip().title() == "Female":
        s *= 0.90

    if vehicle_value > 750_000:   s += 0.50
    elif vehicle_value > 500_000: s += 0.35
    elif vehicle_value > 300_000: s += 0.20
    elif vehicle_value > 150_000: s += 0.10

    if vehicle_type in ("SUV", "Pickup"): s += 0.10

    if vehicle_age > 15:   s -= 0.20
    elif vehicle_age > 10: s -= 0.10
    return max(0.50, s)


def calculate_od_severity_loading_vector(df):
    s = pd.Series(1.0, index=df.index)
    is_female = df["gender"].astype(str).str.strip().str.title() == "Female"
    s *= np.where(is_female, 0.90, 1.00)

    vv = df["vehicle_value"]
    s += np.where(vv > 750_000, 0.50,
         np.where(vv > 500_000, 0.35,
         np.where(vv > 300_000, 0.20,
         np.where(vv > 150_000, 0.10, 0.0))))
    s += np.where(df["vehicle_type"].isin(["SUV", "Pickup"]), 0.10, 0.0)
    va = df["vehicle_age"]
    s -= np.where(va > 15, 0.20, np.where(va > 10, 0.10, 0.0))
    return s.clip(lower=0.50).values


# ============================================================
# THIRD-PARTY LIABILITY — frequency & severity
# ============================================================
def calculate_tpl_frequency_multiplier(age, gender, years_driving, prior_claims, violations,
                                       mileage, credit_score, urban_rural, density=None):
    m = 1.0
    if str(gender).strip().title() == "Female":
        m -= 0.10

    # Very strong Urban vs Rural split for TPL
    if density is not None:
        m += density_tpl_delta(density)
    else:
        m += 0.30 if urban_rural == "Urban" else -0.30

    if prior_claims == 1:   m += 0.30
    elif prior_claims == 2: m += 0.55
    elif prior_claims >= 3: m += 0.80

    if age < 25:   m += 0.25
    elif age > 65: m += 0.08

    if years_driving < 3:    m += 0.30
    elif years_driving < 8:  m += 0.10
    elif years_driving > 15: m -= 0.15

    m += 0.15 * violations

    if mileage > 30000:    m += 0.10
    if credit_score < 550: m += 0.10
    return max(0.40, min(m, 2.20))


def calculate_tpl_frequency_multiplier_vector(df):
    m = pd.Series(1.0, index=df.index)
    is_female = df["gender"].astype(str).str.strip().str.title() == "Female"
    m -= np.where(is_female, 0.10, 0.0)

    urban_fallback_tpl = np.where(df["urban_rural"] == 1, 0.30, -0.30)
    if "population_density" in df.columns:
        dens_delta = density_delta_vector(df["population_density"], kind="tpl")
        dens_delta = np.where(pd.isna(dens_delta), urban_fallback_tpl, dens_delta)
    else:
        dens_delta = urban_fallback_tpl
    m += dens_delta

    pc = df["prior_claims"]
    m += np.where(pc == 1, 0.30,
         np.where(pc == 2, 0.55,
         np.where(pc >= 3, 0.80, 0.0)))

    age = df["age"]
    m += np.where(age < 25, 0.25, np.where(age > 65, 0.08, 0.0))
    yd = df["years_driving"]
    m += np.where(yd < 3, 0.30,
         np.where(yd < 8, 0.10,
         np.where(yd > 15, -0.15, 0.0)))

    m += 0.15 * df["traffic_violations"]
    m += np.where(df["annual_mileage"] > 30000, 0.10, 0.0)
    m += np.where(df["credit_score"] < 550, 0.10, 0.0)
    return m.clip(0.40, 2.20).values


def calculate_tpl_severity_loading(vehicle_type, urban_rural, region, gender):
    s = 1.0
    if str(gender).strip().title() == "Female":
        s *= 0.88

    # Rural has lower severity (less traffic, cheaper legal/medical costs)
    s += 0.20 if urban_rural == "Urban" else -0.20

    if vehicle_type in ("SUV", "Pickup"): s += 0.15
    if vehicle_type == "Minivan":         s += 0.10

    if region in ("Khomas", "Erongo"):    s += 0.10
    return max(0.50, s)


def calculate_tpl_severity_loading_vector(df):
    s = pd.Series(1.0, index=df.index)
    is_female = df["gender"].astype(str).str.strip().str.title() == "Female"
    s *= np.where(is_female, 0.88, 1.00)

    s += np.where(df["urban_rural"] == 1, 0.20, -0.20)
    s += np.where(df["vehicle_type"].isin(["SUV", "Pickup"]), 0.15,
         np.where(df["vehicle_type"] == "Minivan", 0.10, 0.0))
    s += np.where(df["region"].isin(["Khomas", "Erongo"]), 0.10, 0.0)
    return s.clip(lower=0.50).values


# ---------- Card render ----------
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
    risk, risk_color = risk_level(prob, cfg["threshold"])
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


# ============================================================
# BULK INGESTION — TOLERANT OF MISSING COLUMNS
# ============================================================
def prepare_bulk_input(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if df.empty:
        return None, "Uploaded file contains no rows.", []

    if "policy_id" not in df.columns:
        df["policy_id"] = [f"P{i+1:05d}" for i in range(len(df))]
    else:
        df["policy_id"] = df["policy_id"].astype(str)

    filled = []
    for col, default in MODEL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            filled.append(col)

    for col, mapping in BINARY_MAPS.items():
        default_str = str(MODEL_DEFAULTS[col]).strip().title()
        default_val = mapping.get(default_str, 0)
        df[col] = (df[col].astype(str).str.strip().str.title()
                   .map(mapping).fillna(default_val).astype(int))

    for col in NUMERIC_COLUMNS:
        default_val = MODEL_DEFAULTS.get(col, 0)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val)

    area = (pd.to_numeric(df["area_km2"], errors="coerce")
            if "area_km2" in df.columns else pd.Series([np.nan] * len(df)))
    pop  = (pd.to_numeric(df["population"], errors="coerce")
            if "population" in df.columns else pd.Series([np.nan] * len(df)))
    computed_density = np.where((area > 0) & (pop > 0), pop / area, np.nan)

    if "population_density" in df.columns:
        existing = pd.to_numeric(df["population_density"], errors="coerce")
        df["population_density"] = np.where(pd.isna(computed_density),
                                            existing, computed_density)
    else:
        df["population_density"] = computed_density

    if "town" not in df.columns:
        df["town"] = ""
        filled.append("town")

    if "region" not in df.columns:
        df["region"] = MODEL_DEFAULTS["region"]
        filled.append("region")

    return df, None, filled


def run_bulk_prediction(df, od_severity_rate, tpl_base_severity, safety_loading):
    models = MODELS
    for h, cfg in HORIZONS.items():
        if models.get(h) is None:
            raise FileNotFoundError(f"Model file `{cfg['file']}` not found for {h}.")

    out = df.copy()

    out["od_frequency_mult"]  = calculate_od_frequency_multiplier_vector(df)
    out["tpl_frequency_mult"] = calculate_tpl_frequency_multiplier_vector(df)
    out["od_severity_load"]   = calculate_od_severity_loading_vector(df)
    out["tpl_severity_load"]  = calculate_tpl_severity_loading_vector(df)

    out["crowding_band"] = out["population_density"].apply(
        lambda d: density_band(None if pd.isna(d) else d)
    )

    od_severity_nad  = (out["vehicle_value"].astype(float)
                        * od_severity_rate
                        * out["od_severity_load"])
    tpl_severity_nad = tpl_base_severity * out["tpl_severity_load"]

    X = out[MODEL_FEATURES]

    for h, cfg in HORIZONS.items():
        short = cfg["short"]
        base = models[h].predict_proba(X)[:, 1]

        od_prob  = np.minimum(base * OD_CLAIM_SHARE  * out["od_frequency_mult"].values,  0.95)
        tpl_prob = np.minimum(base * TPL_CLAIM_SHARE * out["tpl_frequency_mult"].values, 0.90)

        od_reserve  = (od_prob  * od_severity_nad  * safety_loading).round(2)
        tpl_reserve = (tpl_prob * tpl_severity_nad * safety_loading).round(2)

        combined_prob = np.minimum(od_prob + tpl_prob, 0.95)

        out[f"{short}_od_prob"]      = od_prob
        out[f"{short}_od_reserve"]   = od_reserve
        out[f"{short}_tpl_prob"]     = tpl_prob
        out[f"{short}_tpl_reserve"]  = tpl_reserve
        out[f"{short}_total_prob"]   = combined_prob
        out[f"{short}_total_reserve"] = (od_reserve + tpl_reserve).round(2)
        out[f"{short}_risk_level"]   = [risk_level(p, cfg["threshold"])[0] for p in combined_prob]
        out[f"{short}_decision"]     = np.where(
            combined_prob >= cfg["threshold"], "Claim", "No Claim")

    out["total_reserve"] = out["12M_total_reserve"].round(2)
    return out


def portfolio_summary(enriched):
    rows = []
    for h in HORIZONS:
        short = HORIZONS[h]["short"]
        od_res  = enriched[f"{short}_od_reserve"].sum()
        tpl_res = enriched[f"{short}_tpl_reserve"].sum()
        total   = enriched[f"{short}_total_reserve"].sum()
        rows.append({
            "Horizon": h,
            "OD Reserve (NAD)":  f"{od_res:,.2f}",
            "TPL Reserve (NAD)": f"{tpl_res:,.2f}",
            "Combined Reserve (NAD)": f"{total:,.2f}",
            "Policies Flagged": int((enriched[f"{short}_decision"] == "Claim").sum()),
        })
    return pd.DataFrame(rows)


# ---------- Sidebar Callbacks ----------
def update_age_group():
    a = st.session_state.policyholder_age
    if a < 25:   st.session_state.age_grp = "Young"
    elif a < 40: st.session_state.age_grp = "Adult"
    elif a < 60: st.session_state.age_grp = "Middle-Aged"
    else:        st.session_state.age_grp = "Senior"


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
    ["
