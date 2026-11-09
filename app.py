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

    # --- PROFESSIONAL LOGIN HEADER ---
    st.title("Insurance Claim Risk Prediction System")
    st.markdown("**Secure Access Portal** | University of Namibia Research Thesis by Kennedy N. Hauwanga")
    st.divider()

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

    # DAMPENED PRIOR CLAIMS
    if prior_claims == 1:   m += 0.08
    elif prior_claims == 2: m += 0.15
    elif prior_claims >= 3: m += 0.25

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

    # DAMPENED PRIOR CLAIMS
    pc = df["prior_claims"]
    m += np.where(pc == 1, 0.08,
         np.where(pc == 2, 0.15,
         np.where(pc >= 3, 0.25, 0.0)))

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

    # DAMPENED PRIOR CLAIMS
    if prior_claims == 1:   m += 0.10
    elif prior_claims == 2: m += 0.20
    elif prior_claims >= 3: m += 0.35

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

    # DAMPENED PRIOR CLAIMS
    pc = df["prior_claims"]
    m += np.where(pc == 1, 0.10,
         np.where(pc == 2, 0.20,
         np.where(pc >= 3, 0.35, 0.0)))

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
    ["Young", "Adult", "Middle-Aged", "Senior"], key="age_grp", disabled=True)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
employment_status = st.sidebar.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed", "Retired"])
income_level = st.sidebar.selectbox("Income Level", ["Low", "Medium", "High"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])

# --- Region & Town (all 14 regions + all towns) ---
region = st.sidebar.selectbox("Region", REGIONS)
town = st.sidebar.selectbox("Town", TOWNS_BY_REGION[region])

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
    "home_ownership": home_ownership, "region": region, "town": town,
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
# TAB 1 - Single (PROBABILITIES ONLY, NO RESERVES)
# ============================================================
with tab_single:
    st.subheader("Generate Prediction")
    predict_clicked = st.button("Predict Claim Risk", type="primary", use_container_width=True)

    if predict_clicked:
        # Pass density=None to fall back cleanly to Urban/Rural
        od_freq = calculate_od_frequency_multiplier(
            age=age, gender=gender, years_driving=years_driving,
            prior_claims=prior_claims, violations=traffic_violations,
            mileage=annual_mileage, credit_score=credit_score,
            urban_rural=urban_rural, vehicle_age=vehicle_age,
            safety_features=1 if safety_features == "Yes" else 0,
            density=None,
        )
        tpl_freq = calculate_tpl_frequency_multiplier(
            age=age, gender=gender, years_driving=years_driving,
            prior_claims=prior_claims, violations=traffic_violations,
            mileage=annual_mileage, credit_score=credit_score,
            urban_rural=urban_rural,
            density=None,
        )

        new = {}
        for h, cfg in HORIZONS.items():
            if MODELS.get(h) is None:
                st.warning(f"Model file `{cfg['file']}` not found. Skipping {h}.")
                continue
            try:
                base = float(MODELS[h].predict_proba(input_df[MODEL_FEATURES])[0, 1])
                od_prob  = min(base * OD_CLAIM_SHARE  * od_freq,  0.95)
                tpl_prob = min(base * TPL_CLAIM_SHARE * tpl_freq, 0.90)
                combined = min(od_prob + tpl_prob, 0.95)

                new[h] = {
                    "od_prob": od_prob, "tpl_prob": tpl_prob,
                    "combined": combined,
                    "od_freq": od_freq, "tpl_freq": tpl_freq,
                }
                decision = "Claim" if combined >= cfg["threshold"] else "No Claim"
                log_prediction(age=age, gender=gender, vehicle_val=vehicle_value,
                               premium=premium_amount, horizon=h,
                               decision=decision, prob=combined)
            except Exception as e:
                st.warning(f"Prediction failed for `{h}`: {e}")

        if new:
            st.session_state["last_results"] = new
            st.success("Prediction results have been logged to the database.")

    results = st.session_state["last_results"]
    st.markdown("### Predicted Claim Likelihood")
    c1, c2, c3 = st.columns(3)
    with c1: render_card("3 months",  HORIZONS["3 months"],  results.get("3 months",  {}).get("combined"))
    with c2: render_card("6 months",  HORIZONS["6 months"],  results.get("6 months",  {}).get("combined"))
    with c3: render_card("12 months", HORIZONS["12 months"], results.get("12 months", {}).get("combined"))

    if results:
        st.divider()
        st.subheader("Risk Breakdown — Own Damage vs Third-Party Liability")
        rows = []
        for h in HORIZONS:
            r = results.get(h)
            if not r:
                continue
            
            # CLEAN TABLE: Removed the OD/TPL Freq multipliers completely
            rows.append({
                "Horizon": h,
                "OD Prob":    f"{r['od_prob']:.2%}",
                "TPL Prob":   f"{r['tpl_prob']:.2%}",
                "Combined Prob": f"{r['combined']:.2%}",
            })
                
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# TAB 2 - Bulk (PROBABILITIES + RESERVES)
# ============================================================
with tab_bulk:
    st.subheader("Bulk Policy Risk & Reserve Recommendation")

    template_df = pd.DataFrame([{
        "policy_id": "P0001", "age": 35, "age_group": "Adult", "gender": "Male",
        "marital_status": "Married", "education_level": "Bachelor",
        "employment_status": "Employed", "income_level": "Medium",
        "home_ownership": "Own", "region": "Khomas", "town": "Windhoek",
        "urban_rural": "Urban", "area_km2": 10.0, "population": 30000,
        "population_density": 3000.0, "vehicle_value": 250000,
        "vehicle_age": 5, "vehicle_age_group": "Recent",
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
    with st.expander("📄 Download reference template (all columns)"):
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
                                type=["csv", "xlsx", "xls"], key="bulk_upload")

    st.markdown("#### Reserve Assumptions (Active for Bulk Calculation)")
    ca, cb, cc = st.columns(3)
    with ca:
        bulk_od_rate = st.slider("OD Severity Rate (% of vehicle value)",
            min_value=0.05, max_value=1.00, value=0.30, step=0.05)
    with cb:
        bulk_tpl_base = st.number_input("TPL Base Severity (NAD per claim)",
            min_value=10_000, max_value=1_000_000, value=150_000, step=10_000)
    with cc:
        bulk_safety = st.slider("Safety Loading Multiplier",
            min_value=1.00, max_value=2.00, value=1.25, step=0.05)

    run_bulk = st.button("Run Bulk Prediction & Calculate Reserves", type="primary",
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

            prepared, err, filled = prepare_bulk_input(raw)
            if err:
                st.error(err)
            else:
                if filled:
                    st.warning(
                        f"**Auto-filled {len(filled)} column(s)** with defaults: "
                        f"`{', '.join(filled)}` — predictions will use the "
                        f"training-median defaults for these features."
                    )
                else:
                    st.success("All expected columns present in the upload.")

                if run_bulk:
                    try:
                        with st.spinner("Running predictions and calculating reserves..."):
                            enriched = run_bulk_prediction(prepared, bulk_od_rate,
                                                           bulk_tpl_base, bulk_safety)
                        st.session_state["bulk_results"] = enriched
                        st.success(f"Bulk prediction complete for **{len(enriched)}** policies.")
                    except FileNotFoundError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Bulk prediction failed: {e}")

    if "bulk_results" in st.session_state:
        enriched = st.session_state["bulk_results"]
        st.divider()

        st.subheader("Portfolio Reserve Ladder")
        st.caption(
            "Own Damage and Third-Party Liability reserves are calculated separately. "
            "3M, 6M and 12M horizons are cumulative and must NOT be summed. "
            "The 12M combined reserve is the total amount the insurer should hold."
        )

        t3_od  = enriched["3M_od_reserve"].sum();  t3_tpl  = enriched["3M_tpl_reserve"].sum()
        t6_od  = enriched["6M_od_reserve"].sum();  t6_tpl  = enriched["6M_tpl_reserve"].sum()
        t12_od = enriched["12M_od_reserve"].sum(); t12_tpl = enriched["12M_tpl_reserve"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Policies", f"{len(enriched):,}")
        m2.metric("3M Reserve (OD+TPL)",  f"NAD {t3_od + t3_tpl:,.0f}")
        m3.metric("6M Reserve (OD+TPL)",  f"NAD {t6_od + t6_tpl:,.0f}")
        m4.metric("12M Reserve (OD+TPL)", f"NAD {t12_od + t12_tpl:,.0f}",
                  help="Cumulative expected loss over 12 months — the solvency figure.")

        st.markdown("##### Reserve breakdown by horizon")
        st.dataframe(portfolio_summary(enriched), use_container_width=True, hide_index=True)

        st.markdown("##### 12-Month reserve: OD vs TPL")
        comp = pd.DataFrame({
            "Risk type": ["Own Damage", "Third-Party Liability", "Combined"],
            "12M Reserve (NAD)": [
                f"{t12_od:,.2f}", f"{t12_tpl:,.2f}", f"{t12_od + t12_tpl:,.2f}",
            ],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.markdown("##### Crowding composition of the book")
        crowd = (
            enriched.assign(density=pd.to_numeric(enriched.get("population_density"), errors="coerce"))
            .assign(band=lambda d: d["density"].apply(
                lambda v: density_band(None if pd.isna(v) else v)))
            .groupby("band")
            .agg(policies=("policy_id", "count"),
                 total_12m_reserve=("12M_total_reserve", "sum"))
            .reset_index()
            .rename(columns={"band": "Crowding band",
                             "policies": "Policies",
                             "total_12m_reserve": "12M Reserve (NAD)"})
        )
        crowd["12M Reserve (NAD)"] = crowd["12M Reserve (NAD)"].map(lambda v: f"{v:,.2f}")
        st.dataframe(crowd, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Per-Policy Risk and Reserve")

        show_cols = [
            "policy_id", "region", "town",
            "population_density", "crowding_band",
            "3M_od_prob", "3M_tpl_prob", "3M_od_reserve", "3M_tpl_reserve", "3M_total_reserve",
            "6M_od_prob", "6M_tpl_prob", "6M_od_reserve", "6M_tpl_reserve", "6M_total_reserve",
            "12M_od_prob", "12M_tpl_prob", "12M_od_reserve", "12M_tpl_reserve", "12M_total_reserve",
            "total_reserve",
        ]
        show_cols = [c for c in show_cols if c in enriched.columns]
        view = enriched[show_cols].copy()

        for c in ["3M_od_prob", "3M_tpl_prob", "6M_od_prob", "6M_tpl_prob", "12M_od_prob", "12M_tpl_prob"]:
            if c in view.columns:
                view[c] = view[c].map(lambda v: f"{v:.2%}")
        for c in [x for x in show_cols if "reserve" in x]:
            if c in view.columns:
                view[c] = view[c].map(lambda v: f"{v:,.2f}")
        if "population_density" in view.columns:
            view["population_density"] = view["population_density"].map(
                lambda v: "n/a" if pd.isna(v) else f"{v:,.0f}")

        view = view.rename(columns={
            "policy_id": "Policy", "region": "Region", "town": "Town",
            "population_density": "Density (people/km²)",
            "crowding_band": "Crowding band",
            "3M_od_prob": "3M OD Prob",  "3M_tpl_prob": "3M TPL Prob",
            "3M_od_reserve": "3M OD Res.", "3M_tpl_reserve": "3M TPL Res.",
            "3M_total_reserve": "3M Total",
            "6M_od_prob": "6M OD Prob",  "6M_tpl_prob": "6M TPL Prob",
            "6M_od_reserve": "6M OD Res.", "6M_tpl_reserve": "6M TPL Res.",
            "6M_total_reserve": "6M Total",
            "12M_od_prob": "12M OD Prob", "12M_tpl_prob": "12M TPL Prob",
            "12M_od_reserve": "12M OD Res.", "12M_tpl_reserve": "12M TPL Res.",
            "12M_total_reserve": "12M Total",
            "total_reserve": "Total Reserve (12M)",
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
