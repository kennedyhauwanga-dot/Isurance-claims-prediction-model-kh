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
DB_FILE = "predictions.db"

def init_db():
    """Create the SQLite database and tables if they do not exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Predictions (
            PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
            Age INTEGER,
            Gender TEXT,
            VehicleValue REAL,
            AnnualPremium REAL,
            TimeHorizon TEXT,
            PredictedClass TEXT,
            Probability REAL,
            PredictionDate TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ModelMetadata (
            ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
            ModelName TEXT,
            Hyperparameters TEXT,
            Accuracy REAL,
            AUC_PR REAL,
            LogDate TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(age, gender, vehicle_val, premium, horizon, decision, prob):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO Predictions (Age, Gender, VehicleValue, AnnualPremium, TimeHorizon, PredictedClass, Probability, PredictionDate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (age, gender, vehicle_val, premium, horizon, decision, prob, now))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM Predictions ORDER BY PredictionDate DESC", conn)
    conn.close()
    return df

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    processed_data = output.getvalue()
    return processed_data

init_db()

# ---------- Security & Authentication ----------
def check_credentials():
    """Returns `True` if the user has entered correct username and password."""
    
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

    if "authentication_status" not in st.session_state or st.session_state["authentication_status"] != True:
        st.title("System Login")
         st.subtitle("Only Meant for my Thesis:Kennedy N Hauwanga (218208399)")
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
    else:
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
    """
    Adjusts the base ML probability using real-world actuarial logic.
    """
    multiplier = 1.0
    
    #  Driving Experience ---
    # High experience = lower risk. Low experience = higher risk.
    if years_driving < 3:
        multiplier += 0.4
    elif years_driving < 8:
        multiplier += 0.15
    elif years_driving > 15:
        multiplier -= 0.25
    elif years_driving > 10:
        multiplier -= 0.1
        
    # Location (Urban vs Rural) ---
    # Urban areas have higher accident/theft rates due to congestion. Rural gets a discount.
    if urban_rural == "Rural":
        multiplier -= 0.15
    else:
        multiplier += 0.1
        
    # Prior Claims (within the last 3 years) ---
    # Any prior claim drastically increases risk.
    if prior_claims == 1:
        multiplier += 0.3
    elif prior_claims == 2:
        multiplier += 0.6
    elif prior_claims >= 3:
        multiplier += 1.0

    # ---  Age ---
    if age < 25:
        multiplier += 0.25
    elif age > 65:
        multiplier += 0.1
        
    # ---  Traffic Violations ---
    if violations > 0:
        multiplier += (0.15 * violations)
        
    # ---  Mileage ---
    if mileage > 30000:
        multiplier += 0.2
        
    # ---  Credit Score ---
    if credit_score < 550:
        multiplier += 0.15
        
    # ---  Vehicle Specifics ---
    if vehicle_age > 10:
        multiplier += 0.15
    if vehicle_value > 500000:
        multiplier += 0.1
    if vehicle_type in ["Pickup", "SUV"]:
        multiplier += 0.05
        
    # Cap the multiplier between 0.2 and 3.0
    return max(0.2, min(multiplier, 3.0))

def render_card(horizon, cfg, prob=None):
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
    key="age_grp",
    disabled=True
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
    "University of Namibia research thesis | "
    "Student No. 218208399"
)

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
    risk_multiplier = calculate_business_risk_adjustment(
        age=age, gender=gender, years_driving=years_driving, prior_claims=prior_claims, 
        violations=traffic_violations, mileage=annual_mileage, credit_score=credit_score,
        urban_rural=urban_rural, vehicle_age=vehicle_age, vehicle_value=vehicle_value, 
        vehicle_type=vehicle_type
    )

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
            base_prob = float(model.predict_proba(input_df)[0, 1])
            adjusted_prob = min(base_prob * risk_multiplier, 0.999)
            results[horizon] = adjusted_prob
            
            decision = "Claim" if adjusted_prob >= cfg["threshold"] else "No Claim"
            log_prediction(
                age=age, gender=gender, vehicle_val=vehicle_value, premium=premium_amount, 
                horizon=horizon, decision=decision, prob=adjusted_prob
            )
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

# ---------- Summary table (Horizontal Layout) ----------
if results:
    st.divider()
    st.subheader("Summary")
    
    summary = pd.DataFrame({
        "3 months": [
            f"{results.get('3 months', 0):.2%}", 
            risk_level(results.get('3 months', 0))[0], 
            "Claim" if results.get('3 months', 0) >= HORIZONS["3 months"]["threshold"] else "No Claim", 
            f"{HORIZONS['3 months']['threshold']:.2f}"
        ],
        "6 months": [
            f"{results.get('6 months', 0):.2%}", 
            risk_level(results.get('6 months', 0))[0], 
            "Claim" if results.get('6 months', 0) >= HORIZONS["6 months"]["threshold"] else "No Claim", 
            f"{HORIZONS['6 months']['threshold']:.2f}"
        ],
        "12 months": [
            f"{results.get('12 months', 0):.2%}", 
            risk_level(results.get('12 months', 0))[0], 
            "Claim" if results.get('12 months', 0) >= HORIZONS["12 months"]["threshold"] else "No Claim", 
            f"{HORIZONS['12 months']['threshold']:.2f}"
        ]
    }, index=["Probability", "Risk Level", "Decision", "Threshold"])
    
    st.dataframe(summary, use_container_width=True)
    st.success("Prediction results have been successfully logged to the database.")

# ---------- Reference table (XGBoost only) ----------
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
}), use_container_width=True, hide_index=True, on_select="ignore")

# ---------- Admin: Download Database (Excel) ----------
st.divider()
st.subheader("System Administration (Database Logs)")

df_logs = get_all_predictions()

if not df_logs.empty:
    excel_data = convert_df_to_excel(df_logs)
    
    st.write(f"Total logged predictions in database: **{len(df_logs)}**")
    st.dataframe(df_logs.head(5), use_container_width=True, hide_index=True)
    
    st.download_button(
        label="Click here to Download Prediction Logs (Excel)",
        data=excel_data,
        file_name=f"prediction_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
else:
    st.info("No predictions have been logged to the database yet. Click 'Predict Claim Risk' to generate data.")
