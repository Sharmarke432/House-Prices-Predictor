import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 King County House Price Predictor")
st.markdown("Predictions from **4 ML models** trained on King County Housing Dataset")

FEATURE_ORDER = ['bedrooms', 'bathrooms', 'sqft_living', 
                 'sqft_lot', 'floors', 'waterfront', 
                 'view', 'condition', 'grade', 
                 'sqft_basement', 'yr_built', 
                 'yr_renovated', 'lat', 'long', 
                 'sale_year', 'sale_month', 'zipcode_mean_price']

# ── Load Models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "XGBoost":      joblib.load("xgb_model.pkl"),
        "LightGBM":     joblib.load("light_gbm.pkl"),
        "RandomForest": joblib.load("rf_model.pkl"),
        "Ridge":        joblib.load("ridge_model.pkl"),
        "KNN":          joblib.load("knn_model.pkl"),
    }


@st.cache_data
def load_zipcode_map():
    df = pd.read_csv("zipcode_mean_prices.csv")
    return dict(zip(df["zipcode"], df["price"]))

models      = load_models()
zipcode_map = load_zipcode_map()

# ── Sidebar Inputs ────────────────────────────────────────────────
st.sidebar.header("🏠 Property Details")

col1, col2 = st.sidebar.columns(2)
with col1:
    bedrooms  = st.number_input("Bedrooms",  1, 10, 3)
    floors    = st.number_input("Floors", 1.0, 3.5, 1.0, 0.5)
    condition = st.slider("Condition (1-5)", 1, 5, 3)
with col2:
    bathrooms  = st.number_input("Bathrooms", 0.5, 8.0, 2.0, 0.25)
    waterfront = st.selectbox("Waterfront", [0, 1])
    grade      = st.slider("Grade (1-13)", 1, 13, 7)

col1, col2 = st.sidebar.columns(2)
with col1:
    sqft_living   = st.number_input("Sqft Living",   500,  8000, 2000)
    zipcode = st.sidebar.number_input("Zipcode", 98001, 98200, 98002)

with col2:
    sqft_lot      = st.number_input("Sqft Lot",      500, 100000, 8000)
    sqft_basement = st.number_input("Sqft Basement",   0,   2000,  500)

view = st.sidebar.slider("View Score (0-4)", 0, 4, 0)

col1, col2 = st.sidebar.columns(2)
with col1:
    yr_built     = st.number_input("Year Built",            1900, 2015, 1990)
    lat          = st.number_input("Latitude",  47.1,  47.8, 47.5, format="%.4f")
    sale_year    = st.selectbox("Sale Year", [2014, 2015])
with col2:
    yr_renovated = st.number_input("Year Renovated (0=none)", 0, 2015, 0)
    long         = st.number_input("Longitude", -122.5, -121.0, -122.2, format="%.4f")
    sale_month   = st.slider("Sale Month", 1, 12, 6)

# ── Zipcode Transform ─────────────────────────────────────────────
if zipcode in zipcode_map:
    zipcode_mean_price = zipcode_map[zipcode]
else:
    zipcode_mean_price = np.mean(list(zipcode_map.values()))
    st.sidebar.warning(f"Zipcode {zipcode} not in training data — using global mean.")

# ── Build Input DataFrame in EXACT column order ───────────────────
raw_input = pd.DataFrame([{
    'bedrooms':          bedrooms,
    'bathrooms':         bathrooms,
    'sqft_living':       sqft_living,
    'sqft_lot':          sqft_lot,
    'floors':            floors,
    'waterfront':        waterfront,
    'view':              view,
    'condition':         condition,
    'grade':             grade,
    'sqft_basement':     sqft_basement,
    'yr_built':          yr_built,
    'yr_renovated':      yr_renovated,
    'lat':               lat,
    'long':              long,
    'zipcode_mean_price': zipcode_mean_price,
    'sale_year':         sale_year,
    'sale_month':        sale_month,
}])[FEATURE_ORDER]  # ✅ enforce exact order immediately

# ── Scale Inside Streamlit ────────────────────────────────────────
@st.cache_data
def get_training_stats():
    means = {
        'bedrooms': 3.368131868131868,
        'bathrooms': 2.113794100636206,
        'sqft_living': 2073.838230190862,
        'sqft_lot': 13675.66032388664,
        'floors': 1.4991613649508386,
        'waterfront': 0.007171775592828224,
        'view': 0.23302486986697513,
        'condition': 3.407576633892423,
        'grade': 7.653846153846154,
        'sqft_basement': 287.9327356853673,
        'yr_built': 1971.1083285135917,
        'yr_renovated': 83.00341237709658,
        'lat': 47.56032955465587,
        'long': -122.21413898207057,
        'zipcode_mean_price': 539974.108273377,
        'sale_year': 2014.3218045112783,
        'sale_month': 6.580624638519375
    }
    stds = {
        'bedrooms': 0.9313854099384082,
        'bathrooms': 0.7667900113906938,
        'sqft_living': 905.9916006933574,
        'sqft_lot': 26332.35253480706,
        'floors': 0.5428185387197385,
        'waterfront': 0.08438455468836581,
        'view': 0.7617490399212402,
        'condition': 0.6516976172363061,
        'grade': 1.170354826522681,
        'sqft_basement': 438.7271098728944,
        'yr_built': 29.43560337129589,
        'yr_renovated': 398.5032504016553,
        'lat': 0.13843197869665164,
        'long': 0.1404983028161792,
        'zipcode_mean_price': 233196.5653021182,
        'sale_year': 0.4671819679980777,
        'sale_month': 3.1125763792916787
    }
    return means, stds

means, stds = get_training_stats()

def scale_input(df):
    scaled = df.copy()
    for col in df.columns:
        if col in means and stds[col] > 0:
            scaled[col] = (df[col] - means[col]) / stds[col]
    return scaled  # ✅ numpy array for sklearn models

# ── Predict ───────────────────────────────────────────────────────
if st.button("🚀 Predict Price", type="primary", use_container_width=True):
    scaled_input = scale_input(raw_input)

    st.subheader("Predictions")
    cols = st.columns(5)
    valid_preds = []

    for i, (name, model) in enumerate(models.items()):
        pred = model.predict(scaled_input)[0]

        if not np.isfinite(pred) or pred <= 0:
            cols[i].metric(name, "⚠️ Invalid")
        else:
            pred = float(np.clip(pred, 50_000, 5_000_000))
            valid_preds.append(pred)
            cols[i].metric(name, f"${pred:,.0f}")

    if valid_preds:
        ensemble = np.mean(valid_preds)
        st.success(f"🏆 **Ensemble Average: ${ensemble:,.0f}**")

    with st.expander("🔍 Inputs used by model"):
        st.dataframe(raw_input.T.rename(columns={0: "Value"}))

# ── Model Performance ─────────────────────────────────────────────
with st.expander("📊 Model Performance"):
    st.markdown("""
    | Model | R² | MAE | RMSE |
    |---|---|---|---|
    | **XGBoost** | 0.9066 | $59,377 | $102,520 |
    | **LightGBM** | 0.9026 | $£62,233 | $106,890 |
    | **RandomForest**| 0.8714 | $73,136 | $121,726 |
    | **K-Nearest**| 0.8167 | $83,935 | $148,043 |
    | **Ridge**| 0.8056 | $94,578 | $147,818 |

    """)

st.caption("King County Housing Dataset · Seattle, WA · 2014-2015")
