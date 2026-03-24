import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("King County House Price Predictor")
st.markdown("Predictions from **4 ML models** trained on King County Housing Dataset")

# ── Load Models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "XGBoost":  joblib.load("xgb_model.pkl"),
        "LightGBM": joblib.load("light_gbm.pkl"),
        "Ridge":    joblib.load("ridge_model.pkl"),
        "KNN":      joblib.load("knn_model.pkl"),
    }

@st.cache_data
def load_zipcode_map():
    df = pd.read_csv("zipcode_mean_prices.csv")
    return dict(zip(df["zipcode"], df["price"]))

models      = load_models()
zipcode_map = load_zipcode_map()

# ── Sidebar Inputs ────────────────────────────────────────────────
st.sidebar.header("Property Details")

col1, col2 = st.sidebar.columns(2)
with col1:
    bedrooms  = st.number_input("Bedrooms",  1, 10,   3)
    floors    = st.number_input("Floors",    1.0, 3.5, 1.0, 0.5)
    condition = st.slider("Condition (1-5)", 1, 5, 3)
with col2:
    bathrooms = st.number_input("Bathrooms", 0.5, 8.0, 2.0, 0.25)
    waterfront = st.selectbox("Waterfront", [0, 1])
    grade     = st.slider("Grade (1-13)", 1, 13, 7)

col1, col2 = st.sidebar.columns(2)
with col1:
    sqft_living   = st.number_input("Sqft Living",   500,  8000, 2000)
    sqft_above    = st.number_input("Sqft Above",    500,  5000, 1500)
with col2:
    sqft_lot      = st.number_input("Sqft Lot",      500, 100000, 8000)
    sqft_basement = st.number_input("Sqft Basement",   0,   2000,  500)

col1, col2 = st.sidebar.columns(2)
with col1:
    sale_year = st.selectbox("Sale Year", [2014, 2015])
with col2:
    sale_month = st.slider("Sale Month", 1, 12, 6)

view = st.sidebar.slider("View Score (0-4)", 0, 4, 0)

col1, col2 = st.sidebar.columns(2)
with col1:
    yr_built     = st.number_input("Year Built",     1900, 2015, 1990)
    lat          = st.number_input("Latitude",       47.1,  47.8, 47.5, format="%.4f")
with col2:
    yr_renovated = st.number_input("Year Renovated (0=none)", 0, 2015, 0)
    long         = st.number_input("Longitude",    -122.5, -121.0, -122.2, format="%.4f")

zipcode = st.sidebar.number_input("Zipcode", 98001, 98200, 98002)

# ── Zipcode Transform ─────────────────────────────────────────────
if zipcode in zipcode_map:
    mean_price_by_zipcode = zipcode_map[zipcode]
else:
    mean_price_by_zipcode = np.mean(list(zipcode_map.values()))
    st.sidebar.warning(f"Zipcode {zipcode} not in training data — using global mean.")

# ── Feature Engineering (same as training) ────────────────────────
age          = 2015 - yr_built
was_renovated = 1 if yr_renovated > 0 else 0

# ── Build Input DataFrame ─────────────────────────────────────────
raw_input = pd.DataFrame([{
    "bedrooms":             bedrooms,
    "bathrooms":            bathrooms,
    "sqft_living":          sqft_living,
    "sqft_lot":             sqft_lot,
    "floors":               floors,
    "waterfront":           waterfront,
    "view":                 view,
    "condition":            condition,
    "grade":                grade,
    "sqft_above":           sqft_above,
    "sqft_basement":        sqft_basement,
    "yr_built":             yr_built,
    "yr_renovated":         yr_renovated,
    "lat":                  lat,
    "long":                 long,
    "sale_year":            sale_year,
    "sale_month":           sale_monthyear,
    "mean_price_by_zipcode": mean_price_by_zipcode,
    "age":                  age,
    "was_renovated":        was_renovated
}])

# ── Scale Inside Streamlit (no scaler.pkl needed) ─────────────────
@st.cache_data
def get_training_stats():
    """
    Paste the mean and std from YOUR training data here.
    Run this in your notebook to get them:
        print(X_train.mean().to_dict())
        print(X_train.std().to_dict())
    """
    means = {
        "bedrooms": 3.37, "bathrooms": 2.11, "sqft_living": 2079.9,
        "sqft_lot": 15107.0, "floors": 1.49, "waterfront": 0.007,
        "view": 0.23, "condition": 3.41, "grade": 7.66,
        "sqft_above": 1788.6, "sqft_basement": 291.5,
        "yr_built": 1971.0, "yr_renovated": 84.4,
        "lat": 47.56, "long": -122.21,
        "sqft_living15": 1986.6, "sqft_lot15": 12768.5,
        "mean_price_by_zipcode": 540088.0,
        "age": 44.0, "was_renovated": 0.21
    }
    stds = {
        "bedrooms": 0.93, "bathrooms": 0.77, "sqft_living": 918.4,
        "sqft_lot": 41420.5, "floors": 0.54, "waterfront": 0.085,
        "view": 0.76, "condition": 0.65, "grade": 1.17,
        "sqft_above": 827.8, "sqft_basement": 442.6,
        "yr_built": 29.4, "yr_renovated": 401.7,
        "lat": 0.14, "long": 0.14,
        "sqft_living15": 685.4, "sqft_lot15": 27304.2,
        "mean_price_by_zipcode": 149999.0,
        "age": 29.4, "was_renovated": 0.41
    }
    return means, stds

means, stds = get_training_stats()

def scale_input(df):
    scaled = df.copy()
    for col in df.columns:
        if col in means and stds[col] > 0:
            scaled[col] = (df[col] - means[col]) / stds[col]
    return scaled

# ── Predict ───────────────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):
    scaled_input = scale_input(raw_input)

    st.subheader("💰 Predictions")
    cols = st.columns(4)
    valid_preds = []

    for i, (name, model) in enumerate(models.items()):
        pred = model.predict(scaled_input)[0]

        # Guard against inf / NaN
        if not np.isfinite(pred) or pred <= 0:
            cols[i].metric(name, "⚠️ Invalid")
        else:
            pred = float(np.clip(pred, 50_000, 5_000_000))
            valid_preds.append(pred)
            cols[i].metric(name, f"£{pred:,.0f}")

    # Ensemble
    if valid_preds:
        ensemble = np.mean(valid_preds)
        st.success(f"**Ensemble Average: £{ensemble:,.0f}**")

    # Show transformed values used
    with st.expander("🔍 Inputs used by model"):
        st.dataframe(raw_input.T.rename(columns={0: "Value"}))

# ── Model Performance ─────────────────────────────────────────────
with st.expander("Model Performance"):
    st.markdown("""
    | Model | R² | MAE | RMSE |
    |---|---|---|---|
    | **XGBoost** | 0.893 | £64,545 | £127,023 |
    | **LightGBM** | 0.882 | £67,606 | £133,727 |
    | **Ridge** | 0.789 | £103,340 | £178,565 |
    | **KNN** | 0.819 | £84,779 | £165,606 |
    """)

st.caption("King County Housing Dataset · Seattle, WA · 2014-2015")
