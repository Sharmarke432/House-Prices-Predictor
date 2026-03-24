import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.markdown("Multiple models trained on King County Housing Dataset")

# Load all models and scaler
@st.cache_resource
def load_models():
    models = {
        "Ridge": joblib.load("ridge_model.pkl"),
        "LightGBM": joblib.load("lightgbm_model.pkl"), 
        "XGBoost": joblib.load("xgboost_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
        "KNN": joblib.load("knn_model.pkl")
    }
    scaler = joblib.load("scaler.pkl")
    return models, scaler

@st.cache_data
def load_zipcode_stats():
    return pd.read_csv("zipcode_mean_prices.csv")

models, scaler = load_models()
zipcode_stats = load_zipcode_stats()
zipcode_mean_map = dict(zip(zipcode_stats["zipcode"], zipcode_stats["price"]))

# Sidebar for inputs
st.sidebar.header("🏠 Property Details")
col1, col2 = st.sidebar.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 0.5, 8.0, 2.0, 0.25)
    floors = st.number_input("Floors", 1.0, 3.5, 1.0, 0.5)
with col2:
    sqft_living = st.number_input("Sqft Living", 500, 8000, 2000)
    sqft_above = st.number_input("Sqft Above", 500, 5000, 1500)
    sqft_basement = st.number_input("Sqft Basement", 0, 2000, 500)

col1, col2 = st.sidebar.columns(2)
with col1:
    yr_built = st.number_input("Year Built", 1900, 2020, 1990)
    yr_renovated = st.number_input("Year Renovated (0=none)", 0, 2020, 0)
with col2:
    zipcode = st.number_input("Zipcode", 98001, 98200, 98002)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View Score", 0, 4, 2)

col1, col2 = st.sidebar.columns(2)
with col1:
    condition = st.slider("Condition", 1, 5, 3)
    grade = st.slider("Grade", 1, 13, 7)
with col2:
    sqft_living15 = st.number_input("Sqft Living15", 500, 5000, 1800)
    sqft_lot15 = st.number_input("Sqft Lot15", 1000, 50000, 7500)

lat = st.number_input("Latitude", 47.2, 47.8, 47.5)
long = st.number_input("Longitude", -122.5, -121.0, -122.2)

# Transform zipcode
mean_price_by_zipcode = zipcode_mean_map.get(zipcode, zipcode_stats["price"].mean())
if zipcode not in zipcode_mean_map:
    st.sidebar.warning(f"Zipcode {zipcode} not in training data. Using mean: £{mean_price_by_zipcode:,.0f}")

# Feature engineering (same as training)
age = 2015 - yr_built  # dataset from 2014-2015
was_renovated = 1 if yr_renovated > 0 else 0

# Create input dataframe (CLAMP to prevent inf!)
input_data = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "sqft_living": [sqft_living.clip(500, 8000)],
    "sqft_lot": [sqft_lot15.clip(1000, 50000)],  # using lot15 as proxy
    "floors": [floors],
    "waterfront": [waterfront],
    "view": [view],
    "condition": [condition],
    "grade": [grade],
    "sqft_above": [sqft_above.clip(500, 5000)],
    "sqft_basement": [sqft_basement.clip(0, 2000)],
    "yr_built": [yr_built],
    "yr_renovated": [yr_renovated],
    "zipcode": [zipcode],
    "lat": [lat],
    "long": [long],
    "sqft_living15": [sqft_living15.clip(500, 5000)],
    "sqft_lot15": [sqft_lot15.clip(1000, 50000)],
    "mean_price_by_zipcode": [mean_price_by_zipcode.clip(100000, 2000000)],
    "age": [age],
    "was_renovated": [was_renovated]
})

# Check for issues
if input_data.isnull().any().any() or np.isinf(input_data.select_dtypes(np.number)).any().any():
    st.error("🚨 Invalid input! Check your values.")
    st.stop()

# Predict with ALL models
if st.button("🚀 Predict House Price", type="primary"):
    with st.spinner("Predicting with 5 models..."):
        input_scaled = scaler.transform(input_data)
        
        predictions = {}
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            if np.isfinite(pred):
                predictions[name] = max(50000, min(5000000, pred))  # clamp realistic range
            else:
                predictions[name] = np.nan
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        # Ensemble average (most reliable)
        valid_preds = [v for v in predictions.values() if np.isfinite(v)]
        ensemble_pred = np.mean(valid_preds) if valid_preds else np.nan
        
        with col1:
            st.metric("🏆 Ensemble Average", f"£{ensemble_pred:,.0f}")
        
        # Individual models
        for name, pred in predictions.items():
            if np.isfinite(pred):
                st.metric(name, f"£{pred:,.0f}")
            else:
                st.metric(name, "❌ Failed")

# Model comparison table
with st.expander("📊 Model Performance Comparison"):
    st.markdown("""
    | Model       | Test R² | Test MAE    | Test RMSE   |
    |-------------|---------|-------------|-------------|
    | **XGBoost** | 0.893   | £64,545     | £127,023    |
    | LightGBM    | 0.882   | £67,606     | £133,727    |
    | Random Forest | 0.874 | £71,440  | £138,139    |
    | Ridge       | 0.789   | £103,340    | £178,565    |
    | KNN         | 0.819   | £84,779     | £165,606    |
    """)

st.markdown("---")
st.caption("Trained on King County Housing Dataset (21k samples). Zipcode auto-converted to neighborhood mean price.")
