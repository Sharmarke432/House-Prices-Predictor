import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Load your trained model ───────────────────────────────────────
model = joblib.load("xgb_model.pkl")

st.title("House Price Predictor")

# ── User Inputs ───────────────────────────────────────────────────
bedrooms      = st.slider("Bedrooms", 1, 10, 3)
bathrooms     = st.slider("Bathrooms", 1.0, 8.0, 2.0, step=0.5)
sqft_living   = st.number_input("Sqft Living", 500, 10000, 1800)
sqft_lot      = st.number_input("Sqft Lot", 500, 50000, 5000)
floors        = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0])
waterfront    = st.selectbox("Waterfront", [0, 1])
view          = st.slider("View (0-4)", 0, 4, 0)
condition     = st.slider("Condition (1-5)", 1, 5, 3)
grade         = st.slider("Grade (1-13)", 1, 13, 7)
sqft_above    = st.number_input("Sqft Above", 500, 10000, 1800)
sqft_basement = st.number_input("Sqft Basement", 0, 5000, 0)
yr_built      = st.number_input("Year Built", 1900, 2015, 1990)
yr_renovated  = st.number_input("Year Renovated (0 if never)", 0, 2015, 0)
lat           = st.number_input("Latitude", 47.0, 48.0, 47.5)
long          = st.number_input("Longitude", -122.5, -121.0, -122.0)
sqft_living15 = st.number_input("Sqft Living (neighbours)", 500, 6000, 1800)
sqft_lot15    = st.number_input("Sqft Lot (neighbours)", 500, 30000, 5000)
sale_year     = st.selectbox("Sale Year", [2014, 2015])
sale_month    = st.slider("Sale Month", 1, 12, 6)
zipcode = st.number_input(
    "Zipcode",
    min_value=98001,
    max_value=98200,
    value=98002,
    step=1
)
# ── Predict ───────────────────────────────────────────────────────
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "sqft_living": sqft_living, "sqft_lot": sqft_lot,
        "floors": floors, "waterfront": waterfront,
        "view": view, "condition": condition, "grade": grade,
        "sqft_above": sqft_above, "sqft_basement": sqft_basement,
        "yr_built": yr_built, "yr_renovated": yr_renovated,
        "lat": lat, "long": long,
        "sqft_living15": sqft_living15, "sqft_lot15": sqft_lot15,
        "sale_year": sale_year, "sale_month": sale_month,
        "zipcode_mean_price": zipcode_mean_price
    }])

    log_pred = model.predict(input_data)
    price     = np.expm1(log_pred[0])

    st.success(f"💰 Predicted Price: £{price:,.0f}")
