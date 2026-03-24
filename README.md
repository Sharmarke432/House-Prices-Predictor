# House Price Predictor

A machine learning web app that predicts house prices based on property features using an XGBoost regression model trained on the King County Housing dataset.

## Project Overview

This project covers the full data science pipeline:
- Exploratory Data Analysis (EDA) with Matplotlib and Seaborn
- Feature engineering and data preprocessing
- Training and comparing multiple regression models
- Hyperparameter tuning with Optuna
- Deployment as an interactive Streamlit web app

## 🗂️ Project Structure
├── data/
│ └── housing.csv # King County housing dataset
├── notebooks/
│ └── eda.ipynb # EDA and model training notebook
├── app.py # Streamlit web app
├── xgb_model.pkl # Saved XGBoost model
├── requirements.txt # Project dependencies
└── README.md


## Models Compared

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | £64,545 | £127,023 | 0.8933 |
| LightGBM | £67,606 | £133,727 | 0.8817 |
| Random Forest | £71,440 | £138,139 | 0.8738 |
| KNN | £84,779 | £165,606 | 0.8186 |
| Ridge | £103,340 | £178,565 | 0.7891 |

**XGBoost** was selected as the final model and tuned using Optuna.

## Features Used

- Bedrooms, bathrooms, floors
- Sqft living, sqft lot, sqft above, sqft basement
- Waterfront, view, condition, grade
- Year built, year renovated
- Latitude, longitude
- Zipcode mean price (encoded)
- Sale year, sale month

## Getting Started (locally)

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
```

## Install Dependencies
pip install -r requirements.txt

## Run Streamlit App
streamlit run app.py

## Tech Stack

- Python 3.9
- Scikit-learn — preprocessing and model evaluation
- XGBoost — final prediction model
- LightGBM — compared model
- Optuna — hyperparameter tuning
- Matplotlib / Seaborn — EDA visualisations
- Streamlit — web app deployment
- Joblib — model serialisation

