import streamlit as st
import joblib
import numpy as np

model = joblib.load('rf_crop_model.pkl')

st.title("Crop Stability Predictor (Onion & Garlic)")

soil_ph = st.slider("Soil pH", 5.0, 8.0, 6.5)
avg_temp = st.slider("Avg Temp (°C)", 10.0, 35.0, 22.0)
rainfall = st.slider("Rainfall (mm)", 0, 500, 100)
soil_nutrients = st.number_input("Soil Nutrient Score (0-1)", 0.0, 1.0, 0.5)
seasonal_variability = st.slider("Seasonal Variability Score", 0.0, 1.0, 0.2)

if st.button("Predict Crop Stability"):
    features = np.array([[soil_ph, avg_temp, rainfall, soil_nutrients, seasonal_variability]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Crop Stability: {prediction:.2f}%")
