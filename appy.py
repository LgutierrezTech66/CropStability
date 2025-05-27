import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('model/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🌾 Crop Stability Predictor")

rainfall = st.number_input("Rainfall (mm)")
temperature = st.number_input("Temperature (°C)")
soil_ph = st.number_input("Soil pH")
crop_type = st.selectbox("Crop Type", ["Parsley", "Onion", "Garlic"])

# Map crop type to numerical value
crop_mapping = {"Parsley": 1, "Onion": 2, "Garlic": 3}
crop_val = crop_mapping[crop_type]

if st.button("Predict Stability"):
    input_data = np.array([[rainfall, temperature, soil_ph, crop_val]])
    prediction = model.predict(input_data)
    result = "Stable 🌱" if prediction[0] == 1 else "Unstable ⚠️"
    st.success(f"Predicted Crop Stability: {result}")
