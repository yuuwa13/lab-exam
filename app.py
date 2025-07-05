import streamlit as st
import numpy as np
import joblib

# --- Load model and preprocessor ---
model = joblib.load("xgb_wine_quality_model.pkl")
scaler = joblib.load("xgb_wine_quality_scaler.pkl")
poly = joblib.load("xgb_polynomial_transformer.pkl")

# --- Set page config ---
st.set_page_config(page_title="🍷 Wine Quality Predictor", page_icon="🍇", layout="centered")

# --- App title and subtitle ---
st.title("🍷 Wine Quality Predictor")
st.markdown("""
Welcome to the boutique wine prediction tool!  
Enter the wine's chemical properties below to check if it's **premium quality** 💎 or **not quite there** 🍷.
""")

# --- Input fields for wine attributes ---
with st.form("wine_form"):
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, format="%.2f")
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, format="%.2f")
    citric_acid = st.number_input("Citric Acid", min_value=0.0, format="%.2f")
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, format="%.2f")
    chlorides = st.number_input("Chlorides", min_value=0.0, format="%.4f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, format="%.2f")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, format="%.2f")
    density = st.number_input("Density", min_value=0.0, format="%.5f")
    pH = st.number_input("pH", min_value=0.0, format="%.2f")
    sulphates = st.number_input("Sulphates", min_value=0.0, format="%.2f")
    alcohol = st.number_input("Alcohol", min_value=0.0, format="%.2f")
    
    submitted = st.form_submit_button("🍇 Predict Wine Quality")

# --- Prediction ---
if submitted:
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    # Transform inputs
    input_poly = poly.transform(input_data)
    input_scaled = scaler.transform(input_poly)

    # Predict
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]

    # Display result
    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 This wine is **Good Quality**! Confidence: {confidence:.2%}")
        st.balloons()
    else:
        st.error(f"😔 This wine is **Not Good**. Confidence: {confidence:.2%}")
