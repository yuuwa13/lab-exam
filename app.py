import streamlit as st
import numpy as np
import joblib

# --- Load model and preprocessor ---
model = joblib.load("xgb_wine_quality_model.pkl")
scaler = joblib.load("xgb_wine_quality_scaler.pkl")
poly = joblib.load("xgb_polynomial_transformer.pkl")

# --- Set page config ---
st.set_page_config(page_title="Wine Quality Predictor", page_icon=None, layout="centered")

# --- App title and subtitle ---
st.title("Wine Quality Assessment Tool")
st.markdown("""
Welcome to the wine quality assessment platform.  
Please provide the wine's chemical properties below to evaluate its quality.
""")

# --- Input fields for wine attributes ---
with st.form("wine_form"):
    st.header("Wine Chemical Properties")
    st.markdown("Enter the following measurements for your wine sample:")

    # Acidity Section
    st.markdown("**Acidity**")
    fixed_acidity = st.number_input(
        "Fixed Acidity (g/dm³)", min_value=0.0, max_value=16.0, value=7.0, format="%.2f",
        help="Tartaric acid content, typically between 4–16 g/dm³."
    )
    volatile_acidity = st.number_input(
        "Volatile Acidity (g/dm³)", min_value=0.0, max_value=1.5, value=0.5, format="%.2f",
        help="Acetic acid content, usually less than 1.5 g/dm³."
    )
    citric_acid = st.number_input(
        "Citric Acid (g/dm³)", min_value=0.0, max_value=1.0, value=0.3, format="%.2f",
        help="Citric acid content, typically between 0–1 g/dm³."
    )
    pH = st.number_input(
        "pH", min_value=2.8, max_value=4.0, value=3.3, format="%.2f",
        help="Acidity or basicity, usually between 2.8–4."
    )

    st.markdown("---")

    # Sugar & Chlorides Section
    st.markdown("**Sugar & Chlorides**")
    residual_sugar = st.number_input(
        "Residual Sugar (g/dm³)", min_value=0.0, max_value=15.0, value=2.5, format="%.2f",
        help="Amount of sugar remaining after fermentation."
    )
    chlorides = st.number_input(
        "Chlorides (g/dm³)", min_value=0.0, max_value=0.2, value=0.05, format="%.4f",
        help="Salt content, usually less than 0.2 g/dm³."
    )

    st.markdown("---")

    # Sulfur Dioxide Section
    st.markdown("**Sulfur Dioxide**")
    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide (mg/dm³)", min_value=1.0, max_value=72.0, value=15.0, format="%.2f",
        help="SO₂ not bound to other molecules, typically 1–72 mg/dm³."
    )
    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide (mg/dm³)", min_value=6.0, max_value=289.0, value=46.0, format="%.2f",
        help="Sum of all forms of SO₂, usually 6–289 mg/dm³."
    )

    st.markdown("---")

    # Other Properties
    st.markdown("**Other Properties**")
    density = st.number_input(
        "Density (g/cm³)", min_value=0.99000, max_value=1.00500, value=0.99675, format="%.5f",
        help="Wine density, typically close to 1.0 g/cm³."
    )
    sulphates = st.number_input(
        "Sulphates (g/dm³)", min_value=0.3, max_value=2.0, value=0.65, format="%.2f",
        help="Potassium sulphate content, typically 0.3–2 g/dm³."
    )
    alcohol = st.number_input(
        "Alcohol (% vol)", min_value=8.0, max_value=15.0, value=10.0, format="%.2f",
        help="Alcohol content by volume, usually 8–15%."
    )

    submitted = st.form_submit_button("Predict Wine Quality")

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
        st.success(f"This wine is predicted to be of good quality. (Confidence: {confidence:.2%})")
        st.balloons()
    else:
        st.error(f"This wine is predicted to be of lower quality. (Confidence: {confidence:.2%})")
