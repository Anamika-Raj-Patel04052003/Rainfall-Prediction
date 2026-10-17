
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️"
)

st.title("🌧️ Rainfall Prediction App")
st.write("Enter weather details to predict rainfall.")

model_path = (
    Path(__file__).parent
    / "models"
    / "rainfall_prediction_model.pkl"
)
model = joblib.load(model_path)

st.success("ML model loaded successfully!")

st.subheader("Weather Details")

with st.form("weather_form"):
    day = st.number_input(
    "Day of Year",
    min_value=1,
    max_value=365,
    value=180,
    step=1
    )

    pressure = st.number_input(
        "Pressure", min_value=850.0, max_value=1100.0,
        value=1010.0
    )
    maxtemp = st.number_input(
        "Maximum Temperature (°C)", min_value=-30.0,
        max_value=60.0, value=30.0
    )
    temperature = st.number_input(
        "Temperature (°C)", min_value=-30.0,
        max_value=60.0, value=25.0
    )
    mintemp = st.number_input(
        "Minimum Temperature (°C)", min_value=-40.0,
        max_value=50.0, value=20.0
    )
    dewpoint = st.number_input(
        "Dew Point (°C)", min_value=-50.0,
        max_value=40.0, value=18.0
    )
    humidity = st.slider("Humidity (%)", 0, 100, 70)
    cloud = st.slider("Cloud Cover (%)", 0, 100, 60)
    sunshine = st.number_input(
        "Sunshine (hours)", min_value=0.0,
        max_value=24.0, value=5.0
    )
    winddirection = st.selectbox(
        "Wind Direction",
        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    )
    windspeed = st.number_input(
        "Wind Speed", min_value=0.0,
        max_value=200.0, value=10.0
    )

    submitted = st.form_submit_button("Predict Rainfall")

if submitted:
    weather_data = pd.DataFrame([{
        "day": str(int(day)),
        "pressure": pressure,
        "maxtemp": maxtemp,
        "temperature": temperature,
        "mintemp": mintemp,
        "dewpoint": dewpoint,
        "humidity": humidity,
        "cloud": cloud,
        "sunshine": sunshine,
        "winddirection": winddirection,
        "windspeed": windspeed
    }])

    try:
        prediction = model.predict(weather_data)[0]

        if prediction == "yes":
            st.success("🌧️ Prediction: Rainfall expected")
        else:
            st.info("☀️ Prediction: No rainfall expected")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(weather_data)[0]
            yes_index = list(model.classes_).index("yes")
            st.metric(
                "Estimated Rainfall Probability",
                f"{probabilities[yes_index] * 100:.2f}%"
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.caption("Predictions are estimates, not guaranteed weather forecasts.")
