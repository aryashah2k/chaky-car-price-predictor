import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import base64
import sklearn


# Load the pickled model and scaler
try:
    loaded_model = load('car_price_rf.pkl')
    model = loaded_model['model']
    scaler = loaded_model['scaler']
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model and scaler: {e}")
    st.stop()

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def predict_price(engine, km_driven, max_power, mileage, year):
    input_data = np.array([[engine, km_driven, max_power, mileage, year]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return np.exp(prediction)[0]

def main():
    # Add background image
    add_bg_from_local('garage.jpg')  # Make sure to have a background.jpg file in the same directory

    st.title('Chaky\'s Car Price Predictor')

    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Enter the car details below:</p>', unsafe_allow_html=True)

    # Input fields
    engine = st.number_input('Engine Capacity (cc)', value=1500)
    km_driven = st.number_input('Kilometers Driven', value=50000)
    max_power = st.number_input('Max Power (bhp)', value=100)
    mileage = st.number_input('Mileage (kmpl)', value=20)
    year = st.number_input('Year of Manufacture', value=2020)

    if st.button('Predict Price'):
        price = predict_price(engine, km_driven, max_power, mileage, year)
        st.success(f'The predicted price of the car is ₹{price:,.2f}')

    st.sidebar.header('About')
    st.sidebar.info('This app predicts the price of a used car based on its features using a Random Forest model.')

if __name__ == '__main__':
    main()