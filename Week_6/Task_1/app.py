import streamlit as st
import pandas as pd
import joblib
import os  # Yeh import add kiya gaya hai

st.set_page_config(page_title="AutoPrice AI", page_icon="🚗", layout="wide")

# Load model pipeline with absolute path
@st.cache_resource
def load_model():
    # Is file (app.py) ki current directory ka path nikalen
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Directory path ko joblib file ke naam ke sath jor dain
    model_path = os.path.join(current_dir, 'vehicle_pricing_engine.joblib')
    
    return joblib.load(model_path)

pipeline = load_model()

st.title("🚗 AutoPrice AI: Intelligent Vehicle Valuation")
st.markdown("Enter the vehicle specifications below to instantly generate a data-driven market price estimate.")
st.markdown("---")

# Layout using 3 columns for a professional dashboard look
col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox("Make (Brand)", ["Toyota", "Honda", "Ford", "BMW", "Audi", "Nissan"]) # Add your dataset's actual makes here
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=40, value=5)
    odometer = st.number_input("Odometer (km/miles)", min_value=0, max_value=500000, value=60000, step=1000)
    condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Needs Work"])

with col2:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Truck", "Coupe"])

with col3:
    engine_cc = st.number_input("Engine Capacity (CC)", min_value=500, max_value=8000, value=1500, step=100)
    power_hp = st.number_input("Power (HP)", min_value=40, max_value=1000, value=120, step=10)
    seats = st.selectbox("Seats", [2, 4, 5, 7, 8], index=2)
    seller_type = st.selectbox("Seller Type", ["Dealer", "Private Party"])

st.markdown("---")

# Prediction Execution
if st.button("Calculate Market Value", type="primary", use_container_width=True):
    # Construct input dataframe matching the exact columns of X_train
    input_data = pd.DataFrame({
        'odometer': [odometer],
        'make': [make],
        'fuel': [fuel],
        'transmission': [transmission],
        'condition': [condition],
        'engine_cc': [engine_cc],
        'power_hp': [power_hp],
        'seats': [seats],
        'body_type': [body_type],
        'seller_type': [seller_type],
        'car_age': [car_age]
    })
    
    # Run prediction through the pipeline
    estimated_price = pipeline.predict(input_data)[0]
    
    st.success(f"💰 **Estimated Market Value: {estimated_price:,.2f}**")
    st.info("This valuation is generated using an XGBoost regression model trained on historical market data.")
