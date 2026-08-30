import streamlit as st
import pandas as pd
import joblib

# Set the page configuration
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Load the saved machine learning model
# @st.cache_resource ensures the model is loaded only once, making the app faster
@st.cache_resource
def load_model():
    return joblib.load('titanic_model.joblib')

model = load_model()

# App UI Headers
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger's details below to predict if they would have survived the Titanic disaster.")

# Create a clean layout with columns for user inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Gender", ["Male", "Female"])

with col2:
    age = st.slider("Age (Years)", min_value=0, max_value=100, value=30)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.20)

# Convert categorical inputs to the format our model expects
sex_encoded = 1 if sex == "Female" else 0

# Predict Button
if st.button("Predict Survival", type="primary"):
    
    # Organize the inputs into a dataframe just like the training data
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'Fare': [fare]
    })
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result with professional styling
    st.markdown("---")
    if prediction == 1:
        st.success("🎉 **Prediction: SURVIVED!** This passenger would likely have survived.")
        st.balloons()
    else:
        st.error("😢 **Prediction: DID NOT SURVIVE.** This passenger would likely not have survived.")