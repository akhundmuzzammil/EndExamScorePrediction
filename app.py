import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained Linear Regression model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        # Load the model
        with open("linear_regression_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        # Load the scaler
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error("Error loading the model or scaler files. Ensure both pkl files exist in the current directory.")
        raise e

# Initialize the app
st.title("End-Semester Examination (ESE) Marks Predictor")
st.write("Provide the following details to predict the ESE marks:")

# Input fields for Attendance, MSE, and HRS
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0, help="Enter attendance as a percentage (e.g., 90 for 90%).")
mse_marks = st.number_input("Mid-Semester Examination (MSE) Marks", min_value=0.0, max_value=30.0, step=0.5, help="Enter marks obtained out of 30.")
hrs_studied = st.number_input("Hours Studied (HRS)", min_value=0.0, max_value=24.0, step=1.0, help="Enter the total number of hours studied for the semester.")

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Predict button
if st.button("Predict"):
    try:
        # Input validation
        if not (0 <= attendance <= 100):
            st.error('Attendance must be between 0 and 100')
        elif not (0 <= mse_marks <= 30):
            st.error('MSE marks must be between 0 and 30')
        elif not (0 <= hrs_studied <= 24):
            st.error('Study hours must be between 0 and 24')
        else:
            # Create input array
            input_data = np.array([[attendance, mse_marks, hrs_studied]])
            
            # Scale the input using the saved scaler
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # The prediction is already in the original scale since we didn't scale the target variable
            # Round prediction to nearest integer
            prediction = round(prediction)
            
            # Ensure prediction is within reasonable bounds (0-70 for ESE marks)
            prediction = max(0, min(prediction, 70))

            # Display the prediction
            st.success(f"Predicted End-Semester Examination (ESE) Marks: {prediction}")
            
            # Add interpretation
            if prediction >= 40:
                st.info("✅ The model predicts a passing grade!")
            else:
                st.warning("⚠️ The model predicts a grade below passing threshold.")
                
    except ValueError:
        st.error('Please enter valid numeric values')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
