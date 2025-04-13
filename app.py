import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="ESE Marks Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Sidebar with project information from README
with st.sidebar:
    # Personal Branding
    st.sidebar.markdown("""
    <div>
    <p>
        Built with Streamlit by <a href='https://akhundmuzzammil.com' target='_blank'>Muzzammil Akhund</a> 
        <br>
        Connect: <a href='https://github.com/akhundmuzzammil' target='_blank'>GitHub</a> 
         | <a href='https://linkedin.com/in/akhundmuzzammil' target='_blank'>LinkedIn</a>
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Information
    st.title('About This Project')
    st.markdown("""
    ### [GitHub Repository](https://github.com/akhundmuzzammil/EndExamScorePrediction)
    
    ### Overview
    EndExamScorePrediction is a tool designed to predict end-semester exam scores based on attendance, mid-semester exam marks, and study hours.
    
    ### Features Used
    - Attendance (%)
    - Mid-Semester Examination (MSE) Marks
    - Hours Studied (HRS)
    
    ### Data Source
    The model is trained on a [Kaggle dataset](https://www.kaggle.com/datasets/akiwelekar/predictingese) that contains records of attendance, MSE marks, study hours, and actual ESE scores for 73 entries.
    
    ### Project Details
    This project demonstrates:
    - Data preprocessing and feature normalization
    - Linear regression modeling for academic performance prediction
    - Interactive web application using Streamlit
    - Evaluation using performance metrics such as MSE, RMSE, MAE, and R²
    """)
    
    # Dataset Information
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        The dataset records the attendance (in percentage), MSE marks out of 30, and ESE marks out of 70 for 73 students. 
        
        Data was collected in 2014 while teaching a Software Architecture course at Dr Babasaheb Ambedkar Technological University Lonere.
        
        **Preprocessing Steps:**
        - Missing values are filled using column means
        - Input features are normalized using StandardScaler
        """)

    # Model Information
    with st.expander("🧠 Model Information"):
        st.markdown("""
        - **Algorithm**: Linear Regression
        - **Features**: Attendance percentage, MSE marks, Hours studied
        - **Output**: Predicted ESE marks (out of 70)
        - **Passing threshold**: 40 marks
        
        The model is saved as `linear_regression_model.pkl` and the scaler as `scaler.pkl`.
        """)

# Main content
main_container = st.container()
with main_container:
    # Title and introduction
    st.title("End-Semester Examination (ESE) Marks Predictor")
    st.markdown("### Provide your details below to get a prediction of your ESE marks")
    
    # Input section in a nice card-like container
    input_container = st.container()
    with input_container:
        st.markdown("### 📋 Input Parameters")
        
        # Use columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            attendance = st.slider(
                "Attendance (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=75.0,
                step=1.0, 
                help="Enter attendance as a percentage (e.g., 90 for 90%)."
            )
            
            mse_marks = st.slider(
                "Mid-Semester Examination (MSE) Marks", 
                min_value=0.0, 
                max_value=30.0, 
                value=20.0,
                step=0.5, 
                help="Enter marks obtained out of 30."
            )
        
        with col2:
            hrs_studied = st.slider(
                "Hours Studied (HRS)", 
                min_value=0.0, 
                max_value=24.0, 
                value=10.0,
                step=1.0, 
                help="Enter the total number of hours studied for the semester."
            )
            
            # Add spacer for balance
            st.write("")
            st.write("")
            
            # Load the model and scaler
            model, scaler = load_model_and_scaler()
            
            # Predict button with better styling
            predict_button = st.button("🔮 Predict My ESE Marks", use_container_width=True)

    # Results section
    if predict_button:
        try:
            # Input validation
            if not (0 <= attendance <= 100):
                st.error('Attendance must be between 0 and 100')
            elif not (0 <= mse_marks <= 30):
                st.error('MSE marks must be between 0 and 30')
            elif not (0 <= hrs_studied <= 24):
                st.error('Study hours must be between 0 and 24')
            else:
                # Show a spinner during prediction
                with st.spinner('Analyzing your data...'):
                    # Create input array
                    input_data = np.array([[attendance, mse_marks, hrs_studied]])
                    
                    # Scale the input using the saved scaler
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    
                    # Round prediction to nearest integer
                    prediction = round(prediction)
                    
                    # Ensure prediction is within reasonable bounds (0-70 for ESE marks)
                    prediction = max(0, min(prediction, 70))
                
                # Display the prediction in a stylish container
                st.markdown("### 🎯 Prediction Results")
                
                # Create a progress bar/gauge for the prediction
                st.write(f"**Predicted Score: {prediction}/70**")
                progress = prediction / 70.0  # Normalize to 0-1 scale
                st.progress(progress)
                
                # Apply different styling based on pass/fail
                if prediction >= 40:
                    st.success(f"""
                    ### ✅ Congratulations!
                    The model predicts you will score **{prediction}/70** marks in your ESE.
                    This is above the passing threshold of 40 marks.
                    """)
                else:
                    st.warning(f"""
                    ### ⚠️ Attention Needed
                    The model predicts you will score **{prediction}/70** marks in your ESE.
                    This is below the passing threshold of 40 marks. Consider increasing your study hours!
                    """)
                
                # Show feature importance
                st.markdown("### 📊 How your inputs affect the prediction")
                chart_data = pd.DataFrame({
                    'Parameter': ['Attendance', 'MSE Marks', 'Study Hours'],
                    'Your Value': [f"{attendance}%", f"{mse_marks}/30", f"{hrs_studied} hrs"],
                    'Recommendation': [
                        "✅ Good" if attendance >= 75 else "⚠️ Increase attendance",
                        "✅ Good" if mse_marks >= 15 else "⚠️ Improve MSE performance",
                        "✅ Good" if hrs_studied >= 10 else "⚠️ More study time needed"
                    ]
                })
                st.table(chart_data)
                
        except ValueError:
            st.error('Please enter valid numeric values')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
