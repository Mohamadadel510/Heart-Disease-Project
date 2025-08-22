import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import os

@st.cache_resource
def load_model():
    try:
        model_path = 'Heart_Disease_Project/models/complete_pipeline.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_features():
    try:
        features_path = 'Heart_Disease_Project/models/selected_features.pkl'
        if not os.path.exists(features_path):
            st.error(f"Features file not found at {features_path}")
            return None
        return joblib.load(features_path)
    except Exception as e:
        st.error(f"Error loading features: {e}")
        return None

# Load model and features
model = load_model()
features = load_features()

# Check if both model and features loaded successfully
if model is None or features is None:
    st.error("Failed to load model or features. Please check the file paths.")
    st.stop()

st.title("🫀 Heart Disease Prediction App")
st.write("Enter the patient information below to predict heart disease risk using machine learning.")

# Add some helpful information
with st.expander("ℹ️ How to use this app"):
    st.write("""
    **Instructions:**
    1. Fill in all the patient information in the form below
    2. For numeric values, use realistic ranges (e.g., age 20-100, blood pressure 80-200)
    3. For categorical values, select the most appropriate option from the dropdown
    4. Click 'Predict' to get the heart disease risk assessment
    """)

# Feature descriptions and options for better user understanding
feature_info = {
    'age': {
        'description': 'Age (years)',
        'type': 'numeric',
        'min_value': 1,
        'max_value': 120
    },
    'sex': {
        'description': 'Sex',
        'type': 'categorical',
        'options': {0: 'Female', 1: 'Male'}
    },
    'cp_4': {
        'description': 'Chest Pain Type',
        'type': 'categorical',
        'options': {
            0: 'Typical Angina',
            1: 'Atypical Angina', 
            2: 'Non-Anginal Pain',
            3: 'Asymptomatic'
        }
    },
    'trestbps': {
        'description': 'Resting Blood Pressure (mm Hg)',
        'type': 'numeric',
        'min_value': 80,
        'max_value': 250
    },
    'chol': {
        'description': 'Serum Cholesterol (mg/dl)',
        'type': 'numeric',
        'min_value': 100,
        'max_value': 600
    },
    'fbs': {
        'description': 'Fasting Blood Sugar > 120 mg/dl',
        'type': 'categorical',
        'options': {0: 'False (≤ 120 mg/dl)', 1: 'True (> 120 mg/dl)'}
    },
    'restecg': {
        'description': 'Resting ECG Results',
        'type': 'categorical',
        'options': {
            0: 'Normal',
            1: 'ST-T Wave Abnormality',
            2: 'Left Ventricular Hypertrophy'
        }
    },
    'thalach': {
        'description': 'Maximum Heart Rate Achieved',
        'type': 'numeric',
        'min_value': 60,
        'max_value': 220
    },
    'exang': {
        'description': 'Exercise Induced Angina',
        'type': 'categorical',
        'options': {0: 'No', 1: 'Yes'}
    },
    'oldpeak': {
        'description': 'ST Depression Induced by Exercise',
        'type': 'numeric',
        'min_value': 0.0,
        'max_value': 10.0
    },
    'slope_2': {
        'description': 'Slope of Peak Exercise ST Segment',
        'type': 'categorical',
        'options': {
            0: 'Upsloping',
            1: 'Flat',
            2: 'Downsloping'
        }
    },
    'ca': {
        'description': 'Number of Major Vessels Colored by Fluoroscopy',
        'type': 'categorical',
        'options': {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels'}
    },
    'thal_7.0': {
        'description': 'Thalassemia',
        'type': 'categorical',
        'options': {
            0: 'Normal',
            1: 'Fixed Defect',
            2: 'Reversible Defect'
        }
    }
}

# Create input fields with proper layout
input_data = {}

# Filter out 'num' if it appears in features (it's the target variable)
actual_features = [f for f in features if f != 'num']

# Process ALL features
processed_features = []
for feature in actual_features:
    if feature in feature_info:
        processed_features.append(feature)
    else:
        # Create default info for any missing features
        feature_info[feature] = {
            'description': feature.replace('_', ' ').title(),
            'type': 'categorical',
            'options': {0: 'No', 1: 'Yes'}
        }
        processed_features.append(feature)

# Split features into two columns for better layout
mid_point = len(processed_features) // 2
col1_features = processed_features[:mid_point]
col2_features = processed_features[mid_point:]

st.write(f"**Total features to display: {len(processed_features)}**")
st.write(f"**Actual features: {actual_features}**")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Patient Information - Part 1")
    for feature in col1_features:
        info = feature_info[feature]
        
        if info['type'] == 'numeric':
            input_data[feature] = st.number_input(
                info['description'],
                min_value=float(info['min_value']),
                max_value=float(info['max_value']),
                value=float(info['min_value']),
                key=f"{feature}_input",
                help=f"Enter value for {feature}"
            )
        else:  # categorical
            options_dict = info['options']
            display_options = [f"{k}: {v}" for k, v in options_dict.items()]
            selected_display = st.selectbox(
                info['description'],
                display_options,
                key=f"{feature}_select",
                help=f"Select option for {feature}"
            )
            # Extract the numeric value from the selection
            input_data[feature] = int(selected_display.split(':')[0])

with col2:
    st.subheader(f"Patient Information - Part 2")
    for feature in col2_features:
        info = feature_info[feature]
        
        if info['type'] == 'numeric':
            input_data[feature] = st.number_input(
                info['description'],
                min_value=float(info['min_value']),
                max_value=float(info['max_value']),
                value=float(info['min_value']),
                key=f"{feature}_input",
                help=f"Enter value for {feature}"
            )
        else:  # categorical
            options_dict = info['options']
            display_options = [f"{k}: {v}" for k, v in options_dict.items()]
            selected_display = st.selectbox(
                info['description'],
                display_options,
                key=f"{feature}_select",
                help=f"Select option for {feature}"
            )
            # Extract the numeric value from the selection
            input_data[feature] = int(selected_display.split(':')[0])


if st.button("Predict", type="primary"):
    try:
        # Create DataFrame with the correct column order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present and in the right order
        input_df = input_df.reindex(columns=features, fill_value=0)
        
        # Make predictions
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display result
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"⚠️ High Risk: {probability[1]:.1%} chance of heart disease")
        else: 
            st.success(f"✅ Low Risk: {probability[1]:.1%} chance of heart disease")
        
        # Display probability bar 
        st.subheader("Risk Probability")
        st.progress(probability[1])
        
        # Show detailed probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Low Risk Probability", f"{probability[0]:.1%}")
        with col2:
            st.metric("High Risk Probability", f"{probability[1]:.1%}")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

