"""
Road Accident Severity Prediction Application
==============================================
A Streamlit app for predicting road accident severity risk using a trained RandomForest model.

Author: Saksham Nagar
Date: 2026-01-22
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "Trained_Models/accident_model_20260122_102332.pkl"  # Update this path if your model file has a different name

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """
    Load the trained model and associated metadata from pickle file.
    
    Returns:
        dict: Contains 'model', 'feature_names', 'threshold', and 'metrics'
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        if not Path(MODEL_PATH).exists():
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            st.stop()
        
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        # Validate required keys
        required_keys = ['model', 'feature_names', 'threshold']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            st.error(f"❌ Model file is missing required keys: {missing_keys}")
            st.stop()
        
        return model_data
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def create_input_dataframe(hour, visibility, weather, temperature, humidity, 
                          wind_speed, traffic_signal, junction, crossing):
    """
    Create a DataFrame from user inputs.
    
    Args:
        hour (int): Hour of day (0-23)
        visibility (float): Visibility in km/miles
        weather (str): Weather condition
        temperature (float): Temperature
        humidity (float): Humidity percentage
        wind_speed (float): Wind speed
        traffic_signal (bool): Traffic signal present
        junction (bool): Junction present
        crossing (bool): Crossing present
        
    
    Returns:
        pd.DataFrame: Single row DataFrame with user inputs
    """
    data = {
        'Hour': [hour],
        'Visibility(mi)': [visibility],
        'Weather_Condition': [weather],
        'Temperature(F)': [temperature],
        'Humidity(%)': [humidity],
        'Wind_Speed(mph)': [wind_speed],
        'Traffic_Signal': [traffic_signal],
        'Junction': [junction],
        'Crossing': [crossing]
        
    }
    
    return pd.DataFrame(data)


def preprocess_input(df, feature_names):
    """
    Preprocess input DataFrame to match training feature space.
    
    This function:
    1. Applies one-hot encoding to categorical variables
    2. Aligns columns with training features
    3. Fills missing columns with zeros
    4. Ensures correct column order
    
    Args:
        df (pd.DataFrame): Input DataFrame with user data
        feature_names (list): List of feature names from training
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for prediction
    """
    df['Weather_Condition'] = df['Weather_Condition'].str.strip().str.title()
    # Apply one-hot encoding to categorical columns
    df_encoded = pd.get_dummies(df, columns=['Weather_Condition'], 
                                 prefix=['Weather_Condition'])
    
    # Convert boolean columns to int (if any)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # Get all columns that should exist based on training
    missing_cols = set(feature_names) - set(df_encoded.columns)
    
    # Add missing columns with value 0
    for col in missing_cols:
        df_encoded[col] = 0
    
    # Remove any extra columns that weren't in training
    extra_cols = set(df_encoded.columns) - set(feature_names)
    df_encoded = df_encoded.drop(columns=extra_cols)
    
    # Reorder columns to match training feature order exactly
    df_encoded = df_encoded[feature_names]
    
    return df_encoded

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(model, df_processed, threshold):
    """
    Make prediction using the trained model and custom threshold.
    
    Args:
        model: Trained RandomForest model
        df_processed (pd.DataFrame): Preprocessed input data
        threshold (float): Decision threshold for classification
    
    Returns:
        tuple: (prediction, probability)
            - prediction (str): 'Low Severity' or 'High Severity'
            - probability (float): Probability of high severity (0-1)
    """
    # Get probability of high severity (class 1)
    proba = model.predict_proba(df_processed)[0, 1]
    
    # Apply custom threshold
    if proba >= threshold:
        prediction = "High Severity"
    else:
        prediction = "Low Severity"
    
    return prediction, proba


def get_risk_explanation(prediction, probability):
    """
    Generate human-readable explanation of the prediction.
    
    Args:
        prediction (str): Predicted risk category
        probability (float): Probability of high severity
    
    Returns:
        str: Explanation text
    """
    if prediction == "High Severity":
        if probability >= 0.8:
            explanation = (
                "⚠️ **Critical Risk**: The environmental and road conditions indicate a very high "
                "probability of severe accidents. Immediate safety measures and heightened "
                "vigilance are strongly recommended."
            )
        elif probability >= 0.6:
            explanation = (
                "⚠️ **High Risk**: Current conditions pose a significant risk for severe accidents. "
                "Traffic authorities should monitor the situation closely and consider implementing "
                "safety protocols."
            )
        else:
            explanation = (
                "⚠️ **Moderate-High Risk**: Conditions suggest an elevated risk of severe accidents. "
                "Caution is advised for road users."
            )
    else:
        if probability < 0.3:
            explanation = (
                "✅ **Low Risk**: Current conditions indicate a low probability of severe accidents. "
                "Standard safety practices apply."
            )
        else:
            explanation = (
                "✅ **Moderate-Low Risk**: While classified as low severity risk, conditions are "
                "approaching elevated levels. Continue monitoring weather and traffic conditions."
            )
    
    return explanation

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="Accident Severity Predictor",
        page_icon="🚦",
        layout="centered"
    )
    
    # Title and description
    st.title("🚦 Road Accident Severity Predictor")
    st.markdown("""
    This application predicts the severity risk of road accidents based on environmental, 
    temporal, and infrastructure factors. Enter the current conditions below to assess risk levels.
    """)
    
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading prediction model..."):
        model_data = load_model()
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    threshold = model_data['threshold']
    
    # Display model info (optional)
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Model Type:** RandomForest Classifier")
        st.write(f"**Decision Threshold:** {threshold:.3f}")
        st.write(f"**Number of Features:** {len(feature_names)}")
        if 'metrics' in model_data:
            st.write("**Performance Metrics:**")
            for metric, value in model_data['metrics'].items():
                if isinstance(value, (int, float)):
                    st.write(f"- {metric}: {value:.3f}")
                else:
                    st.write(f"- {metric}: {value}")
    
    st.markdown("---")
    
    # Input form
    st.subheader("📋 Enter Incident Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🕐 Temporal Factors**")
        hour = st.slider("Hour of Day", 0, 23, 12, 
                        help="Select the hour (0-23)")
        
        st.markdown("**🌡️ Environmental Conditions**")
        temperature = st.number_input("Temperature (°F)", 
                                     min_value=-50.0, max_value=150.0, 
                                     value=70.0, step=1.0)
        
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        
        visibility = st.number_input("Visibility (mi)", 
                                    min_value=0.0, max_value=50.0, 
                                    value=10.0, step=0.1)
        
        wind_speed = st.number_input("Wind Speed (mph)", 
                                    min_value=0.0, max_value=200.0, 
                                    value=5.0, step=0.5)
    
    with col2:
        st.markdown("**🌦️ Weather Condition**")
        weather = st.selectbox(
            "Select Weather",
            ["Clear", "Rain", "Fog", "Cloudy", "Snow", "Other"]
        )
        
        st.markdown("**🛣️ Road Infrastructure**")
        traffic_signal = st.checkbox("Traffic Signal Present")
        junction = st.checkbox("Junction Present")
        crossing = st.checkbox("Crossing Present")
        
        
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔍 Predict Severity Risk", type="primary", use_container_width=True):
        try:
            # Create input DataFrame
            input_df = create_input_dataframe(
                hour, visibility, weather, temperature, humidity,
                wind_speed, traffic_signal, junction, crossing
            )
            
            # Preprocess input
            with st.spinner("Processing inputs..."):
                df_processed = preprocess_input(input_df, feature_names)
            
            # Make prediction
            with st.spinner("Generating prediction..."):
                prediction, probability = make_prediction(model, df_processed, threshold)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Risk category with color coding
            if prediction == "High Severity":
                st.error(f"### 🔴 {prediction}")
            else:
                st.success(f"### 🟢 {prediction}")
            
            # Probability display
            st.metric(
                label="Risk Probability", 
                value=f"{probability * 100:.1f}%",
                delta=f"{(probability - threshold) * 100:+.1f}% from threshold"
            )
            
            # Progress bar for visual representation
            st.progress(probability)
            
            # Explanation
            st.markdown("### 📝 Risk Assessment")
            explanation = get_risk_explanation(prediction, probability)
            st.info(explanation)
            
            # Additional details
            with st.expander("🔍 Input Summary"):
                st.write(input_df.transpose())
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.write("Please check your inputs and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "Road Accident Severity Predictor | Traffic Safety Analytics | 2026"
        "</div>", 
        unsafe_allow_html=True
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
