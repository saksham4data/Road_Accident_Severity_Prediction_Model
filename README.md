# Road Accident Severity Prediction Model

## Overview
This project is a data science–based decision support system designed to analyze and predict road accident severity.  
The system uses historical accident data to identify high-risk conditions and predicts whether an accident is likely to be **low severity** or **high severity**.

---

## Problem Statement
Road accidents cause significant loss of life and resources every year.  
Despite the availability of accident data, actionable insights for predicting accident severity under varying conditions are limited.

This project addresses this gap by:
- Analyzing accident patterns
- Identifying high-risk conditions
- Predicting accident severity using machine learning

---

## Key Features
- Binary accident severity prediction (Low / High)
- Safety-oriented threshold tuning
- Random Forest–based prediction model
- Streamlit web application for interactive risk assessment
- Clean separation of experimentation, training, and deployment

---

## Project Structure
accident-predictor/
│
├── train_model.py # Model training and evaluation script
├── streamlit_app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── models/ # Saved models and artifacts
│ ├── accident_model_.pkl
│ └── feature_importance_.csv
├── Data/
│ └── accident_clean_data.csv
├── notebooks/
│ ├── preprocessing.ipynb
│ └── eda.ipynb
└── README.md

---

## Machine Learning Approach
- **Problem Type:** Binary classification  
- **Baseline Model:** Decision Tree  
- **Main Model:** Random Forest  
- **Evaluation Focus:** Recall for high-severity accidents  
- **Threshold Tuning:** Applied to prioritize safety over raw accuracy  

The model is designed as a **decision-support system**, not an automated enforcement tool.

---

## Web Application
The Streamlit application allows users to:
- Enter environmental, temporal, and road conditions
- Receive severity risk prediction
- View probability-based risk score
- Understand risk through human-readable explanations

The application uses a **pre-trained model** and does not perform retraining during inference.

---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```


## Disclaimer
This system is intended for decision support only.
Predictions should not be used as the sole basis for enforcement or safety decisions.

---

## Author
Saksham Nagar