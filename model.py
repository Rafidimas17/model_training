import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Load the saved models and preprocessing objects
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')
knn_model_sys = joblib.load('knn_model_sys.pkl')
knn_model_dia = joblib.load('knn_model_dia.pkl')

def predict_blood_pressure(hr_2, spo2, age):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'age': [age],
        'hr_2': [hr_2],
        'spo2': [spo2]
    })

    # Impute missing values
    input_data_imputed = imputer.transform(input_data)

    # Add condition features based on reasonable defaults or assumptions
    input_data['hr_high_bp_high'] = (input_data['hr_2'] > 100).astype(int)
    input_data['hr_low_bp_low'] = (input_data['hr_2'] < 60).astype(int)
    input_data['spo2_low_bp_low'] = (input_data['spo2'] < 95).astype(int)
    input_data['spo2_low_hr_high'] = (input_data['spo2'] < 95).astype(int)

    # Combine features and condition features
    features_combined = np.hstack((input_data_imputed, input_data[['hr_high_bp_high', 'hr_low_bp_low', 'spo2_low_bp_low', 'spo2_low_hr_high']].values))

    # Scale features
    features_scaled = scaler.transform(features_combined)

    # Add polynomial features
    features_poly = poly.transform(features_scaled)

    # Predict systolic and diastolic blood pressure
    bp_sys_pred = knn_model_sys.predict(features_poly)
    bp_dia_pred = knn_model_dia.predict(features_poly)

    return bp_sys_pred[0], bp_dia_pred[0]

# Example input
hr_2 = 79
spo2 = 96
age = 26

# Get predictions
bp_sys_pred, bp_dia_pred = predict_blood_pressure(hr_2, spo2, age)
print(f'Predicted Systolic Blood Pressure: {bp_sys_pred:.2f}')
print(f'Predicted Diastolic Blood Pressure: {bp_dia_pred:.2f}')
