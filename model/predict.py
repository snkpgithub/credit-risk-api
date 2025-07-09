import joblib
import pandas as pd

# Load model artifacts
MODEL_PATH = "models/xgb_model.pkl"
SCALER_PATH = "models/xgb_scaler.pkl"
ENCODER_PATH = "models/xgb_encoders.pkl"
FEATURE_PATH = "models/xgb_features.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_PATH)

def predict_credit_risk(input_data: dict):
    """
    Predicts default risk for a single input dictionary.
    Returns: risk_score (0–1), and High/Low Risk label.
    """
    df = pd.DataFrame([input_data])

    # Encode categorical fields
    for col, encoder in encoders.items():
        if col in df:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
                df[col] = encoder.transform([encoder.classes_[0]])

    # Add missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[feature_names]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    risk_label = "High Risk" if prob > 0.5 else "Low Risk"

    return {
        "risk_score": float(round(prob, 4)),
        "risk_label": risk_label
    }

def preprocess_for_model(input_data, encoders, scaler, feature_names):
    """
    Shared preprocessing used by both prediction and SHAP explanation.
    Returns: scaled DataFrame with aligned feature columns.
    """
    df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        if col in df:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
                df[col] = encoder.transform([encoder.classes_[0]])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    X_scaled = scaler.transform(df)

    return pd.DataFrame(X_scaled, columns=feature_names)

# Test script
if __name__ == "__main__":
    sample_input = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "OCCUPATION_TYPE": "Laborers",
        "CNT_CHILDREN": 0,
        "DAYS_BIRTH": -12005,
        "DAYS_EMPLOYED": -4542,
        "EXT_SOURCE_1": 0.1,
        "EXT_SOURCE_2": 0.4,
        "EXT_SOURCE_3": 0.5,
        "REGION_RATING_CLIENT": 2
    }

    result = predict_credit_risk(sample_input)
    print("Prediction Result:", result)
