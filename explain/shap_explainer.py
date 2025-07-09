# explain/shap_explainer.py

import shap
import pandas as pd
import joblib
from model.predict import preprocess_for_model

# Load model and assets
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/xgb_scaler.pkl")
encoders = joblib.load("models/xgb_encoders.pkl")
feature_names = joblib.load("models/xgb_features.pkl")

# ✅ Use TreeExplainer for XGBoost
explainer = shap.TreeExplainer(model)

def get_shap_explanation(input_dict):
    df = preprocess_for_model(input_dict, encoders, scaler, feature_names)
    shap_values = explainer(df)

    contributions = []
    for i in range(len(feature_names)):
        contributions.append({
            "feature": feature_names[i],
            "value": float(df.iloc[0, i]),                   # ✅ Ensure value is a native float
            "impact": float(shap_values.values[0, i])        # ✅ Ensure impact is a native float
        })

    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)
    return contributions[:5]

