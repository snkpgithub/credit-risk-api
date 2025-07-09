import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Paths
RAW_DATA_PATH = "data/raw/application_train.csv"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def preprocess(df):
    df = df.drop(columns=["SK_ID_CURR"], errors='ignore')

    # Drop columns with >40% missing values
    df = df.loc[:, df.isnull().mean() < 0.4]

    # Fill missing numeric columns with median
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    # ✅ Save feature names before scaling
    feature_names = list(X.columns)
    joblib.dump(feature_names, f"{MODEL_DIR}/xgb_features.pkl")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders

def train_models(X_train, y_train):
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    return lr, xgb

def evaluate(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

def save_artifacts(model, scaler, encoders, name="xgb"):
    joblib.dump(model, f"{MODEL_DIR}/{name}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{name}_scaler.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/{name}_encoders.pkl")
    print("Artifacts saved.")

if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    X, y, scaler, encoders = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr, xgb = train_models(X_train, y_train)
    evaluate(lr, X_test, y_test, "Logistic Regression")
    evaluate(xgb, X_test, y_test, "XGBoost")

    save_artifacts(xgb, scaler, encoders, name="xgb")
