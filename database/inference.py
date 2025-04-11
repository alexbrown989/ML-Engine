# database/inference.py

import os
import pickle
import pandas as pd
from build_features import calculate_features

MODEL_DIR = "models"

def load_latest_model():
    """Load the latest saved XGBoost model."""
    try:
        model_files = sorted(
            [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
            reverse=True
        )
        if not model_files:
            print("❌ No model files found.")
            return None

        latest_model_path = os.path.join(MODEL_DIR, model_files[0])
        with open(latest_model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Loaded model: {latest_model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def generate_predictions(model, df):
    """Run inference on a DataFrame using the provided model."""
    features = df.drop(columns=["outcome_class"], errors="ignore")
    preds = model.predict(features)
    proba = model.predict_proba(features)

    df['prediction'] = preds
    df['confidence'] = proba.max(axis=1)
    return df

