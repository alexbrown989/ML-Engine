import os
import pickle
import pandas as pd
import xgboost as xgb

MODEL_DIR = "models"

def load_latest_model():
    files = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
        reverse=True
    )
    if not files:
        raise FileNotFoundError("❌ No trained model found in models directory.")

    model_path = os.path.join(MODEL_DIR, files[0])
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    booster = model.get_booster()
    feature_names = booster.feature_names

    print(f"🧠 Loaded model: {model_path} with {len(feature_names)} features")
    return model, feature_names

def load_model_and_features():
    # Alias to maintain compatibility
    return load_latest_model()

def generate_predictions(model, X):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    predictions = pd.DataFrame({
        "prediction": y_pred,
        "confidence": y_proba.max(axis=1)
    })
    return predictions

if __name__ == "__main__":
    # Quick smoke test
    from train_model import train_model
    train_model()
    model, features = load_latest_model()
    print(features)

