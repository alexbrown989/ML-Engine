# /workspaces/ML-Engine/database/train_model.py

import os
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"
MAX_MODELS = 5

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)

    filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved: {path}")

    # Cleanup old models
    all_models = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
        reverse=True
    )
    for old_model in all_models[MAX_MODELS:]:
        os.remove(os.path.join(MODEL_DIR, old_model))
        print(f"🗑️ Deleted old model: {old_model}")

def train_model():
    print(f"\n🔄 Training model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features_ext WHERE outcome_class IS NOT NULL", conn)
    conn.close()

    if df.empty:
        print("❌ No data to train on.")
        return

    print("\n🧠 Column types before preprocessing:")
    print(df.dtypes)
    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())

    df["regime"] = df.get("regime", "calm")
    df["regime"] = df["regime"].astype("category")

    df["vvs_roc_5d"] = pd.to_numeric(df["vvs_roc_5d"], errors="coerce")
    df["vvs_roc_5d"].fillna(df["vvs_roc_5d"].mean(numeric_only=True), inplace=True)

    print("\n🔍 Column types after cleaning:")
    print(df.dtypes)

    X = df.drop(columns=["outcome_class"])
    y = df["outcome_class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"\n📊 Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    model = xgb.XGBClassifier(
        enable_categorical=True,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    print("🔧 Fitting model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model accuracy: {accuracy * 100:.2f}%")

    save_model(model)

if __name__ == "__main__":
    train_model()

