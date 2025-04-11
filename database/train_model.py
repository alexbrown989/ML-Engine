# /workspaces/ML-Engine/database/train_model.py

import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    # Ensure regime exists and is categorical
    if "regime" not in df.columns:
        print("⚠️ 'regime' column missing! Adding default 'calm'")
        df["regime"] = "calm"
    df["regime"] = df["regime"].astype("category")

    # Make sure vvs_roc_5d is numeric
    df["vvs_roc_5d"] = pd.to_numeric(df["vvs_roc_5d"], errors="coerce")
    df["vvs_roc_5d"].fillna(df["vvs_roc_5d"].mean(numeric_only=True), inplace=True)

    print("\n🔍 Column types after cleaning:")
    print(df.dtypes)

    # Prepare features and label
    X = df.drop(columns=["outcome_class"])  # no 'signal_id' present
    y = df["outcome_class"].astype(int)

    # Split
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

    filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {filename}")

if __name__ == "__main__":
    train_model()
