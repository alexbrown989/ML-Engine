import sqlite3
import pandas as pd
import pickle
import numpy as np
import os
from datetime import datetime

def prepare_features(signal_rows):
    print(f"🔍 Preparing features for {len(signal_rows)} signals...")
    df = pd.DataFrame(signal_rows, columns=[
        "signal_id", "vix", "vvix", "skew", "rsi", "regime", "checklist_score"
    ])

    # Show raw incoming data
    print("🧾 First few raw signals:\n", df.head())

    df["vvs_adj"] = (df["vix"] + df["vvix"]) / df["skew"]
    df["vvs_roc_5d"] = np.nan  # Placeholder

    df = pd.get_dummies(df, columns=["regime"])

    df = df.set_index("signal_id")

    print("🧠 Feature DataFrame preview:\n", df.head())
    print("🧠 Feature columns:", df.columns.tolist())
    return df

def map_prediction(pred, conf):
    return {
        "prediction": int(pred),
        "confidence": round(float(conf), 2),
        "entry_readiness": (
            "HIGH" if conf > 0.85 else
            "MEDIUM" if conf > 0.70 else
            "LOW"
        ),
        "suggested_action": "ENTER" if conf > 0.70 else "WAIT"
    }

def run_inference():
    print("🚀 Starting inference at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not os.path.exists("model_xgb.pkl"):
        print("❌ Model not found. Run train_model.py first.")
        return

    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)

    conn = sqlite3.connect("signals.db")
    cursor = conn.cursor()

    # Create predictions table if needed
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            signal_id INTEGER PRIMARY KEY,
            prediction INTEGER,
            confidence REAL,
            entry_readiness TEXT,
            suggested_action TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Get signals that haven't been scored yet
    cursor.execute("""
        SELECT s.id, s.vix, s.vvix, s.skew, s.rsi, s.regime, s.checklist_score
        FROM signals s
        LEFT JOIN predictions p ON s.id = p.signal_id
        WHERE p.signal_id IS NULL
    """)
    rows = cursor.fetchall()

    if not rows:
        print("ℹ️ No new signals to score.")
        conn.close()
        return

    print(f"📥 Fetched {len(rows)} new signals to score.")

    # Prepare matching feature set
    features_df = prepare_features(rows)

    model_features = model.get_booster().feature_names
    print("🔎 Model expects features:", model_features)

    for col in model_features:
        if col not in features_df.columns:
            print(f"⚠️ Missing feature '{col}' — filling with 0")
            features_df[col] = 0

    features_df = features_df[model_features]
    features_df = features_df.fillna(0)

    print("✅ Final feature matrix shape:", features_df.shape)

    proba = model.predict_proba(features_df)
    preds = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)

    print("\n📊 Sample prediction probabilities:")
    for i, row in enumerate(proba[:3]):
        print(f"  Signal {features_df.index[i]}: {row} (pred={preds[i]}, conf={confidences[i]:.2f})")

    # Insert results
    for signal_id, pred, conf in zip(features_df.index, preds, confidences):
        result = map_prediction(pred, conf)
        cursor.execute("""
            INSERT OR REPLACE INTO predictions (
                signal_id, prediction, confidence, entry_readiness, suggested_action
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            signal_id,
            result["prediction"],
            result["confidence"],
            result["entry_readiness"],
            result["suggested_action"]
        ))
        print(f"✅ Stored: signal_id={signal_id}, pred={result['prediction']}, conf={result['confidence']} ({result['entry_readiness']})")

    conn.commit()
    conn.close()
    print("✅ Inference complete and all results stored.")

if __name__ == "__main__":
    run_inference()
