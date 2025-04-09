# database/inference.py

import sqlite3
import pandas as pd
import pickle
from datetime import datetime

def load_model(path="model_xgb.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_unlabeled_signals():
    conn = sqlite3.connect("signals.db")
    query = """
        SELECT id, vix, vvix, skew, rsi, regime, checklist_score
        FROM signals
        WHERE id NOT IN (SELECT signal_id FROM predictions)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def build_features(df_raw):
    df = df_raw.copy()
    df["vvs_adj"] = (df["vix"] + df["vvix"]) / df["skew"]
    df["vvs_roc_5d"] = None  # placeholder
    df["chop_flag"] = 0  # fallback default

    df = pd.get_dummies(df, columns=["regime"], prefix="regime")
    for col in ["regime_calm", "regime_panic", "regime_transition"]:
        if col not in df:
            df[col] = False
    df.set_index("id", inplace=True)
    return df

def map_confidence(prob):
    if prob >= 0.90:
        return "HIGH", "ENTER"
    elif prob >= 0.70:
        return "MEDIUM", "ENTER"
    else:
        return "LOW", "WAIT"

def store_predictions(results):
    conn = sqlite3.connect("signals.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            signal_id INTEGER PRIMARY KEY,
            prediction INTEGER,
            confidence REAL,
            confidence_band TEXT,
            suggested_action TEXT,
            timestamp TEXT
        )
    ''')
    for res in results:
        cursor.execute('''
            INSERT OR REPLACE INTO predictions
            (signal_id, prediction, confidence, confidence_band, suggested_action, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            res["signal_id"], res["prediction"], res["confidence"],
            res["confidence_band"], res["suggested_action"], res["timestamp"]
        ))
        print(f"✅ Stored: signal_id={res['signal_id']}, pred={res['prediction']}, conf={res['confidence']:.2f} ({res['confidence_band']})")
    conn.commit()
    conn.close()

def run_inference():
    print(f"🚀 Starting inference at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df_raw = get_unlabeled_signals()
    print(f"\n📥 Fetched {len(df_raw)} new signals to score.")
    if df_raw.empty:
        print("❌ No new signals to score. Exiting.")
        return

    print(f"🔍 Preparing features for {len(df_raw)} signals...")
    print("🧾 First few raw signals:")
    print(df_raw.head())

    features = build_features(df_raw)
    print("🧠 Feature DataFrame preview:")
    print(features.head())
    print("🧠 Feature columns:", list(features.columns))

    model = load_model()
    expected_cols = model.get_booster().feature_names
    print("🔎 Model expects features:", expected_cols)

    for col in expected_cols:
        if col not in features:
            print(f"⚠️ Missing feature '{col}' — filling with 0")
            features[col] = 0

    features = features[expected_cols]
    print("✅ Final feature matrix shape:", features.shape)

    probs = model.predict_proba(features)
    results = []

    print("\n📊 Sample prediction probabilities:")
    for i, (signal_id, prob) in enumerate(zip(features.index, probs)):
        pred = int(prob.argmax())
        confidence = float(prob[pred])
        band, action = map_confidence(confidence)
        print(f"  Signal {signal_id}: {prob} (pred={pred}, conf={confidence:.2f})")

        results.append({
            "signal_id": signal_id,
            "prediction": pred,
            "confidence": round(confidence, 2),
            "confidence_band": band,
            "suggested_action": action,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    store_predictions(results)
    print("✅ Inference complete and all results stored.")

if __name__ == "__main__":
    run_inference()
