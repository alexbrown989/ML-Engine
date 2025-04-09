import sqlite3
import pandas as pd
from datetime import datetime

def apply_risk_filters():
    print(f"\n🚦 Starting risk filter @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect("signals.db")
    preds = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)

    if preds.empty:
        print("❌ No predictions found. Exiting.")
        return

    print(f"📥 Loaded {len(preds)} predictions")

    # Drop duplicates
    preds = preds.drop_duplicates(subset=["signal_id"])
    print(f"📎 Deduplicated to {len(preds)} signals")

    # JOIN for checklist scores from signals
    signals = pd.read_sql_query("SELECT id, checklist_score FROM signals", conn)
    signals.rename(columns={"id": "signal_id"}, inplace=True)
    preds = preds.merge(signals, on="signal_id", how="left")

    print("🧠 Preview merged predictions:")
    print(preds.head())

    # ✅ Rule 1: Confidence ≥ 0.70
    preds = preds[preds["confidence"] >= 0.70]
    print(f"🎯 Passed confidence filter: {len(preds)} signals")

    # ✅ Rule 2: Checklist Score ≥ 3
    preds = preds[preds["checklist_score"] >= 3]
    print(f"✅ Passed checklist score filter: {len(preds)} signals")

    # ✅ Rule 3: Limit to top N signals by confidence
    N = 3
    preds = preds.sort_values(by="confidence", ascending=False).head(N)
    print(f"⛔️ Throttled to top {N} signals")

    # Decision logic
    preds["final_decision"] = preds["prediction"].apply(
        lambda x: "ENTER" if x == 1 else "WAIT"
    )

    # Save results
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filtered_signals (
            signal_id INTEGER PRIMARY KEY,
            prediction INTEGER,
            confidence REAL,
            confidence_band TEXT,
            final_decision TEXT,
            checklist_score INTEGER,
            timestamp TEXT
        )
    """)

    for _, row in preds.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO filtered_signals
            (signal_id, prediction, confidence, confidence_band, final_decision, checklist_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row["signal_id"], row["prediction"], row["confidence"],
            row["confidence_band"], row["final_decision"],
            row["checklist_score"], row["timestamp"]
        ))
        print(f"✅ Signal {row['signal_id']} → {row['final_decision']} (conf: {row['confidence']:.2f})")

    conn.commit()
    conn.close()
    print("🎉 Risk filter completed and stored results.")

if __name__ == "__main__":
    apply_risk_filters()
