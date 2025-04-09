# database/risk_filter.py

import sqlite3
import pandas as pd
from datetime import datetime

def load_predictions(min_confidence=0.7):
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query('''
        SELECT p.*, s.ticker, s.timestamp
        FROM predictions p
        JOIN signals s ON s.id = p.signal_id
        WHERE p.confidence >= ?
        ORDER BY p.timestamp DESC
    ''', conn, params=(min_confidence,))
    conn.close()
    return df

def apply_rules(df):
    print("🔍 Running risk filters...")

    df["risk_reason"] = ""
    df["risk_pass"] = True

    for idx, row in df.iterrows():
        reasons = []

        # Example filters
        if row["confidence"] < 0.8:
            reasons.append("Below 80% confidence")

        if "tech" in row.get("ticker", "").lower():
            reasons.append("Too many tech tickers")

        if row["confidence_band"] == "LOW":
            reasons.append("Low ML confidence")

        if reasons:
            df.at[idx, "risk_pass"] = False
            df.at[idx, "risk_reason"] = ", ".join(reasons)

    return df

def run_filter():
    print(f"⚙️ Running risk filter at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df = load_predictions(min_confidence=0.7)

    if df.empty:
        print("❌ No predictions meet minimum confidence threshold.")
        return

    df_filtered = apply_rules(df)
    passed = df_filtered[df_filtered["risk_pass"]]
    failed = df_filtered[~df_filtered["risk_pass"]]

    print(f"\n✅ {len(passed)} signals passed filter.")
    print(f"❌ {len(failed)} signals filtered out.")

    if not passed.empty:
        print("\n🚀 Approved Signals:")
        print(passed[["signal_id", "ticker", "confidence", "confidence_band"]])

    if not failed.empty:
        print("\n⚠️ Rejected Signals and Reasons:")
        print(failed[["signal_id", "confidence", "risk_reason"]])

if __name__ == "__main__":
    run_filter()
