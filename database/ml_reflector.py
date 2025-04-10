import sqlite3
import pandas as pd
from datetime import datetime

def run_reflection():
    print(f"\n🪞 Starting reflection engine @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect("signals.db")

    # Load predictions
    preds = pd.read_sql_query("""
        SELECT signal_id, prediction, confidence, confidence_band, timestamp
        FROM predictions
    """, conn)
    print(f"📥 Loaded {len(preds)} predictions")

    # Load true labels
    labels = pd.read_sql_query("""
        SELECT signal_id, outcome_class
        FROM labels
        WHERE outcome_class IS NOT NULL
    """, conn)

    # Load extra info (regime, checklist, etc.)
    signals = pd.read_sql_query("""
        SELECT id AS signal_id, regime, checklist_score
        FROM signals
    """, conn)

    # Debug: Check loaded data
    print(f"\n🧠 Loaded true labels: {len(labels)} labels")
    print(f"\n🧠 Loaded extra signal data: {len(signals)} signals")
    
    # Merge all dataframes
    df = preds.merge(labels, on="signal_id", how="inner")
    df = df.merge(signals, on="signal_id", how="left")

    print(f"🧠 After joins: {len(df)} matched predictions")

    if df.empty:
        print("❌ No matches found between predictions and labels. Exiting.")
        return

    # Debug: Display a preview of the merged dataframe
    print("\n🔎 Sample merged data:")
    print(df.head())

    # Compute reflection columns: correct or incorrect predictions
    df["is_correct"] = (df["prediction"] == df["outcome_class"]).astype(int)
    df["reflect_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Show preview of the reflection data
    print("🔎 Sample reflection results:")
    print(df[["signal_id", "prediction", "outcome_class", "is_correct", "confidence", "confidence_band", "regime"]].head())

    # Create table for reflections if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reflections (
            signal_id INTEGER PRIMARY KEY,
            prediction INTEGER,
            actual INTEGER,
            is_correct INTEGER,
            confidence REAL,
            confidence_band TEXT,
            regime TEXT,
            checklist_score INTEGER,
            prediction_time TEXT,
            reflect_timestamp TEXT
        )
    """)

    # Insert reflections into the reflections table
    try:
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO reflections (
                    signal_id, prediction, actual, is_correct,
                    confidence, confidence_band, regime, checklist_score,
                    prediction_time, reflect_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["signal_id"],
                row["prediction"],
                row["outcome_class"],
                row["is_correct"],
                row["confidence"],
                row["confidence_band"],
                row["regime"],
                row["checklist_score"],
                row["timestamp"],
                row["reflect_timestamp"]
            ))
        
        conn.commit()
        print(f"✅ Reflections saved for {len(df)} signals.")
    except Exception as e:
        print(f"❌ Error during reflections insertion: {str(e)}")
    
    conn.close()

if __name__ == "__main__":
    run_reflection()

