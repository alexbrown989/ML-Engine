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
    df['vvs_adj'] = (df['vix'] + df['vvix']) / df['skew']
    df['vvs_roc_5d'] = None  # placeholder
    df['chop_flag'] = 0  # fallback default

    # One-hot encode 'regime' column
    df['regime'] = df['regime'].astype(str)
    df = pd.get_dummies(df, columns=['regime'], drop_first=True)

    # Handle missing values and other columns as necessary
    for regime in ['regime_calm', 'regime_panic', 'regime_transition']:
        if regime not in df.columns:
            df[regime] = np.nan

    df.set_index("id", inplace=True)
    return df

def run_inference():
    print(f"🚀 Starting inference at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df_raw = get_unlabeled_signals()
    print(f"\n📥 Fetched {len(df_raw)} new signals to score.")
    if df_raw.empty:
        print("❌ No new signals to score. Exiting.")
        return

    print(f"🔍 Preparing features for {len(df_raw)} signals...")

    features = build_features(df_raw)
    print("🧠 Feature DataFrame preview:")
    print(features.head())

    model = load_model()
    expected_cols = model.get_booster().feature_names
    print("🔎 Model expects features:",)

