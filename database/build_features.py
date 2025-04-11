import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# --- Synthetic data generation ---
def generate_synthetic_data(num_samples=100):
    return pd.DataFrame({
        'vix': np.random.uniform(10, 40, num_samples),
        'vvix': np.random.uniform(90, 150, num_samples),
        'skew': np.random.uniform(-2, 2, num_samples),
        'rsi': np.random.uniform(30, 70, num_samples),
        'regime': np.random.choice(['calm', 'panic', 'transition'], num_samples),
        'checklist_score': np.random.randint(1, 5, num_samples),
        'vvs_roc_5d': np.random.uniform(-0.5, 0.5, num_samples),
        'chop_flag': np.random.randint(0, 2, num_samples),
        'outcome_class': np.random.choice([0, 1, 2], num_samples)
    })

# --- Feature Engineering ---
def calculate_features(df):
    print("\n🔧 Calculating features...")

    df['regime'] = df.get('regime', 'calm')
    df['regime'] = df['regime'].fillna('calm').astype(str)

    df['vix'] = pd.to_numeric(df['vix'], errors='coerce')
    df['vvix'] = pd.to_numeric(df['vvix'], errors='coerce')
    df['skew'] = pd.to_numeric(df['skew'], errors='coerce')

    df['skew_normalized'] = (df['skew'] - df['skew'].rolling(30).mean()) / df['skew'].rolling(30).std()
    df['vvs_adj'] = (df['vix'] + df['vvix']) / df['skew_normalized']
    df['vvs_roc_5d'] = df['vvs_adj'] - df['vvs_adj'].shift(5)

    df['rsi'] = pd.to_numeric(df['rsi'], errors='coerce')
    df['macd_hist'] = df['vix'] - df['vvix']
    df['obv_roc_5d'] = df['vix'].diff(5)
    df['volume_change_pct'] = (df['vvix'] - df['vvix'].shift(5)) / df['vvix'].shift(5)
    df['news_sentiment_score'] = np.random.uniform(-1, 1, len(df))
    df['macro_event_proximity'] = np.random.randint(1, 100, len(df))
    df['days_to_earnings'] = np.random.randint(1, 10, len(df))
    df['pre_earnings_flag'] = (df['days_to_earnings'] <= 7).astype(int)
    df['strike_distance_pct'] = np.random.uniform(-0.05, 0.05, len(df))
    df['actual_return_pct_5d'] = (df['vvix'] - df['vix']) / df['vix']

    df['confidence_band'] = pd.cut(
        df['vvs_roc_5d'],
        bins=[-np.inf, 0.5, 0.8, np.inf],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )

    df = pd.get_dummies(df, columns=['regime'], prefix='regime', drop_first=False)

    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

# --- Insert into database ---
def main():
    num_samples = 100
    df = generate_synthetic_data(num_samples)
    df = calculate_features(df)

    conn = sqlite3.connect("signals.db")
    df.to_sql('features_ext', conn, if_exists='append', index=False)
    conn.close()
    print(f"✅ Added {num_samples} synthetic samples to the database.")

if __name__ == "__main__":
    main()


