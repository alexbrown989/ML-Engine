# === database/build_features.py ===
import pandas as pd
import numpy as np


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n\U0001F527 Calculating features...")

    df['regime'] = df.get('regime', 'calm')
    df['regime'] = df['regime'].fillna('calm').astype(str)

    # Fallback defaults
    if 'checklist_score' not in df.columns:
        df['checklist_score'] = 2
    if 'chop_flag' not in df.columns:
        df['chop_flag'] = 0

    # Convert to numeric with coercion
    for col in ['vix', 'vvix', 'skew', 'rsi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rolling features
    df['skew_normalized'] = (df['skew'] - df['skew'].rolling(30).mean()) / df['skew'].rolling(30).std()
    df['vvs_adj'] = (df['vix'] + df['vvix']) / df['skew_normalized']
    df['vvs_roc_5d'] = df['vvs_adj'] - df['vvs_adj'].shift(5)

    # Momentum & volatility
    df['macd_hist'] = df['vix'] - df['vvix']
    df['obv_roc_5d'] = df['vix'].diff(5)
    df['volume_change_pct'] = (df['vvix'] - df['vvix'].shift(5)) / df['vvix'].shift(5)

    # Stubs or external integrations
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

    # One-hot encode regime
    df = pd.get_dummies(df, columns=['regime'], prefix='regime', drop_first=False)

    # Final fill
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df


