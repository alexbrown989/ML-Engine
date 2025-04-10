import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# Feature calculation function
def calculate_features(df):
    print("\n🔧 Calculating features...")

    # 1. Handle Missing Values and Columns
    print("🔍 Handling missing data...")

    # Fill missing data for numerical columns with the mean
    df.fillna(df.mean(), inplace=True)

    # Fill categorical 'regime' with 'calm'
    df['regime'].fillna('calm', inplace=True)  

    # Ensure regime is in the correct format for one-hot encoding
    df['regime'] = df['regime'].astype(str)  # Ensure it's a string before one-hot encoding
    df = pd.get_dummies(df, columns=['regime'], drop_first=True)
    print(f"🧠 One-hot encoded regime. Columns: {df.columns}")

    # Handle missing columns like regime_calm, regime_panic, regime_transition
    for regime in ['regime_calm', 'regime_panic', 'regime_transition']:
        if regime not in df.columns:
            df[regime] = np.nan  # Or fill with 0 if you prefer
    print(f"🧠 Missing regime features handled. Columns: {df.columns}")

    # Ensure all numerical features are numeric
    df['vix'] = pd.to_numeric(df['vix'], errors='coerce')
    df['vvix'] = pd.to_numeric(df['vvix'], errors='coerce')
    df['skew'] = pd.to_numeric(df['skew'], errors='coerce')

    # 2. Volatility + Regime Signals
    print("🔍 Adding Volatility and Regime features...")
    df['skew_normalized'] = (df['skew'] - df['skew'].rolling(30).mean()) / df['skew'].rolling(30).std()
    df['vvs_adj'] = (df['vix'] + df['vvix']) / df['skew_normalized']
    print(f"🧠 vvs_adj calculated. Sample: {df['vvs_adj'].head()}")

    # 5-Day Rate of Change for vvs_adj
    df['vvs_roc_5d'] = df['vvs_adj'] - df['vvs_adj'].shift(5)
    print(f"🧠 vvs_roc_5d calculated. Sample: {df['vvs_roc_5d'].head()}")

    # 3. Momentum, Price, and Volume Behavior
    print("🔍 Adding Momentum, Price, and Volume features...")
    df['rsi'] = pd.to_numeric(df['rsi'], errors='coerce')  # Ensure RSI is numeric
    print(f"🧠 RSI converted. Sample: {df['rsi'].head()}")

    # Moving averages (for trend-following features)
    df['macd_hist'] = df['vix'] - df['vvix']  # Simple MACD proxy
    print(f"🧠 MACD histogram calculated. Sample: {df['macd_hist'].head()}")

    df['obv_roc_5d'] = df['vix'].diff(5)  # On-Balance Volume rate of change (using vix as proxy)
    print(f"🧠 OBV ROC 5d calculated. Sample: {df['obv_roc_5d'].head()}")

    # Price and volume changes
    df['volume_change_pct'] = (df['vvix'] - df['vvix'].shift(5)) / df['vvix'].shift(5)
    print(f"🧠 Volume change % calculated. Sample: {df['volume_change_pct'].head()}")

    # 4. Bollinger Bands and ATR (Average True Range)
    print("🔍 Adding Bollinger Bands and ATR features...")
    df['bollinger_upper'] = df['vix'].rolling(window=20).mean() + 2 * df['vix'].rolling(window=20).std()
    df['bollinger_lower'] = df['vix'].rolling(window=20).mean() - 2 * df['vix'].rolling(window=20).std()

    df['high_low'] = df['vix'] - df['vvix']
    df['high_close'] = abs(df['vix'] - df['vix'].shift(1))
    df['low_close'] = abs(df['vvix'] - df['vvix'].shift(1))
    df['ATR'] = df[['high_low', 'high_close', 'low_close']].mean(axis=1)
    print(f"🧠 Bollinger Bands and ATR calculated. Sample: {df[['bollinger_upper', 'bollinger_lower', 'ATR']].head()}")

    # 5. Sentiment + News signals
    print("🔍 Adding Sentiment + News signals...")
    df['news_sentiment_score'] = np.random.uniform(-1, 1, len(df))  # Simulating news sentiment score
    print(f"🧠 News sentiment score added. Sample: {df['news_sentiment_score'].head()}")

    df['macro_event_proximity'] = np.random.randint(1, 100, len(df))  # Random proximity for macro events
    print(f"🧠 Macro event proximity added. Sample: {df['macro_event_proximity'].head()}")

    # 6. Earnings + Scheduled Event Risk
    print("🔍 Adding Earnings + Event Risk signals...")
    df['days_to_earnings'] = np.random.randint(1, 10, len(df))  # Random days to earnings
    print(f"🧠 Days to earnings added. Sample: {df['days_to_earnings'].head()}")

    df['pre_earnings_flag'] = (df['days_to_earnings'] <= 7).astype(int)
    print(f"🧠 Pre-earnings flag added. Sample: {df['pre_earnings_flag'].head()}")

    # 7. Options Chain Awareness
    print("🔍 Adding Options Chain Awareness features...")
    df['strike_distance_pct'] = np.random.uniform(-0.05, 0.05, len(df))  # Simulating strike distance
    print(f"🧠 Strike distance % calculated. Sample: {df['strike_distance_pct'].head()}")

    # 8. Learning Feedback (ML Reflector)
    print("🔍 Adding Learning Feedback features...")
    df['actual_return_pct_5d'] = (df['vvix'] - df['vix']) / df['vix']  # Simulating 5-day return
    print(f"🧠 Actual return percentage for 5 days calculated. Sample: {df['actual_return_pct_5d'].head()}")

    df['confidence_band'] = np.select(
        [
            df['vvs_roc_5d'] < 0.5,
            (df['vvs_roc_5d'] >= 0.5) & (df['vvs_roc_5d'] < 0.8),
            df['vvs_roc_5d'] >= 0.8
        ],
        ['LOW', 'MEDIUM', 'HIGH'], 
        default='UNKNOWN'
    )
    print(f"🧠 Confidence band assigned. Sample: {df['confidence_band'].head()}")

    print("\n🧠 Feature calculation complete!")
    return df

# Example synthetic data for testing
num_samples = 100  # You can set it to any number of samples you want
synthetic_data = {
    'vix': np.random.uniform(10, 40, num_samples),
    'vvix': np.random.uniform(90, 150, num_samples),
    'skew': np.random.uniform(-2, 2, num_samples),
    'rsi': np.random.uniform(30, 70, num_samples),
    'regime': np.random.choice(['calm', 'panic', 'transition'], num_samples),
    'checklist_score': np.random.randint(1, 5, num_samples),
    'vvs_roc_5d': np.random.uniform(-0.5, 0.5, num_samples),
    'chop_flag': np.random.randint(0, 2, num_samples),
    'outcome_class': np.random.choice([0, 1, 2], num_samples)  # Random outcome
}

df_synthetic = pd.DataFrame(synthetic_data)

# Add new data to the SQL database
conn = sqlite3.connect("signals.db")
df_synthetic.to_sql('features_ext', conn, if_exists='append', index=False)
conn.close()

print(f"✅ Added {num_samples} synthetic samples to the database.")

# Now calculate features
df_synthetic = calculate_features(df_synthetic)

# Print the final DataFrame with features
print(df_synthetic.head())

