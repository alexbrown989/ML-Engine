import pandas as pd
import numpy as np

# Feature calculation function
def calculate_features(df):
    print("\n🔧 Calculating features...")

    # Check columns before processing
    print(f"🧠 Columns before processing: {df.columns}")

    # 1. Volatility + Regime Signals
    print("🔍 Adding Volatility and Regime features...")

    # Ensure all numerical features are numeric
    df['vix'] = pd.to_numeric(df['vix'], errors='coerce')
    df['vvix'] = pd.to_numeric(df['vvix'], errors='coerce')
    df['skew'] = pd.to_numeric(df['skew'], errors='coerce')

    # Slow regime detection (vvs_adj)
    df['skew_normalized'] = (df['skew'] - df['skew'].rolling(30).mean()) / df['skew'].rolling(30).std()
    df['vvs_adj'] = (df['vix'] + df['vvix']) / df['skew_normalized']
    print(f"🧠 vvs_adj calculated. Sample: {df['vvs_adj'].head()}")

    # 5-Day Rate of Change for vvs_adj
    df['vvs_roc_5d'] = df['vvs_adj'] - df['vvs_adj'].shift(5)
    print(f"🧠 vvs_roc_5d calculated. Sample: {df['vvs_roc_5d'].head()}")

    # Fill missing 'regime' data before one-hot encoding
    print("🔍 Handling missing regime data...")
    if 'regime' in df.columns:
        df['regime'].fillna('calm', inplace=True)  # Fill with 'calm' or other logic
        print(f"🧠 Missing regime values filled. Sample: {df['regime'].head()}")
    else:
        print("⚠️ 'regime' column missing! Adding 'calm' as default.")
        df['regime'] = 'calm'

    # One-hot encoding for regime
    df = pd.get_dummies(df, columns=['regime'], drop_first=True)  # One-hot encoding
    print(f"🧠 One-hot encoded regime. Columns: {df.columns}")

    # 2. Momentum, Price, and Volume Behavior
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

    # 3. Sentiment + News signals
    print("🔍 Adding Sentiment + News signals...")
    df['news_sentiment_score'] = np.random.uniform(-1, 1, len(df))  # Simulating news sentiment score
    print(f"🧠 News sentiment score added. Sample: {df['news_sentiment_score'].head()}")

    df['macro_event_proximity'] = np.random.randint(1, 100, len(df))  # Random proximity for macro events
    print(f"🧠 Macro event proximity added. Sample: {df['macro_event_proximity'].head()}")

    # 4. Earnings + Scheduled Event Risk
    print("🔍 Adding Earnings + Event Risk signals...")
    df['days_to_earnings'] = np.random.randint(1, 10, len(df))  # Random days to earnings
    print(f"🧠 Days to earnings added. Sample: {df['days_to_earnings'].head()}")

    df['pre_earnings_flag'] = (df['days_to_earnings'] <= 7).astype(int)
    print(f"🧠 Pre-earnings flag added. Sample: {df['pre_earnings_flag'].head()}")

    # 5. Options Chain Awareness
    print("🔍 Adding Options Chain Awareness features...")
    df['strike_distance_pct'] = np.random.uniform(-0.05, 0.05, len(df))  # Simulating strike distance
    print(f"🧠 Strike distance % calculated. Sample: {df['strike_distance_pct'].head()}")

    # 6. Learning Feedback (ML Reflector)
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

    # Handle missing values
    print("🔍 Handling missing data...")
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill numeric columns with mean
    df['regime'].fillna('calm', inplace=True)  # Fill categorical 'regime' with 'calm' or other logic

    print("\n🧠 Feature calculation complete!")
    return df

