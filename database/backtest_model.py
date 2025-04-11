import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database.build_features import calculate_features
from database.inference import load_latest_model, generate_predictions

# === BACKTEST CONFIG === #
TICKER = "AAPL"
DAYS_BACK = 60

# === BACKTEST FUNCTION === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK)

    # Download historical OHLCV data
    df = yf.download(TICKER, start=start_date, end=end_date)

    # Ensure DataFrame is not empty
    if df.empty:
        print("❌ Failed to fetch data. Exiting.")
        return

    # Add dummy entry_price column (you can improve this to realistic fills)
    df['entry_price'] = df['Open'].shift(-1)

    # Make MultiIndex explicit (simulate realistic complex dataset)
    df.columns = pd.MultiIndex.from_product([["Price"], df.columns])
    print("\n\U0001F9E0 Columns before processing:")
    print(df.columns)

    # Flatten column names cleanly
    df.columns = ['_'.join(filter(None, map(str, col))).lower() for col in df.columns]

    # Feature engineering
    print("\n🔧 Calculating features...")
    df = calculate_features(df)

    # Drop rows with NA from recent indicator calcs
    df.dropna(inplace=True)

    # Load model
    model, expected_features = load_latest_model()
    if model is None:
        print("❌ No model found. Train one first.")
        return

    # Ensure all required features are present
    for feature in expected_features:
        if feature not in df.columns:
            print(f"⚠️ Missing expected feature: {feature}. Adding NaNs.")
            df[feature] = pd.NA

    df = df[expected_features].copy()
    df.fillna(0, inplace=True)

    # Run inference
    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df)
    df['prediction'] = preds

    print("\n📊 Backtest complete. Sample predictions:")
    print(df[['prediction']].tail())

# === ENTRY POINT === #
if __name__ == "__main__":
    backtest()
