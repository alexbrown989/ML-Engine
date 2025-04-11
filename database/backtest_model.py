import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 🔧 Fix for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.build_features import calculate_features
from database.inference import load_model_and_features, generate_predictions

# === CONFIG === #
TICKER = "AAPL"
DAYS_BACK = 60

# === BACKTEST FUNCTION === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK)

    df = yf.download(TICKER, start=start_date, end=end_date)

    if df.empty:
        print("❌ Failed to fetch data. Exiting.")
        return

    df['entry_price'] = df['Open'].shift(-1)

    # Simulate realistic complex dataset structure
    df.columns = pd.MultiIndex.from_product([['Price'], df.columns])
    print("\n\U0001F9E0 Columns before processing:")
    print(df.columns)

    df.columns = ['_'.join(filter(None, map(str, col))).lower() for col in df.columns]

    print("\n🔧 Calculating features...")
    df = calculate_features(df)
    df.dropna(inplace=True)

    model, expected_features = load_model_and_features()
    if model is None:
        print("❌ No model found. Train one first.")
        return

    for feature in expected_features:
        if feature not in df.columns:
            print(f"⚠️ Missing expected feature: {feature}. Adding NaNs.")
            df[feature] = pd.NA

    df = df[expected_features].copy()
    df.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df)
    df = pd.concat([df, preds], axis=1)

    print("\n📊 Backtest complete. Sample predictions:")
    print(df[['prediction', 'confidence']].tail())

# === ENTRY POINT === #
if __name__ == "__main__":
    backtest()

