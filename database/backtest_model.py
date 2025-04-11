import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# === PATCH IMPORT PATH === #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from build_features import calculate_features
from inference import load_model_and_features, generate_predictions

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

    if df.empty:
        print("❌ Failed to fetch data. Exiting.")
        return

    # Add dummy entry_price column (improve later with fill logic)
    df['entry_price'] = df['Open'].shift(-1)

    print("\n\U0001F9E0 Columns before processing:")
    print(df.columns)

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().lower() for col in df.columns.values]
    else:
        df.columns = [col.lower() for col in df.columns]

    # Feature engineering
    print("\n🔧 Calculating features...")
    df = calculate_features(df)
    df.dropna(inplace=True)

    # Load model
    model, expected_features = load_model_and_features()
    if model is None:
        print("❌ No model found. Train one first.")
        return

    for feature in expected_features:
        if feature not in df.columns:
            print(f"⚠️ Missing feature: {feature}. Filling with NaNs.")
            df[feature] = pd.NA

    df = df[expected_features].copy()
    df.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df)
    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    print("\n📊 Sample predictions:")
    print(df[['prediction', 'confidence']].tail())

# === ENTRY POINT === #
if __name__ == "__main__":
    backtest()
