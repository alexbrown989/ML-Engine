# database/backtest_model.py

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from database.build_features import calculate_features
from database.inference import load_latest_model, generate_predictions

TICKER = "AAPL"
DAYS_BACK = 60

def backtest():
    print(f"\n🚀 Starting backtest for {TICKER}...")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK)

    df = yf.download(TICKER, start=start_date, end=end_date)

    if df.empty:
        print("❌ No data returned. Check ticker or date range.")
        return

    df['entry_price'] = df['Open'].shift(-1)

    # Flatten MultiIndex to simple columns
    df.columns = ['_'.join(filter(None, map(str, col))).lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    print("\n🧠 Columns before feature calc:", df.columns.tolist())

    print("🔧 Calculating features...")
    df = calculate_features(df)
    df.dropna(inplace=True)

    model, expected_features = load_latest_model()
    if model is None:
        print("❌ No trained model found.")
        return

    for feature in expected_features:
        if feature not in df.columns:
            print(f"⚠️ Missing feature: {feature}. Filling with NaN.")
            df[feature] = pd.NA

    df = df[expected_features].copy()
    df.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df)
    df = df.join(preds)

    print("\n📊 Top Predictions:")
    print(df[['prediction', 'confidence']].sort_values(by='confidence', ascending=False).head(5))

if __name__ == "__main__":
    backtest()

