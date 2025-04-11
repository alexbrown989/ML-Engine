import os
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from database.build_features import calculate_features
from database.inference import load_model_and_features, generate_predictions

# === CONFIG ===
TICKER = "AAPL"
DAYS_BACK = 60

def backtest():
    print(f"Starting backtest for {TICKER}...")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK)

    df = yf.download(TICKER, start=start_date, end=end_date)

    if df.empty:
        print("❌ Failed to fetch data.")
        return

    # Add dummy entry price (improve later)
    df["entry_price"] = df["Open"].shift(-1)

    print("\n🧠 Columns before processing:")
    print(df.columns)

    # Flatten columns if MultiIndex or named
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]).lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    # Feature engineering
    print("\n🔧 Calculating features...")
    df = calculate_features(df)
    df.dropna(inplace=True)

    model, expected_features = load_model_and_features()
    if model is None:
        print("❌ No model found.")
        return

    # Inject missing columns if needed
    for feature in expected_features:
        if feature not in df.columns:
            print(f"⚠️ Adding missing feature: {feature}")
            df[feature] = 0

    df = df[expected_features]
    df.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df)

    print("\n📊 Predictions sample:")
    print(preds.tail())

if __name__ == "__main__":
    backtest()
