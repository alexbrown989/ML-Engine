import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.build_features import calculate_features
from database.inference import load_model_and_features, generate_predictions


def backtest(ticker="AAPL", lookback_days=60):
    print(f"Starting backtest for {ticker}...")

    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

    if df.empty:
        print(f"❌ No data downloaded for {ticker}.")
        return

    df["entry_price"] = df["Close"]  # Dummy entry price for now

    print("\n🧠 Columns before processing:")
    print(df.columns)

    df.columns = [f"{col.lower()}_{ticker}" if col != "entry_price" else "entry_price" for col in df.columns]

    # Feature engineering
    print("\n🔧 Calculating features...")
    df = calculate_features(df)

    # Predict
    print("\n🧠 Generating predictions...")
    model, required_features = load_model_and_features()
    df_preds = generate_predictions(df, model, required_features)

    print("\n🔍 Sample predictions:")
    print(df_preds[["prediction", "confidence", "confidence_band"]].head())

    print("\n✅ Backtest complete.")


if __name__ == "__main__":
    backtest()
