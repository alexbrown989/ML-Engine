# database/backtest_model.py

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from inference import load_latest_model, generate_predictions
from build_features import calculate_features
from sklearn.metrics import classification_report, confusion_matrix

def backtest(ticker="AAPL"):
    print(f"Starting backtest for {ticker}...")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)

    df = yf.download(ticker, start=start_date, end=end_date)
    df["entry_price"] = df["Open"]

    # 👇 Fix column format
    df.columns = ['_'.join([str(c) for c in col if c]) if isinstance(col, tuple) else str(col) for col in df.columns]
    print(f"\n🧠 Columns before processing:\n{df.columns}")

    df = calculate_features(df)
    model = load_latest_model()

    if model is None:
        print("❌ Model not loaded. Exiting.")
        return

    df = generate_predictions(model, df)

    # 🎯 Print prediction samples
    print("\n🔍 Predictions preview:")
    print(df[['prediction', 'confidence']].head())

    # Metrics only if labels are present
    if 'outcome_class' in df.columns:
        print("\n📊 Classification Report:")
        print(classification_report(df['outcome_class'], df['prediction'], zero_division=0))

        print("\n📉 Confusion Matrix:")
        print(confusion_matrix(df['outcome_class'], df['prediction']))

        print("\n🧠 Accuracy by regime:")
        if 'regime' in df.columns:
            for regime in df['regime'].unique():
                sub = df[df['regime'] == regime]
                acc = (sub['prediction'] == sub['outcome_class']).mean()
                print(f"  - {regime}: {acc:.2%}")

if __name__ == "__main__":
    backtest()
