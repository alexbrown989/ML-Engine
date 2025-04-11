# database/backtest_model.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from database.build_features import calculate_features
from database.inference import load_latest_model, generate_predictions


def fetch_macro_index(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, progress=False)["Close"]
    return data.rename(symbol)


def backtest(ticker="AAPL"):
    print(f"Starting backtest for {ticker}...")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)[
        ["Close", "High", "Low", "Open", "Volume"]
    ]
    df["entry_price"] = df["Close"].shift(-1)

    print("\n🧠 Columns before processing:")
    print(df.columns)

    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0].lower()}_{ticker}" for col in df.columns]
    else:
        df.columns = [col.lower() + f"_{ticker}" for col in df.columns]

    # Add macro fear indicators
    vix = fetch_macro_index("^VIX", start_date, end_date)
    vvix = fetch_macro_index("^VVIX", start_date, end_date)
    skew = fetch_macro_index("^SKEW", start_date, end_date)

    df = df.merge(vix, left_index=True, right_index=True, how="left")
    df = df.merge(vvix, left_index=True, right_index=True, how="left")
    df = df.merge(skew, left_index=True, right_index=True, how="left")

    df.rename(columns={"^VIX": "vix", "^VVIX": "vvix", "^SKEW": "skew"}, inplace=True)

    print("\n🔧 Calculating features...")
    df = calculate_features(df)

    print("\n🔍 Dropping rows with missing entry price...")
    df = df.dropna(subset=["entry_price"])

    print("\n🧠 Running inference...")
    model = load_latest_model()
    preds = generate_predictions(model, df)

    print("\n🔍 Final predictions:")
    print(preds[["prediction", "confidence"]].head())


if __name__ == "__main__":
    backtest()

