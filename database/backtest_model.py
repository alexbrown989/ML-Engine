# database/backtest_model.py

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from build_features import calculate_features
from inference import load_latest_model, generate_predictions


def fetch_market_index_data():
    end = datetime.today()
    start = end - timedelta(days=120)

    index_data = {
        "^VIX": "vix",
        "^VVIX": "vvix",
        "^SKEW": "skew"
    }

    df_combined = pd.DataFrame()

    for symbol, label in index_data.items():
        data = yf.download(symbol, start=start, end=end)["Close"].rename(label)
        df_combined = pd.concat([df_combined, data], axis=1)

    df_combined.fillna(method="ffill", inplace=True)
    return df_combined


def backtest(ticker="AAPL"):
    print(f"Starting backtest for {ticker}...")
    end = datetime.today()
    start = end - timedelta(days=60)

    df = yf.download(ticker, start=start, end=end)
    df["entry_price"] = df["Open"]  # simulate entry next open

    if df.empty:
        print("❌ No price data found.")
        return

    print("\n🧠 Columns before processing:")
    print(df.columns)

    # Normalize column names to lowercase
    df.columns = [col.lower() + f"_{ticker}" for col in df.columns]
    df.rename(columns={
        f"open_{ticker}": "open",
        f"high_{ticker}": "high",
        f"low_{ticker}": "low",
        f"close_{ticker}": "close",
        f"volume_{ticker}": "volume"
    }, inplace=True)

    df["ticker"] = ticker
    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)

    # Merge macro fear data
    macro = fetch_market_index_data().reset_index()
    df = pd.merge(df, macro, left_on="date", right_on="Date", how="left")
    df.drop(columns=["Date"], inplace=True)
    df.fillna(method="ffill", inplace=True)

    print("\n🔧 Calculating features...")
    df = calculate_features(df)

    print("\n🤖 Running inference...")
    model = load_latest_model()
    predictions = generate_predictions(model, df)

    print("\n✅ Backtest complete. Predictions:")
    print(predictions[["date", "ticker", "prediction", "confidence", "confidence_band"]].tail(10))


if __name__ == "__main__":
    backtest()

