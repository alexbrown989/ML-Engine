import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from build_features import calculate_features



def backtest():
    ticker = "AAPL"
    print(f"Starting backtest for {ticker}...")

    end = datetime.today()
    start = end - timedelta(days=60)

    print(f"Fetching data for {ticker} from {start.date()} to {end.date()}")
    df = yf.download(ticker, start=start, end=end)
    df['entry_price'] = df['Close']

    df = df.reset_index()
    
    df.columns.name = None
    print("\n🧠 Columns before MultiIndex:")
    print(df.columns)

    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])

    df = calculate_features(df)
    print(df.tail())


if __name__ == "__main__":
    backtest()


