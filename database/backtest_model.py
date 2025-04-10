import sqlite3
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import pickle
from build_features import calculate_features  # Import from build_features.py

def get_stock_data(ticker):
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")

    data = yf.download(ticker, start=start_date, end=end_date)
    
    if 'Adj Close' in data.columns:
        data['entry_price'] = data['Adj Close']
    else:
        print("Warning: 'Adj Close' not found, using 'Close'")
        data['entry_price'] = data['Close']

    data['Date'] = data.index
    data.set_index('Date', inplace=True)

    return data

def load_model():
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def backtest():
    tickers = ['AAPL', 'GOOG', 'TSLA', 'MSFT', 'META', 'AMZN', 'NVDA', 'PLTR']
    total_profit = 0

    for ticker in tickers:
        print(f"Starting backtest for {ticker}...")
        data = get_stock_data(ticker)
        data = calculate_features(data)  # Use real feature calculations

        model = load_model()
        features = ['rsi', 'vix', 'skew', 'checklist_score']  # Adjust to your actual features
        X = data[features]

        predictions = model.predict(X)
        data['prediction'] = predictions
        data['final_decision'] = data['prediction'].apply(lambda x: 'ENTER' if x == 1 else 'WAIT')

        data['price_change'] = data['Adj Close'].pct_change().shift(-1)
        data['profit'] = np.where(data['final_decision'] == 'ENTER', data['price_change'], 0)

        total_profit += data['profit'].sum()
        print(f"Backtest for {ticker} complete. Profit: {data['profit'].sum() * 100:.2f}%")
    
    print(f"\nTotal Profit from backtesting across all stocks: {total_profit * 100:.2f}%")

if __name__ == "__main__":
    backtest()

