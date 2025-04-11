import sqlite3
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import pickle
from build_features import calculate_features  # Import feature calculation from build_features.py

# Function to fetch stock data from Yahoo Finance for a specific ticker
def get_stock_data(ticker):
    """Fetches stock data from Yahoo Finance for a specific ticker and date range."""
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')  # 60 days ago
    end_date = datetime.now().strftime('%Y-%m-%d')  # Today
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")

    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Print the column names to check the available columns
    print(f"Downloaded data columns for {ticker}: {data.columns}")

    # Use the adjusted closing prices to simulate "entry price"
    if 'Adj Close' in data.columns:
        data['entry_price'] = data['Adj Close']
    else:
        print(f"Warning: 'Adj Close' not found for {ticker}, using 'Close' instead.")
        data['entry_price'] = data['Close']
        
    # Set 'Date' as the index (necessary for backtesting)
    data['Date'] = data.index
    data.set_index('Date', inplace=True)

    return data

# Function to load the pre-trained model
def load_model():
    """Loads the pre-trained model."""
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Backtesting function
def backtest():
    """Main function to simulate backtesting on multiple stocks."""
    tickers = ['AAPL', 'GOOG', 'TSLA', 'MSFT', 'META', 'AMZN', 'NVDA', 'PLTR']
    total_profit = 0  # Variable to accumulate the total profit

    for ticker in tickers:
        print(f"Starting backtest for {ticker}...")
        data = get_stock_data(ticker)

        # Call build_features.py to add real features
        data = calculate_features(data)  # This function should calculate real features like RSI, VIX, etc.

        # Load the pre-trained model
        model = load_model()

        # Define the features that the model uses (ensure this matches with what was used during training)
        features = ['rsi', 'vix', 'skew', 'checklist_score']  # Use the real features
        X = data[features]

        # Make predictions using the model
        predictions = model.predict(X)

        # Simulate trading decisions based on predictions
        data['prediction'] = predictions
        data['final_decision'] = data['prediction'].apply(lambda x: 'ENTER' if x == 1 else 'WAIT')

        # Simulate backtest: If model says "ENTER", check next day's performance
        data['price_change'] = data['Adj Close'].pct_change().shift(-1)  # Next day's price change

        # Calculate profits for "ENTER" decisions
        data['profit'] = np.where(data['final_decision'] == 'ENTER', data['price_change'], 0)

        # Accumulate total profit
        total_profit += data['profit'].sum()

        # Optionally, save results for each stock
        data.to_csv(f'backtest_results_{ticker}.csv', index=True)

        # Show a preview of the results
        print(f"Backtest for {ticker} complete. Profit: {data['profit'].sum() * 100:.2f}%")

    # Total profit from all stocks
    print(f"\nTotal Profit from backtesting across all stocks: {total_profit * 100:.2f}%")

# Running the backtest function
if __name__ == "__main__":
    backtest()


