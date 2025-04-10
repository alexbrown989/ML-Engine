import sqlite3
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import pickle

# Function to fetch random stock data from Yahoo Finance
def get_random_stock_data():
    """Fetches random stock data from Yahoo Finance for a specific date range."""
    # List of sample tickers (you can extend this list)
    tickers = ['AAPL', 'GOOG', 'TSLA', 'MSFT', 'META', 'AMZN', 'NVDA', 'PLTR']
    ticker = random.choice(tickers)  # Randomly pick a stock ticker
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')  # 60 days ago
    end_date = datetime.now().strftime('%Y-%m-%d')  # Today
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")

    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Use the adjusted closing prices to simulate "entry price"
    data['entry_price'] = data['Adj Close']
    return data

# Function to load the pre-trained model
def load_model():
    """Loads the pre-trained model."""
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Function to prepare features for prediction (you should adapt this to match your actual feature engineering)
def prepare_features(df):
    """Prepares features by adding necessary columns (example of preprocessing)."""
    # Create basic features (you can add more based on your actual feature engineering process)
    df['rsi'] = np.random.uniform(30, 70, len(df))  # Simulated RSI
    df['vix'] = np.random.uniform(10, 40, len(df))  # Simulated VIX
    df['skew'] = np.random.uniform(-1, 1, len(df))  # Simulated skew
    df['regime'] = np.random.choice(['calm', 'panic', 'transition'], len(df))  # Random regime
    df['checklist_score'] = np.random.randint(1, 5, len(df))  # Random checklist score
    return df

# Backtesting function
def backtest():
    """Main function to simulate backtesting on fetched data."""
    # Fetch random stock data
    data = get_random_stock_data()

    # Prepare features for the model
    data = prepare_features(data)

    # Load the pre-trained model
    model = load_model()

    # Define the features that the model uses (adjust this according to your feature set)
    features = ['rsi', 'vix', 'skew', 'checklist_score']  # Add all features the model uses
    X = data[features]

    # Make predictions using the model
    predictions = model.predict(X)

    # Simulate trading decisions based on predictions
    data['prediction'] = predictions
    data['final_decision'] = data['prediction'].apply(lambda x: 'ENTER' if x == 1 else 'WAIT')

    # Simulate a simple backtest: If model says "ENTER", check the next day's performance (price change)
    data['price_change'] = data['Adj Close'].pct_change().shift(-1)  # Next day's price change

    # Calculate profits for "ENTER" decisions
    data['profit'] = np.where(data['final_decision'] == 'ENTER', data['price_change'], 0)

    # Backtest Results
    total_profit = data['profit'].sum()
    print(f"Total Profit from backtesting: {total_profit * 100:.2f}%")

    # Optional: Save the results to a CSV file or store them in the database
    data.to_csv('backtest_results.csv', index=False)

    # Show a preview of the results
    print(data[['Date', 'final_decision', 'price_change', 'profit']].head())

# Running the backtest function
if __name__ == "__main__":
    backtest()
