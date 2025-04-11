import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from build_features import calculate_features
from inference import load_latest_model, generate_predictions

# --- Config ---
TICKER = "AAPL"
LOOKBACK_DAYS = 60
ENTRY_THRESHOLD = 0.7  # Confidence threshold for simulated entry


def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    print(f"Fetching data for {TICKER} from {start_date.date()} to {end_date.date()}")
    df = yf.download(TICKER, start=start_date, end=end_date)
    if df.empty:
        print("❌ Failed to fetch data.")
        return

    df['entry_price'] = df['Close']  # Simulate synthetic entry price
    df.reset_index(inplace=True)  # Ensure 'Date' becomes a column

    print("\n🧠 Columns before MultiIndex:")
    print(df.columns)

    # Simulate a MultiIndex structure and flatten it
    df.columns = pd.MultiIndex.from_tuples([(col, TICKER if col != 'Date' else '') for col in df.columns])
    df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]

    # Calculate features from raw price + entry
    df = calculate_features(df)

    # Drop any rows with missing required fields
    df.dropna(inplace=True)

    # Load model
    model = load_latest_model()
    if not model:
        print("❌ No model found.")
        return

    # Generate predictions
    pred_df = generate_predictions(model, df)

    # Entry logic
    pred_df['signal'] = pred_df['confidence'].apply(lambda x: 'ENTER' if x >= ENTRY_THRESHOLD else 'WAIT')

    print("\n🔍 Entry recommendations:")
    print(pred_df[['timestamp', 'confidence', 'prediction', 'signal']].head(10))

    # Strategy performance backtesting logic would go here (P/L, drawdown, etc.)
    print("\n🎯 Backtest completed.")


if __name__ == "__main__":
    backtest()
