import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)

TICKER = "AAPL"
EXTERNAL_TICKERS = {"^VIX": "vix", "^VVIX": "vvix", "^SKEW": "skew"}
DAYS_BACK = 60

def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    try:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"❌ Error downloading primary ticker: {e}")
        return

    if df.empty:
        print("❌ No data fetched for primary ticker.")
        return

    df.columns = [f"{TICKER.lower()}_{col.lower()}" for col in df.columns]

    for symbol, name in EXTERNAL_TICKERS.items():
        try:
            ext = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(ext.columns, pd.MultiIndex):
                close_series = ext[ext.columns.levels[0][0], 'Close']
            else:
                close_series = ext['Close']
            close_series.name = name
            df = df.join(close_series, how='left')
        except Exception as e:
            print(f"⚠️ Could not download or join {symbol}: {e}")
            df[name] = np.nan

    open_col = f"{TICKER.lower()}_open"
    df['entry_price'] = df[open_col].shift(-1) if open_col in df.columns else np.nan

    for col in df.columns:
        df[col] = df[col].ffill().bfill()

    try:
        df = calculate_features(df)
    except Exception as e:
        print(f"❌ Feature calculation failed: {e}")
        return

    df.dropna(inplace=True)
    if df.empty:
        print("❌ No data after feature calculation and cleaning.")
        return

    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0

    df = df[expected_features].copy()
    df.fillna(0, inplace=True)

    try:
        preds = generate_predictions(model, df)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    close_col = f"{TICKER.lower()}_close"
    display_cols = [close_col] if close_col in df.columns else []
    display_cols += ['prediction', 'confidence']
    print("\n📊 Sample predictions:")
    print(df[display_cols].tail())
    print("\n✅ Backtest finished.")

if __name__ == "__main__":
    backtest()
