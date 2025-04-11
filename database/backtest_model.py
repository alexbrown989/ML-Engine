import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# === PATCH IMPORT PATH === #
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
except ImportError as e:
    print(f"❌ Error importing project modules: {e}")
    print(f"   Check if build_features.py and inference.py are in the correct location relative to {script_dir}")
    print(f"   Project root added to path: {project_root}")
    sys.exit(1)

TICKER = "AAPL"
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}
DAYS_BACK = 60

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    print(f"\nDownloading data for primary ticker: {TICKER}...")
    try:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
    except Exception as e:
        print(f"❌ Failed to download data for {TICKER}: {e}")
        return

    if df.empty:
        print(f"❌ No data returned for {TICKER}. Exiting.")
        return

    print("🔧 Flattening primary ticker columns...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns]
    else:
        df.columns = [f"{TICKER.lower()}_{col.lower()}" for col in df.columns]
    print(f"🧠 Columns after flattening: {df.columns.tolist()}")

    for ticker_symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"\nDownloading {ticker_symbol} as '{col_name}'...")
        try:
            ext_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if ext_data.empty:
                raise ValueError("No data")
            if isinstance(ext_data.columns, pd.MultiIndex):
                ticker_key = ext_data.columns.levels[0][0]
                close_series = ext_data[(ticker_key, 'Close')]
            else:
                close_series = ext_data['Close']
            close_series.name = col_name
            df = df.join(close_series, how='left')
        except Exception as e:
            print(f"⚠️ Join failed for {ticker_symbol}: '{col_name}'. Setting {col_name} to NaN.")
            df[col_name] = np.nan

    open_col_name = f"{TICKER.lower()}_open"
    if open_col_name in df.columns:
        df['entry_price'] = df[open_col_name].shift(-1)
    else:
        print(f"⚠️ Could not find '{open_col_name}' column. Setting entry_price to NaN.")
        df['entry_price'] = np.nan

    close_col = f"{TICKER.lower()}_close"
    if close_col in df.columns:
        df['rsi'] = compute_rsi(df[close_col])
        print("🧠 RSI column added.")
    else:
        print(f"❌ {close_col} not found. Cannot compute RSI.")
        return

    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df)
    except Exception as e:
        print(f"❌ KeyError during features: {e}")
        return

    df.dropna(inplace=True)
    if df.empty:
        print("❌ DataFrame empty after cleaning.")
        return

    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    missing = [feat for feat in expected_features if feat not in df.columns]
    if missing:
        print(f"⚠️ Missing features: {missing}. Filling with 0.")
        for feat in missing:
            df[feat] = 0

    X = df[expected_features].copy()
    if X.isnull().any().any():
        print("⚠️ NaNs detected before prediction. Filling with 0.")
        X.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    try:
        preds = generate_predictions(model, X)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    cols_to_display = ['prediction', 'confidence']
    if close_col in df.columns:
        cols_to_display.insert(0, close_col)
    print("\n📊 Sample predictions:")
    print(df[cols_to_display].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()

