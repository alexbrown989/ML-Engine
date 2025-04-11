import os
import sys
from datetime import datetime, timedelta
import traceback

import numpy as np
import pandas as pd
import yfinance as yf

# === PATH SETUP === #
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    print(f"[INFO] Adding project root to sys.path: {project_root}")
    sys.path.append(project_root)

# === SAFE IMPORTS === #
try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
except ImportError as e:
    print(f"[IMPORT ERROR] {e}\nEnsure 'build_features.py' and 'inference.py' are correctly located.")
    sys.exit(1)

# === CONFIG === #
TICKER = "AAPL"
EXTERNAL_TICKERS = {"^VIX": "vix", "^VVIX": "vvix", "^SKEW": "skew"}
DAYS_BACK = 365
RSI_WINDOW = 14

# === HELPERS === #
def compute_rsi(series, window=14):
    if series.isnull().all():
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isinf(rs) & (avg_gain > 0)] = 100
    rsi[np.isinf(rs) & (avg_gain == 0)] = 50
    return pd.Series(rsi, index=series.index)

def retry_download(ticker, start, end, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[WARN] Attempt {attempt + 1} failed for {ticker}: {e}")
    print(f"[FAIL] Could not download data for {ticker} after {retries} attempts.")
    return pd.DataFrame()

# === MAIN === #
def backtest():
    print("\n==================== Starting Backtest ====================")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 50)

    print(f"[INFO] Downloading {TICKER} data...")
    df = retry_download(TICKER, start_date, end_date)
    if df.empty:
        print("[ERROR] Failed to fetch primary ticker data.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[1].lower()}_{col[0].lower()}" for col in df.columns]
    else:
        df.columns = [f"{col.lower()}_{TICKER.lower()}" for col in df.columns]

    close_col = f"close_{TICKER.lower()}"
    open_col = f"open_{TICKER.lower()}"
    df['entry_price'] = df[open_col].shift(-1)
    df['rsi'] = compute_rsi(df[close_col], RSI_WINDOW)

    for ext_symbol, name in EXTERNAL_TICKERS.items():
        ext_data = retry_download(ext_symbol, start_date, end_date)
        if ext_data.empty:
            df[name] = np.nan
        else:
            try:
                df[name] = ext_data['Close']
            except:
                df[name] = np.nan

    print("[INFO] Filling NaNs with forward/backward fill...")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    print("[INFO] Running feature engineering...")
    try:
        df = calculate_features(df)
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        traceback.print_exc()
        return

    print("[INFO] Loading model...")
    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return

    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = np.nan

    df.dropna(subset=expected_features, inplace=True)
    if df.empty:
        print("[ERROR] All rows dropped due to missing features. Check feature engineering.")
        return

    print("[INFO] Generating predictions...")
    try:
        preds = generate_predictions(model, df[expected_features])
        df['prediction'] = preds['prediction']
        df['confidence'] = preds.get('confidence', np.nan)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return

    print("\n📊 Final Predictions Preview:")
    print(df[['prediction', 'confidence']].tail())

# === RUN === #
if __name__ == "__main__":
    try:
        backtest()
    except Exception as ex:
        print("[FATAL] Unexpected crash:", ex)
        traceback.print_exc()

