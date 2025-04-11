# database/backtest_model.py
import os
import sys
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# --- Config ---
TICKER = "AAPL"
DAYS_BACK = 365
RSI_WINDOW = 14
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    print(f"[INFO] Adding project root to path: {project_root}")
    sys.path.append(project_root)

# --- Imports ---
try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
    print("[INFO] Successfully imported custom modules.")
except Exception as e:
    print(f"[IMPORT ERROR] {e}")
    sys.exit(1)


# --- Helper Functions ---
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:window] = np.nan
    return rsi


# --- Backtest ---
def backtest():
    print("\n==================== Starting Backtest ====================")

    end = datetime.today()
    start = end - timedelta(days=DAYS_BACK + 30)

    print(f"[INFO] Downloading {TICKER} data...")
    df = yf.download(TICKER, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker')

    if df.empty:
        print("[ERROR] No data fetched.")
        return

    # --- Flatten Columns ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[1].lower()}_{col[0].lower()}" for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    print(f"[DEBUG] Flattened columns: {df.columns.tolist()}")

    # --- Download External Tickers ---
    for symbol, name in EXTERNAL_TICKERS.items():
        print(f"[INFO] Downloading {symbol} as '{name}'...")
        try:
            data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                close_col = ('Close', symbol)
            else:
                close_col = 'Close'
            df[name] = data[close_col]
        except Exception as e:
            print(f"[WARN] Failed to download {symbol}: {e}")
            df[name] = np.nan

    # --- Entry Price ---
    open_col = f"open_{TICKER.lower()}"
    df['entry_price'] = df[open_col].shift(-1) if open_col in df.columns else np.nan

    # --- RSI ---
    close_col = f"close_{TICKER.lower()}"
    if close_col in df.columns:
        df['rsi'] = compute_rsi(df[close_col], window=RSI_WINDOW)
    else:
        print(f"[ERROR] {close_col} not found. Cannot compute RSI.")
        return

    # --- Fill NaNs ---
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # --- Feature Engineering ---
    print("[INFO] Calculating features...")
    try:
        df = calculate_features(df)
    except Exception as e:
        print(f"[ERROR] Feature calculation failed: {e}")
        return

    # --- Model & Features ---
    print("[INFO] Loading model...")
    try:
        model, expected = load_model_and_features()
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return

    # --- Patch for one-hot regime ---
    if 'regime' in expected:
        expected.remove('regime')
        regime_oh = [col for col in df.columns if col.startswith('regime_')]
        expected.extend(regime_oh)

    for col in expected:
        if col not in df:
            df[col] = np.nan

    df.dropna(subset=expected, inplace=True)

    if df.empty:
        print("[ERROR] DataFrame is empty after dropna.")
        return

    df_predict = df[expected].copy()

    try:
        preds = generate_predictions(model, df_predict)
        df['prediction'] = preds['prediction']
        df['confidence'] = preds.get('confidence', np.nan)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return

    print(df[['prediction', 'confidence']].tail())
    print("\n✅ Backtest complete.")


if __name__ == "__main__":
    backtest()

