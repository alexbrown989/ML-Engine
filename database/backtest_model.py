# Final version of database/backtest_model.py
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import traceback

# --- Configuration ---
TICKER = "AAPL"
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}
DAYS_BACK = 365
RSI_WINDOW = 14

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    print(f"[INFO] Adding project root to path: {project_root}")
    sys.path.append(project_root)

# --- Import Custom Modules ---
try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
    print("[INFO] Successfully imported custom modules.")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# --- Helper ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isinf(rs) & (avg_gain > 0)] = 100
    rsi[np.isinf(rs) & (avg_gain == 0)] = 50
    return pd.Series(rsi, index=series.index, name='rsi')

# --- Main ---
def backtest():
    print("\n==================== Starting Backtest ====================")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 50)

    print(f"[INFO] Downloading {TICKER} data...")
    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        print("[ERROR] No data returned for primary ticker.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns]
    else:
        df.columns = [f"{TICKER.lower()}_{col.lower()}" for col in df.columns]

    print(f"[DEBUG] Flattened columns: {df.columns.tolist()}")

    # Entry price (next day open)
    open_col = f"{TICKER.lower()}_open"
    df['entry_price'] = df[open_col].shift(-1) if open_col in df else np.nan

    # Download external tickers
    for symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"[INFO] Downloading {symbol} as '{col_name}'...")
        try:
            ext = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            close_col = [c for c in ext.columns if isinstance(c, tuple) and c[0] == 'Close']
            series = ext[close_col[0]] if close_col else ext['Close']
            df[col_name] = series
        except Exception as e:
            print(f"[WARN] Join failed for {symbol}: '{col_name}'. Setting {col_name} to NaN.")
            df[col_name] = np.nan

    # RSI
    close_col = f"close_{TICKER.lower()}"

    if close_col in df:
        df['rsi'] = compute_rsi(df[close_col], window=RSI_WINDOW)
    else:
        print(f"[ERROR] {close_col} not found. Cannot compute RSI.")
        return

    # Fill NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print("[INFO] Calculating features...")
    df = calculate_features(df)

    print("[INFO] Loading model...")
    model, expected_features = load_model_and_features()
    if model is None:
        print("[ERROR] No model loaded.")
        return

    for feat in expected_features:
        if feat not in df:
            df[feat] = np.nan

    df.dropna(subset=expected_features, inplace=True)
    if df.empty:
        print("[ERROR] DataFrame is empty after dropna.")
        return

    df_predict = df[expected_features].copy()
    preds = generate_predictions(model, df_predict)

    df['prediction'] = preds['prediction']
    df['confidence'] = preds.get('confidence', np.nan)

    print("\n📊 Sample predictions:")
    cols = [c for c in ['prediction', 'confidence', 'entry_price', close_col] if c in df.columns]
    print(df[cols].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()
