import os
import sys
from datetime import datetime, timedelta
import traceback
import numpy as np
import pandas as pd
import yfinance as yf

# --- Configuration ---
TICKER = "AAPL"
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}
DAYS_BACK = 365
RSI_WINDOW = 14

# --- Setup Paths ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if project_root not in sys.path:
        print(f"[INFO] Adding project root to path: {project_root}")
        sys.path.append(project_root)
except Exception:
    print("[WARN] Could not determine script directory.")

# --- Import Project Modules ---
try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
    print("[INFO] Successfully imported custom modules.")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# --- RSI Helper ---
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
    rsi = pd.Series(rsi, index=series.index, name="rsi")
    rsi.iloc[:window] = np.nan
    return rsi

# --- Column Matching ---
def find_column(df, patterns):
    for col in df.columns:
        for p in patterns:
            if p.lower() in str(col).lower():
                return col
    return None

# --- Backtest Main ---
def backtest():
    print("\n==================== Starting Backtest ====================")
    print(f"[INFO] Downloading {TICKER} data...")
    end = datetime.today()
    start = end - timedelta(days=DAYS_BACK + 50)

    try:
        df = yf.download(TICKER, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{a.lower()}_{b.lower()}" for a, b in df.columns]
        else:
            df.columns = [f"{col.lower()}_{TICKER.lower()}" for col in df.columns]
        print(f"[DEBUG] Flattened columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to download primary data: {e}")
        return

    for symbol, name in EXTERNAL_TICKERS.items():
        print(f"[INFO] Downloading {symbol} as '{name}'...")
        try:
            ext = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(ext.columns, pd.MultiIndex):
                col = find_column(ext, ['close'])
            else:
                col = 'close'
            df[name] = ext[col] if col in ext.columns else np.nan
        except Exception as e:
            print(f"[WARN] Failed to fetch {symbol}: {e}. Setting {name} to NaN.")
            df[name] = np.nan

    close_col = find_column(df, [f"close_{TICKER.lower()}"])
    if close_col:
        df['rsi'] = compute_rsi(df[close_col], RSI_WINDOW)
    else:
        print("[ERROR] aapl_close not found. Cannot compute RSI.")
        return

    open_col = find_column(df, [f"open_{TICKER.lower()}"])
    if open_col:
        df['entry_price'] = df[open_col].shift(-1)

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    print("[INFO] Calculating features...")
    try:
        df = calculate_features(df)
    except Exception as e:
        print(f"[ERROR] Feature calculation failed: {e}")
        return

    print("[INFO] Loading model...")
    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return

    for col in expected_features:
        if col not in df:
            df[col] = np.nan

    df.dropna(subset=expected_features, inplace=True)
    if df.empty:
        print("[ERROR] DataFrame is empty after dropna.")
        return

    preds = generate_predictions(model, df[expected_features])
    df['prediction'] = preds['prediction']
    df['confidence'] = preds.get('confidence', np.nan)

    print("\n📊 Final Predictions:")
    print(df[['prediction', 'confidence']].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()
