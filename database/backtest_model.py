# backtest_model.py
import os
import sys
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# === Setup ===
TICKER = "AAPL"
EXTERNAL_TICKERS = {"^VIX": "vix", "^VVIX": "vvix", "^SKEW": "skew"}
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
except ImportError as e:
    print(f"[ERROR] Failed to import custom modules: {e}")
    sys.exit(1)

# --- Helpers ---
def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isinf(rs) & (avg_gain > 0)] = 100
    rsi[np.isinf(rs) & (avg_gain == 0)] = 50
    out = pd.Series(rsi, index=series.index, name="rsi")
    out.iloc[:window] = np.nan
    return out

# --- Backtest ---
def backtest():
    print("\n==================== Starting Backtest ====================")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 30)

    print(f"[INFO] Downloading {TICKER} data...")
    df = yf.download(TICKER, start=start_date, end=end_date, auto_adjust=True, progress=False, group_by='ticker')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns]
    else:
        df.columns = [f"{TICKER.lower()}_{col.lower()}" for col in df.columns]
    print(f"[DEBUG] Flattened columns: {df.columns.tolist()}")

    for symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"[INFO] Downloading {symbol} as '{col_name}'...")
        try:
            ext_df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if isinstance(ext_df.columns, pd.MultiIndex):
                ext_col = ext_df['Close', symbol] if ('Close', symbol) in ext_df.columns else None
            else:
                ext_col = ext_df['Close'] if 'Close' in ext_df.columns else None
            if ext_col is not None:
                df[col_name] = ext_col
        except Exception as e:
            print(f"[WARN] Failed to download {symbol}: {e}")
            df[col_name] = np.nan

    if f"{TICKER.lower()}_close" in df.columns:
        df['rsi'] = compute_rsi(df[f"{TICKER.lower()}_close"], window=RSI_WINDOW)
    else:
        print("[ERROR] aapl_close not found. Cannot compute RSI.")
        return

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    print("[INFO] Calculating features...")
    df = calculate_features(df)

    print("[INFO] Loading model...")
    model, expected_features = load_model_and_features()
    print(f"🧠 Loaded model: {model} with {len(expected_features)} features")

    # Fix 'regime' issue
    if 'regime' in expected_features:
        regime_onehots = [col for col in df.columns if col.startswith('regime_')]
        if regime_onehots:
            print(f"🔁 Converting one-hot back to 'regime' for model compatibility.")
            def decode_regime(row):
                for col in regime_onehots:
                    if row.get(col) == 1:
                        return col.replace("regime_", "")
                return 'calm'
            df['regime'] = df.apply(decode_regime, axis=1)

    for col in expected_features:
        if col not in df.columns:
            df[col] = np.nan

    df.dropna(subset=expected_features, inplace=True)
    if df.empty:
        print("[ERROR] DataFrame is empty after dropna.")
        return

    try:
        preds = generate_predictions(model, df[expected_features])
        df['prediction'] = preds['prediction']
        df['confidence'] = preds.get('confidence', np.nan)
        print("\n📊 Sample Predictions:")
        print(df[[f"{TICKER.lower()}_close", 'prediction', 'confidence']].tail())
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return

    print("\n✅ Backtest complete.")

if __name__ == '__main__':
    backtest()
