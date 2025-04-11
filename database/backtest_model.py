# Final version of database/backtest_model.py (patched for regime encoding fix)
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import traceback

TICKER = "AAPL"
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}
DAYS_BACK = 365
RSI_WINDOW = 14

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if project_root not in sys.path:
        print(f"[INFO] Adding project root to path: {project_root}")
        sys.path.append(project_root)
except NameError:
    print("[WARN] Could not automatically determine project root.")
    script_dir = "."

try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
    print("[INFO] Successfully imported custom modules.")
except ImportError as e:
    print(f"[ERROR] Import issue: {e}")
    sys.exit(1)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isinf(rs) & (avg_gain > 0)] = 100
    rsi[np.isinf(rs) & (avg_gain == 0)] = 50
    rsi = pd.Series(rsi, index=series.index)
    rsi.iloc[:window] = np.nan
    return rsi


def backtest():
    print("\n==================== Starting Backtest ====================")
    print(f"[INFO] Downloading {TICKER} data...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 50)

    try:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty:
            print("[ERROR] Ticker data is empty.")
            return
        df.columns = [f"{col.lower()}_{TICKER.lower()}" for col in df.columns]
        print(f"[DEBUG] Flattened columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch ticker: {e}")
        return

    for ext_symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"[INFO] Downloading {ext_symbol} as '{col_name}'...")
        try:
            ext_data = yf.download(ext_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            close_col = next((c for c in ext_data.columns if c[0].lower() == 'close' or c == 'Close'), None)
            if isinstance(close_col, tuple):
                df[col_name] = ext_data[close_col]
            else:
                df[col_name] = ext_data['Close']
        except Exception:
            df[col_name] = np.nan

    close_col = f"close_{TICKER.lower()}"
    open_col = f"open_{TICKER.lower()}"
    if close_col in df.columns:
        df['rsi'] = compute_rsi(df[close_col], RSI_WINDOW)
    else:
        print(f"[ERROR] {close_col} not found. Cannot compute RSI.")
        return

    if open_col in df.columns:
        df['entry_price'] = df[open_col].shift(-1)
    else:
        df['entry_price'] = np.nan

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    print("[INFO] Calculating features...")
    df = calculate_features(df)

    print("[INFO] Loading model...")
    try:
        model, expected_features = load_model_and_features()
        print(f"🧠 Loaded model: models/model_xgb_20250411044311.pkl with {len(expected_features)} features")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Replace 'regime' with one-hot cols if needed
    if 'regime' in expected_features:
        onehot_regimes = [col for col in df.columns if col.startswith('regime_')]
        if onehot_regimes:
            expected_features.remove('regime')
            expected_features += onehot_regimes
            print(f"[DEBUG] Replaced 'regime' with one-hot columns: {onehot_regimes}")

    for col in expected_features:
        if col not in df:
            df[col] = np.nan

    df.dropna(subset=expected_features, inplace=True)
    if df.empty:
        print("[ERROR] DataFrame is empty after dropna.")
        return

    df_predict = df[expected_features].copy()

    try:
        preds = generate_predictions(model, df_predict)
        df['prediction'] = preds['prediction']
        df['confidence'] = preds.get('confidence', np.nan)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return

    print("\n📊 Sample Predictions:")
    print(df[[close_col, 'prediction', 'confidence']].tail())

    print("\n==================== Backtest Complete ====================")


if __name__ == "__main__":
    try:
        backtest()
    except Exception as e:
        print(f"[FATAL] Crash: {e}")
        traceback.print_exc()
        sys.exit(1)
