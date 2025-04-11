import os
import sys
import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# === PATCH IMPORT PATH === #
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# === IMPORT PROJECT MODULES === #
try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
except ImportError as e:
    print(f"❌ Error importing modules: {e}\n   From: {script_dir}\n   Added path: {project_root}")
    sys.exit(1)

# === CONFIG === #
TICKER = "AAPL"
EXTERNAL_TICKERS = {"^VIX": "vix", "^VVIX": "vvix", "^SKEW": "skew"}
DAYS_BACK = 60
DB_PATH = os.path.join(project_root, "signals.db")

# === UTILS === #
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === BACKTEST === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        print("❌ No data fetched.")
        return

    print("🔧 Flattening primary ticker columns...")
    df.columns = [f"{TICKER.lower()}_{col[1].lower()}" if isinstance(col, tuple) else f"{TICKER.lower()}_{col.lower()}" for col in df.columns]
    print("🧠 Columns after flattening:", df.columns.tolist())

    for ticker_symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"\nDownloading {ticker_symbol} as '{col_name}'...")
        try:
            ext_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            series = ext_data['Close'] if 'Close' in ext_data.columns else ext_data[ext_data.columns[0]]
            df[col_name] = series
        except Exception as e:
            print(f"⚠️ Join failed for {ticker_symbol}: '{col_name}'. Setting {col_name} to NaN.")
            df[col_name] = np.nan

    close_col = f"{TICKER.lower()}_close"
    if close_col in df.columns:
        df['rsi'] = compute_rsi(df[close_col])
        print("🧠 RSI column added.")
    else:
        print(f"❌ {close_col} not found. Cannot compute RSI.")
        return

    open_col = f"{TICKER.lower()}_open"
    df['entry_price'] = df[open_col].shift(-1) if open_col in df.columns else np.nan

    print("🔧 Filling NaNs...")
    for col in df.columns:
        df[col] = df[col].ffill().bfill()

    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df)
    except Exception as e:
        print(f"❌ Feature error: {e}\nColumns passed: {df.columns.tolist()}")
        return

    df.dropna(inplace=True)
    if df.empty:
        print("❌ No data after feature calc.")
        return

    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    missing = set(expected_features) - set(df.columns)
    for feat in missing:
        df[feat] = 0
    df_predict = df[expected_features].copy()
    df_predict.fillna(0, inplace=True)

    print("\n🔮 Predicting...")
    try:
        preds = generate_predictions(model, df_predict)
        df['prediction'] = preds['prediction']
        df['confidence'] = preds['confidence']
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return

    print("\n📊 Sample predictions:")
    preview_cols = [close_col, 'prediction', 'confidence'] if close_col in df.columns else ['prediction', 'confidence']
    print(df[preview_cols].tail())

    # === SAVE TO SQLITE === #
    conn = sqlite3.connect(DB_PATH)
    df['ticker'] = TICKER
    df['backtest_start'] = start_date.strftime('%Y-%m-%d')
    df['backtest_end'] = end_date.strftime('%Y-%m-%d')
    df.to_sql("backtest_signals", conn, if_exists="replace", index=False)
    conn.close()
    print("✅ Saved to SQLite → backtest_signals table.")

# === ENTRY === #
if __name__ == "__main__":
    backtest()

