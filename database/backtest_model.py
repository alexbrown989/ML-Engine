import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Patch import path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

TICKER = "AAPL"
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}
DAYS_BACK = 120

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        print("❌ Failed to download data.")
        return

    print("🔧 Flattening primary ticker columns...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns]
    else:
        df.columns = [f"{TICKER.lower()}_{col.lower()}" for col in df.columns]
    print(f"🧠 Columns after flattening: {df.columns.tolist()}")

    for symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"\nDownloading {symbol} as '{col_name}'...")
        try:
            ext = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(ext.columns, pd.MultiIndex):
                level = ext.columns.levels[0][0]
                close_series = ext[(level, 'Close')]
            else:
                close_series = ext['Close']
            close_series.name = col_name
            df = df.join(close_series, how='left')
            if df[col_name].isnull().all():
                print(f"⚠️ Column '{col_name}' is all NaN after join.")
        except Exception:
            print(f"⚠️ Join failed for {symbol}: '{col_name}'. Setting {col_name} to NaN.")
            df[col_name] = np.nan

    open_col = f"{TICKER.lower()}_open"
    if open_col in df.columns:
        df['entry_price'] = df[open_col].shift(-1)
    else:
        print(f"⚠️ Could not find '{open_col}'. Adding NaNs.")
        df['entry_price'] = np.nan

    # Calculate RSI
    close_col = f"{TICKER.lower()}_close"
    if close_col in df.columns:
        df['rsi'] = compute_rsi(df[close_col])
        print("🧠 RSI column added.")
    else:
        print(f"❌ {close_col} not found. Cannot compute RSI.")
        return

    print(f"\n🧠 Final columns before feature calculation: {df.columns.tolist()}")

    print("🔧 Filling NaNs with ffill + bfill...")
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df)
    except Exception as e:
        print(f"❌ Feature calculation error: {e}")
        return

    print("\n🔍 NaN summary before drop:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return

    df.dropna(subset=[f for f in expected_features if f in df.columns], inplace=True)

    if df.empty:
        print("❌ DataFrame empty after cleaning.")
        return

    df_predict = df[expected_features].copy()
    df_predict.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df_predict)
    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    close_col = f'{TICKER.lower()}_close'
    cols_to_show = ['prediction', 'confidence']
    if close_col in df.columns:
        cols_to_show.insert(0, close_col)

    print("\n📊 Sample predictions:")
    print(df[cols_to_show].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()
