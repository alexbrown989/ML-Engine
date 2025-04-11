import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === PATCH PATH === #
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
except ImportError as e:
    print(f"❌ Import error: {e}\n   Check relative paths or sys.path setup.")
    sys.exit(1)

# === CONFIG === #
TICKER = "AAPL"
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}
DAYS_BACK = 60

# === UTILS === #
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === BACKTEST === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    # Primary ticker
    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        print("❌ Primary ticker returned empty DataFrame.")
        return

    # Flatten columns
    print("🔧 Flattening primary ticker columns...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{str(col[0]).lower()}_{str(col[1]).lower()}"
            if not str(col[0]).lower() == str(col[1]).lower()
            else str(col[1]).lower()
            for col in df.columns
        ]
    else:
        df.columns = [col.lower() for col in df.columns]
    print(f"🧠 Columns after flattening: {df.columns.tolist()}")

    # External data joins
    for symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"\nDownloading {symbol} as '{col_name}'...")
        try:
            ext = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if ext.empty:
                raise ValueError("Empty")
            ext_col = ext['Close'] if 'Close' in ext.columns else None
            if ext_col is None:
                raise KeyError("Missing Close")
            df[col_name] = ext_col
        except Exception as e:
            print(f"⚠️ Join failed for {symbol}: '{col_name}'. Setting {col_name} to NaN.")
            df[col_name] = np.nan

    # entry price
    open_col = f"{TICKER.lower()}_open"
    df['entry_price'] = df[open_col].shift(-1) if open_col in df.columns else np.nan

    # Inject RSI before calling calculate_features
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
        print("❌ All rows dropped during feature cleanup.")
        return

    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    # Align columns
    for feat in expected_features:
        if feat not in df.columns:
            print(f"⚠️ Adding missing feature '{feat}' as 0")
            df[feat] = 0

    df = df[expected_features].copy()
    df.fillna(0, inplace=True)

    # Predict
    print("\n🔮 Generating predictions...")
    try:
        preds = generate_predictions(model, df)
        df['prediction'] = preds['prediction']
        df['confidence'] = preds['confidence']
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return

    print("\n📊 Sample predictions:")
    print(df[['prediction', 'confidence']].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()
