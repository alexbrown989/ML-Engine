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
    print(f"❌ Import error: {e}")
    sys.exit(1)

TICKER = "AAPL"
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
}
DAYS_BACK = 60

def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    try:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"❌ Error downloading primary ticker: {e}")
        return

    if df.empty:
        print("❌ No data returned for primary ticker.")
        return

    print("🔧 Flattening primary ticker columns...")
    df.columns = [f"{TICKER.lower()}_{col.lower()}" for col in df.columns]
    print(f"🧠 Columns after flattening {TICKER}: {df.columns.tolist()}")

    for ext_ticker, alias in EXTERNAL_TICKERS.items():
        print(f"\nDownloading {ext_ticker} as '{alias}'...")
        try:
            ext_data = yf.download(ext_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(ext_data.columns, pd.MultiIndex):
                close_series = ext_data.iloc[:, 0]
            else:
                close_series = ext_data['Close']
            close_series.name = alias
            df = df.join(close_series, how='left')
        except Exception:
            print(f"⚠️ Join failed for {ext_ticker}: '{alias}'. Setting {alias} to NaN.")
            df[alias] = np.nan

    df['entry_price'] = df[f"{TICKER.lower()}_open"].shift(-1)

    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df)
    except KeyError as e:
        print(f"❌ KeyError during features: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error in feature calculation: {e}")
        return

    df.dropna(inplace=True)
    if df.empty:
        print("❌ DataFrame empty after feature processing.")
        return

    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features].fillna(0)

    print("\n🔮 Generating predictions...")
    try:
        preds = generate_predictions(model, df)
        df['prediction'] = preds['prediction']
        df['confidence'] = preds['confidence']
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    print("\n📊 Sample predictions:")
    print(df[['prediction', 'confidence']].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()

