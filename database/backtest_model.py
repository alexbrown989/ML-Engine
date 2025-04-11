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
    print(f"\n❌ Import error: {e}\n   Check if files exist and are accessible. Path tried: {project_root}")
    sys.exit(1)

TICKER = "AAPL"
EXTERNAL_TICKERS = {"^VIX": "vix", "^VVIX": "vvix", "^SKEW": "skew"}
DAYS_BACK = 60

def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    try:
        df = yf.download(TICKER, start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            print("❌ No data for primary ticker.")
            return
        df['entry_price'] = df['Open'].shift(-1)
        df.columns = [f"{TICKER.lower()}_{col.lower()}" for col in df.columns]
    except Exception as e:
        print(f"❌ Error downloading primary ticker: {e}")
        return

    for symbol, name in EXTERNAL_TICKERS.items():
        try:
            ext = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
            if 'Close' in ext:
                df[name] = ext['Close']
            else:
                print(f"⚠️ {symbol} has no 'Close' column. Using NaN.")
                df[name] = np.nan
        except Exception as e:
            print(f"⚠️ Failed to fetch {symbol}: {e}. Filling {name} with NaNs.")
            df[name] = np.nan

    # Fill missing values
    for col in df.columns:
        df[col].ffill(inplace=True)
        df[col].bfill(inplace=True)

    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df)
    except KeyError as e:
        print(f"❌ KeyError in feature generation: {e}\n   Columns available: {df.columns.tolist()}")
        return
    except Exception as e:
        print(f"❌ General error in feature generation: {e}")
        return

    df.dropna(inplace=True)
    if df.empty:
        print("❌ All rows dropped after feature calc. Exiting.")
        return

    try:
        model, expected_features = load_model_and_features()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    missing = set(expected_features) - set(df.columns)
    for col in missing:
        df[col] = 0.0
    df = df[expected_features]

    try:
        preds = generate_predictions(model, df)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return

    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']
    print("\n📊 Sample predictions:")
    print(df[['prediction', 'confidence']].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()
