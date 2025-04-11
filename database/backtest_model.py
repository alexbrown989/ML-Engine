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
    "^SKEW": "skew"
}
DAYS_BACK = 60

def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 15)

    try:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
    except Exception as e:
        print(f"❌ Download error for {TICKER}: {e}")
        return

    if df.empty:
        print(f"❌ No data for {TICKER}.")
        return

    print("🔧 Flattening primary ticker columns...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0].lower().replace('^','')}_{col[1].lower()}" for col in df.columns]
    else:
        df.columns = [f"{TICKER.lower()}_{col.lower()}" if not col.lower().startswith(TICKER.lower()) else col.lower() for col in df.columns]

    print(f"🧠 Columns after flattening {TICKER}: {df.columns.tolist()}")

    for symbol, name in EXTERNAL_TICKERS.items():
        print(f"\nDownloading {symbol} as '{name}'...")
        try:
            ext = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            close_series = ext['Close'] if 'Close' in ext else ext.iloc[:, 0]
            close_series.name = name
            df = df.join(close_series, how='left')
            if df[name].isnull().all():
                print(f"⚠️ {name} data is all NaN.")
        except Exception as e:
            print(f"⚠️ Join failed for {symbol}: {e}. Setting {name} to NaN.")
            df[name] = np.nan

    open_col = f"{TICKER.lower()}_open"
    df['entry_price'] = df[open_col].shift(-1) if open_col in df.columns else np.nan

    cols_to_fill = list(EXTERNAL_TICKERS.values()) + ['entry_price'] + [col for col in df.columns if col.startswith(TICKER.lower())]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df)
    except KeyError as e:
        print(f"❌ KeyError during features: {e}")
        return
    except Exception as e:
        print(f"❌ Unknown feature error: {e}")
        return

    df.dropna(inplace=True)
    if df.empty:
        print("❌ All rows dropped after feature calc.")
        return

    try:
        model, features = load_model_and_features()
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    for f in features:
        if f not in df:
            print(f"⚠️ Missing: {f}, filling with 0")
            df[f] = 0

    df_pred = df[features].copy().fillna(0)

    print("\n🔮 Generating predictions...")
    try:
        preds = generate_predictions(model, df_pred)
    except Exception as e:
        print(f"❌ Predict error: {e}")
        return

    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    show = ['prediction', 'confidence']
    close_col = f"{TICKER.lower()}_close"
    if close_col in df.columns:
        show.insert(0, close_col)

    print("\n📊 Sample predictions:")
    print(df[show].tail())
    print("\n✅ Backtest complete.")

if __name__ == "__main__":
    backtest()

