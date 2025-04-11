import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# === PATCH IMPORT PATH === #
# Ensure this points correctly to your project structure if files are in different dirs
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Assuming these imports are now found due to sys.path modification
try:
    from build_features import calculate_features
    from inference import load_model_and_features
except ImportError as e:
    print(f"❌ Error importing project modules: {e}")
    print(f"   Check if build_features.py and inference.py are in the correct location relative to {script_dir}")
    print(f"   Project root added to path: {project_root}")
    sys.exit(1) # Exit if imports fail

# === BACKTEST CONFIG === #
TICKER = "AAPL"
# Dictionary of external tickers and their desired column names in the DataFrame
EXTERNAL_TICKERS = {
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
    # "^GSPC": "gspc", # Uncomment to add S&P 500 data as 'gspc' column
    # Add other required tickers here if calculate_features needs more
}
DAYS_BACK = 60 # Fetch roughly 60 calendar days of data

# === BACKTEST FUNCTION === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    # Use today's date for end_date; yfinance handles fetching up to the latest available data
    end_date = datetime.today()
    # Fetch extra days for feature calculation lookback periods & weekend gaps
    start_date = end_date - timedelta(days=DAYS_BACK + 15) # Increased buffer slightly

    # --- Download primary ticker data ---
    print(f"\nDownloading data for primary ticker: {TICKER}...")
    try:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
        # Using auto_adjust=True simplifies things by providing adjusted prices & removing Adj Close column
    except Exception as e:
        print(f"❌ Failed to download data for {TICKER}: {e}")
        return

    if df.empty:
        print(f"❌ No data returned for {TICKER} in the specified date range. Exiting.")
        return

    # --- Flatten df columns immediately ---
    print("🔧 Flattening primary ticker columns...")
    if isinstance(df.columns, pd.MultiIndex):
        # Format: ('AAPL', 'Close') -> 'aapl_close'
        df.columns = [f"{col[0].lower().replace('^','')}_{col[1].lower()}" for col in df.columns.values]
    else:
        # Format: 'Close' -> 'close' (if only one ticker symbol was somehow returned flat)
        # We'll prefix with the main ticker name for consistency if needed
        ticker_prefix = f"{TICKER.lower()}_"
        df.columns = [ticker_prefix + col.lower() if not col.lower().startswith(ticker_prefix) else col.lower() for col in df.columns]

    print(f"🧠 Columns after flattening {TICKER}: {df.columns.tolist()}")

    # --- Download and Join External Ticker Data ---
    for ticker_symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"\nDownloading data for external ticker: {ticker_symbol} (as '{col_name}')...")
        try:
            ext_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
        except Exception as e:
             print(f"⚠️ Could not download data for {ticker_symbol}: {e}. Adding NaN column for '{col_name}'.")
             df[col_name] = np.nan
             continue # Move to next ticker

        if ext_data.empty:
            print(f"⚠️ No data returned for {ticker_symbol}. Adding NaN column for '{col_name}'.")
            df[col_name] = np.nan
            continue

        # --- Extract Close Series ---
        try:
            close_series = None
            if isinstance(ext_data.columns, pd.MultiIndex):
                 # Handles (TICKER, 'Close') structure
                 # Need to handle potential variations in ticker casing returned by yf
                 ticker_level = ext_data.columns.levels[0][0] # Get the actual ticker string from MultiIndex
                 close_series = ext_data[(ticker_level, 'Close')]
            elif 'Close' in ext_data.columns:
                 # Handles flat 'Close' column structure
                 close_series = ext_data['Close']
            else:
                 raise KeyError("Could not find 'Close' column in standard locations.")

            close_series.name = col_name # Rename the Series itself

            # --- Join the RENAMED SERIES to the FLATTENED df ---
            print(f"🔧 Joining '{col_name}' series to DataFrame...")
            df = df.join(close_series, how='left')
            print(f"✅ Successfully joined {ticker_symbol} data as '{col_name}'.")

            # Check if join resulted in all NaNs (possible date mismatch)
            if df[col_name].isnull().all():
                 print(f"⚠️ Column '{col_name}' contains only NaNs after join. Check date alignment between {TICKER} and {ticker_symbol}.")

        except KeyError as e_key:
            print(f"❌ Could not find or extract 'Close' column for {ticker_symbol}: {e_key}. Columns found: {ext_data.columns.tolist()}")
            print(f"⚠️ Adding NaN column for '{col_name}'.")
            df[col_name] = np.nan
        except Exception as e_join:
             print(f"❌ Error processing or joining data for {ticker_symbol}: {e_join}")
             print(f"⚠️ Adding NaN column for '{col_name}'.")
             df[col_name] = np.nan


    # --- Add dummy entry_price column (using flattened column name) ---
    open_col_name = f'{TICKER.lower()}_open' # Assumes flattened name format
    if open_col_name in df.columns:
        df['entry_price'] = df[open_col_name].shift(-1)
    else:
         print(f"⚠️ Could not find '{open_col_name}' column to create 'entry_price'. Adding NaNs.")
         df['entry_price'] = np.nan

    print(f"\n🧠 Final columns before feature calculation: {df.columns.tolist()}")

    # --- Ensure required columns exist ---
    required_cols = list(EXTERNAL_TICKERS.values()) # Check all external cols were added
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        # This shouldn't happen if the loop added NaN columns on failure, but good safety check
        print(f"❌ FATAL: Required external columns missing before calling calculate_features: {missing_required}")
        return

    # --- Fill NaNs (Optional but recommended before feature calculation) ---
    # Simple forward fill might be appropriate for market indices
    print("🔧 Filling NaNs using forward fill...")
    cols_to_fill = list(EXTERNAL_TICKERS.values()) + ['entry_price']
    # Also fill NaNs in the primary ticker's OHLCV data if any exist
    ohlcv_cols = [col for col in df.columns if col.startswith(TICKER.lower())]
    cols_to_fill.extend(ohlcv_cols)

    for col in cols_to_fill:
        if col in df.columns:
            if df[col].isnull().any():
                 df[col].fillna(method='ffill', inplace=True)
                 # Still might have NaNs at the beginning if ffill wasn't enough
                 df[col].fillna(method='bfill', inplace=True) # Backfill remaining start NaNs
    print(f"    NaN count after fill: {df.isnull().sum().sum()}")


    # --- Feature engineering ---
    print("\n🔧 Calculating features...")
    try:
        # Pass the DataFrame which should now contain primary ticker data + all external indices
        df = calculate_features(df)
    except KeyError as e:
        print(f"❌ KeyError during feature calculation: {e}")
        print("   Likely requires another column not in EXTERNAL_TICKERS or not generated correctly.")
        print(f"   Columns passed into calculate_features: {df.columns.tolist()}")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during feature calculation: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Post-Feature Calculation ---
    print("🧹 Dropping rows with any remaining NaNs...")
    original_rows = len(df)
    # Drop rows if *any* column still has NaN (e.g., from feature calc or initial backfill failure)
    df.dropna(inplace=True)
    print(f"   Rows before dropna: {original_rows}, after: {len(df)}")

    if df.empty:
       print("❌ DataFrame is empty after feature calculation and dropna. Check feature logic or input data.")
       return

    # --- Load Model and Predict ---
    try:
        model, expected_features = load_model_and_features()
    except FileNotFoundError as e:
         print(f"❌ {e} - Train a model first.")
         return
    except Exception as e_load:
        print(f"❌ Error loading model or features: {e_load}")
        return
    if model is None:
        print("❌ No model found or loaded. Train one first.")
        return

    print("\n✨ Aligning DataFrame columns with model features...")
    print(f"   Model expects features: {expected_features}")
    # Ensure all features the model expects are present
    missing_in_df = set(expected_features) - set(df.columns)
    if missing_in_df:
        print(f"⚠️ Missing features needed by model: {missing_in_df}. Filling with 0.")
        for feature in missing_in_df:
            df[feature] = 0 # Or pd.NA

    # Select only expected features in the correct order
    final_features_for_prediction = [feat for feat in expected_features if feat in df.columns]
    if len(final_features_for_prediction) != len(expected_features):
        print("❌ Cannot proceed: Not all expected model features are present after alignment.")
        still_missing = set(expected_features) - set(final_features_for_prediction)
        print(f"   Specifically Missing: {still_missing}")
        return

    df_predict = df[final_features_for_prediction].copy()
    # Final NaN fill before prediction (shouldn't be needed if dropna worked, but safety)
    if df_predict.isnull().sum().sum() > 0:
         print(f"⚠️ NaNs detected before prediction ({df_predict.isnull().sum().sum()}). Filling with 0.")
         df_predict.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    try:
        preds = generate_predictions(model, df_predict)
    except Exception as e_pred:
         print(f"❌ Error during prediction generation: {e_pred}")
         print(f"   Features passed to prediction: {df_predict.columns.tolist()}")
         print(f"   Sample data passed: \n{df_predict.head()}")
         return


    # Add predictions back to the main df using index alignment
    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    print("\n📊 Sample predictions:")
    close_col_name = f'{TICKER.lower()}_close'
    cols_to_show = ['prediction', 'confidence']
    if close_col_name in df.columns:
        cols_to_show.insert(0, close_col_name)
    print(df[cols_to_show].tail())
    print("\n✅ Backtest script finished.")

# === ENTRY POINT === #
if __name__ == "__main__":
    # Optional: Add basic argument parsing here later if needed (e.g., specify ticker)
    backtest()
