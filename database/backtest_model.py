import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# === PATCH IMPORT PATH === #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from build_features import calculate_features # Assuming build_features.py is in the parent dir
from inference import load_model_and_features

# === BACKTEST CONFIG === #
TICKER = "AAPL"
VIX_TICKER = "^VIX" # VIX index ticker in Yahoo Finance
DAYS_BACK = 60 # Fetch roughly 60 calendar days of data

# === BACKTEST FUNCTION === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    # Get current date and calculate start date
    # Note: yfinance often needs T+1 to get data for T, especially for VIX.
    # Consider fetching slightly more data if needed for feature calculation lags.
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK + 5) # Fetch a bit extra for feature calc lookback

    # --- Download primary ticker data ---
    print(f"Downloading data for {TICKER}...")
    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, group_by='ticker')

    if df.empty:
        print(f"❌ Failed to fetch data for {TICKER}. Exiting.")
        return

    # --- Flatten df columns immediately ---
    print("🔧 Flattening primary ticker columns...")
    if isinstance(df.columns, pd.MultiIndex):
        # Format: ('AAPL', 'Close') -> 'aapl_close'
        df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns.values]
    else:
        # If already flat (e.g., 'Close'), just make lowercase
        df.columns = [col.lower() for col in df.columns]
    print(f"🧠 Columns after flattening {TICKER}: {df.columns.tolist()}")

    # --- Download VIX data ---
    print(f"Downloading data for {VIX_TICKER}...")
    vix_data = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False, group_by='ticker')

    vix_col_name = 'vix' # Desired final column name

    if vix_data.empty:
        print(f"⚠️ Failed to fetch data for {VIX_TICKER}. Adding NaN column for '{vix_col_name}'.")
        df[vix_col_name] = np.nan
    else:
        # --- Extract VIX Close Series ---
        try:
            if isinstance(vix_data.columns, pd.MultiIndex):
                 # Handles ('^VIX', 'Close') structure
                 vix_close_series = vix_data[(VIX_TICKER, 'Close')]
            else:
                 # Handles flat 'Close' column structure
                 vix_close_series = vix_data['Close']

            vix_close_series.name = vix_col_name # Rename the Series itself

            # --- Join the RENAMED SERIES to the FLATTENED df ---
            print(f"🔧 Joining '{vix_col_name}' series to flattened DataFrame...")
            df = df.join(vix_close_series, how='left') # Now joining 1-level series to 1-level df
            print("✅ Successfully joined VIX data.")

            if df[vix_col_name].isnull().all():
                 print(f"⚠️ Warning: VIX column '{vix_col_name}' contains only NaNs after join. Check date alignment.")

        except KeyError:
            print(f"❌ Could not find 'Close' column in VIX data. Columns: {vix_data.columns}")
            print(f"⚠️ Adding NaN column for '{vix_col_name}'.")
            df[vix_col_name] = np.nan

    # --- Add dummy entry_price column (using flattened column name) ---
    # Construct the expected flattened open price column name
    open_col_name = f'{TICKER.lower()}_open'
    if open_col_name in df.columns:
        df['entry_price'] = df[open_col_name].shift(-1)
    else:
         print(f"❌ Could not find '{open_col_name}' column for {TICKER} to create 'entry_price'. Columns: {df.columns}")
         df['entry_price'] = np.nan # Add placeholder

    print(f"\n🧠 Final columns before feature calculation: {df.columns.tolist()}")

    # --- Ensure 'vix' column exists ---
    if vix_col_name not in df.columns:
        print(f"❌ FATAL: '{vix_col_name}' column is missing before calling calculate_features.")
        return

    # Fill potential NaNs in VIX before feature calculation if needed by the function
    # Or let calculate_features handle them if it's designed to
    # Example: df[vix_col_name].fillna(method='ffill', inplace=True) # Forward fill VIX NaNs

    # --- Feature engineering ---
    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df) # Pass the DataFrame that now includes 'vix' and other flat columns
    except KeyError as e:
        print(f"❌ KeyError during feature calculation: {e}")
        print("Check if build_features.py expects other columns that were not prepared or were misnamed during flattening.")
        print(f"Columns passed to calculate_features: {df.columns.tolist()}")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during feature calculation: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Post-Feature Calculation ---
    original_rows = len(df)
    df.dropna(inplace=True) # Drop rows with NaNs generated during/after feature calculation
    print(f"ℹ️ Rows before dropna: {original_rows}, after: {len(df)}")

    if df.empty:
       print("❌ DataFrame is empty after feature calculation and dropna. Check feature logic or input data.")
       return

    # --- Load Model and Predict ---
    try:
        model, expected_features = load_model_and_features()
    except FileNotFoundError as e:
         print(f"❌ {e} - Train a model first.")
         return
    if model is None:
        print("❌ No model found or loaded. Train one first.")
        return

    print("\n✨ Aligning DataFrame columns with model features...")
    missing_in_df = set(expected_features) - set(df.columns)
    if missing_in_df:
        print(f"⚠️ Missing features in DataFrame needed by model: {missing_in_df}. Filling with 0.")
        for feature in missing_in_df:
            df[feature] = 0

    final_features_for_prediction = [feat for feat in expected_features if feat in df.columns]
    if len(final_features_for_prediction) != len(expected_features):
        print("❌ Cannot proceed: Not all expected model features are present in the final DataFrame after alignment.")
        still_missing = set(expected_features) - set(final_features_for_prediction)
        print(f"   Model Expected: {expected_features}")
        print(f"   DataFrame Has : {df.columns.tolist()}")
        print(f"   Specifically Missing: {still_missing}")
        return

    df_predict = df[final_features_for_prediction].copy()
    df_predict.fillna(0, inplace=True)

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df_predict)

    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    print("\n📊 Sample predictions:")
    # Show relevant columns
    close_col_name = f'{TICKER.lower()}_close' # Construct expected close column name
    cols_to_show = ['prediction', 'confidence']
    if close_col_name in df.columns:
        cols_to_show.insert(0, close_col_name)
    elif 'close' in df.columns: # Fallback
         cols_to_show.insert(0, 'close')
    print(df[cols_to_show].tail())


# === ENTRY POINT === #
if __name__ == "__main__":
    backtest()
