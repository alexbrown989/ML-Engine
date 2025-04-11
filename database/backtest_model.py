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
DAYS_BACK = 60

# === BACKTEST FUNCTION === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK)

    # --- Download primary ticker data ---
    print(f"Downloading data for {TICKER}...")
    # Ensure yfinance returns a predictable structure, group_by='ticker' helps
    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, group_by='ticker')

    if df.empty:
        print(f"❌ Failed to fetch data for {TICKER}. Exiting.")
        return

    # --- Download VIX data ---
    print(f"Downloading data for {VIX_TICKER}...")
    vix_data = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False, group_by='ticker')

    if vix_data.empty:
        print(f"⚠️ Failed to fetch data for {VIX_TICKER}. Adding NaN column for 'vix'.")
        df['vix'] = np.nan # Add a single NaN column named 'vix'
    else:
        # --- Extract VIX Close Series and rename the SERIES itself ---
        try:
            # yf.download with group_by='ticker' for single ticker might return single index or multi-index
            if isinstance(vix_data.columns, pd.MultiIndex):
                 # If MultiIndex (e.g., ('^VIX', 'Close')), access correctly
                 vix_close_series = vix_data[(VIX_TICKER, 'Close')]
            else:
                 # If single index (e.g., 'Close'), access directly
                 vix_close_series = vix_data['Close']

            vix_close_series.name = 'vix' # Rename the Series itself to 'vix'
            # --- Join the RENAMED SERIES ---
            # Joining a named Series should add a single column with that name
            df = df.join(vix_close_series, how='left')
            print("✅ Successfully joined VIX data as 'vix' column.")

        except KeyError:
            print(f"❌ Could not find 'Close' column in VIX data. Columns: {vix_data.columns}")
            print("⚠️ Adding NaN column for 'vix'.")
            df['vix'] = np.nan # Add placeholder if extraction failed

    # Add dummy entry_price column
    # Accessing Open might depend on whether df has MultiIndex or not after group_by='ticker'
    try:
        if isinstance(df.columns, pd.MultiIndex):
             # If df still has MultiIndex (e.g. ('AAPL', 'Open'))
             df[('entry_price', '')] = df[(TICKER, 'Open')].shift(-1) # Keep MultiIndex structure for now
        else:
             # If df has flat columns (e.g. 'Open' if only one ticker was downloaded initially - less likely now)
             df['entry_price'] = df['Open'].shift(-1) # Add as flat column
    except KeyError:
         print(f"❌ Could not find 'Open' column for {TICKER} to create 'entry_price'. Columns: {df.columns}")
         # Decide how to proceed, e.g., add NaNs or stop
         df['entry_price'] = np.nan


    print("\n🧠 Columns before flattening:")
    print(df.columns) # Check structure: should have AAPL columns (maybe multi) and a single 'vix' column

    # --- Refined Flattening Logic ---
    print("\n🔧 Flattening columns...")
    flat_cols = []
    for col in df.columns.values:
        if isinstance(col, tuple):
            # If it's a tuple from MultiIndex (e.g., ('AAPL', 'Close') or ('entry_price', ''))
            # Join non-empty parts, make lower case
            flat_name = '_'.join(filter(None, col)).strip().lower()
            # Special case: ensure entry_price ends up as 'entry_price'
            if flat_name == 'entry_price_':
                flat_name = 'entry_price'
            flat_cols.append(flat_name)
        else:
            # If it's already a flat string (like 'vix' added from the series)
            flat_cols.append(str(col).lower()) # Just ensure lowercase

    df.columns = flat_cols

    print("\n🧠 Columns after flattening:")
    print(df.columns) # CRITICAL: Verify 'vix' column exists with the correct name here

    # --- Feature engineering ---
    # Ensure 'vix' column exists before calling calculate_features
    if 'vix' not in df.columns:
        print("❌ FATAL: 'vix' column is missing before calling calculate_features even after adjustments.")
        print(f"   Available columns: {df.columns.tolist()}")
        # Add a dummy 'vix' with NaNs if you want calculate_features to handle it,
        # but it indicates a problem in the data prep phase.
        # df['vix'] = np.nan
        return # Stop execution

    print("\n🔧 Calculating features...")
    try:
        # calculate_features might add new columns (like 'regime' seen in the error)
        df = calculate_features(df)
    except KeyError as e:
        # If KeyError still happens here, it might be for a DIFFERENT column expected by calculate_features
        print(f"❌ KeyError during feature calculation: {e}")
        print("Check if build_features.py expects other columns that were not prepared.")
        print(f"Columns passed to calculate_features: {df.columns.tolist()}") # Show columns *before* the error inside the function
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during feature calculation: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Post-Feature Calculation ---
    # Drop rows with NaNs that might have been generated during feature calculation
    # or existed due to failed VIX join/calculation
    original_rows = len(df)
    df.dropna(inplace=True)
    print(f"ℹ️ Rows before dropna: {original_rows}, after: {len(df)}")


    if df.empty:
       print("❌ DataFrame is empty after feature calculation and dropna. Check feature logic or input data (e.g., too many NaNs).")
       return

    # --- Load Model and Predict (rest of the code remains similar) ---
    try:
        model, expected_features = load_model_and_features()
    except FileNotFoundError as e:
         print(f"❌ {e}")
         print("Train a model first using the training script.")
         return
    if model is None:
        print("❌ No model found or loaded. Train one first.")
        return

    print("\n✨ Aligning DataFrame columns with model features...")
    # Ensure all features the model expects are present in df
    missing_in_df = set(expected_features) - set(df.columns)
    if missing_in_df:
        print(f"⚠️ Missing features in DataFrame needed by model: {missing_in_df}. Filling with 0.")
        for feature in missing_in_df:
            df[feature] = 0

    # Select only the features the model expects IN THE CORRECT ORDER
    final_features_for_prediction = [feat for feat in expected_features if feat in df.columns]

    # Verify *all* expected features were found before proceeding
    if len(final_features_for_prediction) != len(expected_features):
        print("❌ Cannot proceed: Not all expected model features are present in the final DataFrame after alignment.")
        print(f"   Model Expected: {expected_features}")
        print(f"   DataFrame Has : {df.columns.tolist()}")
        # Find which ones are *actually* missing from expected list
        still_missing = set(expected_features) - set(final_features_for_prediction)
        print(f"   Specifically Missing: {still_missing}")
        return

    df_predict = df[final_features_for_prediction].copy()
    df_predict.fillna(0, inplace=True) # Fill any remaining NaNs in the prediction set with 0

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df_predict)

    # Add predictions back for inspection
    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    print("\n📊 Sample predictions:")
    # Show relevant columns including close price for context if available
    cols_to_show = ['prediction', 'confidence']
    if 'close_aapl' in df.columns: # Add the ticker's close price if available
        cols_to_show.insert(0, 'close_aapl')
    elif 'close' in df.columns: # Fallback if only 'close' exists
         cols_to_show.insert(0, 'close')
    print(df[cols_to_show].tail())


# === ENTRY POINT === #
if __name__ == "__main__":
    backtest()
