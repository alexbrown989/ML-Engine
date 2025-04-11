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
VIX_TICKER = "^VIX"  # VIX index ticker
VVIX_TICKER = "^VVIX" # VVIX index ticker
DAYS_BACK = 60 # Fetch roughly 60 calendar days of data

# === BACKTEST FUNCTION === #
def backtest():
    print(f"Starting backtest for {TICKER}...")
    end_date = datetime.today()
    # Fetch extra days for feature calculation lookback periods
    start_date = end_date - timedelta(days=DAYS_BACK + 10)

    # --- Download primary ticker data ---
    print(f"Downloading data for {TICKER}...")
    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, group_by='ticker')

    if df.empty:
        print(f"❌ Failed to fetch data for {TICKER}. Exiting.")
        return

    # --- Flatten df columns immediately ---
    print("🔧 Flattening primary ticker columns...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns.values]
    else:
        df.columns = [col.lower() for col in df.columns]
    print(f"🧠 Columns after flattening {TICKER}: {df.columns.tolist()}")

    # --- Download and Join VIX data ---
    vix_col_name = 'vix'
    print(f"Downloading data for {VIX_TICKER}...")
    vix_data = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False, group_by='ticker')

    if vix_data.empty:
        print(f"⚠️ Failed to fetch data for {VIX_TICKER}. Adding NaN column for '{vix_col_name}'.")
        df[vix_col_name] = np.nan
    else:
        try:
            if isinstance(vix_data.columns, pd.MultiIndex):
                 vix_close_series = vix_data[(VIX_TICKER, 'Close')]
            else:
                 vix_close_series = vix_data['Close']
            vix_close_series.name = vix_col_name
            print(f"🔧 Joining '{vix_col_name}' series to flattened DataFrame...")
            df = df.join(vix_close_series, how='left')
            print("✅ Successfully joined VIX data.")
            if df[vix_col_name].isnull().all():
                 print(f"⚠️ VIX column '{vix_col_name}' contains only NaNs after join.")
        except KeyError:
            print(f"❌ Could not find 'Close' column in VIX data. Adding NaN column for '{vix_col_name}'.")
            df[vix_col_name] = np.nan

    # --- Download and Join VVIX data ---
    vvix_col_name = 'vvix'
    print(f"Downloading data for {VVIX_TICKER}...")
    vvix_data = yf.download(VVIX_TICKER, start=start_date, end=end_date, progress=False, group_by='ticker')

    if vvix_data.empty:
        print(f"⚠️ Failed to fetch data for {VVIX_TICKER}. Adding NaN column for '{vvix_col_name}'.")
        df[vvix_col_name] = np.nan
    else:
        try:
            if isinstance(vvix_data.columns, pd.MultiIndex):
                 vvix_close_series = vvix_data[(VVIX_TICKER, 'Close')]
            else:
                 vvix_close_series = vvix_data['Close']
            vvix_close_series.name = vvix_col_name
            print(f"🔧 Joining '{vvix_col_name}' series to flattened DataFrame...")
            df = df.join(vvix_close_series, how='left') # Join VVIX
            print("✅ Successfully joined VVIX data.")
            if df[vvix_col_name].isnull().all():
                 print(f"⚠️ VVIX column '{vvix_col_name}' contains only NaNs after join.")
        except KeyError:
            print(f"❌ Could not find 'Close' column in VVIX data. Adding NaN column for '{vvix_col_name}'.")
            df[vvix_col_name] = np.nan


    # --- Add dummy entry_price column (using flattened column name) ---
    open_col_name = f'{TICKER.lower()}_open'
    if open_col_name in df.columns:
        df['entry_price'] = df[open_col_name].shift(-1)
    else:
         print(f"❌ Could not find '{open_col_name}' column to create 'entry_price'.")
         df['entry_price'] = np.nan

    print(f"\n🧠 Final columns before feature calculation: {df.columns.tolist()}")

    # --- Ensure required columns exist ---
    required_cols = [vix_col_name, vvix_col_name] # Add any other absolutely required cols here
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        print(f"❌ FATAL: Required columns missing before calling calculate_features: {missing_required}")
        return

    # Optional: Fill NaNs before feature calculation if needed
    # df[vix_col_name].fillna(method='ffill', inplace=True)
    # df[vvix_col_name].fillna(method='ffill', inplace=True)
    # df['entry_price'].fillna(method='ffill', inplace=True) # Or handle differently

    # --- Feature engineering ---
    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df)
    except KeyError as e:
        print(f"❌ KeyError during feature calculation: {e}")
        print("Check build_features.py - does it expect other columns? (e.g., other indicators, rates?)")
        print(f"Columns passed into calculate_features: {df.columns.tolist()}") # Show columns just before the call might fail
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during feature calculation: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Post-Feature Calculation ---
    original_rows = len(df)
    # Consider which columns' NaNs should cause a row drop. Maybe only target-related?
    # If features can be calculated with some NaNs, maybe drop later or based on specific cols.
    df.dropna(inplace=True) # This drops rows if *any* column has NaN
    print(f"ℹ️ Rows before dropna: {original_rows}, after: {len(df)}")

    if df.empty:
       print("❌ DataFrame is empty after feature calculation and dropna. Check feature logic, input data, or dropna strategy.")
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
        print(f"   Model Expected: {expected_features}")
        print(f"   DataFrame Has : {df.columns.tolist()}")
        print(f"   Specifically Missing: {still_missing}")
        return

    df_predict = df[final_features_for_prediction].copy()
    # Final NaN fill before prediction
    df_predict.fillna(0, inplace=True) # Use 0 or another strategy like mean/median if appropriate

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df_predict)

    # Add predictions back to the main df using index alignment
    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    print("\n📊 Sample predictions:")
    close_col_name = f'{TICKER.lower()}_close'
    cols_to_show = ['prediction', 'confidence']
    if close_col_name in df.columns:
        cols_to_show.insert(0, close_col_name)
    print(df[cols_to_show].tail())

# === ENTRY POINT === #
if __name__ == "__main__":
    backtest()
