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
    df = yf.download(TICKER, start=start_date, end=end_date, progress=False) # progress=False to avoid double progress bars

    if df.empty:
        print(f"❌ Failed to fetch data for {TICKER}. Exiting.")
        return

    # --- Download VIX data ---
    print(f"Downloading data for {VIX_TICKER}...")
    vix_df = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False)

    if vix_df.empty:
        print(f"⚠️ Failed to fetch data for {VIX_TICKER}. VIX features might be missing.")
        # Decide how to handle missing VIX: exit, fill with 0/mean, or proceed letting calculate_features handle NaNs
        # For now, we'll proceed and add an empty column to avoid immediate error,
        # but calculate_features might still fail or produce NaNs.
        df['vix'] = np.nan # Add an empty column if VIX download fails
    else:
        # Select VIX Close and rename it appropriately
        vix_df = vix_df[['Close']].rename(columns={'Close': 'vix'})
        # Join VIX data into the main DataFrame based on the Date index
        df = df.join(vix_df, how='left') # Use left join to keep all AAPL dates

    # Add dummy entry_price column (improve later with fill logic)
    # Do this *after* potential joins to avoid MultiIndex issues if done before
    df['entry_price'] = df['Open'].shift(-1)

    print("\n🧠 Columns before processing:")
    print(df.columns) # Should now show VIX columns if join was successful

    # Flatten columns if MultiIndex - adjust if VIX join created multi-level columns unexpectedly
    # (yf.download usually returns single-level column names unless multiple tickers are requested at once)
    if isinstance(df.columns, pd.MultiIndex):
        # This part needs careful handling if VIX added extra levels.
        # Assuming yf.download gave single-level for TICKER and vix_df had 'vix':
        # It's more likely df.columns is NOT a MultiIndex after the join in this specific case.
        # Let's refine the flattening logic for robustness.
        flat_cols = []
        for col in df.columns.values:
            if isinstance(col, tuple):
                # Handle potential tuples from MultiIndex (e.g., ('Close', 'AAPL'))
                 flat_cols.append('_'.join(filter(None, col)).strip().lower())
            else:
                 # Handle single string column names (like 'vix' after the join)
                 flat_cols.append(str(col).lower())
        df.columns = flat_cols
    else:
         df.columns = [col.lower() for col in df.columns]

    print("\n🧠 Columns after flattening:")
    print(df.columns) # Check if 'vix' is present and correctly named

    # Feature engineering - NOW df should contain the 'vix' column
    print("\n🔧 Calculating features...")
    try:
        df = calculate_features(df) # Pass the DataFrame that now includes 'vix'
    except KeyError as e:
        print(f"❌ KeyError during feature calculation: {e}")
        print("Likely cause: A required column (possibly 'vix' despite adding it) is still missing or misnamed after flattening.")
        print("Columns passed to calculate_features:", df.columns)
        return # Stop execution
    except Exception as e:
        print(f"❌ An unexpected error occurred during feature calculation: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for other errors
        return # Stop execution

    df.dropna(inplace=True) # Drop rows with NaNs generated during feature calculation (or from VIX join)

    if df.empty:
       print("❌ DataFrame is empty after feature calculation and dropna. Check feature logic or input data.")
       return

    # Load model
    try:
        model, expected_features = load_model_and_features()
    except FileNotFoundError as e:
         print(f"❌ {e}")
         print("Train a model first using the training script.")
         return
    if model is None:
        print("❌ No model found or loaded. Train one first.")
        return

    # Align columns with model's expected features
    print("\n✨ Aligning DataFrame columns with model features...")
    missing_in_df = set(expected_features) - set(df.columns)
    if missing_in_df:
        print(f"⚠️ Missing features in DataFrame needed by model: {missing_in_df}. Filling with 0.")
        for feature in missing_in_df:
            df[feature] = 0 # Or use pd.NA and ensure fillna(0) below handles it

    extra_in_df = set(df.columns) - set(expected_features)
    if extra_in_df:
        print(f"ℹ️ Extra features in DataFrame not used by model: {extra_in_df}")
        # Keep only expected features for prediction
        # df = df[expected_features].copy() # Re-enable if you want exactly matching columns

    # Ensure all expected features are present before prediction
    final_features_for_prediction = [feat for feat in expected_features if feat in df.columns]
    if len(final_features_for_prediction) != len(expected_features):
        print("❌ Cannot proceed: Not all expected model features are present in the final DataFrame.")
        print(f"   Expected: {expected_features}")
        print(f"   Available: {df.columns.tolist()}")
        return

    df_predict = df[final_features_for_prediction].copy()
    df_predict.fillna(0, inplace=True) # Fill any remaining NaNs with 0

    print("\n🔮 Generating predictions...")
    preds = generate_predictions(model, df_predict) # Use the aligned and filled DataFrame

    # Add predictions back to the original df for inspection (use index alignment)
    df['prediction'] = preds['prediction']
    df['confidence'] = preds['confidence']

    print("\n📊 Sample predictions:")
    print(df[['prediction', 'confidence']].tail())

# === ENTRY POINT === #
if __name__ == "__main__":
    backtest()
