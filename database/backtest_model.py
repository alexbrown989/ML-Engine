import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import traceback # For detailed error logging

# --- Configuration ---
TICKER = "AAPL"
EXTERNAL_TICKERS = {
    # Symbol: Desired column name
    "^VIX": "vix",
    "^VVIX": "vvix",
    "^SKEW": "skew",
    # Add more external tickers if needed, e.g.:
    # "GC=F": "gold", # Gold futures
    # "CL=F": "oil",  # Crude Oil futures
    # "EURUSD=X": "eurusd" # Forex
}
DAYS_BACK = 365 # Increased backlog for more testing (adjust as needed)
RSI_WINDOW = 14

# --- Path Setup ---
# Ensure project root is in the path to find custom modules
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if project_root not in sys.path:
        print(f"[INFO] Adding project root to path: {project_root}")
        sys.path.append(project_root)
except NameError:
    # Handle case where __file__ is not defined (e.g., interactive session)
    print("[WARN] Could not automatically determine project root. Make sure 'build_features' and 'inference' are importable.")
    script_dir = "." # Assume current directory


# --- Import Custom Modules ---
try:
    from build_features import calculate_features
    from inference import load_model_and_features, generate_predictions
    print("[INFO] Successfully imported custom modules: 'build_features', 'inference'.")
except ImportError as e:
    print(f"[ERROR] Failed to import custom modules: {e}")
    print("[ERROR] Please ensure 'build_features.py' and 'inference.py' exist and are in the Python path.")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] An unexpected error occurred during custom module import: {e}")
    sys.exit(1)


# --- Helper Functions ---

def find_column(df: pd.DataFrame, possible_patterns: list) -> str | None:
    """
    Tries to find a column in the DataFrame matching a list of possible patterns.
    Case-insensitive search. Handles string and tuple column names. Returns the first match found or None.
    """
    df_columns_processed = {}
    for col in df.columns:
        if isinstance(col, tuple):
            # If tuple, use the first element (usually the field name like 'Close')
            # Ensure it's a string before lowercasing
            col_name = str(col[0]).lower() if len(col) > 0 else ""
            df_columns_processed[col_name] = col # Store original tuple column name
        elif isinstance(col, str):
            # If string, use it directly
            df_columns_processed[col.lower()] = col # Store original string column name
        # else: ignore other types if necessary

    for pattern in possible_patterns:
        pattern_lower = pattern.lower()
        if pattern_lower in df_columns_processed:
            found_col = df_columns_processed[pattern_lower]
            print(f"[DEBUG] Found column matching pattern '{pattern}': '{found_col}'")
            return found_col # Return the original column name (string or tuple)

    print(f"[WARN] Could not find any column matching patterns: {possible_patterns}")
    return None

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) with safety checks."""
    if not isinstance(series, pd.Series):
        print("[ERROR] Input to compute_rsi must be a pandas Series.")
        return pd.Series(index=series.index, dtype=np.float64) # Return empty series of correct type
    if series.empty:
         print("[WARN] Input series for compute_rsi is empty.")
         return pd.Series(index=series.index, dtype=np.float64)

    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0) # Fill NaN from diff with 0
    loss = -delta.clip(upper=0).fillna(0) # Fill NaN from diff with 0

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Avoid division by zero
    # Use .values to ensure calculations happen at numpy level if needed, but keep series for index alignment
    rs_values = np.where(avg_loss.values == 0, np.inf, avg_gain.values / avg_loss.values)

    rsi_values = 100 - (100 / (1 + rs_values))

    # Handle infinite RS (where loss was 0, gain > 0) -> RSI should be 100
    rsi_values[np.isinf(rs_values) & (avg_gain.values > 0)] = 100
    # Handle case where both avg_gain and avg_loss are 0 -> RSI is undefined (often set to 50 or NaN)
    rsi_values[np.isinf(rs_values) & (avg_gain.values == 0)] = 50 # Or np.nan if preferred

    # *** FIX: Convert numpy array back to Series with original index ***
    rsi = pd.Series(rsi_values, index=series.index, name='rsi')

    # Ensure the first 'window' periods are NaN as they don't have enough data
    # Note: rolling mean already induces NaNs, but this ensures alignment if calculation changes
    if len(rsi) >= window:
         rsi.iloc[:window-1] = np.nan # diff introduces 1 NaN, rolling introduces window-1 more

    return rsi


# --- Main Backtesting Function ---

def backtest():
    """
    Performs the backtest: downloads data, calculates features,
    loads model, generates predictions, and prints results.
    """
    print(f"\n{'='*20} Starting Backtest {'='*20}")
    print(f"[INFO] Primary Ticker: {TICKER}")
    print(f"[INFO] Lookback Period: {DAYS_BACK} days")
    print(f"[INFO] External Tickers: {list(EXTERNAL_TICKERS.keys())}")

    ticker_lower = TICKER.lower()
    end_date = datetime.today()
    # Add extra buffer for rolling calculations & shifts
    start_date = end_date - timedelta(days=DAYS_BACK + 50)

    # --- Download Primary Ticker Data ---
    print(f"\n[INFO] Downloading primary ticker data for {TICKER}...")
    try:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty:
            print(f"[ERROR] Failed to download data for {TICKER}. DataFrame is empty.")
            return
        print(f"[INFO] Downloaded {len(df)} rows for {TICKER}.")
        # Make index timezone naive if it's timezone aware (common issue)
        if df.index.tz is not None:
            print("[DEBUG] Converting index to timezone naive.")
            df.index = df.index.tz_localize(None)

    except Exception as e:
        print(f"[ERROR] Failed to download data for {TICKER}: {e}")
        print(traceback.format_exc())
        return

    # --- Flatten Primary Ticker Columns ---
    print("[INFO] Processing primary ticker columns...")
    original_columns = df.columns.tolist()
    if isinstance(df.columns, pd.MultiIndex):
        print("[DEBUG] Detected MultiIndex columns. Flattening to 'feature_ticker' format.")
        df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns]
    else:
        # Check if columns already have ticker prefix/suffix or are standard OHLCV
        has_prefix = any(col.lower().startswith(ticker_lower + '_') for col in df.columns)
        has_suffix = any(col.lower().endswith('_' + ticker_lower) for col in df.columns)
        is_standard = all(col.lower() in ['open', 'high', 'low', 'close', 'volume'] for col in df.columns)

        if not has_prefix and not has_suffix and is_standard:
             print("[DEBUG] Detected standard OHLCV columns. Adding ticker suffix: 'feature_ticker'.")
             df.columns = [f"{col.lower()}_{ticker_lower}" for col in df.columns]
        elif not has_suffix: # If it has prefix or some other format, just ensure lower case
             print("[DEBUG] Columns detected with non-standard naming. Converting to lower case.")
             df.columns = [col.lower() for col in df.columns]
        else: # Already seems to have suffix or other format
             print("[DEBUG] Columns seem to have ticker suffix or other format. Converting to lower case.")
             df.columns = [col.lower() for col in df.columns]

    print(f"[DEBUG] Original columns: {original_columns}")
    print(f"[DEBUG] Columns after processing: {df.columns.tolist()}")

    # --- Find Essential Columns ---
    print("[INFO] Locating essential OHLCV columns...")
    open_col = find_column(df, [f'open_{ticker_lower}', f'{ticker_lower}_open', 'Open'])
    high_col = find_column(df, [f'high_{ticker_lower}', f'{ticker_lower}_high', 'High'])
    low_col = find_column(df, [f'low_{ticker_lower}', f'{ticker_lower}_low', 'Low'])
    close_col = find_column(df, [f'close_{ticker_lower}', f'{ticker_lower}_close', 'Close'])
    volume_col = find_column(df, [f'volume_{ticker_lower}', f'{ticker_lower}_volume', 'Volume'])

    essential_cols = {'open': open_col, 'high': high_col, 'low': low_col, 'close': close_col, 'volume': volume_col}
    if not all(essential_cols.values()):
        missing = [k for k, v in essential_cols.items() if v is None]
        print(f"[ERROR] Could not find essential columns: {missing}. Aborting.")
        return
    print("[INFO] Essential columns located successfully.")

    # --- Download and Join External Tickers ---
    print("\n[INFO] Downloading and joining external ticker data...")
    for symbol, col_name in EXTERNAL_TICKERS.items():
        print(f"[DEBUG] Processing external ticker: {symbol} as '{col_name}'")
        try:
            ext_df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if ext_df.empty:
                print(f"[WARN] No data downloaded for {symbol}. Setting column '{col_name}' to NaN.")
                df[col_name] = np.nan
                continue

            # Make index timezone naive if needed
            if ext_df.index.tz is not None:
                ext_df.index = ext_df.index.tz_localize(None)

            # Find the 'Close' column in the external data
            ext_close_col = find_column(ext_df, ['Close', 'close'])
            if ext_close_col is None:
                 print(f"[WARN] Could not find 'Close' column for {symbol}. Setting column '{col_name}' to NaN.")
                 df[col_name] = np.nan
                 continue

            # Select and rename the series before joining
            close_series = ext_df[ext_close_col].rename(col_name)

            # Perform the join
            df = df.join(close_series, how='left')
            print(f"[DEBUG] Joined '{col_name}' from {symbol}.")

            # Check if join resulted in all NaNs (can happen with index mismatches)
            if df[col_name].isnull().all():
                print(f"[WARN] Column '{col_name}' is all NaN after joining {symbol}. Check date alignment or data availability.")
            elif df[col_name].isnull().any():
                 print(f"[DEBUG] Column '{col_name}' contains some NaNs after join, likely due to differing trading days.")

        except Exception as e:
            print(f"[ERROR] Failed to download or join data for {symbol}: {e}")
            print(f"[WARN] Setting column '{col_name}' to NaN due to error.")
            # print(traceback.format_exc()) # Uncomment for full error details
            df[col_name] = np.nan # Ensure column exists even if download failed

    # --- Calculate Entry Price ---
    print("\n[INFO] Calculating entry price (next day's open)...")
    if open_col:
        df['entry_price'] = df[open_col].shift(-1)
        print(f"[DEBUG] Calculated 'entry_price' based on '{open_col}'.")
    else:
        # This case should have been caught earlier, but as a failsafe:
        print(f"[ERROR] Cannot calculate entry price because open column was not found.")
        df['entry_price'] = np.nan


    # --- Calculate RSI ---
    print(f"\n[INFO] Calculating RSI (window={RSI_WINDOW})...")
    if close_col and close_col in df.columns:
        try:
            df['rsi'] = compute_rsi(df[close_col], window=RSI_WINDOW)
            print(f"[DEBUG] RSI column added using '{close_col}'.")
            if df['rsi'].isnull().all():
                 print(f"[WARN] RSI calculation resulted in all NaNs. Check '{close_col}' data.")
            elif df['rsi'].isnull().sum() > RSI_WINDOW: # Expect ~RSI_WINDOW NaNs at start
                 print(f"[DEBUG] RSI contains {df['rsi'].isnull().sum()} NaNs.")

        except Exception as e:
            print(f"[ERROR] Failed to compute RSI: {e}")
            print(traceback.format_exc())
            # Don't necessarily exit, features might handle missing RSI
            df['rsi'] = np.nan # Add NaN column if calc fails
    else:
        print(f"[ERROR] Cannot compute RSI because close column ('{close_col}') not found or invalid.")
        # Depending on model requirements, you might exit or add NaNs
        df['rsi'] = np.nan

    print(f"\n[DEBUG] Columns before feature calculation: {df.columns.tolist()}")
    print(f"[DEBUG] NaN counts before filling:\n{df.isnull().sum().sort_values(ascending=False).head(15)}")

    # --- Fill NaNs ---
    # Careful filling: ffill().bfill() can propagate stale data forward then backward.
    # Consider domain-specific filling (e.g., fill volume NaNs with 0?)
    # For simplicity, using ffill -> bfill as in original idea.
    print("\n[INFO] Filling NaN values (forward fill then backward fill)...")
    original_nan_counts = df.isnull().sum()
    for col in df.columns:
        if df[col].isnull().any():
            pre_fill_nans = df[col].isnull().sum()
            if pre_fill_nans > 0:
                 # Fill forward first to carry last known value
                 df[col] = df[col].ffill()
                 # Then fill backward to handle NaNs at the beginning
                 df[col] = df[col].bfill()
                 post_fill_nans = df[col].isnull().sum()
                 if post_fill_nans < pre_fill_nans:
                      print(f"[DEBUG] Filled {pre_fill_nans - post_fill_nans} NaNs in column '{col}'. {post_fill_nans} NaNs remain.")
                 if post_fill_nans > 0:
                      print(f"[WARN] Column '{col}' still contains {post_fill_nans} NaNs after ffill/bfill. Consider imputation or dropping.")
    print(f"[DEBUG] NaN counts after filling:\n{df.isnull().sum().sort_values(ascending=False).head(15)}")


    # --- Calculate Features ---
    print("\n[INFO] Calculating features using 'build_features.calculate_features'...")
    try:
        df = calculate_features(df.copy()) # Pass a copy to avoid modifying original in place if function does that
        print("[INFO] Feature calculation complete.")
        print(f"[DEBUG] Columns after feature calculation: {df.columns.tolist()}")
        print(f"[DEBUG] NaN counts after feature calculation:\n{df.isnull().sum().sort_values(ascending=False).head(15)}")
    except Exception as e:
        print(f"[ERROR] Feature calculation failed: {e}")
        print(traceback.format_exc())
        return # Cannot proceed without features

    # --- Load Model and Expected Features ---
    print("\n[INFO] Loading model and expected features...")
    try:
        model, expected_features = load_model_and_features()
        if model is None or expected_features is None:
             print("[ERROR] Failed to load model or expected features. Check 'inference.py'.")
             return
        print(f"[INFO] Model loaded. Expected features: {len(expected_features)}")
        print(f"[DEBUG] Expected features list: {expected_features[:5]}...{expected_features[-5:]}") # Print sample
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        print(traceback.format_exc())
        return

    # --- Align DataFrame Columns with Expected Features ---
    print("[INFO] Aligning DataFrame columns with model's expected features...")
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        print(f"[WARN] The following expected features are missing from the DataFrame: {missing_features}")
        print("[WARN] Adding missing feature columns with NaN values.")
        for f in missing_features:
            df[f] = np.nan

    extra_features = [f for f in df.columns if f not in expected_features]
    if extra_features:
        print(f"[DEBUG] The following columns are in the DataFrame but not expected by the model (will be kept for now): {extra_features}")

        # --- Handle Remaining NaNs in Feature Columns ---
    # Option 1: Drop rows with NaNs in expected features (might lose data)
    # Option 2: Impute NaNs (e.g., with 0, mean, median)
    print("\n[INFO] Handling NaNs in expected feature columns...")
    features_to_check_for_na = [f for f in expected_features if f in df.columns] # Only check cols that actually exist
    
    # --- INSERT DEBUG PRINTS HERE ---
    print("[DEBUG] DataFrame columns before final dropna:")
    print(df.columns.tolist())
    print(f"[DEBUG] NaN counts for expected features subset ({len(features_to_check_for_na)} features):")
    # Only print NaN counts for features we are actually checking/dropping based on
    if features_to_check_for_na: 
         print(df[features_to_check_for_na].isnull().sum().sort_values(ascending=False).head(15))
    else:
         print("[DEBUG] No expected features found in DataFrame columns to check for NaNs.")
    # --- END DEBUG PRINTS ---

    # Using dropna for simplicity, as in original logic. Consider imputation if data loss is too high.
    initial_rows = len(df)
    features_to_check_for_na = [f for f in expected_features if f in df.columns] # Only check cols that actually exist
    df.dropna(subset=features_to_check_for_na, inplace=True)
    rows_after_dropna = len(df)
    print(f"[INFO] Dropped {initial_rows - rows_after_dropna} rows containing NaNs in expected feature columns.")

    if df.empty:
        print("[ERROR] DataFrame is empty after removing rows with NaNs in features. Cannot generate predictions.")
        return

    # Failsafe: Impute any remaining NaNs in feature columns with 0 (e.g., if a column was all NaN initially)
    # This assumes the model can handle 0s. Adjust if needed.
    df_predict = df[expected_features].copy() # Select only expected features for prediction
    if df_predict.isnull().values.any():
         print("[WARN] NaNs still detected in prediction input AFTER dropna. Imputing with 0.")
         print(f"[DEBUG] NaN counts before imputation:\n{df_predict.isnull().sum().sort_values(ascending=False).head(15)}")
         df_predict.fillna(0, inplace=True)


    # --- Generate Predictions ---
    print("\n[INFO] Generating predictions...")
    try:
        preds_df = generate_predictions(model, df_predict) # Assuming returns dict/df with 'prediction' and 'confidence'

        # Check if prediction output is valid
        if not isinstance(preds_df, (pd.DataFrame, pd.Series, dict)) or 'prediction' not in preds_df:
             print("[ERROR] Prediction function did not return expected format (dict or DataFrame with 'prediction' key).")
             return

        # Align prediction index with df_predict index before assigning
        if isinstance(preds_df, (pd.DataFrame, pd.Series)):
             preds_df = preds_df.reindex(df_predict.index) # Ensure index matches

        df['prediction'] = preds_df['prediction']
        if 'confidence' in preds_df:
            df['confidence'] = preds_df['confidence']
            print("[INFO] Predictions generated with confidence scores.")
        else:
            df['confidence'] = np.nan # Add NaN confidence if not provided
            print("[INFO] Predictions generated (no confidence scores provided).")

    except Exception as e:
        print(f"[ERROR] Prediction generation failed: {e}")
        print(traceback.format_exc())
        return

    # --- Display Results ---
    print("\n[INFO] Backtest Results:")

    # Try to show relevant columns
    cols_to_show = []
    if close_col and close_col in df.columns: cols_to_show.append(close_col)
    if 'prediction' in df.columns: cols_to_show.append('prediction')
    if 'confidence' in df.columns: cols_to_show.append('confidence')
    if 'entry_price' in df.columns: cols_to_show.append('entry_price') # Useful for simulating trades

    if not cols_to_show:
        print("[WARN] Could not find standard columns (close, prediction, confidence) to display results.")
        print(df.tail()) # Show last few rows of whatever exists
    else:
        print("\n📊 Sample Predictions (Last 5 rows):")
        print(df[cols_to_show].tail())

    # Add more analysis here if needed (e.g., calculate hypothetical PnL, accuracy, etc.)


    print(f"\n{'='*20} Backtest Complete {'='*20}")

# --- Run Backtest ---
if __name__ == "__main__":
    try:
        backtest()
    except Exception as e:
        print("\n[FATAL] An unhandled exception occurred during backtest execution!")
        print(f"[FATAL] Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)
