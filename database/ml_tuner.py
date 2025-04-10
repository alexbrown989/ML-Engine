import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    print(f"\n🔄 Retraining model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features WHERE outcome_class IS NOT NULL", conn)
    conn.close()

    if df.empty:
        print("❌ No data to train on. Exiting.")
        return

    print("\n🧠 Column types before preprocessing:")
    print(df.dtypes)

    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())

    # Handle missing regime data
    print("\n🔍 Handling missing 'regime' data...")
    if 'regime' in df.columns:
        df['regime'].fillna('calm', inplace=True)  # Fill with 'calm' if missing
        print(f"🧠 Missing regime values filled. Sample: {df['regime'].head()}")
    else:
        print("⚠️ 'regime' column missing! Adding 'calm' as default.")
        df['regime'] = 'calm'

    # One-hot encoding for regime column
    print("🔍 One-hot encoding 'regime' column...")
    df = pd.get_dummies(df, columns=['regime'], drop_first=True)
    print(f"🧠 One-hot encoded regime. Columns: {df.columns}")

    # Ensure 'vvs_roc_5d' is numeric and fill missing values
    print("\n🔍 Ensuring 'vvs_roc_5d' is numeric and filling missing values...")
    df['vvs_roc_5d'] = pd.to_numeric(df['vvs_roc_5d'], errors='coerce')
    df['vvs_roc_5d'].fillna(df['vvs_roc_5d'].mean(), inplace=True)

    # Handle missing features (e.g., regime_calm, regime_panic)
    print("\n🔍 Handling missing features...")
    for regime in ['regime_calm', 'regime_panic', 'regime_transition']:
        if regime not in df.columns:
            df[regime] = np.nan  # Or you can use df[regime] = 0 depending on your logic
    print(f"🧠 Missing features filled. Columns: {df.columns}")

    print("\n🧠 Column types after preprocessing:")
    print(df.dtypes)

    # Preparing features and target
    X = df.drop(columns=["signal_id", "outcome_class"])
    y = df["outcome_class"].astype(int) - 1  # Assumes outcome_class is in [1, 2] range

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 Training with {len(X_train)} rows, testing with {len(X_test)} rows.")
    print("\n🧠 Preview of X_train:")
    print(X_train.head())

    # Train XGBoost model
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        enable_categorical=True
    )

    try:
        print("🔧 Training XGBoost model...")
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"❌ Error during training: {str(e)}")
        return

    # Evaluate model performance
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    confidence_band = ['HIGH' if prob[1] > 0.7 else 'MEDIUM' if prob[1] > 0.4 else 'LOW' for prob in y_proba]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy after retraining: {accuracy * 100:.2f}%")

    # Save model
    model_filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ New model saved as {model_filename}")

if __name__ == "__main__":
    train_model()
