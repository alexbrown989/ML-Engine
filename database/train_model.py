import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    print(f"\n🔄 Training model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features_ext WHERE outcome_class IS NOT NULL", conn)
    conn.close()

    if df.empty:
        print("❌ No data to train on. Exiting.")
        return

    print("\n🧠 Column types before preprocessing:")
    print(df.dtypes)

    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())

    # Handle regime properly
    print("\n🔍 Handling 'regime' column...")
    if 'regime' in df.columns:
        df['regime'] = df['regime'].fillna('calm').astype('category')
    else:
        print("⚠️ 'regime' column missing! Adding 'calm' as default.")
        df['regime'] = pd.Series(['calm'] * len(df), dtype='category')

    # Ensure 'vvs_roc_5d' is numeric and filled
    df['vvs_roc_5d'] = pd.to_numeric(df['vvs_roc_5d'], errors='coerce')
    df['vvs_roc_5d'].fillna(df['vvs_roc_5d'].mean(numeric_only=True), inplace=True)

    print("\n🔍 Column types after cleaning:")
    print(df.dtypes)

    # Prepare features/target
    X = df.drop(columns=["signal_id", "outcome_class"])
    y_raw = df["outcome_class"].astype(int)
    y = y_raw - y_raw.min()  # Ensure 0-based class labels

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"\n📊 Training with {len(X_train)} rows, testing with {len(X_test)} rows.")
    print("\n🧠 Preview of X_train:")
    print(X_train.head())

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        enable_categorical=True  # critical for 'category' dtype
    )

    try:
        print("🔧 Training XGBoost model...")
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"❌ Error during training: {str(e)}")
        return

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    confidence_band = ['HIGH' if prob[1] > 0.7 else 'MEDIUM' if prob[1] > 0.4 else 'LOW' for prob in y_proba]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy after retraining: {accuracy * 100:.2f}%")

    model_filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ New model saved as {model_filename}")

if __name__ == "__main__":
    train_model()
