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

    # Fill missing regime columns
    for regime in ['regime_calm', 'regime_panic', 'regime_transition']:
        if regime not in df.columns:
            df[regime] = np.nan  # Or fill with 0 if you prefer
    
    print(f"\n🧠 Column types before preprocessing:")
    print(df.dtypes)

    # One-hot encode the regime column
    df['regime'] = df['regime'].astype(str)  # Ensure it's a string before one-hot encoding
    df = pd.get_dummies(df, columns=['regime'], drop_first=True)

    # Ensure 'vvs_roc_5d' is numeric
    df['vvs_roc_5d'] = pd.to_numeric(df['vvs_roc_5d'], errors='coerce')
    df['vvs_roc_5d'].fillna(df['vvs_roc_5d'].mean(), inplace=True)

    print("\n🧠 Column types after preprocessing:")
    print(df.dtypes)

    # Prepare features and target
    X = df.drop(columns=["signal_id", "outcome_class"])
    y = df["outcome_class"].astype(int) - 1  # Assumes outcome_class is in [1, 2] range

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 Training with {len(X_train)} rows, testing with {len(X_test)} rows.")
    
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
    print(f"

