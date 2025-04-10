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
    df = pd.read_sql_query("SELECT * FROM features_ext WHERE outcome_class IS NOT NULL", conn)
    conn.close()

    if df.empty:
        print("❌ No data to train on. Exiting.")
        return

    print("\n🧠 Column types before preprocessing:")
    print(df.dtypes)

    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())

    if 'regime' not in df.columns:
        print("⚠️ 'regime' column missing! Adding default value 'calm'.")
        df['regime'] = 'calm'

    df['regime'] = df['regime'].astype(str).fillna('calm')
    df = pd.get_dummies(df, columns=['regime'], drop_first=True)

    df['vvs_roc_5d'] = pd.to_numeric(df['vvs_roc_5d'], errors='coerce')
    df['vvs_roc_5d'] = df['vvs_roc_5d'].fillna(df['vvs_roc_5d'].mean())

    for col in ['regime_calm', 'regime_panic', 'regime_transition']:
        if col not in df.columns:
            df[col] = 0

    print("\n🧠 Column types after preprocessing:")
    print(df.dtypes)

    y = df['outcome_class'].astype(int) - 1
    X = df.drop(columns=['signal_id', 'outcome_class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 Training with {len(X_train)} rows, testing with {len(X_test)} rows.")

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy after retraining: {accuracy * 100:.2f}%")

    filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved as {filename}")


if __name__ == "__main__":
    train_model()

