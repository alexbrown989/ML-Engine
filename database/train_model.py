import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

def train_model():
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("""
        SELECT * FROM features
        WHERE outcome_class IS NOT NULL
    """, conn)
    conn.close()

    # Drop unused or problematic columns
    if 'vvs_roc_5d' in df.columns:
        df = df.drop(columns=['vvs_roc_5d'])

    # One-hot encode categorical variables like 'regime'
    if 'regime' in df.columns:
        df = pd.get_dummies(df, columns=['regime'])

    df = df.dropna()

    X = df.drop(columns=['signal_id', 'outcome_class'])
    y = df['outcome_class'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    joblib.dump(model, "model_xgb.pkl")
    print("✅ Model trained and saved as model_xgb.pkl")

if __name__ == "__main__":
    train_model()
