import sqlite3
import pandas as pd
import xgboost as xgb
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    print(f"\n🔄 Training model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features_ext", conn)
    conn.close()

    df['outcome_class'] = pd.to_numeric(df['outcome_class'], errors='coerce')
    df.dropna(subset=['outcome_class'], inplace=True)

    y = df['outcome_class'].astype(int)
    X = df.drop(columns=['outcome_class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.2%}")

    filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved as {filename}")


if __name__ == "__main__":
    train_model()
