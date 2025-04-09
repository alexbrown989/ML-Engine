import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_model():
    conn = sqlite3.connect("signals.db")
    
    df = pd.read_sql_query("""
        SELECT f.*, l.label_5p_win_d5
        FROM features f
        JOIN labels l ON f.signal_id = l.signal_id
        WHERE l.label_5p_win_d5 IS NOT NULL
    """, conn)

    conn.close()

    if df.empty:
        print("❌ No data available for training.")
        return

    # Define X and y
    X = df.drop(columns=["signal_id", "label_5p_win_d5", "outcome_class"])
    y = df["label_5p_win_d5"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "model_xgb_label5p_d5.joblib")
    print("✅ Model saved to model_xgb_label5p_d5.joblib")

if __name__ == "__main__":
    train_model()
