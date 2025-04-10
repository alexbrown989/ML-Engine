import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import pickle
from datetime import datetime

def train_model():
    print(f"\n🔄 Starting model training @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect("signals.db")
    
    # Load data
    df = pd.read_sql_query("""SELECT * FROM features WHERE outcome_class IS NOT NULL""", conn)
    conn.close()

    if df.empty:
        print("❌ No data to train on. Exiting.")
        return

    # Preparing features and target
    X = df.drop(columns=["signal_id", "outcome_class"])
    y = df["outcome_class"]

    # Ensure outcome_class is integer-encoded [0, 1]
    y = y.astype(int) - 1  # Assumes that outcome_class is in [1, 2] range

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 Training with {len(X_train)} rows, testing with {len(X_test)} rows.")

    # Train XGBoost model
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # Get predicted probabilities and calculate confidence band
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate confidence bands using np.select
    probabilities_class1 = y_pred_proba[:, 1]  # Probabilities for the positive class

    conditions = [
        probabilities_class1 < 0.50,                         # Condition for LOW
        (probabilities_class1 >= 0.50) & (probabilities_class1 < 0.80), # Condition for MEDIUM
        probabilities_class1 >= 0.80                          # Condition for HIGH
    ]
    choices = ['LOW', 'MEDIUM', 'HIGH'] # Corresponding category for each condition

    X_test['confidence_band'] = np.select(conditions, choices, default='UNKNOWN')

    # Derive y_pred from probabilities (0 if proba < 0.5, 1 if proba >= 0.5)
    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy after retraining: {accuracy * 100:.2f}%")

    # Save the new model
    model_filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ New model saved as {model_filename}")

    # Optionally, log retraining metrics
    with open("retraining_log.txt", "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Accuracy: {accuracy * 100:.2f}%\n")

if __name__ == "__main__":
    train_model()

