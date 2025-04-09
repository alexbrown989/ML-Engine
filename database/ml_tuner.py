import sqlite3
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

def train_model():
    print(f"\n🔄 Retraining model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect("signals.db")
    
    # Load data
    df = pd.read_sql_query("""
        SELECT * FROM features
        WHERE outcome_class IS NOT NULL
    """, conn)
    conn.close()

    if df.empty:
        print("❌ No data to train on. Exiting.")
        return

    # Preparing features and target
    X = df.drop(columns=["signal_id", "outcome_class"])
    y = df["outcome_class"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 Training with {len(X_train)} rows, testing with {len(X_test)} rows.")

    # Train XGBoost model
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy after retraining: {accuracy * 100:.2f}%")

    # Log accuracy per regime and confidence level
    log_performance(accuracy, X_test, y_test, y_pred)

    # Save the new model
    model_filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ New model saved as {model_filename}")

    # Optionally, log retraining metrics
    with open("retraining_log.txt", "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Accuracy: {accuracy * 100:.2f}%\n")

def log_performance(accuracy, X_test, y_test, y_pred):
    # Calculate accuracy per regime and confidence level
    regimes = X_test['regime']  # Assume 'regime' is a column in X_test DataFrame
    confidence_levels = ['LOW', 'MEDIUM', 'HIGH']
    
    accuracy_per_confidence = {}
    accuracy_per_regime = {}

    for conf in confidence_levels:
        mask = y_pred.confidence_band == conf
        conf_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        accuracy_per_confidence[conf] = conf_accuracy
        print(f"📊 Accuracy for {conf} confidence: {conf_accuracy * 100:.2f}%")

    for regime in regimes.unique():
        mask = regimes == regime
        regime_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        accuracy_per_regime[regime] = regime_accuracy
        print(f"📊 Accuracy for {regime} regime: {regime_accuracy * 100:.2f}%")
    
    # Save these logs to database or file
    with open("performance_log.txt", "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Accuracy: {accuracy * 100:.2f}%\n")
        log.write(f"Accuracy by confidence: {accuracy_per_confidence}\n")
        log.write(f"Accuracy by regime: {accuracy_per_regime}\n")

if __name__ == "__main__":
    train_model()
