import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
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

    # Ensure outcome_class is integer-encoded [0, 1]
    y = y.astype(int) - 1  # Assumes that outcome_class is in [1, 2] range

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 Training with {len(X_train)} rows, testing with {len(X_test)} rows.")

    # Store original regimes for accuracy calculation later
    X_test_original_regimes = X_test['regime'].copy()  # Store before modification

    # Process categorical 'regime' column (one-hot encoding)
    X_train = process_categorical(X_train)
    X_test = process_categorical(X_test)

    # Train XGBoost model
    model = xgb.XGBClassifier(
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

    # Log accuracy per regime and confidence level
    log_performance(accuracy, model, X_test, y_test, y_pred, X_test_original_regimes)

    # Save the new model
    model_filename = f"model_xgb_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ New model saved as {model_filename}")

    # Optionally, log retraining metrics
    with open("retraining_log.txt", "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Accuracy: {accuracy * 100:.2f}%\n")

def process_categorical(df):
    # Handle 'vvs_roc_5d' column by converting it to numeric and filling NaNs
    df['vvs_roc_5d'] = pd.to_numeric(df['vvs_roc_5d'], errors='coerce')
    mean_vvs_roc = df['vvs_roc_5d'].mean()
    df['vvs_roc_5d'] = df['vvs_roc_5d'].fillna(mean_vvs_roc)
    
    # Process 'regime' column with one-hot encoding
    df = pd.get_dummies(df, columns=["regime"], drop_first=True)
    return df

def log_performance(accuracy, model, X_test, y_test, y_pred, original_regimes):
    # Calculate accuracy per regime and confidence level
    confidence_levels = ['LOW', 'MEDIUM', 'HIGH']
    accuracy_per_confidence = {}
    accuracy_per_regime = {}

    # Calculate accuracy per confidence level
    for conf in confidence_levels:
        mask = X_test['confidence_band'] == conf
        if mask.any():
            conf_accuracy = accuracy_score(y_test[mask], y_pred[mask])
            accuracy_per_confidence[conf] = conf_accuracy
            print(f"📊 Accuracy for {conf} confidence: {conf_accuracy * 100:.2f}%")
        else:
            accuracy_per_confidence[conf] = 'N/A'
            print(f"📊 Accuracy for {conf} confidence: N/A")

    # Calculate accuracy per regime using original regimes (before one-hot encoding)
    for regime_value in original_regimes.unique():
        mask = original_regimes == regime_value
        if mask.any():
            regime_accuracy = accuracy_score(y_test[mask], y_pred[mask])
            accuracy_per_regime[regime_value] = regime_accuracy
            print(f"📊 Accuracy for {regime_value} regime: {regime_accuracy * 100:.2f}%")
        else:
            accuracy_per_regime[regime_value] = 'N/A'
            print(f"📊 Accuracy for {regime_value} regime: N/A")

    # Optionally, plot feature importance
    plot_feature_importance(model)

    # Save these logs to database or file
    with open("performance_log.txt", "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Accuracy: {accuracy * 100:.2f}%\n")
        log.write(f"Accuracy by confidence: {accuracy_per_confidence}\n")
        log.write(f"Accuracy by regime: {accuracy_per_regime}\n")

def plot_feature_importance(model):
    # Plot feature importance
    xgb.plot_importance(model, importance_type="weight", max_num_features=10)
    plt.show()

if __name__ == "__main__":
    train_model()


