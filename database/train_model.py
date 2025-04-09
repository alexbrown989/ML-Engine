import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import pickle

def train_model():
    # Connect and load
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()

    # Ensure outcome_class is valid integer
    df = df[df["outcome_class"].notna()]
    df["outcome_class"] = df["outcome_class"].astype(int)

    print("✅ Raw outcome_class counts:")
    print(df["outcome_class"].value_counts(), "\n")

    # Drop incomplete rows
    df = df.dropna()
    print("🔍 Missing values per column:")
    print(df.isnull().sum())

    if df.empty:
        print("❌ No data left to train on after filtering. Exiting.")
        return

    # Features + target
    y = df["outcome_class"]
    X = df.drop(columns=["signal_id", "outcome_class"])

    # Encode categorical
    if "regime" in X.columns:
        X = pd.get_dummies(X, columns=["regime"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model (multiclass)
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n🧪 Evaluation:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    # Save model
    with open("model_xgb.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved as model_xgb.pkl")

if __name__ == "__main__":
    train_model()
