import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import pickle

def train_model():
    # Load data
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()

    df = df[df["outcome_class"].notna()]
    df["outcome_class"] = df["outcome_class"].astype(int)  # Already 0, 1, 2

    print("✅ Raw outcome_class counts:")
    print(df["outcome_class"].value_counts(), "\n")

    if "vvs_roc_5d" in df.columns:
        df = df.drop(columns=["vvs_roc_5d"])

    df = df.dropna()
    print("🔍 Missing values per column:")
    print(df.isnull().sum())

    if df.empty:
        print("❌ No data left to train on after filtering. Exiting.")
        return

    # Features and labels
    y = df["outcome_class"]
    X = df.drop(columns=["signal_id", "outcome_class"])

    if "regime" in X.columns:
        X = pd.get_dummies(X, columns=["regime"])

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n📊 y_train distribution:")
    print(y_train.value_counts())
    print("\n📊 y_test distribution:")
    print(y_test.value_counts())

    # Train model
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softmax",
        num_class=3
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("\n🧪 Evaluation:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    with open("model_xgb.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved as model_xgb.pkl")

if __name__ == "__main__":
    train_model()

