import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import pickle

def train_model():
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()

    df = df[df["outcome_class"].notna()]

    # ✅ Cast and reindex to [0, 1] instead of [1, 2]
    df["outcome_class"] = df["outcome_class"].astype(int) - 1

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

    y = df["outcome_class"]
    X = df.drop(columns=["signal_id", "outcome_class"])

    if "regime" in X.columns:
        X = pd.get_dummies(X, columns=["regime"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("\n📊 y_train distribution:")
    print(y_train.value_counts())

    print("\n📊 y_test distribution:")
    print(y_test.value_counts())

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n🧪 Evaluation:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    with open("model_xgb.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved as model_xgb.pkl")

if __name__ == "__main__":
    train_model()
