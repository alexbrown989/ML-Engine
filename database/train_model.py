import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle

def train_model():
    # Step 1: Connect to DB and load features
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()

    # Step 2: Filter for labeled rows only (binary classification)
    df = df[df['outcome_class'].isin([0, 1])]
    print("🔍 Label counts before dropna:")
    print(df['outcome_class'].value_counts(dropna=False), "\n")

    # Step 3: Drop rows with missing values
    df = df.dropna()
    print(f"📦 Remaining rows after dropna: {len(df)}")
    if len(df) == 0:
        print("❌ No data left to train on after filtering. Exiting.")
        return

    # Step 4: Type casting
    df['regime'] = df['regime'].astype('category')
    df['vvs_roc_5d'] = df['vvs_roc_5d'].astype(float)

    # Step 5: Define features and labels
    y = df['outcome_class'].astype(int)
    X = df.drop(columns=['signal_id', 'outcome_class'])

    # Step 6: One-hot encode categorical columns
    X = pd.get_dummies(X)

    # Step 7: Train/test split with safety
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
    except ValueError as e:
        print(f"❌ Train/test split failed: {e}")
        return

    if len(X_train) == 0 or len(X_test) == 0:
        print("⚠️ Not enough data to split into train and test. Exiting.")
        return

    # Step 8: Train the XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Step 9: Evaluate model
    y_pred = model.predict(X_test)
    print("\n🔍 Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Step 10: Save model
    with open("model_xgb.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as model_xgb.pkl")
    print(f"📊 Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")

if __name__ == "__main__":
    train_model()
