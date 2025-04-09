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

    # Step 2: Filter valid labeled rows
    df = df[df['outcome_class'].isin([0, 1])]

    # Step 3: Handle missing values + type cleanup
    df = df.dropna()
    df['regime'] = df['regime'].astype('category')
    df['vvs_roc_5d'] = df['vvs_roc_5d'].astype(float)  # if nulls removed

    # Step 4: Define features and target
    y = df['outcome_class']
    X = df.drop(columns=['signal_id', 'outcome_class'])

    # Step 5: Convert categorical features
    X = pd.get_dummies(X)

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Step 7: Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Step 8: Evaluate
    y_pred = model.predict(X_test)
    print("🔍 Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Step 9: Save model
    with open("model_xgb.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved as model_xgb.pkl")

if __name__ == "__main__":
    train_model()
