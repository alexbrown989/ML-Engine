import sqlite3
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load model
with open("model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
conn = sqlite3.connect("signals.db")
df = pd.read_sql_query("SELECT * FROM features", conn)
conn.close()

df = df[df["outcome_class"].notna()]
df["outcome_class"] = df["outcome_class"].astype(int)

print("✅ Outcome class breakdown:")
print(df["outcome_class"].value_counts(), "\n")

# Drop unused or placeholder features
if "vvs_roc_5d" in df.columns:
    df = df.drop(columns=["vvs_roc_5d"])

# Encode categorical
df = pd.get_dummies(df, columns=["regime"])
df = df.dropna()

# Prepare features
y = df["outcome_class"]
X = df.drop(columns=["signal_id", "outcome_class"])
X = X[[col for col in X.columns if col in model.feature_names_in_]]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predict
y_pred = model.predict(X_test)

print(f"✅ Prediction (first sample): {y_pred[0]}")
print(f"📊 Actual (first sample): {y_test.iloc[0]}")
print(f"\n🔢 Test Set Size: {len(y_test)} sample(s)")

print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
