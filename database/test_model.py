import sqlite3
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load trained model
with open("model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Connect and load features
conn = sqlite3.connect("signals.db")
df = pd.read_sql_query("SELECT * FROM features", conn)
conn.close()

# Drop rows with no labels
df = df[df["outcome_class"].notna()]
df["outcome_class"] = df["outcome_class"].astype(int)

# Show class distribution
print("✅ Outcome class breakdown:")
print(df["outcome_class"].value_counts(), "\n")

# One-hot encode regime
df_encoded = pd.get_dummies(df, columns=["regime"])

# Drop unused or non-numeric columns
if "vvs_roc_5d" in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=["vvs_roc_5d"])

# Split features and label
y = df_encoded["outcome_class"]
X = df_encoded.drop(columns=["signal_id", "outcome_class"])

# Align with model training
X = X[[col for col in X.columns if col in model.feature_names_in_]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Predict
y_pred = model.predict(X_test)

# 🧠 Fix: Convert from softprob to class
import numpy as np
y_pred = np.argmax(y_pred, axis=1)

# Evaluation
print("✅ Prediction (first sample):", y_pred[0])
print("📊 Actual (first sample):", y_test.iloc[0])
print(f"\n🔢 Test Set Size: {len(y_test)} sample(s)")

if len(y_test) >= 2:
    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
else:
    print("⚠️ Not enough data to generate a full classification report.")
