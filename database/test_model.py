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
df = pd.read_sql_query("SELECT * FROM features WHERE outcome_class IS NOT NULL", conn)
conn.close()

# Clean and prepare data
df["outcome_class"] = df["outcome_class"].astype(int)
print("✅ Outcome class breakdown:")
print(df["outcome_class"].value_counts(), "\n")

# Handle missing features
expected_features = model.feature_names_in_
for feature in expected_features:
    if feature not in df.columns:
        print(f"⚠️ Missing feature '{feature}', adding with NaN values.")
        df[feature] = pd.NA  # Fill missing features with NaN or 0 based on your need

# Encode categorical 'regime' column if needed
df = pd.get_dummies(df, columns=["regime"], drop_first=True)

# Prepare features and target
y = df["outcome_class"]
X = df.drop(columns=["signal_id", "outcome_class"])

# Ensure all model features are available
X = X[[col for col in X.columns if col in expected_features]]

# Split data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Make predictions
y_pred = model.predict(X_test)

# Display results
print(f"✅ Prediction (first sample): {y_pred[0]}")
print(f"📊 Actual (first sample): {y_test.iloc[0]}")
print(f"\n🔢 Test Set Size: {len(y_test)} sample(s)")

# Classification report and accuracy
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

