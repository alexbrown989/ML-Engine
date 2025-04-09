import sqlite3
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load model
with open("model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Connect to DB and load features
conn = sqlite3.connect("signals.db")
df = pd.read_sql_query("SELECT * FROM features", conn)
conn.close()

# Drop rows without a label and convert label to int
df = df[df['outcome_class'].notna()]
df['outcome_class'] = df['outcome_class'].astype(int)

# 🧠 Label counts
print("✅ Outcome class breakdown:")
print(df['outcome_class'].value_counts(), "\n")

# One-hot encode regime column
df_encoded = pd.get_dummies(df, columns=["regime"])

# Separate X and y
y = df_encoded["outcome_class"]
X = df_encoded.drop(columns=["signal_id", "outcome_class"])

# Keep only columns model expects
X = X[[col for col in X.columns if col in model.feature_names_in_]]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Predict
y_pred = model.predict(X_test)

# Ensure predictions are ints
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

# Show results
print(f"✅ Prediction: {int(y_pred[0])}")
print(f"📊 Actual: {int(y_test.iloc[0])}")
print(f"\n🔢 Test Set Size: {len(y_test)} sample(s)")

if len(y_test) >= 2:
    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
else:
    print("⚠️ Not enough data to generate a classification report.")
