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

# Drop any rows with missing outcome_class
df = df[df['outcome_class'].notna()]

# 🧠 Optional: Print label counts
print("✅ Outcome class breakdown:")
print(df['outcome_class'].value_counts(), "\n")

# Preprocessing — one-hot encode regime (since it's categorical)
df_encoded = pd.get_dummies(df, columns=["regime"])

# Separate features (X) and target (y)
X = df_encoded.drop(["signal_id", "outcome_class"], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
y = df_encoded["outcome_class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Predict
y_pred = model.predict(X_test)

# Results
print("✅ Prediction:", int(y_pred[0]))
print("📊 Actual:", int(y_test.iloc[0]))
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred))
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
