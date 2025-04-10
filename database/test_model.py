import sqlite3
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Load model
with open("model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
conn = sqlite3.connect("signals.db")
df = pd.read_sql_query("SELECT * FROM features WHERE outcome_class IS NOT NULL", conn)
conn.close()

# Handle missing values and convert categorical columns
df.fillna(df.mean(), inplace=True)
df['regime'] = df['regime'].astype(str)
df = pd.get_dummies(df, columns=['regime'], drop_first=True)

# Ensure missing columns are added
for regime in ['regime_calm', 'regime_panic', 'regime_transition']:
    if regime not in df.columns:
        df[regime] = np.nan  # Or fill with 0 if you prefer

# Prepare features and target
y = df["outcome_class"].astype(int) - 1  # Assumes outcome_class is in [1, 2] range
X = df.drop(columns=["signal_id", "outcome_class"])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost model
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
