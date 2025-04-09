import sqlite3
import pandas as pd

conn = sqlite3.connect("signals.db")
df = pd.read_sql_query("SELECT * FROM features", conn)
conn.close()

# Check where we *should* be training
df_filtered = df[df['outcome_class'].isin([0, 1])]
print("✅ Raw outcome_class counts:")
print(df_filtered['outcome_class'].value_counts(dropna=False))
print()

# Show which columns have missing data
missing_summary = df_filtered.isna().sum()
print("🔍 Missing values per column:")
print(missing_summary[missing_summary > 0])

