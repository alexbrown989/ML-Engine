import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
from modules.labeler import label_trade

# Connect to the database
conn = sqlite3.connect("signals.db")
cursor = conn.cursor()

# Step 1: Fetch all rows from signals table
cursor.execute("SELECT signal_id, ticker, timestamp, entry_price FROM signals")
rows = cursor.fetchall()

for signal_id, ticker, timestamp, entry_price in rows:
    print(f"🔍 Labeling {ticker} from {timestamp}...")

    result = label_trade(ticker, timestamp, entry_price)

    if "error" in result:
        print(f"❌ Error for {ticker}: {result['error']}")
        continue

    # Build dynamic insert based on result keys
    cols = ", ".join(result.keys())
    placeholders = ", ".join(["?"] * len(result))
    values = list(result.values())

    cursor.execute(f"""
        INSERT OR REPLACE INTO labels (
            signal_id, {cols}
        ) VALUES (
            ?, {placeholders}
        )
    """, [signal_id] + values)

    print(f"✅ Signal {signal_id} labeled.")

conn.commit()
conn.close()
print("✅ All signals labeled and stored.")


