import sqlite3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.labeler import label_trade

conn = sqlite3.connect("signals.db")
cursor = conn.cursor()

# Get all signals
cursor.execute("SELECT id, ticker, timestamp, entry_price FROM signals")
rows = cursor.fetchall()

for row in rows:
    signal_id, ticker, timestamp, entry_price = row
    result = label_trade(ticker, timestamp, entry_price)

    if "error" in result:
        print(f"❌ Error for {ticker}: {result['error']}")
        continue

    outcome_class = (
        1 if result.get("label_5p_win_d5") else
        2 if result.get("label_2p_loss_d5") else
        0
    )
    result["outcome_class"] = int(outcome_class)  # ✅ Ensure it’s INT

    # Build dynamic insert
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

    print(f"✅ Signal {signal_id} labeled as {outcome_class}.")

conn.commit()
conn.close()
print("✅ All signals labeled and stored.")
