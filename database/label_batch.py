import sqlite3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.labeler import label_trade

conn = sqlite3.connect("signals.db")
cursor = conn.cursor()

cursor.execute("SELECT id, ticker, timestamp, entry_price FROM signals")
rows = cursor.fetchall()

for row in rows:
    signal_id, ticker, timestamp, entry_price = row

    # Clean up any existing labels for this signal
    cursor.execute("DELETE FROM labels WHERE signal_id = ?", (signal_id,))

    result = label_trade(ticker, timestamp, entry_price)

    if "error" in result:
        print(f"❌ Error for {ticker}: {result['error']}")
        continue

    # Confirm label keys exist
    if "label_5p_win_d5" not in result or "label_2p_loss_d5" not in result:
        print(f"⚠️ Missing expected label keys for signal {signal_id}: {result}")
        continue

    # Define class: 1 = win, 2 = loss, 0 = chop/neutral
    if result["label_5p_win_d5"] and not result["label_2p_loss_d5"]:
        outcome_class = 1  # win
    elif result["label_2p_loss_d5"] and not result["label_5p_win_d5"]:
        outcome_class = 2  # loss
    else:
        outcome_class = 0  # chop/neutral

    result["outcome_class"] = outcome_class

    cols = ", ".join(result.keys())
    placeholders = ", ".join(["?"] * len(result))
    values = list(result.values())

    cursor.execute(f"""
        INSERT INTO labels (signal_id, {cols})
        VALUES (?, {placeholders})
    """, [signal_id] + values)

    print(f"✅ Signal {signal_id} labeled as {outcome_class}.")

conn.commit()
conn.close()
print("✅ All signals labeled and stored.")

