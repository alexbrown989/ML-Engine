# Connect to DB
import sqlite3

conn = sqlite3.connect("signals.db")
cursor = conn.cursor()

# ✅ FIXED: Correct field name is id, not signal_id
cursor.execute("SELECT id, ticker, timestamp, entry_price FROM signals")
rows = cursor.fetchall()

for row in rows:
    signal_id, ticker, timestamp, entry_price = row
    result = label_trade(ticker, timestamp, entry_price)

    if "error" in result:
        print(f"❌ Error for {ticker}: {result['error']}")
        continue

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

    print(f"✅ Signal {signal_id} labeled.")


conn.commit()
conn.close()
print("✅ All signals labeled and stored.")
