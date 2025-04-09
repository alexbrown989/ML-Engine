from modules.labeler import label_trade
import sqlite3

# Connect to DB
conn = sqlite3.connect("signals.db")
cursor = conn.cursor()

# Fetch signals to label
cursor.execute("SELECT signal_id, ticker, timestamp, entry_price FROM signals")
rows = cursor.fetchall()

for row in rows:
    signal_id, ticker, timestamp, entry_price = row
    result = label_trade(ticker, timestamp, entry_price)

    if "error" in result:
        print(f"❌ Signal {signal_id} failed: {result['error']}")
        continue

    cursor.execute("""
        INSERT OR REPLACE INTO labels (
            signal_id,
            label_3p_win, label_5p_win, label_10p_win,
            label_2p_loss, chop_flag,
            max_gain_pct, max_drawdown_pct,
            label_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal_id,
        result['label_3p_win'],
        result['label_5p_win'],
        result['label_10p_win'],
        result['label_2p_loss'],
        result['chop_flag'],
        float(result['max_gain_pct']),
        float(result['max_drawdown_pct']),
        result['label_reason']
    ))

    print(f"✅ Signal {signal_id} labeled.")

conn.commit()
conn.close()
print("✅ All signals labeled and saved.")


conn.commit()
conn.close()
print("✅ All signals labeled and saved.")
