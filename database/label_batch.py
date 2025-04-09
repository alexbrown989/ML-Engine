from labeler import label_trade
import sqlite3

# Test signal
sample_signals = [
    {"signal_id": 1, "ticker": "AAPL", "timestamp": "2025-04-01", "entry_price": 170.25},
]

# Connect to DB
conn = sqlite3.connect("signals.db")
c = conn.cursor()

for signal in sample_signals:
    result = label_trade(signal['ticker'], signal['timestamp'], signal['entry_price'])

    if "error" in result:
        print(f"❌ Error labeling signal {signal['signal_id']}: {result['error']}")
        continue

    # Insert results
    c.execute("""
    INSERT OR REPLACE INTO labels (
        signal_id,
        label_3p_win, label_5p_win, label_10p_win, label_2p_loss,
        max_gain_pct, max_drawdown_pct,
        days_to_max_gain, days_to_max_loss,
        label_reason
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal['signal_id'],
        result['label_3p_win'],
        result['label_5p_win'],
        result['label_10p_win'],
        result['label_2p_loss'],
        result['max_gain_pct'],
        result['max_drawdown_pct'],
        result['days_to_max_gain'],
        result['days_to_max_loss'],
        result['label_reason']
    ))

conn.commit()
conn.close()
print("✅ All signals labeled and saved.")
