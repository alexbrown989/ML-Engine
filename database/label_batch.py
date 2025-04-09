# label_batch.py
# Applies labels to all unlabeled signals in SQLite
# Location: /database/

import sqlite3
from modules.labeler import label_trade_outcome
import time

def label_unlabeled_signals():
    conn = sqlite3.connect("trades.db")  # Must match your actual DB name
    cursor = conn.cursor()

    # Step 1: Get all signals not already labeled
    cursor.execute('''
        SELECT id, ticker, timestamp, entry_price 
        FROM signals 
        WHERE id NOT IN (SELECT signal_id FROM labels)
    ''')
    rows = cursor.fetchall()

    print(f"🧠 Found {len(rows)} unlabeled signals...")

    # Step 2: Loop through each signal
    for row in rows:
        signal_id, ticker, date, entry_price = row
        print(f"🔍 Labeling {ticker} from {date} @ ${entry_price}")

        result = label_trade_outcome(ticker, date, entry_price)
        if "error" in result:
            print(f"❌ Skipping {ticker} due to error: {result['error']}")
            continue

        # Step 3: Insert labeled result into `labels` table
        cursor.execute('''
            INSERT INTO labels (
                signal_id, 
                label_3p_win, label_5p_win, label_10p_win, 
                label_2p_loss, chop_flag, 
                max_gain_pct, max_drawdown_pct, 
                days_to_max_gain, days_to_max_loss,
                net_return_pct, gain_to_loss_ratio,
                volatility_range_pct, outcome_class,
                directional_bias, days_held,
                label_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_id,
            result.get("label_3p_win"),
            result.get("label_5p_win"),
            result.get("label_10p_win"),
            result.get("label_2p_loss"),
            result.get("chop_flag"),
            result.get("max_gain_pct"),
            result.get("max_drawdown_pct"),
            result.get("days_to_max_gain"),
            result.get("days_to_max_loss"),
            result.get("net_return_pct"),
            result.get("gain_to_loss_ratio"),
            result.get("volatility_range_pct"),
            result.get("outcome_class"),
            result.get("directional_bias"),
            result.get("days_held"),
            result.get("label_reason")
        ))

        print(f"✅ Saved label for signal ID: {signal_id}")
        time.sleep(0.5)

    conn.commit()
    conn.close()
    print("🎉 All signals labeled!")

if __name__ == "__main__":
    label_unlabeled_signals()
