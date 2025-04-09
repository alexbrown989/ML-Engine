import sqlite3

def build_features():
    conn = sqlite3.connect("signals.db")
    cursor = conn.cursor()

    # Ensure proper table schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            signal_id INTEGER PRIMARY KEY,
            vix REAL,
            vvix REAL,
            skew REAL,
            vvs_adj REAL,
            vvs_roc_5d REAL,
            rsi REAL,
            regime TEXT,
            checklist_score INTEGER,
            chop_flag INTEGER,
            outcome_class INTEGER,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    ''')

    # Pull signal data + labels
    cursor.execute('''
        SELECT s.id, s.vix, s.vvix, s.skew, s.rsi, s.regime, s.checklist_score,
               l.chop_flag, l.outcome_class
        FROM signals s
        JOIN labels l ON s.id = l.signal_id
    ''')
    rows = cursor.fetchall()

    for row in rows:
        signal_id, vix, vvix, skew, rsi, regime, checklist, chop_flag, outcome = row
        try:
            # ⚠️ Skip rows with NULL labels
            if outcome is None:
                print(f"⚠️ Skipping signal {signal_id}: no outcome_class.")
                continue

            # Derived features
            vvs_adj = (vix + vvix) / skew if skew else None
            vvs_roc_5d = None  # Placeholder

            cursor.execute('''
                INSERT OR REPLACE INTO features (
                    signal_id, vix, vvix, skew, vvs_adj, vvs_roc_5d,
                    rsi, regime, checklist_score, chop_flag, outcome_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, vix, vvix, skew, vvs_adj, vvs_roc_5d,
                rsi, regime, checklist, chop_flag, int(outcome)
            ))
        except Exception as e:
            print(f"❌ Error processing signal {signal_id}: {e}")

    conn.commit()
    conn.close()
    print("✅ Features built and stored.")

if __name__ == "__main__":
    build_features()

if __name__ == "__main__":
    build_features()
