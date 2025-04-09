import sqlite3

def build_features():
    conn = sqlite3.connect("signals.db")
    cursor = conn.cursor()

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
            outcome_class TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    ''')

    # Fetch from valid, labeled signals only
    cursor.execute('''
        SELECT s.id, s.vix, s.vvix, s.skew, s.rsi, s.regime, s.checklist_score,
               l.chop_flag, l.outcome_class
        FROM signals s
        INNER JOIN labels l ON s.id = l.signal_id
        WHERE l.outcome_class IS NOT NULL
    ''')
    rows = cursor.fetchall()

    skipped = 0

    for row in rows:
        signal_id, vix, vvix, skew, rsi, regime, checklist, chop_flag, outcome = row
        try:
            if outcome is None:
                skipped += 1
                continue

            vvs_adj = (vix + vvix) / skew if skew else None
            vvs_roc_5d = None

            cursor.execute('''
                INSERT OR REPLACE INTO features (
                    signal_id, vix, vvix, skew, vvs_adj, vvs_roc_5d,
                    rsi, regime, checklist_score, chop_flag, outcome_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, vix, vvix, skew, vvs_adj, vvs_roc_5d,
                rsi, regime, checklist, chop_flag, outcome
            ))
        except Exception as e:
            print(f"❌ Error processing signal {signal_id}: {e}")
            skipped += 1

    conn.commit()
    conn.close()
    print(f"✅ Features built and stored. Skipped {skipped} rows due to missing or invalid data.")

if __name__ == "__main__":
    build_features()

