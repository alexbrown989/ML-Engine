import sqlite3

def init_db():
    conn = sqlite3.connect("signals.db")
    cursor = conn.cursor()

    # Table: signals
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            entry_price REAL,
            rsi REAL,
            vix REAL,
            vvix REAL,
            skew REAL,
            regime TEXT,
            checklist_score INTEGER
        )
    ''')

    # Table: labels
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            signal_id INTEGER,
            label_3p_win INTEGER,
            label_5p_win INTEGER,
            label_10p_win INTEGER,
            label_2p_loss INTEGER,
            chop_flag INTEGER,
            max_gain_pct REAL,
            max_drawdown_pct REAL,
            days_to_max_gain INTEGER,
            days_to_max_loss INTEGER,
            label_reason TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ SQLite DB initialized.")

if __name__ == "__main__":
    init_db()
Add SQLite schema: signals and labels tables
