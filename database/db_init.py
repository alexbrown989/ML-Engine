import sqlite3

def init_db():
    conn = sqlite3.connect("trades.db")
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

    # Table: labels (expanded)
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
            days_to_max_gain TEXT,
            days_to_max_loss TEXT,
            net_return_pct REAL,
            gain_to_loss_ratio REAL,
            volatility_range_pct REAL,
            outcome_class TEXT,
            directional_bias TEXT,
            days_held INTEGER,
            label_reason TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ SQLite DB initialized.")

if __name__ == "__main__":
    init_db()
