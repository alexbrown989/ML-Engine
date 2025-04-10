import sqlite3

def init_db():
    conn = sqlite3.connect("signals.db")  # change name if needed
    cursor = conn.cursor()

    # 🧹 Clean slate
    cursor.execute("DROP TABLE IF EXISTS signals;")
    cursor.execute("DROP TABLE IF EXISTS labels;")

    # 🧠 Create signals table
    cursor.execute('''
        CREATE TABLE signals (
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

    # 🧠 Create labels table (merged with multi-horizon + original features)
    cursor.execute('''
        CREATE TABLE labels (
            signal_id INTEGER,
            label_3p_win_d3 INTEGER,
            label_5p_win_d3 INTEGER,
            label_10p_win_d3 INTEGER,
            label_2p_loss_d3 INTEGER,
            label_3p_win_d5 INTEGER,
            label_5p_win_d5 INTEGER,
            label_10p_win_d5 INTEGER,
            label_2p_loss_d5 INTEGER,
            label_3p_win_d7 INTEGER,
            label_5p_win_d7 INTEGER,
            label_10p_win_d7 INTEGER,
            label_2p_loss_d7 INTEGER,
            label_3p_win_d10 INTEGER,
            label_5p_win_d10 INTEGER,
            label_10p_win_d10 INTEGER,
            label_2p_loss_d10 INTEGER,
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
    print("✅ signals.db schema reset and created.")

# CLI usage
if __name__ == "__main__":
    init_db()
