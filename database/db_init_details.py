import sqlite3

def init_db():
    conn = sqlite3.connect("signals.db")  # change name if needed
    cursor = conn.cursor()

    # 🧹 Clean slate
    cursor.execute("DROP TABLE IF EXISTS signals;")
    cursor.execute("DROP TABLE IF EXISTS labels;")
    cursor.execute("DROP TABLE IF EXISTS features;")  # Ensure features table is included

    # 🧠 Create signals table
    cursor.execute('''
        CREATE TABLE signals (
            signal_id INTEGER PRIMARY KEY,
            ticker TEXT,
            timestamp TEXT,
            entry_price REAL
        )
    ''')

    # 🧠 Create features table (storing feature-related data for each signal)
    cursor.execute('''
        CREATE TABLE features (
            signal_id INTEGER,
            vix REAL,
            vvix REAL,
            skew REAL,
            rsi REAL,
            regime TEXT,
            bollinger_upper REAL,
            bollinger_lower REAL,
            ATR REAL,
            macd_hist REAL,
            obv_roc_5d REAL,
            volume_change_pct REAL,
            news_sentiment_score REAL,
            macro_event_proximity INTEGER,
            strike_distance_pct REAL,
            actual_return_pct_5d REAL,
            confidence_band TEXT,
            FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized and tables created.")

if __name__ == "__main__":
    init_db()
