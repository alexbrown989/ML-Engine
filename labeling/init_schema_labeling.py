import sqlite3

conn = sqlite3.connect('signals.db')
c = conn.cursor()

# Signal metadata
c.execute("""
CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    ticker TEXT,
    direction TEXT
)
""")

# Feature values
c.execute("""
CREATE TABLE IF NOT EXISTS features (
    signal_id INTEGER,
    VIX REAL,
    VVIX REAL,
    SKEW REAL,
    RSI REAL,
    IV_Rank REAL,
    regime TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
)
""")

# Outcome labels
c.execute("""
CREATE TABLE IF NOT EXISTS labels (
    signal_id INTEGER,
    label_3p_win INTEGER,
    label_5p_win INTEGER,
    label_10p_win INTEGER,
    label_2p_loss INTEGER,
    max_gain_pct REAL,
    max_drawdown_pct REAL,
    days_to_max_gain INTEGER,
    days_to_max_loss INTEGER,
    label_reason TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
)
""")

conn.commit()
conn.close()
print("✅ Database initialized: signals.db")
print("✅ DB and tables created at:", conn)
