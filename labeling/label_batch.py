import sqlite3
import sys
import os

# Ensure correct path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.labeler import label_trade

def label_signals():
    # Connect to the database
    conn = sqlite3.connect("signals.db")
    cursor = conn.cursor()

    # Retrieve all signals that need labeling
    cursor.execute("SELECT id, ticker, timestamp, entry_price FROM signals")
    rows = cursor.fetchall()

    for row in rows:
        signal_id, ticker, timestamp, entry_price = row

        # Clean up any existing labels for this signal (for new labeling process)
        cursor.execute("DELETE FROM labels WHERE signal_id = ?", (signal_id,))

        # Call label_trade to get the result
        result = label_trade(ticker, timestamp, entry_price)

        if "error" in result:
            print(f"❌ Error for {ticker}: {result['error']}")
            continue

        # Ensure that all expected label keys are in the result
        expected_labels = [
            "label_3p_win_d3", "label_5p_win_d3", "label_10p_win_d3", "label_2p_loss_d3",
            "label_3p_win_d5", "label_5p_win_d5", "label_10p_win_d5", "label_2p_loss_d5",
            "label_3p_win_d7", "label_5p_win_d7", "label_10p_win_d7", "label_2p_loss_d7",
            "label_3p_win_d10", "label_5p_win_d10", "label_10p_win_d10", "label_2p_loss_d10",
            "max_gain_pct", "max_drawdown_pct", "days_to_max_gain", "days_to_max_loss",
            "chop_flag", "outcome_class", "label_reason"
        ]
        
        # Check for missing expected label keys
        missing_labels = [label for label in expected_labels if label not in result]
        if missing_labels:
            print(f"⚠️ Missing expected label keys for signal {signal_id}: {missing_labels}")
            continue

        # Define the outcome class based on win/loss criteria
        if result["label_5p_win_d5"] and not result["label_2p_loss_d5"]:
            outcome_class = 1  # Win
        elif result["label_2p_loss_d5"] and not result["label_5p_win_d5"]:
            outcome_class = 2  # Loss
        else:
            outcome_class = 0  # Chop/neutral

        # Add outcome_class to the result
        result["outcome_class"] = outcome_class

        # Prepare columns and values for the insert query
        cols = ", ".join(result.keys())
        placeholders = ", ".join(["?"] * len(result))
        values = list(result.values())

        # Insert the result into the labels table
        cursor.execute(f"""
            INSERT INTO labels (signal_id, {cols})
            VALUES (?, {placeholders})
        """, [signal_id] + values)

        print(f"✅ Signal {signal_id} labeled as {outcome_class}.")

    # Commit and close the connection
    conn.commit()
    conn.close()
    print("✅ All signals labeled and stored.")

# Run the labeling process
if __name__ == "__main__":
    label_signals()


