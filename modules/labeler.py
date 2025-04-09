import yfinance as yf
from datetime import datetime, timedelta

def label_trade(ticker, entry_date, entry_price):
    try:
        entry = datetime.strptime(entry_date, "%Y-%m-%d")
        lookahead_days = [3, 5, 7, 10]

        result = {
            "label_3p_win": 0,
            "label_5p_win": 0,
            "label_10p_win": 0,
            "label_2p_loss": 0,
            "max_gain_pct": 0,
            "max_drawdown_pct": 0,
            "label_reason": "",
            "chop_flag": False
        }

        high_water = []
        low_water = []

        for days in lookahead_days:
            end = entry + timedelta(days=days)
            data = yf.download(ticker, start=entry.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)

            if data.empty:
                continue

            highs = data["High"]
            lows = data["Low"]

            max_gain = (highs.max() - entry_price) / entry_price
            max_drawdown = (lows.min() - entry_price) / entry_price

            high_water.append(max_gain)
            low_water.append(max_drawdown)

            # Only label for 7-day window (same as before)
            if days == 7:
                result["label_3p_win"] = int(max_gain >= 0.03)
                result["label_5p_win"] = int(max_gain >= 0.05)
                result["label_10p_win"] = int(max_gain >= 0.10)
                result["label_2p_loss"] = int(max_drawdown <= -0.02)

                result["max_gain_pct"] = round(max_gain * 100, 2)
                result["max_drawdown_pct"] = round(abs(max_drawdown * 100), 2)
                result["label_reason"] = f"High {max_gain:.2%}, Low {max_drawdown:.2%}"

                # Chop flag (if it's dead flat)
                if abs(max_gain) < 0.015 and abs(max_drawdown) < 0.015:
                    result["chop_flag"] = True

        return result

    except Exception as e:
        return {"error": str(e)}
