
import yfinance as yf
import datetime

def label_trade(ticker, entry_date, entry_price):
    try:
        start_date = datetime.datetime.strptime(entry_date, "%Y-%m-%d")
        end_date = start_date + datetime.timedelta(days=7)
        hist = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        max_gain = ((hist['High'] - entry_price) / entry_price).max()
        max_loss = ((hist['Low'] - entry_price) / entry_price).min()
        label_3p_win = int(max_gain >= 0.03)
        label_5p_win = int(max_gain >= 0.05)
        label_10p_win = int(max_gain >= 0.10)
        label_2p_loss = int(max_loss <= -0.02)

        result = {
            "label_3p_win": label_3p_win,
            "label_5p_win": label_5p_win,
            "label_10p_win": label_10p_win,
            "label_2p_loss": label_2p_loss,
            "max_gain_pct": round(max_gain * 100, 2),
            "max_drawdown_pct": round(max_loss * 100, 2),
            "label_reason": f"High {round(max_gain * 100, 2)}%, Low {round(max_loss * 100, 2)}%"
        }

        return result
    except Exception as e:
        return {"error": str(e)}
