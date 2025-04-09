import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def label_trade(ticker, timestamp, entry_price):
    try:
        lookahead_days = [3, 5, 7, 10]
        entry_date = datetime.strptime(timestamp, "%Y-%m-%d")
        end_date = entry_date + timedelta(days=max(lookahead_days) + 1)

        data = yf.download(ticker, start=timestamp, end=end_date.strftime("%Y-%m-%d"))
        if data.empty:
            return {"error": f"No data for {ticker} from {timestamp}."}

        future_prices = data["Close"]
        returns = (future_prices - entry_price) / entry_price

        # Calculate max gain/drawdown over the period
        max_gain = returns.max()
        max_loss = returns.min()

        # Multi-horizon labels
        labels = {}
        for d in lookahead_days:
            if len(returns) >= d:
                sub_ret = returns[:d]
                labels[f'label_3p_win_d{d}'] = int(sub_ret.max().item() >= 0.03)
                labels[f'label_5p_win_d{d}'] = int(sub_ret.max().item() >= 0.05)
                labels[f'label_10p_win_d{d}'] = int(sub_ret.max().item() >= 0.10)
                labels[f'label_2p_loss_d{d}'] = int(sub_ret.min().item() <= -0.02) # This line appears twice in your snippet

        # Chop detection
        chop_flag = max_gain < 0.015 and max_loss > -0.015

        return {
            **labels,
            "max_gain_pct": round(max_gain * 100, 2),
            "max_drawdown_pct": round(max_loss * 100, 2),
            "chop_flag": chop_flag,
            "label_reason": f"High: {round(max_gain*100,2)}%, Low: {round(max_loss*100,2)}%"
        }

    except Exception as e:
        return {"error": str(e)}
