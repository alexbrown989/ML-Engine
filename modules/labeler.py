import yfinance as yf
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

        max_gain = float(returns.max())
        max_loss = float(returns.min())

        labels = {}
        for d in lookahead_days:
            if len(returns) >= d:
                sub_ret = returns[:d]
                max_ret = float(sub_ret.max())
                min_ret = float(sub_ret.min())
                labels[f'label_3p_win_d{d}'] = int(max_ret >= 0.03)
                labels[f'label_5p_win_d{d}'] = int(max_ret >= 0.05)
                labels[f'label_10p_win_d{d}'] = int(max_ret >= 0.10)
                labels[f'label_2p_loss_d{d}'] = int(min_ret <= -0.02)

        chop_flag = max_gain < 0.015 and max_loss > -0.015

        # 🎯 Outcome class assignment
        # 1 = bullish success, 0 = loss, 2 = chop (optional)
        if labels.get("label_5p_win_d5", 0) == 1:
            outcome_class = 1
        elif labels.get("label_2p_loss_d5", 0) == 1:
            outcome_class = 0
        elif chop_flag:
            outcome_class = 2  # you can drop this line if you want to ignore chop for classification
        else:
            outcome_class = None

        return {
            **labels,
            "max_gain_pct": round(max_gain * 100, 2),
            "max_drawdown_pct": round(max_loss * 100, 2),
            "chop_flag": chop_flag,
            "label_reason": f"High: {round(max_gain*100,2)}%, Low: {round(max_loss*100,2)}%",
            "outcome_class": outcome_class
        }

    except Exception as e:
        return {"error": str(e)}
