from datetime import datetime, timedelta
import yfinance as yf

def label_trade_outcome(ticker, signal_date, entry_price):
    try:
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        end_dt = signal_dt + timedelta(days=7)

        df = yf.download(ticker, start=signal_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
        if df.empty:
            return {"error": "No data returned for labeler."}

        high = df['High']
        low = df['Low']
        close = df['Close']

        gain_pct = ((high.max() - entry_price) / entry_price) * 100
        loss_pct = ((low.min() - entry_price) / entry_price) * 100
        net_return = ((close.iloc[-1] - entry_price) / entry_price) * 100 if not df.empty else 0

        max_gain_day = df['High'].idxmax().strftime("%Y-%m-%d") if not df['High'].empty else None
        max_loss_day = df['Low'].idxmin().strftime("%Y-%m-%d") if not df['Low'].empty else None

        gain_to_loss_ratio = round(gain_pct / abs(loss_pct), 2) if loss_pct != 0 else None
        volatility_range = ((high.max() - low.min()) / entry_price) * 100

        outcome_class = (
            "WIN" if gain_pct >= 5 and loss_pct > -2 else
            "CHOP" if gain_pct < 3 and loss_pct > -2 else
            "LOSS"
        )

        directional_bias = (
            "UP" if net_return >= 1 else
            "DOWN" if net_return <= -1 else
            "NEUTRAL"
        )

        label = {
            "target": 1 if gain_pct >= 3 else 0,
            "label_3p_win": gain_pct >= 3,
            "label_5p_win": gain_pct >= 5,
            "label_10p_win": gain_pct >= 10,
            "label_2p_loss": loss_pct <= -2,
            "chop_flag": gain_pct < 3 and loss_pct > -2,
            "max_gain_pct": round(gain_pct, 2),
            "max_drawdown_pct": round(loss_pct, 2),
            "days_to_max_gain": max_gain_day,
            "days_to_max_loss": max_loss_day,
            "net_return_pct": round(net_return, 2),
            "gain_to_loss_ratio": gain_to_loss_ratio,
            "volatility_range_pct": round(volatility_range, 2),
            "outcome_class": outcome_class,
            "directional_bias": directional_bias,
            "days_held": 7,
            "label_reason": f"Hit {round(gain_pct, 2)}% gain on {max_gain_day}. Max loss was {round(loss_pct, 2)}%."
        }

        return label

    except Exception as e:
        return {"error": str(e)}
