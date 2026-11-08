import pandas as pd


def next_trading_days(calendar: pd.DatetimeIndex, t: pd.Timestamp, k: int) -> pd.DatetimeIndex:
    """The k trading days strictly after t, from the exchange calendar (weekends and holidays skipped)."""
    cal = pd.DatetimeIndex(calendar).sort_values()
    i = cal.searchsorted(pd.Timestamp(t), side="right")
    out = cal[i:i + k]
    if len(out) < k:
        raise ValueError(f"calendar ends before {k} trading days after {t:%Y-%m-%d}")
    return out
