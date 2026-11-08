"""Nikkei 225 daily OHLC (PREREG §2). Primary: Yahoo ^N225 (longer history). Check: Nikkei Inc.'s official CSV."""
import io

import numpy as np
import pandas as pd
import requests

OFFICIAL = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_jp.csv"
COLS = ["open", "high", "low", "close"]


def parse_official(raw: bytes, end: str) -> pd.DataFrame:
    """Columns データ日付, 終値, 始値, 高値, 安値 (Shift-JIS). Rows after `end` are dropped before anything is kept."""
    df = pd.read_csv(io.StringIO(raw.decode("shift_jis", errors="replace")))
    df.columns = ["date", "close", "open", "high", "low"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] <= pd.Timestamp(end)].set_index("date").sort_index()
    return df[COLS].astype(float)


def official_daily(end: str) -> pd.DataFrame:
    r = requests.get(OFFICIAL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return parse_official(r.content, end)


def yahoo_daily(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    stop = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance `end` is exclusive
    df = yf.download("^N225", start=start, end=stop, auto_adjust=False, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[df.index <= pd.Timestamp(end)]
    return df[COLS].astype(float).dropna()


def cross_check(a: pd.DataFrame, b: pd.DataFrame, n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Max relative OHLC difference on n random common dates."""
    common = a.index.intersection(b.index)
    pick = pd.DatetimeIndex(np.random.default_rng(seed).choice(common, size=min(n, len(common)), replace=False))
    rel = (a.loc[pick, COLS] / b.loc[pick, COLS] - 1).abs()
    return rel.assign(max_rel=rel.max(axis=1)).sort_index()
