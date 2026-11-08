import numpy as np
import pandas as pd
import pytest

from kx.jquants import JQuants, panel, trading_days
from kx.nikkei import parse_official
from kx.universe import universe

IDX = pd.bdate_range("2025-01-01", periods=80)
T = IDX[-1]


def _close(codes, n=80):
    return pd.DataFrame(100.0, IDX[-n:], codes).reindex(IDX)


def _master(rows):
    return pd.DataFrame(rows, columns=["Code", "ScaleCat", "Mkt"])


def test_scale_filter_core_large_mid():
    m = _master([("A", "TOPIX Core30", "0111"), ("B", "TOPIX Mid400", "0111"), ("C", "TOPIX Small 1", "0111"),
                 ("D", "-", "0112")])
    c = _close(list("ABCD"))
    assert universe(m, c, c, T) == ["A", "B"]


def test_fallback_top500_by_turnover():
    m = _master([("A", "-", "0111"), ("B", "-", "0111"), ("C", "-", "0111"), ("D", "-", "0112")])
    c = _close(list("ABCD"))
    v = pd.DataFrame({"A": 3.0, "B": 1.0, "C": 2.0, "D": 9.0}, IDX)
    assert universe(m, c, v, T, fallback_n=2) == ["A", "C"]


def test_missing_bar_excluded():
    m = _master([("A", "TOPIX Core30", "0111"), ("B", "TOPIX Core30", "0111")])
    c = _close(["A", "B"])
    c.loc[T, "B"] = np.nan
    assert universe(m, c, c, T) == ["A"]


def test_delisted_drops_after_date():
    m = _master([("A", "TOPIX Core30", "0111")])  # B was delisted, so the master as of T no longer lists it
    c = _close(["A", "B"])
    assert universe(m, c, c, T) == ["A"]


def test_min_history_60():
    m = _master([("A", "TOPIX Core30", "0111"), ("B", "TOPIX Core30", "0111")])
    c = pd.concat([_close(["A"]), _close(["B"], n=59)], axis=1)
    assert universe(m, c, c, T) == ["A"]


def test_split_adjusted_continuous():
    bars = pd.DataFrame({"Date": ["2025-03-03", "2025-03-04", "2025-03-05"], "Code": "X",
                         "C": [1000.0, 1004.0, 502.0], "AdjC": [500.0, 502.0, 502.0]})
    r = panel(bars, "AdjC")["X"].pct_change().dropna()
    assert ((r > -0.05) & (r < 0.05)).all()


def test_trading_days_include_half_days():
    cal = pd.DataFrame({"Date": ["2025-12-30", "2025-12-31", "2026-01-05"], "HolDiv": ["1", "0", "2"]})
    assert list(trading_days(cal)) == list(pd.to_datetime(["2025-12-30", "2026-01-05"]))


def test_client_refuses_sealed_dates():
    with pytest.raises(ValueError):
        JQuants(key="x").daily_bars("2026-05-01")


def test_parse_official_truncates_at_end():
    raw = "データ日付,終値,始値,高値,安値\n2026/04/30,1,1,1,1\n2026/05/01,2,2,2,2\n本資料は,,,,\n".encode("shift_jis")
    df = parse_official(raw, "2026-04-30")
    assert list(df.index) == [pd.Timestamp("2026-04-30")] and df["close"].iloc[0] == 1.0
