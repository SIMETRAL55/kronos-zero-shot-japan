import numpy as np
import pandas as pd

from kx.backtest import vol_target_backtest


def _close(days=60, start="2025-02-03"):
    idx = pd.bdate_range(start, periods=days)
    return pd.Series(40000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, days))), idx)


def test_contracts_integer_and_capped():
    c = _close()
    bt = vol_target_backtest(c, pd.Series(1e-8, c.index))  # tiny forecast variance wants many contracts
    assert bt["contracts"].dtype.kind == "i" and bt["contracts"].max() == 2


def test_zero_variance_guard():
    c = _close()
    bt = vol_target_backtest(c, pd.Series(0.0, c.index))
    assert (bt["contracts"] == 0).all() and np.isfinite(bt["net"]).all()


def test_roll_cost_quarterly():
    c = _close(days=60, start="2025-02-03")  # spans the March 2025 SQ (2nd Friday: 2025-03-14)
    bt = vol_target_backtest(c, pd.Series(1e-8, c.index), max_contracts=1)
    roll_day = pd.Timestamp("2025-03-13")
    assert bt.loc[roll_day, "roll_cost"] == 1 * 2 * (11.0 + 50.0)
    assert (bt["roll_cost"] > 0).sum() == 1
