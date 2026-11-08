import numpy as np
import pandas as pd

from kx.backtest import tranche_backtest

IDX = pd.bdate_range("2025-01-06", periods=12)


def _flat(codes, value=100.0):
    return pd.DataFrame(value, IDX, codes)


def test_limit_up_entry_skipped():
    o, c = _flat(["A", "B"]), _flat(["A", "B"])
    c.loc[IDX[1]:, "A"] = 110.0  # A gains 10% on its entry day
    sig = pd.DataFrame({"A": [1.0], "B": [0.0]}, [IDX[0]])
    blocked = pd.DataFrame(False, IDX, ["A", "B"])
    free = tranche_backtest(sig, o, c, blocked, h=1, q=0.5, cost_bp=0.0)
    blocked.loc[IDX[1], "A"] = True
    held = tranche_backtest(sig, o, c, blocked, h=1, q=0.5, cost_bp=0.0)
    assert np.isclose(free.loc[IDX[1], "gross"], 0.10) and held.loc[IDX[1], "gross"] == 0.0


def test_cost_per_tranche():
    o, c = _flat(["A", "B"]), _flat(["A", "B"])
    sig = pd.DataFrame(np.tile([1.0, 0.0], (6, 1)), IDX[:6], ["A", "B"])
    bt = tranche_backtest(sig, o, c, pd.DataFrame(False, IDX, ["A", "B"]), h=3, q=0.5, cost_bp=30.0)
    assert np.allclose(bt.loc[IDX[1]:IDX[6], "net"], -0.003 / 3)


def test_overlap_average():
    o = _flat(["A", "B"])
    c = pd.DataFrame({"A": 100 * 1.01 ** np.arange(12), "B": 100.0}, IDX)
    o["A"] = c["A"].shift(1).fillna(100.0)  # A opens at the prior close, so each day is +1% for A
    sig = pd.DataFrame(np.tile([1.0, 0.0], (6, 1)), IDX[:6], ["A", "B"])
    bt = tranche_backtest(sig, o, c, pd.DataFrame(False, IDX, ["A", "B"]), h=3, q=0.5, cost_bp=0.0)
    assert np.isclose(bt.loc[IDX[4], "gross"], 0.01)  # three active tranches, each +1% that day
    # bench: equal weight at entry, then buy-and-hold drift; tranches on day 3, 2 and 1 of their holding
    w = [1.01 ** k / (1.01 ** k + 1) for k in (2, 1, 0)]
    assert np.isclose(bt.loc[IDX[4], "bench"], np.mean([0.01 * x for x in w]))
