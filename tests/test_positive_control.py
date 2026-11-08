import numpy as np
import pandas as pd

from kx.stats import daily_rank_ic, nw_tstat
from kx.synthetic import make_random_walks


def test_planted_ic_recovered():
    """P6: a signal with a planted rank IC of 0.05 is recovered by the evaluation code."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-07-01", periods=460)
    cols = [f"c{i}" for i in range(480)]
    fwd = rng.normal(size=(460, 480))
    rho = 0.0524  # Pearson rho giving Spearman ~0.05 for normals: (6/pi) * arcsin(rho/2)
    sig = rho * fwd + np.sqrt(1 - rho ** 2) * rng.normal(size=(460, 480))
    ic = daily_rank_ic(pd.DataFrame(sig, idx, cols), pd.DataFrame(fwd, idx, cols))
    mean, t = nw_tstat(ic, lag=12)
    assert abs(mean - 0.05) < 0.01 and t > 3


def test_random_walks_shape_and_ohlc_order():
    w = make_random_walks(3, 100, np.zeros(3), np.full(3, 0.02), seed=1)
    df = w["s0000"]
    assert len(w) == 3 and df.shape == (100, 6)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
