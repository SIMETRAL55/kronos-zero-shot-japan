import math

import numpy as np
import pandas as pd
from scipy import stats as st

from kx.stats import daily_rank_ic, deflated_sharpe, dm_test, fama_macbeth, nw_tstat, pbo_cscv, qlike


def _panel(rng, days=300, names=400):
    idx = pd.bdate_range("2025-01-01", periods=days)
    return idx, [f"c{i}" for i in range(names)], rng


def test_ic_perfect():
    idx, cols, rng = _panel(np.random.default_rng(0))
    f = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), idx, cols)
    assert np.allclose(daily_rank_ic(f, f), 1.0)


def test_ic_noise():
    idx, cols, rng = _panel(np.random.default_rng(0))
    s = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), idx, cols)
    f = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), idx, cols)
    assert abs(daily_rank_ic(s, f).mean()) < 0.02


def test_ic_min_names_drops_thin_days():
    idx, cols, rng = _panel(np.random.default_rng(0), days=5, names=400)
    s = pd.DataFrame(rng.normal(size=(5, 400)), idx, cols)
    s.iloc[0, 150:] = np.nan
    assert idx[0] not in daily_rank_ic(s, s, min_names=300).index


def test_nw_smaller_than_naive_on_ar1():
    rng = np.random.default_rng(0)
    e = np.zeros(1000)
    for i in range(1, 1000):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    x = pd.Series(0.2 + e)
    naive = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    assert abs(nw_tstat(x, lag=12)[1]) < abs(naive)


def test_fm_recovers_planted():
    idx, cols, rng = _panel(np.random.default_rng(1), days=200)
    z1 = pd.DataFrame(rng.normal(size=(200, 400)), idx, cols)
    z2 = pd.DataFrame(rng.normal(size=(200, 400)), idx, cols)
    y = 0.3 * z1 + pd.DataFrame(rng.normal(size=(200, 400)), idx, cols)
    res = fama_macbeth(y, {"k": z1, "b": z2}, lag=1)
    assert res.loc["k", "t"] > 3 and abs(res.loc["b", "t"]) < 2


def test_dm_detects_better():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.uniform(1, 2, 500))
    a = b - 0.1 + rng.normal(0, 0.01, 500)
    mean, p = dm_test(a, b, lag=12)
    assert mean < 0 and p < 0.01


def test_qlike_zero_at_truth():
    rv = pd.Series([0.5, 1.0, 2.0])
    assert np.allclose(qlike(rv, rv), 0.0)


def test_dsr_equals_psr_when_one_trial():
    r = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 500))
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    psr = st.norm.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    assert math.isclose(deflated_sharpe(r, [sr]), psr, rel_tol=1e-9)


def test_dsr_falls_with_more_trials():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.002, 0.01, 500))
    few, many = list(rng.normal(0, 0.05, 2)), list(rng.normal(0, 0.05, 50))
    assert deflated_sharpe(r, many) < deflated_sharpe(r, few)


def test_pbo_dominant_low():
    rng = np.random.default_rng(5)
    R = pd.DataFrame(rng.normal(0, 0.01, (480, 6)))
    R[0] += 0.004
    assert pbo_cscv(R) < 0.1


def test_pbo_noise_mid():
    R = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (480, 6)))
    assert 0.2 < pbo_cscv(R) < 0.8
