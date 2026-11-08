import numpy as np
import pandas as pd

from kx.signals import (baseline_signals, ewma_var, forward_returns, har_coefficients, har_design, kronos_vol,
                        rank_signals, realized_var12, vol_baselines)


def _prices(days=300, names=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, names)), 0)), idx,
                        [f"c{i}" for i in range(names)])


def test_b1_zero_on_constant():
    c = pd.DataFrame(100.0, pd.bdate_range("2024-01-01", periods=60), ["a"])
    assert baseline_signals(c)["B1"].iloc[-1, 0] == 0.0


def test_b3_uses_t21_and_t252():
    c = _prices()
    t = 280
    assert np.isclose(baseline_signals(c)["B3"].iloc[t, 0], c.iloc[t - 21, 0] / c.iloc[t - 252, 0] - 1)


def test_forward_returns_window_end():
    c = _prices()
    o = c * 0.999
    end = c.index[250]
    fr = forward_returns(o, c, 12, str(end.date()))
    assert fr.index.max() == c.index[238]
    assert np.isclose(fr.iloc[0, 0], c.iloc[12, 0] / o.iloc[1, 0] - 1)


def test_signals_ignore_future_rows():
    c = _prices()
    c2 = c.copy()
    c2.iloc[201:] *= 1.5
    a, b = baseline_signals(c), baseline_signals(c2)
    for k in a:
        pd.testing.assert_frame_equal(a[k].iloc[:201], b[k].iloc[:201])


def test_k2_in_unit_interval():
    rng = np.random.default_rng(1)
    paths = 100 * np.exp(rng.normal(0, 0.02, (4, 50, 12, 6)))
    out = rank_signals(paths, np.full(4, 100.0), h=12)
    assert ((out["K2"] >= 0) & (out["K2"] <= 1)).all() and set(out) == {"K1", "K2", "K3"}


def test_v1_v2_nonnegative():
    rng = np.random.default_rng(2)
    paths = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (100, 12, 6)), 1))
    v = kronos_vol(paths, 100.0)
    assert v["V1"] >= 0 and v["V2"] >= 0


def test_ewma_toy_value():
    assert np.isclose(ewma_var(pd.Series([0.01, -0.02, 0.03])).iloc[-1], 1.6492e-4)


def test_har_recovers_coefficients():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 2000))
    X = har_design(r).dropna()
    y = 1e-5 + 0.3 * X["d"] + 0.4 * X["w"] + 0.2 * X["m"] + rng.normal(0, 1e-6, len(X))
    assert np.allclose(har_coefficients(X, y)[1:], [0.3, 0.4, 0.2], atol=0.05)


def test_realized_var12_alignment():
    c = _prices(days=40, names=1)["c0"]
    r = np.log(c).diff()
    assert np.isclose(realized_var12(c).iloc[5], (r.iloc[6:18] ** 2).sum())


def test_vol_baselines_positive_and_causal():
    c = _prices(days=800, names=1, seed=4)["c0"]
    first = str(c.index[740].date())
    a = vol_baselines(c, first)
    c2 = c.copy()
    c2.iloc[771:] *= 1.3
    b = vol_baselines(c2, first)
    assert (a > 0).all().all() and list(a.columns) == ["EWMA", "GARCH", "HAR"]
    pd.testing.assert_frame_equal(a.loc[:c.index[770]], b.loc[:c.index[770]])
