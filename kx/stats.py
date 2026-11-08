"""Evaluation statistics (PREREG §4, §5). DataFrames are date x code."""
import itertools
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

EULER = 0.5772156649


def daily_rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 300) -> pd.Series:
    """Spearman IC per date across names; dates with fewer than min_names valid pairs are dropped."""
    s, f = signal.align(fwd, join="inner")
    out = {}
    for d in s.index:
        a, b = s.loc[d], f.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= min_names:
            out[d] = np.corrcoef(a[m].rank(), b[m].rank())[0, 1]
    return pd.Series(out, dtype=float)


def nw_tstat(x: pd.Series, lag: int) -> tuple[float, float]:
    """Mean of x and its Newey-West t (HAC, maxlags=lag)."""
    x = pd.Series(x).dropna().to_numpy(float)
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(res.params[0]), float(res.tvalues[0])


def rank_normal(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise ranks mapped to standard-normal scores; NaN stays NaN."""
    r = df.rank(axis=1)
    return pd.DataFrame(st.norm.ppf((r - 0.5).div(df.notna().sum(axis=1), axis=0)), df.index, df.columns)


def fama_macbeth(y: pd.DataFrame, X: dict[str, pd.DataFrame], lag: int) -> pd.DataFrame:
    """Daily OLS of rank-normal y on rank-normal regressors (missing regressor -> 0, the median score)."""
    yn = rank_normal(y)
    xn = {k: rank_normal(v.reindex_like(y)).fillna(0.0) for k, v in X.items()}
    coefs = {}
    for d in yn.index:
        yd = yn.loc[d].dropna()
        if len(yd) < 30:
            continue
        A = np.column_stack([np.ones(len(yd))] + [xn[k].loc[d, yd.index].to_numpy() for k in X])
        coefs[d] = np.linalg.lstsq(A, yd.to_numpy(), rcond=None)[0][1:]
    C = pd.DataFrame(coefs, index=list(X)).T
    return pd.DataFrame({k: nw_tstat(C[k], lag) for k in X}, index=["mean", "t"]).T


def dm_test(loss_a: pd.Series, loss_b: pd.Series, lag: int) -> tuple[float, float]:
    """Diebold-Mariano: mean(loss_a - loss_b) and the one-sided p-value for 'a has lower loss'."""
    d = (pd.Series(loss_a) - pd.Series(loss_b)).dropna()
    mean, t = nw_tstat(d, lag)
    return mean, float(st.norm.cdf(t))


def qlike(rv: pd.Series, f: pd.Series) -> pd.Series:
    q = rv / f
    return q - np.log(q) - 1


def deflated_sharpe(r: pd.Series, sr_trials: list[float]) -> float:
    """Bailey & Lopez de Prado (2014) DSR with per-period Sharpe ratios; N = len(sr_trials)."""
    r = pd.Series(r).dropna()
    T, sr = len(r), r.mean() / r.std(ddof=1)
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    n = len(sr_trials)
    sr0 = 0.0
    if n > 1:
        sr0 = math.sqrt(np.var(sr_trials, ddof=1)) * (
            (1 - EULER) * st.norm.ppf(1 - 1 / n) + EULER * st.norm.ppf(1 - 1 / (n * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(st.norm.cdf(z))


def pbo_cscv(R: pd.DataFrame, s: int = 16) -> float:
    """Probability of backtest overfitting via CSCV (Bailey et al. 2017). R: days x variants."""
    X = R.dropna().to_numpy(float)
    blocks = np.array_split(np.arange(len(X)), s)
    nv = X.shape[1]
    logits = []
    for comb in itertools.combinations(range(s), s // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(s) if i not in comb])
        sr_is = X[is_idx].mean(0) / X[is_idx].std(0, ddof=1)
        sr_oos = X[oos_idx].mean(0) / X[oos_idx].std(0, ddof=1)
        best = int(np.argmax(sr_is))
        omega = ((sr_oos < sr_oos[best]).sum() + 1) / (nv + 1)
        logits.append(math.log(omega / (1 - omega)))
    return float(np.mean(np.array(logits) <= 0))
