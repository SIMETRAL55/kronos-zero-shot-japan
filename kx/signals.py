"""Kronos signals, price-based baselines, forward returns and volatility baselines (PREREG §4, §5)."""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

CLOSE = 3
VAR_FLOOR = 1e-10  # ponytail: floor so QLIKE stays defined if OLS-HAR goes non-positive; never binds at real vol levels


def rank_signals(paths: np.ndarray, last_close: np.ndarray, h: int) -> dict[str, np.ndarray]:
    """paths (B, n, 12, 6). K1 = mean path's return to h; K2 = P(up); K3 = mean/std of path returns."""
    c = paths[:, :, h - 1, CLOSE]
    ret = c / last_close[:, None] - 1
    sd = ret.std(1, ddof=1) if ret.shape[1] > 1 else np.zeros(len(ret))
    return {"K1": c.mean(1) / last_close - 1,
            "K2": (c > last_close[:, None]).mean(1),
            "K3": np.divide(ret.mean(1), sd, out=np.zeros(len(ret)), where=sd > 0)}


def baseline_signals(adj_close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """B1-B4 from rows up to t only. Rows are consecutive trading days (missing bars stay NaN)."""
    c = adj_close
    return {"B1": -(c / c.rolling(40).mean() - 1),
            "B2": -(c / c.shift(21) - 1),
            "B3": c.shift(21) / c.shift(252) - 1,
            "B4": -(c / c.shift(5) - 1)}


def forward_returns(adj_open: pd.DataFrame, adj_close: pd.DataFrame, h: int, window_end: str) -> pd.DataFrame:
    """r(t,h) = close(t+h) / open(t+1) - 1, keeping only t whose t+h is on or before window_end."""
    fr = adj_close.shift(-h) / adj_open.shift(-1) - 1
    last = adj_close.index.searchsorted(pd.Timestamp(window_end), side="right") - 1
    return fr.iloc[:max(last - h + 1, 0)]


def kronos_vol(paths: np.ndarray, last_close: float) -> dict[str, float]:
    """paths (n, 12, 6). V1 = mean over paths of summed squared log returns; V2 = variance of 12-day log return."""
    c = np.concatenate([np.full((len(paths), 1), last_close), paths[:, :, CLOSE]], axis=1)
    lr = np.diff(np.log(c), axis=1)
    return {"V1": float((lr ** 2).sum(1).mean()), "V2": float(np.log(c[:, -1] / last_close).var(ddof=1))}


def realized_var12(close: pd.Series) -> pd.Series:
    r2 = np.log(close).diff() ** 2
    return r2.rolling(12).sum().shift(-12)


def ewma_var(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """Variance estimate after observing r_t (the forecast for t+1); starts at r_0 squared."""
    out = np.empty(len(r))
    v = r.iloc[0] ** 2
    for i, x in enumerate(r.to_numpy()):
        v = x * x if i == 0 else lam * v + (1 - lam) * x * x
        out[i] = v
    return pd.Series(out, r.index)


def har_design(r: pd.Series) -> pd.DataFrame:
    r2 = r ** 2
    return pd.DataFrame({"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()})


def har_coefficients(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    return np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]


def vol_baselines(close: pd.Series, first_forecast: str) -> pd.DataFrame:
    """12-day variance forecasts at each t >= first_forecast, using data up to t; GARCH/HAR refit each month start."""
    r = np.log(close).diff().dropna()
    dates = r.index[r.index >= pd.Timestamp(first_forecast)]
    ewma = 12 * ewma_var(r)
    X, y = har_design(r), realized_var12(close).reindex(r.index) / 12
    out, params, beta, month = {}, None, None, None
    for t in dates:
        if (t.year, t.month) != month:
            month = (t.year, t.month)
            past = r[r.index < t]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = arch_model(100 * past, vol="GARCH", p=1, q=1, dist="t").fit(disp="off").params
            # HAR targets must be fully observed before the refit date: s + 12 < position of t
            pos = r.index.get_loc(t)
            fit = pd.concat([X, y.rename("y")], axis=1).iloc[:max(pos - 12, 0)].dropna()
            beta = har_coefficients(fit[["d", "w", "m"]], fit["y"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = arch_model(100 * r.loc[:t], vol="GARCH", p=1, q=1, dist="t").fix(params)
            garch = g.forecast(horizon=12, reindex=False).variance.iloc[-1].sum() / 1e4
        har = 12 * float(beta @ np.r_[1.0, X.loc[t].to_numpy()])
        out[t] = (ewma.loc[t], garch, max(har, VAR_FLOOR))
    return pd.DataFrame.from_dict(out, orient="index", columns=["EWMA", "GARCH", "HAR"])
