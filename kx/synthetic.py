"""Synthetic OHLCV random walks for placebos (PREREG §6 P4)."""
import numpy as np
import pandas as pd


def make_random_walks(n_names: int, n_days: int, mu: np.ndarray, sigma: np.ndarray, seed: int,
                      start: str = "2020-01-01") -> dict[str, pd.DataFrame]:
    """Geometric random walks with drift mu and vol sigma per name (daily, log terms); no predictability."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    out = {}
    for i in range(n_names):
        lr = rng.normal(mu[i], sigma[i], n_days)
        close = 100 * np.exp(np.cumsum(lr))
        prev = np.r_[100.0, close[:-1]]
        open_ = prev * np.exp(rng.normal(0, sigma[i] / 4, n_days))
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma[i] / 2, n_days)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma[i] / 2, n_days)))
        vol = rng.lognormal(12, 0.5, n_days)
        out[f"s{i:04d}"] = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                                         "volume": vol, "amount": vol * close}, idx)
    return out
