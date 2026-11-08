"""P4 drift-bias record (PREREG §6): Kronos-small K1 (S1) on 30 random walks, with and without drift. CPU."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kx.forecast import forecast_paths  # noqa: E402
from kx.kronos_env import load_kronos  # noqa: E402
from kx.signals import rank_signals  # noqa: E402
from kx.synthetic import make_random_walks  # noqa: E402


def run(mu: float, sigma: float = 0.015, n_names: int = 30, seed: int = 0) -> dict:
    walks = make_random_walks(n_names, 300, np.full(n_names, mu), np.full(n_names, sigma), seed)
    tok, model, _ = load_kronos()
    bars, xd, yd, last = [], [], [], []
    for df in walks.values():
        x = df.iloc[220:260]
        bars.append(x.reset_index(drop=True))
        xd.append(pd.DatetimeIndex(x.index))
        yd.append(pd.DatetimeIndex(df.index[260:272]))
        last.append(x["close"].iloc[-1])
    paths = forecast_paths(tok, model, bars, xd, yd, T=0.6, top_p=0.9, n=10, seed=seed)
    k1 = rank_signals(paths, np.array(last), h=12)["K1"]
    return {"mu_per_day": mu, "sigma": sigma, "n_names": n_names,
            "true_mean_log_return_12d": 12 * mu,
            "true_mean_simple_return_12d": float(np.exp(12 * mu + 6 * sigma ** 2) - 1),
            "kronos_mean_pred_return_12d": float(k1.mean()),
            "share_predicted_down": float((k1 < 0).mean())}


if __name__ == "__main__":
    print(json.dumps([run(0.001), run(0.0)], indent=1))
