"""H2 evaluation (PREREG §5) on one run's Nikkei 225 forecasts.  uv run python scripts/eval_h2.py h2dev"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kx.backtest import summary, vol_target_backtest  # noqa: E402
from kx.signals import kronos_vol, realized_var12, vol_baselines  # noqa: E402
from kx.stats import dm_test, qlike  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "h2dev"
out = ROOT / "runs" / run / "out"
z, m = np.load(out / "paths.npz"), json.loads((out / "manifest.json").read_text())
assert hashlib.sha256(np.ascontiguousarray(z["paths"]).tobytes()).hexdigest() == m["output_sha256"]
close = pd.read_parquet(ROOT / "data" / "nikkei225_yahoo.parquet")["close"]
dates = pd.to_datetime([k.split("|")[0] for k in z["keys"]])
assert dates.max() <= close.index[-1]

kv = pd.DataFrame([kronos_vol(z["paths"][i], close.loc[d]) for i, d in enumerate(dates)], index=dates)
base = vol_baselines(close, str(dates[0].date())).reindex(dates)
F = pd.concat([kv, base], axis=1)
rv = realized_var12(close).reindex(dates)
assert rv.notna().all(), "a forecast date's 12-day outcome is missing"

loss = {k: qlike(rv, F[k]) for k in F}
dm = {k: dict(zip(["mean_diff", "p_one_sided"], dm_test(loss[k], loss["HAR"], lag=12))) for k in ["V1", "V2"]}
order = sorted(dm, key=lambda k: dm[k]["p_one_sided"])  # Holm over V1, V2
holm, running = {}, 0.0
for i, k in enumerate(order):
    running = max(running, min(1.0, (2 - i) * dm[k]["p_one_sided"]))
    holm[k] = running
for k in dm:
    dm[k]["p_holm"] = holm[k]
best = min(["V1", "V2"], key=lambda k: loss[k].mean())


def mz(f: pd.Series) -> dict:
    res = sm.OLS(np.log(rv), sm.add_constant(np.log(f))).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return {"a": float(res.params.iloc[0]), "b": float(res.params.iloc[1]), "b_se": float(res.bse.iloc[1])}


bt_close = close.loc[dates[0]:dates[-1]]
bts = {"kronos_" + best: vol_target_backtest(bt_close, F[best]),
       "HAR": vol_target_backtest(bt_close, F["HAR"]),
       # a tiny forecast variance pins the position at the cap, i.e. always exactly 1 contract
       "constant_1": vol_target_backtest(bt_close, pd.Series(1e-12, bt_close.index), max_contracts=1)}
sizing = {k: summary(v) for k, v in bts.items()}
mdd_ok = abs(sizing["kronos_" + best]["max_drawdown"]) <= 1.2 * abs(sizing["HAR"]["max_drawdown"])
split = {name: {k: float(loss[k][mask].mean()) for k in F}
         for name, mask in [("before_2024-11-05", dates < "2024-11-05"), ("from_2024-11-05", dates >= "2024-11-05")]}
res = {"run": run, "n_dates": len(dates), "first": str(dates[0].date()), "last": str(dates[-1].date()),
       "mean_qlike": {k: float(v.mean()) for k, v in loss.items()}, "dm_vs_HAR": dm, "best_kronos": best,
       "mincer_zarnowitz": {k: mz(F[k]) for k in F}, "sizing": sizing, "mdd_no_harm": bool(mdd_ok),
       "qlike_split": split, "manifest": {k: m[k] for k in ["output_sha256", "device_name", "seconds", "cfg"]
                                          if k != "cfg"} | {"cfg": {k: v for k, v in m["cfg"].items() if k != "csv"}},
       "pass": bool(dm[best]["p_holm"] < 0.05 and dm[best]["mean_diff"] < 0 and mdd_ok)}
(ROOT / "runs" / f"{run}_h2_result.json").write_text(json.dumps(res, indent=1, default=float))


def col(s: pd.Series) -> list:
    """Per-date column for the dashboard (DESIGN.md D1): aligned to `dates`, NaN -> null."""
    return [None if not np.isfinite(x) else float(x) for x in s.reindex(dates).to_numpy(float)]


series = {"run": run, "dates": [str(d.date()) for d in dates], "rv12": col(rv),
          "forecast": {k: col(F[k]) for k in F}, "qlike": {k: col(loss[k]) for k in loss},
          "sizing": {k: {"contracts": col(b["contracts"]), "net_yen": col(b["net"]), "equity_yen": col(b["equity"])}
                     for k, b in bts.items()}}
(ROOT / "runs" / f"{run}_series.json").write_text(json.dumps(series))
print(json.dumps(res, indent=1, default=float))
