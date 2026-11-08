"""P4 (PREREG §6): K1 rank IC on synthetic random walks must be within ±2 SE of 0."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kx.signals import rank_signals  # noqa: E402
from kx.stats import daily_rank_ic, nw_tstat  # noqa: E402

out = ROOT / "runs" / "p4" / "out"
z, m = np.load(out / "paths.npz"), json.loads((out / "manifest.json").read_text())
assert hashlib.sha256(np.ascontiguousarray(z["paths"]).tobytes()).hexdigest() == m["output_sha256"]
dates = pd.to_datetime(z["dates"])
pos = {d: i for i, d in enumerate(dates)}
keys = [k.split("|") for k in z["keys"]]
last_close = z["inputs"][:, -1, 3]
k1 = rank_signals(z["paths"], last_close, h=12)["K1"]
codes = list(z["codes"])
fwd = {(d, c): z["close"][pos[pd.Timestamp(d)] + 12, codes.index(c)] / z["open"][pos[pd.Timestamp(d)] + 1, codes.index(c)] - 1
       for d, c in keys}
sig = pd.Series(k1, pd.MultiIndex.from_tuples([tuple(k) for k in keys])).unstack()
ret = pd.Series(fwd).unstack().reindex_like(sig)
ic = daily_rank_ic(sig, ret, min_names=300)
mean, t = nw_tstat(ic, lag=2)  # forecast dates 6 bars apart, 12-bar horizon: adjacent windows overlap
res = {"n_dates": len(ic), "n_names": sig.shape[1], "mean_ic": mean, "nw_t": t, "pass_within_2se": abs(t) <= 2,
       "device": m["device_name"], "seconds": m["seconds"], "output_sha256": m["output_sha256"]}
(ROOT / "runs" / "p4_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
