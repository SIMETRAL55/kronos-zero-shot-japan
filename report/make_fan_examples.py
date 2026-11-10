"""Write report/fan_examples.json: 25 Kronos forecast fans for the Nikkei 225 (H2 dev run), for the interactive report.

Needs the local H2 outputs (runs/h2dev/out/paths.npz) and the Nikkei data (data/nikkei225_yahoo.parquet), which are not
in the repository. Run once after reproducing H2:  uv run python report/make_fan_examples.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ser = json.loads((ROOT / "runs/h2dev_series.json").read_text())
z = np.load(ROOT / "runs/h2dev/out/paths.npz")
close = pd.read_parquet(ROOT / "data/nikkei225_yahoo.parquet")["close"]
dates = pd.to_datetime(ser["dates"])


def ann(v):  # 12-day variance -> annualised volatility in %
    return None if v is None else round(float(np.sqrt(max(v, 0) / 12 * 245) * 100), 1)


# 24 dates spread over the window, plus 2 Aug 2024 (the Friday before the 5 Aug crash)
pick = sorted(set(np.linspace(0, len(dates) - 1, 24).astype(int))
              | {int(dates.get_indexer([pd.Timestamp("2024-08-02")])[0])})
pos = {d: i for i, d in enumerate(close.index)}
out = []
for i in pick:
    t, p = dates[i], pos[dates[i]]
    paths = z["paths"][i, :, :, 3]
    out.append({"date": str(t.date()),
                "hist_dates": [str(d.date()) for d in close.index[p - 39:p + 1]],
                "fut_dates": [str(d.date()) for d in close.index[p + 1:p + 13]],
                "hist": [round(float(x)) for x in close.iloc[p - 39:p + 1]],
                "real": [round(float(x)) for x in close.iloc[p + 1:p + 13]],
                "paths": [[round(float(x)) for x in row] for row in paths[:50]],
                "mean": [round(float(x)) for x in paths.mean(0)],
                "vol": {"kronos": ann(ser["forecast"]["V1"][i]), "har": ann(ser["forecast"]["HAR"][i]),
                        "real": ann(ser["rv12"][i])}})
(ROOT / "report/fan_examples.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"{len(out)} examples written")
