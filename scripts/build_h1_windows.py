"""Build H1 dev windows (PREREG §1-§2, Amendment 1) from the cached J-Quants Free data.

Writes (gitignored) data/h1/windows.npz -- bars + date indices only, no issue codes -- for upload to a private
Kaggle dataset, and data/h1/index.parquet (key -> date, code, last_close) which stays on this machine.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kx.jquants import DATA as JQ, DEV_END, panel, trading_days  # noqa: E402
from kx.universe import universe  # noqa: E402

FIELDS = ["AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "Va"]  # -> open, high, low, close, volume, amount

bars = pd.concat([pd.read_parquet(f) for f in sorted((JQ / "bars").glob("*.parquet"))], ignore_index=True)
P = {f: panel(bars, f).astype(float) for f in FIELDS}
dates = P["AdjC"].index
cal = trading_days(pd.concat([pd.read_parquet(f) for f in JQ.glob("calendar_*.parquet")]).drop_duplicates("Date"))
last = dates.searchsorted(pd.Timestamp(DEV_END), side="right") - 1
out_bars, t_idx, y_idx, rows = [], [], [], []
skipped_gaps = 0
for i, t in enumerate(dates):
    mfile = JQ / "master" / f"{t:%Y-%m-%d}.parquet"
    if i < 39 or i + 12 > last or not mfile.exists():
        continue
    codes = universe(pd.read_parquet(mfile), P["AdjC"], P["Va"], t)
    block = np.stack([P[f].iloc[i - 39:i + 1].reindex(columns=codes).to_numpy() for f in FIELDS], axis=-1)
    ok = ~np.isnan(block).any(axis=(0, 2))  # a missing bar inside the 40-bar window excludes the name (no ffill)
    skipped_gaps += int((~ok).sum())
    sel = np.array(codes)[ok]
    out_bars.append(np.transpose(block[:, ok, :], (1, 0, 2)).astype(np.float32))
    t_idx += [i] * len(sel)
    y_idx += [int(cal.searchsorted(t, side="right"))] * len(sel)
    rows += [(t, c, float(P["AdjC"].iloc[i][c])) for c in sel]

idx = pd.DataFrame(rows, columns=["date", "code", "last_close"])
idx.index.name = "key"
(ROOT / "data" / "h1").mkdir(parents=True, exist_ok=True)
idx.to_parquet(ROOT / "data" / "h1" / "index.parquet")
np.savez_compressed(ROOT / "data" / "h1" / "windows.npz", bars=np.concatenate(out_bars), t_idx=np.array(t_idx),
                    y_idx=np.array(y_idx), dates=np.array(dates.strftime("%Y-%m-%d"), dtype=str),
                    cal=np.array(cal.strftime("%Y-%m-%d"), dtype=str))
print(f"{len(idx)} windows, {idx['date'].nunique()} dates ({idx['date'].min():%Y-%m-%d}..{idx['date'].max():%Y-%m-%d}),"
      f" median {int(idx.groupby('date').size().median())} names/date, {skipped_gaps} name-dates dropped for gaps")
