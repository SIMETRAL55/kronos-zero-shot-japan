"""Push one H1 forecast job to Kaggle (PREREG §1: S1 = 10 averaged paths; S2 = 50 kept paths, split in 4 blocks).

  export KAGGLE_USERNAME=your-name
  uv run python scripts/push_h1_jobs.py s1      # ~4 h on 2x T4
  uv run python scripts/push_h1_jobs.py s2a     # ~4.5 h each; then s2b, s2c, s2d (Kaggle runs about 2 at a time)
  uv run python scripts/kaggle_run.py fetch h1s1   # when a job is COMPLETE; likewise h1s2a ... h1s2d

Needs data/h1/windows.npz (scripts/build_h1_windows.py) uploaded with `scripts/kaggle_dataset.py create`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_run import USER, push  # noqa: E402

if not USER:
    sys.exit("Set KAGGLE_USERNAME to your Kaggle username first.")
n = len(np.load(ROOT / "data" / "h1" / "windows.npz")["t_idx"])
edges = np.linspace(0, n, 5).astype(int)  # the four S2 blocks (170,613 windows gave 0/42653/85306/127959/170613)
base = {"builder": "npz", "input": "/kaggle/input/kronos-h1-dev/windows.npz", "batch_rows": 4096, "multi_gpu": True,
        "datasets": [f"{USER}/kronos-h1-dev"]}
jobs = {"s1": dict(base, start=0, stop=n, T=0.6, top_p=0.9, n=10, seed=0, keep="close")}
for i, b in enumerate("abcd"):
    jobs[f"s2{b}"] = dict(base, start=int(edges[i]), stop=int(edges[i + 1]), T=1.0, top_p=1.0, n=50, seed=1_000_000,
                          keep="close_h5_h12")
which = sys.argv[1]
push(f"h1{which}", jobs[which])
