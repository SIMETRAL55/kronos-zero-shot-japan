"""Push the H2 dev forecast job (Nikkei 225, S2: T=1.0, top_p=1.0, N=100) to Kaggle. PREREG §5."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
y = pd.read_parquet(ROOT / "data" / "nikkei225_yahoo.parquet")
assert y.index.max() <= pd.Timestamp("2026-04-30")
csv = y.rename_axis("date").reset_index().to_csv(index=False, float_format="%.4f")
cfg = {"builder": "series", "name": "nikkei225", "csv": csv, "first": "2024-07-01", "last_outcome": "2026-04-30",
       "T": 1.0, "top_p": 1.0, "n": 100, "seed": 0, "batch_rows": 4096}
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_run import push  # noqa: E402

push(sys.argv[1] if len(sys.argv) > 1 else "h2dev", cfg)
