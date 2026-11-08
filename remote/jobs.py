"""Forecast jobs run on a remote GPU (or locally on CPU) by remote/run.py. Output: <out>/paths.npz + manifest.json."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kx.forecast import COLS, forecast_paths
from kx.kronos_env import VENDOR, load_kronos, make_predictor
from kx.synthetic import make_random_walks


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_p4(cfg: dict):
    """P4: random walks with no predictability; forecast dates every 6 bars."""
    n, days = cfg["n_names"], cfg["n_days"]
    walks = make_random_walks(n, days, np.zeros(n), np.linspace(0.01, 0.03, n), seed=cfg["data_seed"])
    positions = [59 + 6 * k for k in range(cfg["n_dates"])]
    bars, xd, yd, keys = [], [], [], []
    for code, df in walks.items():
        for p in positions:
            x = df.iloc[p - 39:p + 1]
            bars.append(x[COLS].reset_index(drop=True))
            xd.append(pd.DatetimeIndex(x.index))
            yd.append(pd.DatetimeIndex(df.index[p + 1:p + 13]))
            keys.append(f"{df.index[p]:%Y-%m-%d}|{code}")
    extras = {"open": np.stack([w["open"].to_numpy() for w in walks.values()], 1),
              "close": np.stack([w["close"].to_numpy() for w in walks.values()], 1),
              "dates": np.array(walks["s0000"].index.strftime("%Y-%m-%d"), dtype=str),
              "codes": np.array(list(walks), dtype=str)}
    return bars, xd, yd, keys, extras


def build_parity(cfg: dict):
    df = pd.read_csv(VENDOR / "tests/data/regression_input.csv", parse_dates=["timestamps"])
    bars, xd, yd, keys = [], [], [], []
    for s in (0, 50, 100):
        x = df.iloc[s:s + 40]
        bars.append(x[COLS].reset_index(drop=True))
        xd.append(pd.DatetimeIndex(x["timestamps"]))
        yd.append(pd.DatetimeIndex(df["timestamps"].iloc[s + 40:s + 52]))
        keys.append(str(s))
    return bars, xd, yd, keys, {}


def build_series(cfg: dict):
    """One index series (CSV text: date,open,high,low,close). Forecast dates t >= first whose t+12 <= last_outcome."""
    import io
    df = pd.read_csv(io.StringIO(cfg["csv"]), parse_dates=["date"], index_col="date").sort_index()
    df["volume"], df["amount"] = 0.0, 0.0  # indices: OHLC only (PREREG §1)
    last = df.index.searchsorted(pd.Timestamp(cfg["last_outcome"]), side="right") - 1
    bars, xd, yd, keys = [], [], [], []
    for p in range(39, last - 11):
        t = df.index[p]
        if t < pd.Timestamp(cfg["first"]):
            continue
        x = df.iloc[p - 39:p + 1]
        bars.append(x[COLS].reset_index(drop=True))
        xd.append(pd.DatetimeIndex(x.index))
        yd.append(pd.DatetimeIndex(df.index[p + 1:p + 13]))
        keys.append(f"{t:%Y-%m-%d}|{cfg['name']}")
    return bars, xd, yd, keys, {}


def build_npz(cfg: dict):
    """Windows from an uploaded npz (scripts/build_h1_windows.py); rows [start, stop) only. Keys are row numbers."""
    z = np.load(cfg["input"])
    dates, cal = pd.DatetimeIndex(z["dates"]), pd.DatetimeIndex(z["cal"])
    lo, hi = cfg["start"], cfg["stop"]
    bars, xd, yd = [], [], []
    for b, ti, yi in zip(z["bars"][lo:hi], z["t_idx"][lo:hi], z["y_idx"][lo:hi]):
        bars.append(pd.DataFrame(b, columns=COLS))
        xd.append(dates[ti - 39:ti + 1])
        yd.append(cal[yi:yi + 12])
    return bars, xd, yd, [str(k) for k in range(lo, hi)], {}


BUILDERS = {"p4": build_p4, "parity": build_parity, "series": build_series, "npz": build_npz}
KEEP = {"close": lambda p: p[..., 3], "close_h5_h12": lambda p: p[:, :, [4, 11], 3]}  # cut outputs to what H1 uses


def run_multi(job: str, cfg: dict, out: Path) -> dict:
    """Split rows [start, stop) across all visible GPUs (one process each), then merge into out/paths.npz."""
    import os
    import subprocess
    import sys
    n = max(torch.cuda.device_count(), 1)
    edges = np.linspace(cfg["start"], cfg["stop"], n + 1).astype(int)
    procs = []
    for g in range(n):
        part = dict(cfg, start=int(edges[g]), stop=int(edges[g + 1]))
        code = (f"import json,sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r}); "
                f"from pathlib import Path; from remote.jobs import run; "
                f"run({job!r}, json.loads({json.dumps(part)!r}), Path({str(out / f'part{g}')!r}), 'cuda')")
        procs.append(subprocess.Popen([sys.executable, "-c", code], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}))
    if any(p.wait() for p in procs):
        raise RuntimeError("a GPU worker failed")
    parts = [np.load(out / f"part{g}" / "paths.npz") for g in range(n)]
    paths = np.concatenate([p["paths"] for p in parts])
    np.savez_compressed(out / "paths.npz", paths=paths, keys=np.concatenate([p["keys"] for p in parts]))
    metas = [json.loads((out / f"part{g}" / "manifest.json").read_text()) for g in range(n)]
    manifest = {**metas[0], "cfg": cfg, "gpus": n, "output_sha256": _sha(paths), "n_windows": int(len(paths)),
                "seconds": max(m["seconds"] for m in metas), "input_sha256": [m["input_sha256"] for m in metas]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def run(job: str, cfg: dict, out: Path, device: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bars, xd, yd, keys, extras = BUILDERS[cfg.get("builder", job)](cfg)
    tok, model, meta = load_kronos(device=device)
    inputs = np.stack([b.to_numpy(np.float32) for b in bars])
    t0 = time.time()
    if job == "parity":  # greedy decoding (top_k=1) so CPU and GPU can be compared despite different RNG streams
        torch.manual_seed(0)
        dfs = make_predictor(tok, model, device).predict_batch(bars, [pd.Series(x) for x in xd], [pd.Series(y) for y in yd],
                                                               pred_len=12, T=1.0, top_k=1, top_p=1.0, sample_count=1,
                                                               verbose=False)
        paths = np.stack([d[COLS].to_numpy(np.float32) for d in dfs])[:, None]
    else:
        paths = forecast_paths(tok, model, bars, xd, yd, T=cfg["T"], top_p=cfg["top_p"], n=cfg["n"],
                               seed=cfg["seed"], device=device, batch_rows=cfg.get("batch_rows", 2048))
    secs = time.time() - t0
    if cfg.get("keep"):
        paths = KEEP[cfg["keep"]](paths)
        inputs = inputs[:, -1:, :]  # the last bar is enough downstream; full inputs live in the input npz
    np.savez_compressed(out / "paths.npz", paths=paths, keys=np.array(keys), inputs=inputs, **extras)
    manifest = {"job": job, "cfg": cfg, **meta, "device": device,
                "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
                "torch": torch.__version__, "input_sha256": _sha(inputs), "output_sha256": _sha(paths),
                "n_windows": len(bars), "seconds": round(secs, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
