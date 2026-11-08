"""Kronos forecasts that keep every sample path (upstream averages them away, #395)."""
import numpy as np
import pandas as pd
import torch

from kx.kronos_env import make_predictor

COLS = ["open", "high", "low", "close", "volume", "amount"]
PRED_LEN = 12


def forecast_paths(tokenizer, model, bars: list[pd.DataFrame], x_dates: list[pd.DatetimeIndex],
                   y_dates: list[pd.DatetimeIndex], *, T: float, top_p: float, n: int, seed: int,
                   device: str = "cpu", batch_rows: int = 2048) -> np.ndarray:
    """Return paths of shape (len(bars), n, 12, 6), columns COLS.

    Each series is repeated n times and sent through upstream predict_batch with sample_count=1, so every
    row is one independent path; torch.manual_seed(seed + chunk) makes each chunk reproducible.
    """
    pred = make_predictor(tokenizer, model, device)
    rows = [(b[COLS], pd.Series(x), pd.Series(y)) for b, x, y in zip(bars, x_dates, y_dates) for _ in range(n)]
    out = []
    for chunk, start in enumerate(range(0, len(rows), batch_rows)):
        part = rows[start:start + batch_rows]
        torch.manual_seed(seed + chunk)
        with torch.inference_mode():
            dfs = pred.predict_batch([r[0] for r in part], [r[1] for r in part], [r[2] for r in part],
                                     pred_len=PRED_LEN, T=T, top_k=0, top_p=top_p, sample_count=1, verbose=False)
        out.extend(d[COLS].to_numpy(np.float32) for d in dfs)
    return np.stack(out).reshape(len(bars), n, PRED_LEN, len(COLS))
