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
    base = [(b[COLS], pd.Series(x), pd.Series(y)) for b, x, y in zip(bars, x_dates, y_dates)]  # once per series
    total, out = len(base) * n, []
    for chunk, start in enumerate(range(0, total, batch_rows)):
        ids = [i // n for i in range(start, min(start + batch_rows, total))]  # series-major, path-minor rows
        torch.manual_seed(seed + chunk)
        with torch.inference_mode():
            dfs = pred.predict_batch([base[i][0] for i in ids], [base[i][1] for i in ids], [base[i][2] for i in ids],
                                     pred_len=PRED_LEN, T=T, top_k=0, top_p=top_p, sample_count=1, verbose=False)
        out.extend(d[COLS].to_numpy(np.float32) for d in dfs)
    return np.stack(out).reshape(len(bars), n, PRED_LEN, len(COLS))
