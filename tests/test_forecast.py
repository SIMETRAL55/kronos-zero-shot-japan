import numpy as np
import pandas as pd
import torch

from kx.forecast import COLS, forecast_paths
from kx.kronos_env import make_predictor
from kx.tsecal import next_trading_days
from tests.conftest import window


def test_next_trading_days_golden_week():
    hol = pd.to_datetime(["2025-04-29", "2025-05-05", "2025-05-06"])
    cal = pd.bdate_range("2025-04-21", "2025-05-16").difference(hol)
    got = next_trading_days(cal, pd.Timestamp("2025-04-30"), 3)
    assert list(got) == list(pd.to_datetime(["2025-05-01", "2025-05-02", "2025-05-07"]))


def _inputs(df, starts):
    w = [window(df, s) for s in starts]
    return [a for a, _, _ in w], [b for _, b, _ in w], [c for _, _, c in w]


def test_shape(kronos, reg_df):
    tok, model, _ = kronos
    out = forecast_paths(tok, model, *_inputs(reg_df, [0, 50, 100]), T=1.0, top_p=1.0, n=4, seed=0)
    assert out.shape == (3, 4, 12, 6)


def test_seed_determinism(kronos, reg_df):
    tok, model, _ = kronos
    a = forecast_paths(tok, model, *_inputs(reg_df, [0, 50]), T=1.0, top_p=1.0, n=3, seed=5)
    b = forecast_paths(tok, model, *_inputs(reg_df, [0, 50]), T=1.0, top_p=1.0, n=3, seed=5)
    assert np.array_equal(a, b)


def test_paths_differ(kronos, reg_df):
    tok, model, _ = kronos
    out = forecast_paths(tok, model, *_inputs(reg_df, [0]), T=1.0, top_p=1.0, n=5, seed=1)
    assert not all(np.array_equal(out[0, 0], out[0, i]) for i in range(1, 5))


def test_n1_matches_upstream(kronos, reg_df):
    tok, model, _ = kronos
    x, x_ts, y_ts = window(reg_df, 0)
    ours = forecast_paths(tok, model, [x], [x_ts], [y_ts], T=1.0, top_p=1.0, n=1, seed=7)
    torch.manual_seed(7)
    up = make_predictor(tok, model).predict_batch([x], [pd.Series(x_ts)], [pd.Series(y_ts)], pred_len=12, T=1.0,
                                                  top_p=1.0, sample_count=1, verbose=False)
    assert np.array_equal(ours[0, 0], up[0][COLS].to_numpy(np.float32))
