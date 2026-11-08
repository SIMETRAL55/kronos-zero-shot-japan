import numpy as np
import pandas as pd
import pytest
import torch

from kx.kronos_env import VENDOR, load_kronos, make_predictor

COLS = ["open", "high", "low", "close", "volume", "amount"]


@pytest.fixture(scope="module")
def kronos():
    tok, model, meta = load_kronos()
    return tok, model, meta


def _window():
    df = pd.read_csv(VENDOR / "tests/data/regression_input.csv", parse_dates=["timestamps"])
    x = df.iloc[:40]
    y_ts = df["timestamps"].iloc[40:52].reset_index(drop=True)
    return x[COLS].reset_index(drop=True), x["timestamps"].reset_index(drop=True), y_ts


def test_eval_mode(kronos):
    tok, model, _ = kronos
    assert not any(m.training for m in [*tok.modules(), *model.modules()])


def test_same_seed_same_forecast(kronos):
    tok, model, _ = kronos
    pred = make_predictor(tok, model)
    x, x_ts, y_ts = _window()
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        outs.append(pred.predict(x, x_ts, y_ts, pred_len=12, T=1.0, top_p=1.0, sample_count=1, verbose=False).values)
    assert np.array_equal(outs[0], outs[1])
