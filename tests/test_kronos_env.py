import numpy as np
import pandas as pd
import torch

from kx.kronos_env import make_predictor
from tests.conftest import window


def test_eval_mode(kronos):
    tok, model, _ = kronos
    assert not any(m.training for m in [*tok.modules(), *model.modules()])


def test_same_seed_same_forecast(kronos, reg_df):
    tok, model, _ = kronos
    pred = make_predictor(tok, model)
    x, x_ts, y_ts = window(reg_df, 0)
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        outs.append(pred.predict(x, pd.Series(x_ts), pd.Series(y_ts), pred_len=12, T=1.0, top_p=1.0,
                                 sample_count=1, verbose=False).values)
    assert np.array_equal(outs[0], outs[1])
