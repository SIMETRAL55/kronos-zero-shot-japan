import json

import numpy as np

from remote.jobs import run, run_multi
from tests.conftest import COLS, window


def _npz(tmp_path, reg_df):
    """Six tiny windows in the build_h1_windows.py format."""
    dates = reg_df["timestamps"].iloc[:120]
    bars = np.stack([window(reg_df, s)[0].to_numpy(np.float32) for s in range(0, 60, 10)])
    p = tmp_path / "w.npz"
    np.savez_compressed(p, bars=bars, t_idx=np.arange(39, 99, 10), y_idx=np.arange(40, 100, 10),
                        dates=np.array(dates.dt.strftime("%Y-%m-%d %H:%M"), dtype=str),
                        cal=np.array(dates.dt.strftime("%Y-%m-%d %H:%M"), dtype=str))
    assert bars.shape[2] == len(COLS)
    return p


def test_run_multi_matches_parts_and_keeps_closes(tmp_path, reg_df, kronos):
    cfg = {"builder": "npz", "input": str(_npz(tmp_path, reg_df)), "start": 0, "stop": 6, "T": 1.0,
           "top_p": 1.0, "n": 2, "seed": 3, "keep": "close_h5_h12"}
    m = run_multi("npz", cfg, tmp_path / "multi", device="cpu", n_procs=2)
    z = np.load(tmp_path / "multi" / "paths.npz")
    assert z["paths"].shape == (6, 2, 2) and list(z["keys"]) == [str(k) for k in range(6)]
    assert m["n_windows"] == 6 and m["gpus"] == 2
    # the second part alone, with its own seed offset, reproduces rows 3..5 exactly
    run("npz", dict(cfg, start=3, stop=6, seed=3 + 3), tmp_path / "solo", "cpu")
    solo = np.load(tmp_path / "solo" / "paths.npz")["paths"]
    assert np.array_equal(z["paths"][3:], solo)
    assert json.loads((tmp_path / "multi" / "manifest.json").read_text())["output_sha256"] == m["output_sha256"]
