"""Build docs/index.html (the interactive report) from committed results only.

  uv run python report/build_report.py

Inputs: runs/h2dev_series.json, runs/h2dev_h2_result.json, runs/p4_result.json, runs/h1_costs.json, runs/h1_dev.json,
runs/h1_dev_series.json, runs/h1_p5.json, runs/h1_p5diag.json, report/fan_examples.json. No raw market data is embedded
except the 25 short Nikkei 225 windows in fan_examples.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kx.stats import daily_rank_ic, nw_tstat  # noqa: E402

R = ROOT / "runs"
load = lambda name: json.loads((R / name).read_text())  # noqa: E731
ser, res = load("h2dev_series.json"), load("h2dev_h2_result.json")


def ann(v):
    return None if v is None else round(float(np.sqrt(max(v, 0) / 12 * 245) * 100), 1)


def rnd(xs, k=5):
    return [None if x is None else round(x, k) for x in xs]


# P6, rebuilt exactly as tests/test_positive_control.py does it
rng = np.random.default_rng(0)
idx, cols = pd.bdate_range("2024-07-01", periods=460), [f"c{i}" for i in range(480)]
fwd = rng.normal(size=(460, 480))
sig = 0.0524 * fwd + np.sqrt(1 - 0.0524 ** 2) * rng.normal(size=(460, 480))
p6_mean, p6_t = nw_tstat(daily_rank_ic(pd.DataFrame(sig, idx, cols), pd.DataFrame(fwd, idx, cols)), lag=12)

p4, costs = load("p4_result.json"), load("h1_costs.json")
h1, h1s, p5, p5d = load("h1_dev.json"), load("h1_dev_series.json"), load("h1_p5.json"), load("h1_p5diag.json")
data = {
    "examples": json.loads((ROOT / "report/fan_examples.json").read_text()),
    "vol": {"dates": ser["dates"], "real": [ann(v) for v in ser["rv12"]],
            **{k: [ann(v) for v in ser["forecast"][k]] for k in ["V1", "HAR", "GARCH"]}},
    "qlike": {k: round(v, 3) for k, v in res["mean_qlike"].items()},
    "sizing": {k: {"equity_k": [round(v / 1000, 1) for v in s["equity_yen"]], **res["sizing"][k]}
               for k, s in ser["sizing"].items()},
    "checks": {"p4_mean_ic": p4["mean_ic"], "p4_se": abs(p4["mean_ic"] / p4["nw_t"]), "p6_mean_ic": p6_mean,
               "p6_se": abs(p6_mean / p6_t)},
    "costs": {"ar_median": costs["abdi_ranaldo"]["median_bp"], "tick1": costs["tick_floor"]["median_one_tick_bp"],
              "tick2": costs["tick_floor"]["median_two_tick_round_trip_bp"], "prereg": costs["prereg_assumption_bp"]},
    "h1": {"variants": {k: {f: v[f] for f in ["mean_ic", "ic_nw_t", "fm_t", "excess_sharpe_10bp", "dsr",
                                              "monthly_pos_share", "excess_ann_by_cost_bp"]}
                        for k, v in h1["variants"].items()},
           "pbo": h1["pbo_all_variants"], "b3_coverage": h1["b3_coverage"],
           "series": {"dates": h1s["dates"],
                      "variants": {k: {f: rnd(v[f]) for f in ["rank_ic", "portfolio_10bp", "benchmark"]}
                                   for k, v in h1s["variants"].items()}},
           "p5": {k: {"date_mean": v["mean_ic"], "date_se": v["nw_se"],
                      "stock_mean": p5d[k]["stock_shuffled_mean_ic"],
                      "stock_se": abs(p5d[k]["stock_shuffled_mean_ic"] / p5d[k]["stock_shuffled_nw_t"])}
                  for k, v in p5.items()}},
}
blob = json.dumps(data, separators=(",", ":"))
assert "</" not in blob
html = (ROOT / "report/template.html").read_text().replace("/*DATA*/", blob, 1)
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/index.html").write_text(html)
(ROOT / "docs/.nojekyll").write_text("")
print(f"docs/index.html written ({len(html):,} bytes)")
