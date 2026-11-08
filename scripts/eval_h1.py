"""H1 evaluation (PREREG §4, §10, Amendment 1). Run in order, committing between steps:

  uv run python scripts/eval_h1.py costs   # §10: round-trip cost check (Abdi-Ranaldo spread) -- no signals used
  uv run python scripts/eval_h1.py p5      # P5 shuffled-date placebo + minimum detectable IC -- no real IC computed
  uv run python scripts/eval_h1.py dev     # the six pre-registered variants
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kx.backtest import tranche_backtest  # noqa: E402
from kx.jquants import DATA as JQ, DEV_END, panel  # noqa: E402
from kx.signals import baseline_signals, forward_returns  # noqa: E402
from kx.stats import daily_rank_ic, deflated_sharpe, dm_test, fama_macbeth, nw_tstat, pbo_cscv  # noqa: E402

H = (5, 12)
IDX = pd.read_parquet(ROOT / "data" / "h1" / "index.parquet")


def panels() -> dict[str, pd.DataFrame]:
    bars = pd.concat([pd.read_parquet(f) for f in sorted((JQ / "bars").glob("*.parquet"))], ignore_index=True)
    return {f: panel(bars, f).astype(float) for f in ["AdjO", "AdjH", "AdjL", "AdjC", "UL"]}


def wide(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": IDX["date"], "code": IDX["code"], "v": values}).pivot(
        index="date", columns="code", values="v")


def load_paths(prefix: str) -> np.ndarray:
    zs = [np.load(p) for p in sorted((ROOT / "runs").glob(f"{prefix}*/out/paths.npz"))]
    keys, paths = np.concatenate([z["keys"].astype(int) for z in zs]), np.concatenate([z["paths"] for z in zs])
    paths = paths[np.argsort(keys)]
    assert len(paths) == len(IDX) and (np.sort(keys) == np.arange(len(IDX))).all(), "missing forecast rows"
    return paths


def kronos_signals() -> dict[str, pd.DataFrame]:
    s1, s2 = load_paths("h1s1"), load_paths("h1s2")  # S1 closes (B,10,12); S2 closes at h5,h12 (B,50,2)
    last = IDX["last_close"].to_numpy()
    out = {}
    for h in H:
        out[f"K1_h{h}"] = wide(s1[:, :, h - 1].mean(1) / last - 1)
        r = s2[:, :, 0 if h == 5 else 1] / last[:, None] - 1
        sd = r.std(1, ddof=1)
        out[f"K2_h{h}"] = wide((r > 0).mean(1))
        out[f"K3_h{h}"] = wide(np.divide(r.mean(1), sd, out=np.zeros(len(r)), where=sd > 0))
    return out


def costs(P) -> dict:
    """Abdi & Ranaldo (2017): s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})], eta = mid-range in logs. Per stock, dev window."""
    c, eta = np.log(P["AdjC"]), (np.log(P["AdjH"]) + np.log(P["AdjL"])) / 2
    prod = ((c - eta) * (c - eta.shift(-1))).loc[IDX["date"].min():DEV_END]
    s2 = 4 * prod[sorted(IDX["code"].unique())].mean()
    bp = np.sqrt(s2.clip(lower=0)) * 1e4
    return {"names": int(bp.notna().sum()), "median_round_trip_bp": float(bp.median()),
            "p25_bp": float(bp.quantile(0.25)), "p75_bp": float(bp.quantile(0.75)),
            "prereg_assumption_bp": 10.0, "amend_needed_gt_15bp": bool(bp.median() > 15)}


def p5(P, sig) -> dict:
    rng = np.random.default_rng(0)
    res = {}
    for name, s in sig.items():
        h = int(name.split("_h")[1])
        fwd = forward_returns(P["AdjO"], P["AdjC"], h, DEV_END).reindex(s.index)
        perm = pd.Index(rng.permutation(s.index))
        shuffled = pd.DataFrame(fwd.loc[perm].to_numpy(), s.index, fwd.columns)  # date t gets another date's returns
        ic = daily_rank_ic(s, shuffled)
        mean, t = nw_tstat(ic, lag=h)
        se = abs(mean / t) if t else float("nan")
        res[name] = {"mean_ic": mean, "nw_t": t, "pass_within_2se": abs(t) <= 2, "ic_sd": float(ic.std()),
                     "nw_se": se, "min_detectable_ic_t3": 3 * se, "n_days": len(ic)}
    return res


def dev(P, sig) -> dict:
    B = baseline_signals(P["AdjC"])
    blocked = (P["UL"] == 1) & (P["AdjO"] >= P["AdjH"])  # opened limit-up: cannot buy at the open
    rows, excess = {}, {}
    for name, s in sig.items():
        h = int(name.split("_h")[1])
        fwd = forward_returns(P["AdjO"], P["AdjC"], h, DEV_END).reindex(s.index)
        ic = daily_rank_ic(s, fwd)
        m, t = nw_tstat(ic, lag=h)
        fm = fama_macbeth(fwd, {"K": s, **{b: B[b].reindex(s.index) for b in B}}, lag=h)
        bts = {c: tranche_backtest(s, P["AdjO"], P["AdjC"], blocked, h, cost_bp=c) for c in (5, 10, 15, 25)}
        ex = bts[10]["excess"].loc[s.index.min():].iloc[1:]
        excess[name] = ex
        monthly = ic.groupby(ic.index.to_period("M")).mean()
        row = {"mean_ic": m, "ic_nw_t": t, "fm_t": float(fm.loc["K", "t"]), "fm_coef": float(fm.loc["K", "mean"]),
               "monthly_pos_share": float((monthly > 0).mean()), "n_days": len(ic),
               "excess_ann_by_cost_bp": {c: float(b["excess"].mean() * 245) for c, b in bts.items()},
               "excess_sharpe_10bp": float(ex.mean() / ex.std(ddof=1) * np.sqrt(245))}
        if name.startswith("K1"):
            lk = ((fwd - s) ** 2).mean(axis=1)
            l0 = (fwd ** 2).mean(axis=1)
            row["dm_vs_zero_forecast"] = dict(zip(["mean_diff", "p_one_sided"], dm_test(lk, l0, lag=h)))
        rows[name] = row
    R = pd.DataFrame(excess).dropna()
    srs = list((R.mean() / R.std(ddof=1)).values)
    pbo = pbo_cscv(R)
    for name in rows:
        rows[name]["dsr"] = deflated_sharpe(R[name], srs)
        rows[name]["pass"] = bool(rows[name]["fm_t"] >= 3 and rows[name]["dsr"] >= 0.95 and pbo <= 0.2
                                  and rows[name]["monthly_pos_share"] >= 0.6)
    b3 = B["B3"].reindex(sig["K1_h5"].index)[sig["K1_h5"].notna()]
    return {"variants": rows, "pbo_all_variants": pbo, "b3_coverage": float(b3.notna().sum().sum() /
                                                                            sig["K1_h5"].notna().sum().sum())}


if __name__ == "__main__":
    step = sys.argv[1]
    P = panels()
    res = costs(P) if step == "costs" else p5(P, kronos_signals()) if step == "p5" else dev(P, kronos_signals())
    (ROOT / "runs" / f"h1_{step}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
