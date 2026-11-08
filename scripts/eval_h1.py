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


TICK_TOPIX100 = [(1000, 0.1), (3000, 0.5), (10000, 1), (30000, 5), (100000, 10), (300000, 50), (np.inf, 100)]
TICK_OTHER = [(3000, 1), (5000, 5), (30000, 10), (50000, 50), (300000, 100), (500000, 500), (np.inf, 1000)]


def tick(price: float, topix100: bool) -> float:
    """TSE tick size (呼値の単位) in force through the dev window; TOPIX100 constituents use the finer table."""
    return next(t for cap, t in (TICK_TOPIX100 if topix100 else TICK_OTHER) if price <= cap)


def costs(P) -> dict:
    """§10 cost check, two ways (no signals involved).
    1. Abdi & Ranaldo (2017) spread: s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})], eta = log mid-range, per stock.
    2. Tick floor: one TSE tick / raw price per stock; a round trip costs at least ~1-2 ticks."""
    c, eta = np.log(P["AdjC"]), (np.log(P["AdjH"]) + np.log(P["AdjL"])) / 2
    prod = ((c - eta) * (c - eta.shift(-1))).loc[IDX["date"].min():DEV_END]
    codes = sorted(IDX["code"].unique())
    s2 = 4 * prod[codes].mean()
    ar = np.sqrt(s2.clip(lower=0)) * 1e4
    bars = pd.concat([pd.read_parquet(f) for f in sorted((JQ / "bars").glob("*.parquet"))], ignore_index=True)
    raw = panel(bars, "C").astype(float).loc[IDX["date"].min():DEV_END, codes].median()
    master = pd.read_parquet(JQ / "master" / f"{IDX['date'].max():%Y-%m-%d}.parquet").set_index("Code")["ScaleCat"]
    t100 = master.reindex(codes).isin(["TOPIX Core30", "TOPIX Large70"])
    tick_bp = pd.Series({k: tick(raw[k], bool(t100[k])) / raw[k] * 1e4 for k in codes if np.isfinite(raw[k])})
    return {"names": int(ar.notna().sum()),
            "abdi_ranaldo": {"median_bp": float(ar.median()), "p25_bp": float(ar.quantile(0.25)),
                             "p75_bp": float(ar.quantile(0.75)), "share_clipped_at_zero": float((s2 <= 0).mean())},
            "tick_floor": {"median_one_tick_bp": float(tick_bp.median()), "p75_one_tick_bp": float(tick_bp.quantile(0.75)),
                           "median_two_tick_round_trip_bp": float(2 * tick_bp.median())},
            "prereg_assumption_bp": 10.0}


def series_json(cols: dict[str, dict[str, pd.Series]]) -> dict:
    """Per-date columns for the dashboard (its DESIGN.md D1): one shared date axis, NaN/missing -> null."""
    dates = sorted(set().union(*[s.index for v in cols.values() for s in v.values()]))
    return {"dates": [str(pd.Timestamp(d).date()) for d in dates],
            "variants": {k: {f: [None if not np.isfinite(x) else float(x) for x in s.reindex(dates).to_numpy(float)]
                             for f, s in v.items()} for k, v in cols.items()}}


def p5(P, sig) -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    res, cols = {}, {}
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
        cols[name] = {"shuffled_ic": ic}
    return res, series_json(cols)


def dev(P, sig) -> tuple[dict, dict]:
    B = baseline_signals(P["AdjC"])
    blocked = (P["UL"] == 1) & (P["AdjO"] >= P["AdjH"])  # opened limit-up: cannot buy at the open
    rows, excess, cols = {}, {}, {}
    for name, s in sig.items():
        h = int(name.split("_h")[1])
        fwd = forward_returns(P["AdjO"], P["AdjC"], h, DEV_END).reindex(s.index)
        ic = daily_rank_ic(s, fwd)
        m, t = nw_tstat(ic, lag=h)
        fm = fama_macbeth(fwd, {"K": s, **{b: B[b].reindex(s.index) for b in B}}, lag=h)
        bts = {c: tranche_backtest(s, P["AdjO"], P["AdjC"], blocked, h, cost_bp=c) for c in (5, 10, 14, 15, 25, 37)}
        ex = bts[10]["excess"].loc[s.index.min():].iloc[1:]
        excess[name] = ex
        live = bts[10].loc[ex.index]
        cols[name] = {"rank_ic": ic, "excess_10bp": ex, "portfolio_10bp": live["net"], "benchmark": live["bench"]}
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
    return ({"variants": rows, "pbo_all_variants": pbo,
             "b3_coverage": float(b3.notna().sum().sum() / sig["K1_h5"].notna().sum().sum())}, series_json(cols))


if __name__ == "__main__":
    step = sys.argv[1]
    P = panels()
    res, series = (costs(P), None) if step == "costs" else p5(P, kronos_signals()) if step == "p5" \
        else dev(P, kronos_signals())
    (ROOT / "runs" / f"h1_{step}.json").write_text(json.dumps(res, indent=1, default=float))
    if series is not None:  # dashboard per-date series: commit together with h1_<step>.json, never earlier
        (ROOT / "runs" / f"h1_{step}_series.json").write_text(json.dumps(series))
    print(json.dumps(res, indent=1, default=float))
