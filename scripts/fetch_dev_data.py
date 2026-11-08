"""Fetch dev-stage data only (PREREG §2): nothing after 2026-04-30. Hashes go to data/SNAPSHOT.json (committed).

  uv run python scripts/fetch_dev_data.py nikkei      # Yahoo ^N225 (primary) + official CSV cross-check
  uv run python scripts/fetch_dev_data.py jquants     # needs JQUANTS_API_KEY in ~/.config/kronos-explore/jquants.env
  uv run python scripts/fetch_dev_data.py jquants --free   # Free plan (PREREG Amendment 1), ~3 h at 5 req/min
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kx.jquants import CRED, DEV_END, JQuants, trading_days  # noqa: E402
from kx.nikkei import cross_check, official_daily, yahoo_daily  # noqa: E402

DATA = ROOT / "data"


def nikkei() -> None:
    y, o = yahoo_daily("2015-01-01", DEV_END), official_daily(DEV_END)
    assert y.index.max() <= pd.Timestamp(DEV_END) and o.index.max() <= pd.Timestamp(DEV_END)
    y.to_parquet(DATA / "nikkei225_yahoo.parquet")
    o.to_parquet(DATA / "nikkei225_official.parquet")
    cc = cross_check(y, o)
    out = {"source_primary": "Yahoo ^N225 via yfinance", "source_check": "Nikkei Inc. official daily CSV",
           "primary_first": str(y.index.min().date()), "primary_rows": len(y),
           "primary_rows_before_2024_07": int((y.index < "2024-07-01").sum()),
           "check_first": str(o.index.min().date()), "n_dates": len(cc),
           "max_rel_diff": float(cc["max_rel"].max()), "pass_le_0.05pct": bool(cc["max_rel"].max() <= 0.0005),
           "dates": [str(d.date()) for d in cc.index]}
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "nikkei_crosscheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "dates"}, indent=1))


def jquants() -> None:
    if not CRED.exists():
        sys.exit(f"No J-Quants key at {CRED}: add JQUANTS_API_KEY=... (mode 600) and re-run")
    free = "--free" in sys.argv  # PREREG Amendment 1: Free plan = 2 years, 12-week delay, 5 req/min, no TOPIX
    jq = JQuants(min_interval=12.5 if free else 1.1)
    start = "2022-01-01"
    try:
        cal = jq.calendar(start, DEV_END)
    except requests.HTTPError as e:  # plan window: "Your subscription covers the following dates: 2024-08-16 ~ ..."
        m = re.search(r"covers the following dates: (\d{4}-\d{2}-\d{2})", e.response.text)
        if not m:
            raise
        start = m.group(1)
        cal = jq.calendar(start, DEV_END)
    days = trading_days(cal)
    if not free:
        jq.topix("2021-01-01", DEV_END)
    stock_days = days[days >= "2023-01-01"]
    print(f"calendar {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}; {len(stock_days)} bar days to fetch", flush=True)
    for i, d in enumerate(stock_days):
        jq.daily_bars(f"{d:%Y-%m-%d}")
        if i % 50 == 0:
            print(f"bars {i}/{len(stock_days)}", flush=True)
    first_forecast = days[60] if free else pd.Timestamp("2024-07-01")  # Free: needs 60 days of history first
    for d in days[days >= first_forecast]:
        jq.master(f"{d:%Y-%m-%d}")
    print("J-Quants dev data fetched", flush=True)


def snapshot() -> None:
    files = sorted(DATA.rglob("*.parquet"))
    snap = {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (DATA / "SNAPSHOT.json").write_text(json.dumps(snap, indent=1, sort_keys=True))
    print(f"{len(snap)} files hashed into data/SNAPSHOT.json")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    {"nikkei": nikkei, "jquants": jquants}[sys.argv[1]]()
    snapshot()
