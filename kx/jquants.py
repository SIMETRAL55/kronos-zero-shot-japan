"""J-Quants API v2 client (x-api-key auth). Pages are cached as parquet under data/jq/ (gitignored, licence)."""
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.jquants.com/v2"
DATA = Path(__file__).resolve().parent.parent / "data" / "jq"
CRED = Path.home() / ".config" / "kronos-explore" / "jquants.env"
DEV_END = "2026-04-30"  # PREREG §2: nothing after this is fetched before the holdout step


def api_key() -> str:
    for line in CRED.read_text().splitlines():
        if line.startswith("JQUANTS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"JQUANTS_API_KEY not found in {CRED}")


class JQuants:
    def __init__(self, key: str | None = None, max_date: str = DEV_END, min_interval: float = 1.1):
        self.s = requests.Session()
        self.s.headers["x-api-key"] = key or api_key()
        self.max_date = pd.Timestamp(max_date)
        self.gap, self.last = min_interval, 0.0  # Light plan: 60 requests/minute

    def _guard(self, *dates: str) -> None:
        for d in dates:
            if pd.Timestamp(d) > self.max_date:
                raise ValueError(f"{d} is after max_date {self.max_date:%Y-%m-%d} (sealed window)")

    def _get(self, path: str, **params) -> pd.DataFrame:
        rows, key = [], None
        while True:
            q = dict(params, **({"pagination_key": key} if key else {}))
            time.sleep(max(0.0, self.gap - (time.monotonic() - self.last)))
            r = self.s.get(f"{BASE}{path}", params=q, timeout=60)
            self.last = time.monotonic()
            if r.status_code == 429:  # sliding one-minute window, no Retry-After header
                time.sleep(65)
                continue
            r.raise_for_status()
            j = r.json()
            rows += j.get("data", [])
            key = j.get("pagination_key")
            if not key:
                return pd.DataFrame(rows)

    def _cached(self, name: str, path: str, **params) -> pd.DataFrame:
        f = DATA / f"{name}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        df = self._get(path, **params)
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f)
        return df

    def daily_bars(self, date: str) -> pd.DataFrame:
        self._guard(date)
        return self._cached(f"bars/{date}", "/equities/bars/daily", date=date)

    def master(self, date: str) -> pd.DataFrame:
        self._guard(date)
        return self._cached(f"master/{date}", "/equities/master", date=date)

    def calendar(self, start: str, end: str) -> pd.DataFrame:
        self._guard(end)
        return self._cached(f"calendar_{start}_{end}", "/markets/calendar", **{"from": start, "to": end})

    def topix(self, start: str, end: str) -> pd.DataFrame:
        self._guard(end)
        return self._cached(f"topix_{start}_{end}", "/indices/bars/daily/topix", **{"from": start, "to": end})


def trading_days(cal: pd.DataFrame) -> pd.DatetimeIndex:
    """HolDiv 1 = business day, 2 = half-day session; both trade."""
    return pd.DatetimeIndex(pd.to_datetime(cal.loc[cal["HolDiv"].astype(str).isin(["1", "2"]), "Date"])).sort_values()


def panel(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    """Long bars (Date, Code, field) -> date x code panel; a missing bar stays NaN (never forward-filled)."""
    p = bars.pivot_table(index="Date", columns="Code", values=field, aggfunc="last")
    p.index = pd.to_datetime(p.index)
    return p.sort_index()
