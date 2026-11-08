"""Backtests (PREREG §4 portfolio, §5 economic check)."""
import numpy as np
import pandas as pd


def _roll_days(index: pd.DatetimeIndex) -> set:
    """Trading day before the 2nd Friday of Mar/Jun/Sep/Dec (OSE SQ) for each quarter in the index."""
    out = set()
    for y in sorted(set(index.year)):
        for m in (3, 6, 9, 12):
            fridays = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
            sq = [d for d in fridays if d.month == m and d.weekday() == 4][1]
            before = index[index < sq]
            if len(before) and before[-1] >= index[0] and (sq <= index[-1]):
                out.add(before[-1])
    return out


def vol_target_backtest(close: pd.Series, f12: pd.Series, capital: float = 1_000_000, target_vol: float = 0.10,
                        max_contracts: int = 2, multiplier: int = 10, fee_side: float = 11.0,
                        tick_value: float = 50.0) -> pd.DataFrame:
    """Long Nikkei micro sized at each close t from the 12-day variance forecast F; P&L over t -> t+1.

    contracts_t = clip(round(target_vol/sqrt(245) * capital / (sqrt(F/12) * multiplier * C_t)), 0, max).
    Costs: (fee + 1 tick) per contract per side on every change; quarterly roll = 2 legs per held contract.
    """
    c, f = close.astype(float), f12.reindex(close.index).astype(float)
    daily_vol_yen = np.sqrt(np.clip(f, 0, None) / 12) * multiplier * c
    want = np.where(daily_vol_yen > 0, target_vol / np.sqrt(245) * capital / daily_vol_yen, 0.0)
    n = pd.Series(np.clip(np.round(np.nan_to_num(want)), 0, max_contracts).astype(int), c.index)
    pnl = (n * multiplier * c.diff().shift(-1)).fillna(0.0)
    trade_cost = n.diff().fillna(n).abs() * (fee_side + tick_value)
    rolls = _roll_days(c.index)
    roll_cost = pd.Series([n[d] * 2 * (fee_side + tick_value) if d in rolls else 0.0 for d in c.index], c.index)
    net = pnl - trade_cost - roll_cost
    return pd.DataFrame({"contracts": n, "pnl": pnl, "trade_cost": trade_cost, "roll_cost": roll_cost,
                         "net": net, "equity": capital + net.cumsum()})


def _book(sel: pd.DataFrame, adj_open: pd.DataFrame, adj_close: pd.DataFrame, h: int) -> tuple[pd.Series, pd.Series]:
    """Jegadeesh-Titman book: each start date's equal-weight basket bought at the t+1 open, held to the t+h close,
    with 1/h of capital per tranche. Returns (daily book return, count of non-empty tranches started per entry day)."""
    dates = adj_close.index
    oc = (adj_close / adj_open - 1).fillna(0.0)          # entry day: open -> close
    cc = adj_close.pct_change(fill_method=None).fillna(0.0)  # later days: close -> close (missing bar = flat)
    book = np.zeros(len(dates))
    starts = np.zeros(len(dates))
    for t, row in sel.iterrows():
        p = dates.get_loc(t)
        names = row.index[row.to_numpy(bool)]
        if len(names) == 0 or p + h >= len(dates):
            continue
        R = np.column_stack([oc.iloc[p + 1][names].to_numpy()] + [cc.iloc[p + k][names].to_numpy() for k in range(2, h + 1)])
        V = np.cumprod(1 + R, axis=1).mean(0)
        book[p + 1:p + h + 1] += (V / np.r_[1.0, V[:-1]] - 1) / h
        starts[p + 1] += 1
    return pd.Series(book, dates), pd.Series(starts, dates)


def tranche_backtest(signal: pd.DataFrame, adj_open: pd.DataFrame, adj_close: pd.DataFrame,
                     entry_blocked: pd.DataFrame, h: int, q: float = 0.2, cost_bp: float = 10.0) -> pd.DataFrame:
    """Long-only top-q by signal (PREREG §4). entry_blocked[t+1] marks names that cannot be bought at the t+1 open
    (limit-up). Each tranche pays cost_bp per round trip on its full weight. bench = equal-weight of all scored names."""
    nxt = entry_blocked.shift(-1).reindex(signal.index).fillna(False).astype(bool)
    valid = signal.notna()
    top = signal.rank(axis=1, ascending=False, pct=True).le(q) & valid & ~nxt.reindex(columns=signal.columns,
                                                                                         fill_value=False)
    gross, starts = _book(top, adj_open, adj_close, h)
    bench, _ = _book(valid, adj_open, adj_close, h)
    net = gross - starts * (cost_bp / 1e4) / h
    return pd.DataFrame({"gross": gross, "net": net, "bench": bench, "excess": net - bench})


def summary(bt: pd.DataFrame, capital: float = 1_000_000) -> dict:
    r = bt["net"] / capital
    dd = bt["equity"] / bt["equity"].cummax() - 1
    return {"sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(245)) if r.std() > 0 else 0.0,
            "max_drawdown": float(dd.min()), "worst_day_yen": float(bt["net"].min()),
            "total_net_yen": float(bt["net"].sum()), "avg_contracts": float(bt["contracts"].mean())}
