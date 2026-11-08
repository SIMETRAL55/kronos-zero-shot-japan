# Results

Specification: `PREREG.md`, pre-registered as commit `808d459`. Harness checks: `HARNESS.md`.

## H2: Nikkei 225 volatility. Dev verdict: **not shown, stopped (PREREG §8)**

**Setup:**
- 435 forecast dates, 2024-07-01 to 2026-04-13, all with outcomes by 2026-04-30.
- Kronos-small, S2 sampling (T = 1.0, top_p = 1.0, 100 paths), run on a Kaggle Tesla T4 in 719 s.
- Output hash `4d219183…5775`.
- Target: realised 12-day variance, close to close.
- Reproduce: `uv run python scripts/eval_h2.py h2dev`. Full numbers: `runs/h2dev_h2_result.json`.

| Forecaster | Mean QLIKE (lower is better) | Mincer-Zarnowitz slope (SE) |
|---|---|---|
| GARCH(1,1)-t | **0.724** | 0.44 (0.14) |
| HAR-RV | 0.728 | 0.50 (0.17) |
| EWMA (λ = 0.94) | 1.005 | 0.17 (0.13) |
| Kronos V1 (mean of path variances) | 1.592 | 0.34 (0.16) |
| Kronos V2 (spread of path endpoints) | 11.986 | 0.21 (0.12) |

- **Pass rule: the best Kronos variant beats HAR-RV, Holm-adjusted Diebold-Mariano p < 0.05.**
  - V1 is *worse* than HAR by +0.864 QLIKE (one-sided p for "better" = 0.967; Holm 1.0).
  - V2 is worse by +11.26 (p = 0.999).
  - **Fail.**
- **Why:** Kronos under-forecasts volatility.
  - V1 is below the realised value on 63% of dates (median ratio 0.77).
  - V2, the spread of its own sample paths, is far too narrow: below realised on 95% of dates (median ratio 0.14). This
    matches the overconfidence reported in upstream #395.
  - The median level is close (18.3% annualised against 19.8% realised). The problem is that Kronos reacts too little
    when volatility jumps.
- **Robustness:** Kronos is worse than HAR both before and after the 2024-11-05 TSE close change (V1 3.54 vs 1.75,
  then 1.12 vs 0.48).

**Economic check** (¥1M, Nikkei micro, 10% volatility target, ≤ 2 contracts):

| Sizing | Sharpe | Max drawdown | Worst day | Net P&L | Avg contracts |
|---|---|---|---|---|---|
| Kronos V1 | 0.76 | −20.5% | −¥52,880 | +¥189,730 | 1.38 |
| HAR-RV | 0.62 | −15.4% | −¥44,513 | +¥124,252 | 0.99 |
| Constant 1 contract | 0.82 | −10.8% | −¥44,513 | +¥167,802 | 1.00 |

- **No-harm rule: Kronos maximum drawdown ≤ 1.2 × HAR's.** Kronos is 1.33×. **Fail.**
- Kronos sizing made more money only because it under-forecast volatility. That meant holding more contracts in a
  market that rose over the window, which is leverage, not skill. A constant single contract beat both on Sharpe.

**Decision, per §8:** H2 stops here. No holdout, no re-tuning. Kronos-small's zero-shot volatility forecasts are not
usable for position sizing on the Nikkei 225. HAR or GARCH are better and cost nothing to run.

## H1: TOPIX500 ranking. **Not run yet**

H1 needs J-Quants stock data (Light plan) and a decision on processing it on Kaggle; see `HARNESS.md`. Its harness
pieces are built and tested: universe, signals, backtest, statistics.
