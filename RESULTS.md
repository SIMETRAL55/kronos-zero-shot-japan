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

## H1: TOPIX500 ranking. Dev verdict: **not shown, stopped (PREREG §8)**

**Setup (PREREG §4 + Amendment 1):**
- 343 forecast dates, 2024-11-14 to 2026-04-13 (J-Quants Free plan), about 497 TOPIX500 names per date, 170,613
  windows.
- Kronos-small zero-shot.
  - S1: T = 0.6, top_p = 0.9, 10 paths.
  - S2: T = 1.0, top_p = 1.0, 50 paths.
  - Kaggle, 2× T4, ~23 GPU-hours. Output hashes are in `runs/h1s*/out/manifest.json`.
- The §10 cost check and P5 were committed before this run (`4a40012`, `b7999eb`).
- Reproduce: `uv run python scripts/eval_h1.py dev`. Full numbers: `runs/h1_dev.json`. Per-date series:
  `runs/h1_dev_series.json`.

| Variant | Mean rank IC (Newey-West t) | Fama-MacBeth t beyond B1–B4 (needs ≥ 3) | Top-quintile excess/yr after 10 bp | Excess Sharpe | Deflated Sharpe (needs ≥ 0.95) | Months with IC > 0 | Pass |
|---|---|---|---|---|---|---|---|
| K1 h5 | +0.0058 (0.89) | −0.47 | −4.4% | −1.11 | 0.016 | 61% | no |
| K2 h5 | +0.0049 (0.46) | −1.03 | −6.3% | −1.52 | 0.003 | 67% | no |
| K3 h5 | +0.0042 (0.38) | −1.02 | −6.1% | −1.40 | 0.006 | 61% | no |
| K1 h12 | +0.0076 (0.51) | +0.05 | +0.04% | +0.01 | 0.198 | 61% | no |
| K2 h12 | −0.0005 (−0.02) | −0.58 | −5.5% | −1.12 | 0.015 | 50% | no |
| K3 h12 | −0.0012 (−0.06) | −0.55 | −4.2% | −0.81 | 0.035 | 50% | no |

- PBO across the 6 variants: 0.14. The 12-1 momentum control (B3) had history for 44% of name-dates; the rest took the
  neutral rank, per §4.
- **No variant passes.**
  - Raw ICs are between −0.001 and +0.008 and none is significant, although the test can detect about 0.018–0.034
    (P5 power), or 0.007–0.009 on the true null.
  - After controlling for simple price rules (reversal, momentum), Kronos adds nothing: every Fama-MacBeth t is at most
    +0.05.
  - Its top-quintile portfolio trails the equal-weight universe after costs in 5 of 6 variants. The sixth breaks even
    at 10 bp.
- **Cost sensitivity.** At 5 bp, five variants are still negative and K1 h12 makes +0.9% a year. At 14 bp and above,
  all six are negative.
- **Point accuracy** (reported only). K1's predicted returns are slightly closer to realised returns than a flat zero
  (Diebold-Mariano p < 0.001 at both horizons). The pass rules test ranking, the thing that picks stocks, and there
  Kronos found nothing.
- This agrees with the outside US test (upstream #375: IC +0.022, t 1.39) and with H2.

**Decision, per §8:** H1 stops. No holdout, no Kronos-base run, no re-tuning. The private Kaggle dataset of windows is
deleted.

## Overall conclusion

On data it never saw (July 2024 to April 2026), zero-shot Kronos-small did not provide tradeable information for:
- **What to hold:** TOPIX500 cross-sectional ranking at 5 or 12 days.
- **How much to hold:** Nikkei 225 12-day volatility. HAR and GARCH did better.

Both pre-registered tests stopped at the dev stage, and neither touched the sealed holdout. Money spent: ¥0. The
harness (data layer, Kaggle GPU runner, statistics, placebos, backtests) is tested and reusable for the next idea.
