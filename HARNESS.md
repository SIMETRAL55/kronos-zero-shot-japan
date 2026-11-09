# Phase 2 harness report

Specification: `PREREG.md` §6, pre-registered as commit `808d459` on 2026-11-08.

## Environment and pinned versions

- **Code:** Kronos upstream `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, fetched by `scripts/fetch_kronos.sh` into
  `vendor/`.
- **Weights:** `NeoQuasar/Kronos-small` @ `901c26c1332695a2a8f243eb2f37243a37bea320`;
  `NeoQuasar/Kronos-Tokenizer-base` @ `0e0117387f39004a9016484a186a908917e22426`. These are the revisions upstream's
  own regression test pins.
- **Packages:** Python 3.12.13, torch 2.15.0+cpu, pandas 2.3.3, numpy 2.5.4, huggingface_hub 0.33.1. Locked in
  `uv.lock`.
- **Inference mode:** `kx/kronos_env.load_kronos` forces `.eval()` and refuses to return if any module is still
  training (upstream #415).

## Checks

| # | Check | Result | Reproduce |
|---|---|---|---|
| P1 | Inference mode; same seed gives same forecast | **Pass** (2 tests) | `uv run pytest tests/test_kronos_env.py` |
| P2 | Upstream regression suite @ 67b630e | **Pass** (4 tests: 2 regression, 2 MSE) | `cd vendor/Kronos && PYTHONPATH=. uv run --project ../.. pytest tests/test_kronos_regression.py` |
| P3 | Path wrapper with n = 1 equals upstream `predict_batch(sample_count=1)` | **Pass** | `uv run pytest tests/test_forecast.py` |
| P6 | Planted IC 0.05 is recovered (460 days × 480 names) | **Pass** (0.05 ± 0.01, Newey-West t > 3) | `uv run pytest tests/test_positive_control.py` |
| P4 | Random-walk placebo: 500 synthetic stocks × 40 dates, K1 under S1, Kaggle T4 | **Pass**: mean IC −0.0012, Newey-West t −0.18 (within ±2 SE) | `uv run python scripts/eval_p4.py` (`runs/p4_result.json`) |
| P7 | Reproduce upstream #375 | **Not reproducible**: the issue does not publish its 182-ticker list (only examples). Non-blocking per §6 | n/a |
| GPU | GPU vs CPU parity: greedy decoding on 3 windows, Kaggle Tesla T4 (torch 2.11+cu130) vs laptop CPU | **Exact** (max relative difference 0.0) | `scripts/kaggle_run.py push/fetch parity` |
| Data | Nikkei 225: Yahoo `^N225` (primary, from 2015-01-05) vs Nikkei Inc. official CSV, 20 random dates | **Pass**: max OHLC difference 5.6e-8 (limit 5e-4) | `uv run python scripts/fetch_dev_data.py nikkei` (`runs/nikkei_crosscheck.json`) |

Full test suite: `uv run pytest`, 45 passed.

## Compute (measured)

- **Kaggle Tesla T4, one GPU:** ≈ 17 ms per sample path. P4: 200,000 paths in 3,356 s. H2 dev: 43,500 paths in
  719 s.
- **H1 dev estimate at that rate** (≈ 480 names × 435 dates):
  - S1 (10 paths): ≈ 2.1M paths ≈ 10 GPU-hours.
  - S2 (50 paths): ≈ 10.4M paths ≈ 49 GPU-hours.
  - **Total ≈ 58 GPU-hours.** That is about 2 weeks of Kaggle's free quota (≈ 30 h/week), or about 1 week if each
    session uses both of its T4s.

## J-Quants terms (decision needed before H1 data goes to a cloud GPU)

- J-Quants is for individuals' **private use only**. Passing data to third parties, or building apps on it, is
  prohibited, commercial or not ([usage help](https://jpx-jquants.com/ja/help/usage);
  [terms](https://jpx-jquants.com/ja/termsofservice)).
- Processing it in one's own private Kaggle session is not addressed explicitly. Lowest-exposure option: send only
  Kronos-normalised 40-bar windows with no issue codes, in a private kernel, deleted after the run. The user decides.

## H1 data (PREREG Amendment 1: J-Quants Free plan)

- **Coverage:** the plan reported 2024-08-16 to 2026-08-16. Fetching stopped at 2026-04-30, with the client refusing
  any later date.
- **Fetched:** prices for 415 trading days (2024-08-16 to 2026-04-30), and 355 daily listed-issue master snapshots
  (2024-11-14 to 2026-04-30).
- **Checksums:** all 773 data files are hashed in `data/SNAPSHOT.json`.
- **Windows:** 170,613 windows on **343 forecast dates (2024-11-14 to 2026-04-13)**.
  - The first date is the 61st trading day of Free-plan history.
  - Median 497 names per date. 0 name-dates were dropped for gaps inside a 40-bar window.
  - Built by `uv run python scripts/build_h1_windows.py`.
- **What goes to the private Kaggle dataset:** only prices and date indices, never issue codes. The dataset is deleted
  after the runs.
- **Seeds:** each GPU worker uses the run seed plus its first row, so reruns reproduce.
  `tests/test_run_multi.py` checks the split-and-merge path. This is deterministic, as §1 requires, but keyed by row
  block rather than forecast date.

## §10 cost check (done before any H1 signal was evaluated)

Reproduce with `uv run python scripts/eval_h1.py costs`; output in `runs/h1_costs.json`. 517 TOPIX500 names over the
dev window.

| Method | Median round trip | Notes |
|---|---|---|
| Abdi & Ranaldo (2017) daily high-low spread estimator | 36.7 bp (p25 0, p75 61) | **27.7% of names clipped at zero.** Noise-dominated when spreads are tiny relative to daily volatility, as for liquid large caps |
| TSE tick floor (official tick tables; TOPIX100 use the finer table) | one tick = **6.9 bp**; two ticks = **13.8 bp** | A round trip with a one-tick spread costs one tick. Two ticks allows for a wider spread or slippage |

**Decision.** The realistic round trip for the median name is about 7–14 bp. That is below §10's 15 bp trigger, so the
pre-registered **10 bp** stays the pass-rule cost and no amendment is needed. Commissions are zero at the major
zero-commission routes (Phase 0). The 14 bp (two ticks) and 37 bp (Abdi-Ranaldo) figures are reported as extra
sensitivities next to 5/15/25 bp.

## P5 and the power figure (committed before any real H1 IC was computed)

Reproduce with `uv run python scripts/eval_h1.py p5` and `... p5diag`. Outputs: `runs/h1_p5.json`,
`runs/h1_p5_series.json`, `runs/h1_p5diag.json`.

**P5 as pre-registered (pair each date's signals with a random other date's returns): fails on 5 of 6 variants.**

| Variant | Date-shuffled mean IC | Newey-West t | Within ±2 SE? | Power: min. detectable IC (t = 3) |
|---|---|---|---|---|
| K1 h5 | +0.0047 | +0.80 | yes | 0.018 |
| K2 h5 | −0.0146 | −2.08 | no | 0.021 |
| K3 h5 | −0.0232 | −3.96 | no | 0.018 |
| K1 h12 | −0.0239 | −3.10 | no | 0.023 |
| K2 h12 | −0.0438 | −3.90 | no | 0.034 |
| K3 h12 | −0.0351 | −4.35 | no | 0.024 |

**Diagnosis (`p5diag`; it never pairs a signal with its own date's returns):**

1. **A true null for the pipeline passes on all 6 variants.** Signals were permuted across stocks within each date, and
   every IC is within ±2 SE (t from −0.70 to +0.94). So the code does not leak, and it scores pure noise as zero.
2. **Both inputs persist at the stock level.**
   - Signal rank persistence over 20 days: −0.04 to +0.14 (K2 and K3 at h12 are the most persistent).
   - Forward-return rank persistence across non-overlapping horizons: +0.013 (h5) and +0.023 (h12).
   - Shuffling dates keeps each stock's identity, so stock-level traits (volatility, for example) survive the shuffle.
     That is what pushes the date-shuffled IC away from zero. It is a property of this data, not a bug, and it means
     the date shuffle is not a valid null here.
3. **Power with the true null:** minimum detectable IC 0.007–0.009 at t = 3.

**Decision.** §6's stop rule covers P1–P4 and P6, not P5, so the evaluation proceeds exactly as pre-registered. This
stock-level component stays inside the real ICs and portfolio returns, as it would for a real portfolio. The power
figure is reported both ways.

## Closed

- **Done before the H1 dev results:** the §10 cost check (`4a40012`) and P5 with its power figure (`b7999eb`).
- **H1 dev results:** `f51931c`. No variant passes, so H1 stops per §8.
- **Cleanup (2026-11-09):** the private Kaggle dataset `<kaggle-user>/kronos-h1-dev` is deleted.
  - The private notebooks `kronos-eval-*` still hold forecast outputs: numbers only, no issue codes.
  - The J-Quants data stays only in this machine's `data/`.


## P4, part 1: pull-to-mean bias record (no pass rule)

Setup: 30 synthetic random walks, σ = 1.5%/day, 300 bars; K1 under S1 from bars 220–259. Reproduce with
`uv run python scripts/placebo_synthetic.py`.

| Drift | True mean 12-day return (log / simple) | Kronos mean predicted 12-day return | Share predicted down |
|---|---|---|---|
| +0.1%/day | +1.20% / +1.34% | **−0.09%** | 43% |
| 0 | 0.00% / +0.14% | +0.80% (≈1.5 SE across 30 names) | 33% |

Reading:
- Kronos does not carry a steady trend forward. It forecasts about zero where the true drift is +1.3%.
- That fits the pull-toward-recent-levels concern (upstream #422), but it is far milder than #422's −10.4%. #422 used
  20 bars and its own setup.
- The pre-registered B1 control (reversal to the 40-day mean) is in every Fama-MacBeth test, so H1 cannot pass on
  this bias alone.
