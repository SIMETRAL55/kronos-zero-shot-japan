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

## Still open before H1 dev results

- **§10 broker-cost check:** confirm or replace the 10 bp round-trip assumption from official broker pages. Done as an
  amendment before H1 dev is opened, if the figure changes.
- **P5 and the power figure:** H1 only; run on H1's dev forecasts before any real IC is computed.


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
