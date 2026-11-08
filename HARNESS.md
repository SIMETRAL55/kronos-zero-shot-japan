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
