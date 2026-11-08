# Kronos Evaluation (Phases 1–3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the pre-registered evaluation of Kronos-small. H1 ranks TOPIX500 stocks; H2 forecasts Nikkei 225
volatility. The outputs are `HARNESS.md` and `RESULTS.md` with a go/no-go for each hypothesis.

**Architecture:** A small Python package `kx/` holds the data clients, a Kronos wrapper that keeps individual sample
paths, signals and baselines, statistics, and backtests. It is tested with pytest on synthetic fixtures, with no
network calls in tests. GPU-heavy forecasting runs as a single script on Kaggle (or Colab) that turns an input `.npz`
of bar windows into an output `.npz` of paths plus a hash manifest. All evaluation runs on the laptop CPU.

**Tech Stack:** Python 3.12 via uv; torch (CPU locally); pandas, numpy, scipy, statsmodels, arch, pyarrow, requests,
pytest; upstream Kronos at `67b630e` (fetched by a script, gitignored); J-Quants API (Light); Kaggle CLI.

**Spec:** `PREREG.md` (with `PROMPT.md` and `FEASIBILITY.md`). § numbers below refer to `PREREG.md`.

## Global Constraints

- Kronos upstream commit `67b630e`. Call `.eval()` on the tokenizer and the model. Record the HF revision hashes in
  `HARNESS.md`.
- Kronos-small zero-shot. Kronos-base only per §7. No fine-tuning.
- Lookback 40, horizon 12. S1: T=0.6, top_p=0.9, N=10 (averaged). S2: T=1.0, top_p=1.0, paths kept (N=50 for H1,
  N=100 for H2).
- Dev forecast dates from 2024-07-01, with outcomes by 2026-04-30. Fetch no data after 2026-04-30 before Task 13.
- ≤ 12 variants in total. Every run is logged with a manifest. No re-tuning after dev results.
- Spend ≤ ¥3,000/month (J-Quants Light ¥1,650 + Kaggle free). Ask before any other spend.
- No live orders. No git remote, no push. No `Co-Authored-By` lines in commits.
- J-Quants data is never committed (`data/` is gitignored). Only `data/SNAPSHOT.json` hashes are committed.
- Execute with superpowers:executing-plans, one task at a time (the laptop has run out of memory with parallel jobs).
- Stop for the user's review after Task 9 (end of Phase 2) and Task 13 (end of Phase 3).

## Review Focus

1. **Missing bars, trading halts, listings and delistings inside the window.** The name drops out for that date and is
   never forward-filled. Test: Task 6 `test_missing_bar_excluded`, `test_delisted_drops_after_date`.
2. **Splits.** Kronos inputs and forward returns both use adjusted prices, so a 1:2 split must not show as a −50%
   return. Test: Task 6 `test_split_adjusted_continuous`.
3. **Future timestamps.** They must be the next TSE trading days, skipping weekends and holidays. Test: Task 3
   `test_next_trading_days_golden_week`.
4. **Limit-up at the next open.** The backtest cannot buy those names. Test: Task 11 `test_limit_up_entry_skipped`.
5. **Window edges and look-ahead.** No dev forecast's outcome may cross 2026-04-30, and no data after t may change a
   signal at t. Test: Task 5 `test_forward_returns_window_end`, `test_signals_ignore_future_rows`.

---

### Task 1: Repository and pre-registration commit (Phase 1)

**Files:** Create `.gitignore` (`data/`, `vendor/`, `.venv/`, `__pycache__/`, `*.npz`, `*.parquet`); commit
`PROMPT.md`, `FEASIBILITY.md`, `phase0/factbase.json`, `PREREG.md` and this plan.

- [ ] **Step 1:** `git init -b main ~/kronos-explore` and write `.gitignore`.
- [ ] **Step 2:** Run `git status --short`. Expected: the files above and nothing under `data/` or `vendor/`.
- [ ] **Step 3:** Commit with message `Pre-register Kronos evaluation (Phase 1)`. Record the commit hash in the
  report. That commit is the pre-registration timestamp.

### Task 2: Environment, pinned Kronos and inference-mode checks (P1, P2)

**Files:** Create `pyproject.toml`, `scripts/fetch_kronos.sh`, `kx/__init__.py`, `kx/kronos_env.py`,
`tests/test_kronos_env.py`.

**Interfaces:**
- Produces: `load_kronos(model_id: str = "NeoQuasar/Kronos-small", tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base", device: str = "cpu") -> tuple[KronosTokenizer, Kronos, dict]`.
  - The dict holds the `model_revision` and `tokenizer_revision` commit hashes.
  - It puts `vendor/Kronos` on `sys.path`, calls `.eval()` on both, and asserts no module has `training=True`.

- [ ] **Step 1:** `pyproject.toml`: Python 3.12; deps torch (CPU), numpy, pandas, scipy, statsmodels, arch, pyarrow,
  requests, einops==0.8.1, huggingface_hub==0.33.1, safetensors, tqdm, pytest.
  `scripts/fetch_kronos.sh` clones `shiyu-coder/Kronos` into `vendor/Kronos` and checks out `67b630e`.
- [ ] **Step 2:** Run `uv sync && bash scripts/fetch_kronos.sh`. Expected: `git -C vendor/Kronos rev-parse --short HEAD`
  prints `67b630e`.
- [ ] **Step 3:** Write the failing tests:
  - `test_eval_mode`: every module in the tokenizer and the model has `training is False`.
  - `test_same_seed_same_forecast`: two `predict` calls on `vendor/Kronos/tests/data/regression_input.csv` (last 40
    rows, pred_len 12, seed 0) are `np.array_equal`.
- [ ] **Step 4:** Implement `load_kronos`. Run `uv run pytest tests/test_kronos_env.py -v`. Expected: 2 passed.
- [ ] **Step 5:** Run the upstream regression test (P2): `cd vendor/Kronos && PYTHONPATH=. uv run --project ../.. pytest tests/test_kronos_regression.py -v`.
  Expected: passed. Note the result for `HARNESS.md`.
- [ ] **Step 6:** Commit `pyproject.toml uv.lock scripts/ kx/ tests/` with message `Add pinned Kronos environment (P1, P2)`.

### Task 3: Path-keeping forecast wrapper and trading-day timestamps (P3)

**Files:** Create `kx/tsecal.py`, `kx/forecast.py`, `tests/test_forecast.py`.

**Interfaces:**
- Consumes: `load_kronos` (Task 2).
- Produces:
  - `next_trading_days(calendar: pd.DatetimeIndex, t: pd.Timestamp, k: int) -> pd.DatetimeIndex`.
  - `forecast_paths(tokenizer, model, bars: list[pd.DataFrame], x_dates: list[pd.DatetimeIndex], y_dates: list[pd.DatetimeIndex], *, T: float, top_p: float, n: int, seed: int, device: str = "cpu", batch_rows: int = 2048) -> np.ndarray`.
  - Shape `(len(bars), n, 12, 6)`; columns open, high, low, close, volume, amount.
  - Method: replicate each series n times into upstream `KronosPredictor.predict_batch(..., sample_count=1)` in chunks
    of `batch_rows`, calling `torch.manual_seed(seed + chunk_index)` per chunk. This needs no upstream code change.

- [ ] **Step 1:** Write the failing tests:
  - `test_next_trading_days_golden_week`. With the fixture calendar, 2025-04-30 and k=3 give `[2025-05-01, 2025-05-02, 2025-05-07]`.
  - `test_shape`. Output shape is `(3, 4, 12, 6)` for 3 series and n=4.
  - `test_seed_determinism`. Two calls with the same seed are equal.
  - `test_paths_differ`. With n=5, not all paths are equal.
  - `test_n1_matches_upstream` (P3). n=1 equals `predict_batch(sample_count=1)` under the same seed.
- [ ] **Step 2:** Run `uv run pytest tests/test_forecast.py -v`. Expected: FAIL (not defined).
- [ ] **Step 3:** Implement both functions as specified.
- [ ] **Step 4:** Run the same command. Expected: 5 passed.
- [ ] **Step 5:** Commit with message `Add path-keeping Kronos wrapper (P3)`.

### Task 4: Statistics library

**Files:** Create `kx/stats.py`, `tests/test_stats.py`.

**Interfaces** (DataFrames are date × code):
- `daily_rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 300) -> pd.Series`
- `nw_tstat(x: pd.Series, lag: int) -> tuple[float, float]` returns (mean, t). It uses statsmodels OLS on a constant
  with `cov_type="HAC"` and `maxlags=lag`.
- `fama_macbeth(y: pd.DataFrame, X: dict[str, pd.DataFrame], lag: int) -> pd.DataFrame`. Each day it regresses
  rank-normal y on rank-normal X; the result's index is the regressor and its columns are `mean` and `t`.
- `dm_test(loss_a: pd.Series, loss_b: pd.Series, lag: int) -> tuple[float, float]` returns (mean of a−b, one-sided p for
  a < b).
- `qlike(rv: pd.Series, f: pd.Series) -> pd.Series`, where `rv/f - log(rv/f) - 1`.
- `deflated_sharpe(r: pd.Series, sr_trials: list[float]) -> float`. Per-period Sharpe ratios; Bailey & López de Prado
  (2014):
  - DSR = Φ((SR − SR0)·√(T−1) / √(1 − γ3·SR + (γ4−1)/4·SR²))
  - SR0 = √Var(sr_trials) · ((1−e)·Φ⁻¹(1−1/N) + e·Φ⁻¹(1−1/(N·exp(1)))), with e = 0.5772156649.
  - Here γ3 is skewness, γ4 is kurtosis (not excess), and N = len(sr_trials). SR0 = 0 when N = 1.
- `pbo_cscv(R: pd.DataFrame, s: int = 16) -> float`. R is days × variants. Use all C(16,8) block splits. Take the
  in-sample best variant's out-of-sample relative rank ω, compute λ = ln(ω/(1−ω)), and return the share of λ ≤ 0.

- [ ] **Step 1:** Write the failing tests:
  - `test_ic_perfect`: mean 1.0.
  - `test_ic_noise`: |mean| < 0.02 with seed 0 and 300 days × 400 names.
  - `test_nw_smaller_than_naive_on_ar1`: φ = 0.8.
  - `test_fm_recovers_planted`: coefficient t > 3 on the planted regressor, |t| < 2 on noise.
  - `test_dm_detects_better`: `loss_a = loss_b - 0.1` gives p < 0.01.
  - `test_qlike_zero_at_truth`.
  - `test_dsr_equals_psr_when_one_trial` and `test_dsr_falls_with_more_trials`.
  - `test_pbo_dominant_low`: < 0.1.
  - `test_pbo_noise_mid`: between 0.3 and 0.7 with seed 0.
- [ ] **Step 2:** Run `uv run pytest tests/test_stats.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the functions.
- [ ] **Step 4:** Run the same command. Expected: all pass.
- [ ] **Step 5:** Commit with message `Add evaluation statistics`.

### Task 5: Signals, baselines and forward returns

**Files:** Create `kx/signals.py`, `tests/test_signals.py`.

**Interfaces:**
- Consumes: `forecast_paths` output (Task 3).
- Produces:
  - `rank_signals(paths: np.ndarray, last_close: np.ndarray, h: int) -> dict[str, np.ndarray]`. From S1 paths it
    returns `K1`; from S2 paths, `K2` and `K3`. Close is index 3 of the last axis. Definitions are in §4.
  - `baseline_signals(adj_close: pd.DataFrame) -> dict[str, pd.DataFrame]`: `B1`–`B4` per §4.
  - `forward_returns(adj_open: pd.DataFrame, adj_close: pd.DataFrame, h: int, window_end: str) -> pd.DataFrame`.
  - `kronos_vol(paths: np.ndarray, last_close: float) -> dict[str, float]`: `V1`, `V2` per §5.
  - `realized_var12(close: pd.Series) -> pd.Series`.
  - `vol_baselines(close: pd.Series, first_forecast: str) -> pd.DataFrame`, with columns `EWMA`, `GARCH`, `HAR`.
    These are expanding-window fits refitted each month start, using the `arch` package for GARCH(1,1) with
    Student-t.

- [ ] **Step 1:** Write the failing tests:
  - `test_b1_zero_on_constant`.
  - `test_b3_uses_t21_and_t252`.
  - `test_forward_returns_window_end`. For h=12, no row whose t+12 is after `window_end`.
  - `test_signals_ignore_future_rows`. Changing rows after t leaves all B-signals at t unchanged.
  - `test_k2_in_unit_interval`.
  - `test_v1_v2_nonnegative`.
  - `test_ewma_toy_value`. Hand-computed for 3 returns.
  - `test_har_recovers_coefficients`. On a simulated HAR process, within ±0.05.
- [ ] **Step 2:** Run `uv run pytest tests/test_signals.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the functions.
- [ ] **Step 4:** Run the same command. Expected: all pass.
- [ ] **Step 5:** Commit with message `Add Kronos signals and baselines`.

### Task 6: Data layer (J-Quants, Nikkei 225) and universe

**Prerequisite (the user):** subscribe to J-Quants Light and put the refresh token in
`~/.config/kronos-explore/jquants.env` (mode 600; never committed). Before writing code, read the J-Quants terms on
processing data in cloud notebooks. Record in `HARNESS.md` whether uploading to a private Kaggle dataset is allowed.
If it isn't, Task 8 uses Colab upload or CPU instead.

**Files:** Create `kx/jquants.py`, `kx/nikkei.py`, `kx/universe.py`, `tests/test_universe.py`, `tests/fixtures/`, and
`data/SNAPSHOT.json` (committed).

**Interfaces:**
- `JQuants(refresh_token: str)` with methods:
  - `.daily_quotes(start: str, end: str) -> pd.DataFrame`. Columns are Date, Code, AdjustmentOpen/High/Low/Close/Volume,
    TurnoverValue, UpperLimit, LowerLimit. Paginated, with parquet cache per month under `data/jq/`.
  - `.listed_master(date: str) -> pd.DataFrame`. Columns are Code, ScaleCategory, MarketCode.
  - `.trading_calendar(start, end) -> pd.DatetimeIndex`.
  - `.topix(start, end) -> pd.DataFrame`.
- `nikkei225_daily(start: str, end: str) -> pd.DataFrame`. Source chosen in this task; cross-check 20 random dates
  against a second source within 0.05%.
- `universe(master: pd.DataFrame, quotes: pd.DataFrame, t: pd.Timestamp) -> list[str]` per §2, including the fallback.
- `write_snapshot(paths: list[Path]) -> dict[str, str]`, which writes SHA-256 hashes to `data/SNAPSHOT.json`.

- [ ] **Step 1:** Write the failing tests (fixtures are small JSON samples; no network):
  - `test_scale_filter_core_large_mid`.
  - `test_fallback_top500_by_turnover`.
  - `test_missing_bar_excluded`.
  - `test_delisted_drops_after_date`.
  - `test_min_history_60`.
  - `test_split_adjusted_continuous`. A 1:2 split fixture gives an adjusted-close day return between −5% and +5%.
- [ ] **Step 2:** Run `uv run pytest tests/test_universe.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Run the same command. Expected: all pass.
- [ ] **Step 4:** Run `uv run python scripts/fetch_dev_data.py` (create it in this task). It fetches dev-stage data only:
  - stocks from 2023-01-01 to 2026-04-30;
  - TOPIX and Nikkei 225 from the earliest available date (at least 500 trading days before 2024-07-01) to
    2026-04-30.
  - Then run `write_snapshot`. Expected: `SNAPSHOT.json` lists every file, and no file holds a date after 2026-04-30
    (assert in the script).
- [ ] **Step 5:** Commit with message `Add data layer, universe, dev data snapshot`. This covers code, fixtures and
  `SNAPSHOT.json`, but no data.

### Task 7: Synthetic placebo and positive control (P6, and P4's drift bias)

**Files:** Create `scripts/placebo_synthetic.py`, `tests/test_positive_control.py`, `HARNESS.md` (create).

**Interfaces:**
- Consumes: `daily_rank_ic`, `nw_tstat` (Task 4); `forecast_paths` (Task 3).
- Produces: `make_random_walks(n_names: int, n_days: int, mu: np.ndarray, sigma: np.ndarray, seed: int) -> pd.DataFrame`,
  which returns daily OHLCV bars.

- [ ] **Step 1:** Write the failing test `test_planted_ic_recovered` (P6). A synthetic signal with a planted IC of 0.05,
  over 460 days × 480 names, is recovered at 0.05 ± 0.01 with Newey-West t > 3.
- [ ] **Step 2:** Implement until it passes. Run `uv run pytest tests/test_positive_control.py -v`.
- [ ] **Step 3:** Run the drift-bias measurement on CPU: 30 walks with μ = +0.1%/day and σ = 1.5%, K1 under S1.
  Record the mean predicted 12-day return against the true +1.2% in `HARNESS.md`. This records P4's bias; it has no
  pass rule.
- [ ] **Step 4:** Commit with message `Add positive control and drift-bias record`.

### Task 8: GPU runner, P4 random-walk placebo and #375 reproduction (P7)

**Files:** Create `remote/run_forecasts.py`, `remote/kernel-metadata.json`, `scripts/kaggle_push.sh`,
`scripts/build_windows.py`; update `HARNESS.md`.

**Interfaces:**
- Input `.npz`:
  - `bars` (B,40,6) float32;
  - `x_dates` (B,40) and `y_dates` (B,12) as `datetime64[D]`;
  - `keys` (B,) str, written as `"{date}|{code}"`;
  - `cfg` JSON: T, top_p, n, seed.
- Output `.npz`: `paths` (B,n,12,6) float32 and `keys`.
- Plus `manifest.json` with input_sha256, output_sha256, model_revision, cfg, git_commit, seconds and device.
- `run_forecasts.py` uses `forecast_paths` (Task 3) and needs no other repo code; vendor `kx/forecast.py` into the
  kernel bundle.

- [ ] **Step 1:** Run the runner locally on CPU with B=4, n=2. Expected: shapes match, and the manifest hashes match
  `sha256sum`.
- [ ] **Step 2:** Push to Kaggle (`bash scripts/kaggle_push.sh`, GPU on, internet on for HF weights) and run the same
  tiny input. Expected: max |GPU − CPU| < 1e-3 relative on the closes. Record the runtime.
- [ ] **Step 3 (P4):**
  - Run 500 synthetic random-walk stocks × 40 dates under S1 on Kaggle.
  - Compute the rank IC of K1 against the synthetic forward returns.
  - Expected: the mean is within ±2 SE of 0.
- [ ] **Step 4 (P7):**
  - Get #375's universe and dates from the issue's attachments.
  - Pull yfinance daily bars (`auto_adjust=True`) and run under S1 with that issue's timing.
  - Expected: IC within +0.022 ± 0.015. If the list is unavailable, write "not reproducible".
- [ ] **Step 5:** Commit with message `Add GPU runner; P4 and P7 results`.

### Task 9: Phase 2 report, then STOP for review

**Files:** Complete `HARNESS.md`:
- P1–P4, P6, P7, model revision hashes, the compute route and its quota use, and the J-Quants terms decision;
- the broker-cost check: confirm or replace §10's 10 bp from 2–3 official broker pages, as a dated amendment to
  `PREREG.md` if it changes;
- the Nikkei source and its cross-check.

- [ ] **Step 1:** Write `HARNESS.md`. Each check gets its pass/fail line and the command that reproduces it.
- [ ] **Step 2:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 3:** Commit with message `Phase 2 harness report`. **Stop. Ask the user to review `HARNESS.md`.**

### Task 10: Dev forecasts, P5 and the power figure (Phase 3)

**Files:** Create `scripts/make_dev_forecasts.py`; update `HARNESS.md` (P5 and the power figure) and `runs/`
(manifests, committed).

- [ ] **Step 1:** Build windows for every (t, code) with t in dev and code in U(t); and for the Nikkei 225 and TOPIX
  dev dates.
- [ ] **Step 2:** Run on GPU:
  - H1: S1 N=10 and S2 N=50 (≈ 24 GPU-hours on Kaggle T4×2; split across sessions by date blocks).
  - H2: S2 N=100.
  - Expected: one manifest per block, with all hashes verified on download.
- [ ] **Step 3 (P5):**
  - Pair each date's K signals with a different, randomly chosen date's forward returns (seed 0).
  - Expected: the mean IC is within ±2 SE of 0.
  - Compute the minimum detectable IC, the mean IC giving Newey-West t = 3 from the P5 IC standard deviation, for
    h = 5 and h = 12.
- [ ] **Step 4:** Commit the `HARNESS.md` addendum and `runs/`, with message `Dev forecasts, P5, power figure`. This
  commit must come before Task 11.

### Task 11: H1 dev evaluation

**Files:** Create `kx/backtest.py`, `tests/test_backtest.py`, `scripts/eval_h1.py`, `RESULTS.md`.

**Interfaces:**
- `tranche_backtest(signal: pd.DataFrame, adj_open: pd.DataFrame, adj_close: pd.DataFrame, upper_limit: pd.DataFrame, h: int, q: float = 0.2, cost_bp: float = 10.0) -> pd.DataFrame`.
  It returns daily `gross`, `net`, `bench` and `excess` series.

- [ ] **Step 1:** Write the failing tests:
  - `test_limit_up_entry_skipped`.
  - `test_cost_per_tranche`. Each new tranche is charged `cost_bp` on its full weight.
  - `test_overlap_average`. With h=3, the daily book equals the mean of 3 tranches.
- [ ] **Step 2:** Implement until they pass. Run `uv run pytest tests/test_backtest.py -v`.
- [ ] **Step 3:** Run `uv run python scripts/eval_h1.py`. For all 6 variants, write to `RESULTS.md`:
  - rank IC with its Newey-West t;
  - Fama-MacBeth incremental t;
  - DM against B0;
  - portfolio excess returns at 5/10/15/25 bp;
  - Deflated Sharpe, PBO, monthly consistency, the before/after 2024-11-05 split, and the ¥3M lot-rounding check.
  - Apply the pass rule in §4 exactly.
- [ ] **Step 4:** If any variant passes, run Kronos-base on the passing variants only (≤ 4; §7) and re-run the
  evaluation with all trials counted.
- [ ] **Step 5:** Commit with message `H1 dev results`.

### Task 12: H2 dev evaluation

**Files:** Create `scripts/eval_h2.py` and `tests/test_vol_target.py`; add `vol_target_backtest` to `kx/backtest.py`;
update `RESULTS.md`.

**Interfaces:**
- `vol_target_backtest(close: pd.Series, f12: pd.Series, capital: float = 1_000_000, target_vol: float = 0.10, max_contracts: int = 2, multiplier: int = 10, fee_side: float = 11.0, tick_value: float = 50.0) -> pd.DataFrame`.
  It returns daily P&L, contracts and costs.

- [ ] **Step 1:** Write the failing tests:
  - `test_contracts_integer_and_capped`.
  - `test_zero_variance_guard`. F = 0 gives 0 contracts, not an error.
  - `test_roll_cost_quarterly`.
- [ ] **Step 2:** Implement until they pass.
- [ ] **Step 3:** Run `uv run python scripts/eval_h2.py`. Write to `RESULTS.md`:
  - QLIKE for V1, V2, EWMA, GARCH and HAR;
  - DM against HAR (Holm), and Mincer-Zarnowitz;
  - the sizing comparison (Sharpe, max drawdown, worst day);
  - the TOPIX secondary results.
  - Apply the pass rule in §5.
- [ ] **Step 4:** Commit with message `H2 dev results`.

### Task 13: Open the holdout once, then STOP for review

- [ ] **Step 1:** Commit `RESULTS.md` naming the single variant per hypothesis to confirm, or "none, killed per §8".
  Use the message `Name holdout variants`.
- [ ] **Step 2:** Only for the named variants:
  - fetch data for 2026-05-01..2026-10-31;
  - extend `SNAPSHOT.json`;
  - build windows and run the forecasts;
  - run the §4/§5 holdout tests once.
- [ ] **Step 3:** Write the final go/no-go and a one-page plain-language summary in `RESULTS.md`. Commit with message
  `Holdout results`. **Stop. Ask the user to review.**
