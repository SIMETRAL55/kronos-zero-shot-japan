# Pre-registration: Kronos on Tokyo stocks (H1) and Nikkei 225 volatility (H2)

**Version 1, 8 Nov 2026.** Committed before any data from the test windows is downloaded or any model is run on it.
Inputs: `PROMPT.md` (approved task) and `FEASIBILITY.md` (approved market pick). Any change after this commit is a
dated amendment at the end of this file. An amendment may only be made before the data it affects is opened.

## 0. Hypotheses

- **H1, what to hold.** Kronos-small's zero-shot forecasts rank TOPIX500 stocks' forward 5- and 12-day returns better
  than free price-based baselines, by enough to matter after costs.
- **H2, how much to hold.** Kronos-small's zero-shot volatility forecasts for the Nikkei 225 beat HAR-RV, without making
  a volatility-targeted Nikkei micro futures position worse.

A documented "not shown" is a valid result for either.

## 1. Model and inference (fixed)

- **Code and weights.** Upstream code at commit `67b630e`. Weights are `NeoQuasar/Kronos-small` and
  `NeoQuasar/Kronos-Tokenizer-base`, pinned to the Hugging Face revision hashes recorded in `HARNESS.md` at first
  download.
- **Inference mode.** `.eval()` is called explicitly on the tokenizer and the model (upstream issue #415), under
  `torch.inference_mode()`, in fp32, with seeds derived deterministically from (forecast date, batch index).
- **Bars.** Daily bars, lookback 40, horizon 12 (the paper's daily setting). Inputs are open, high, low, close, volume
  and amount (amount = J-Quants TurnoverValue). Indices get OHLC only, with volume and amount as zeros.
- **Timing.** A forecast made at day t uses bars up to and including t's close. The future timestamps are the next 12
  TSE trading days from the JPX calendar.
- **Sampling configs.**
  - **S1** (the paper's investment setting): T = 0.6, top_p = 0.9, 10 paths, averaged.
  - **S2** (the full distribution): T = 1.0, top_p = 1.0. Every path is kept: 50 paths for H1 and 100 for H2. N = 50
    for H1 keeps the dev compute within one week of Kaggle's free GPU quota.
- **Model sizes.** Kronos-base runs only the variants Kronos-small passes on dev (§7). No fine-tuning.

## 2. Data

- **Stocks: J-Quants API, Light plan.**
  - Split-adjusted daily OHLCV, TurnoverValue and limit flags.
  - The listed-issue master as of each date, delisted issues included.
  - The JPX trading calendar and the TOPIX index.
- **Universe U(t).** Issues whose ScaleCategory on date t is TOPIX Core30, Large70 or Mid400.
  - **Fallback**, if ScaleCategory as of a date is unavailable: the 500 Prime issues with the highest median
    TurnoverValue over the prior 60 trading days.
  - **Excluded at t:** fewer than 60 trading days of history, or no bar on t.
  - **Days with fewer than 300 valid names** are dropped.
- **Indices.**
  - Nikkei 225 daily OHLC is the primary H2 series. Its source is chosen in Phase 2 and cross-checked against a second
    source on 20 random dates (difference ≤ 0.05%).
  - TOPIX daily OHLC from J-Quants is the secondary H2 series.
- **Integrity.**
  - Raw data is never committed (licence). The SHA-256 of every data file is committed in `data/SNAPSHOT.json` before
    evaluation.
  - Phase 2 and dev work fetch data only up to 2026-04-30. Holdout data is fetched in §9's step only.

## 3. Windows

- **Pretraining cutoff:** June 2024. No test uses an outcome before 2024-07-01.
- **Dev:** forecast dates from 2024-07-01 whose whole forward window ends by 2026-04-30. Dates whose outcome would cross
  into the holdout are dropped.
- **Holdout (sealed):** forecast dates from 2026-05-01 whose forward window ends by 2026-10-31. It is opened once, under
  §9.
- **Baselines** (EWMA, GARCH, HAR, B1–B4) may fit on any data before t, back to the earliest available date (at least
  500 trading days before 2024-07-01).
- **Close-time change.** The TSE close moved to 15:30 on 2024-11-05. Dev results are also reported split at that date,
  as a robustness check only.

## 4. H1: ranking test (Tokyo stocks)

**Signals.** For horizon h ∈ {5, 12}, with Ĉ = predicted close and C = actual close:
- **K1:** the S1 mean path, Ĉ(t+h) / C(t) − 1.
- **K2:** P(up), the share of S2 paths with Ĉ(t+h) > C(t).
- **K3:** forecast Sharpe, the mean of the S2 path returns to h divided by their standard deviation.

That gives 6 Kronos-small variants.

**Forward return:** r(t, h) = AdjClose(t+h) / AdjOpen(t+1) − 1.

**Baselines (controls)**, each computed from data up to t:
- **B0:** persistence, a zero predicted return. Used only for the point-forecast Diebold-Mariano (DM) test.
- **B1:** reversal to the 40-day mean, −(C_t / mean(C_{t−39..t}) − 1). This controls for Kronos's known pull toward the
  window mean (upstream #422).
- **B2:** 1-month reversal, −(C_t / C_{t−21} − 1).
- **B3:** 12-1 momentum, C_{t−21} / C_{t−252} − 1.
- **B4:** 5-day reversal, −(C_t / C_{t−5} − 1).

A missing baseline value gets the cross-sectional median rank.

**Metrics** (Newey-West lags = h throughout):
1. **Daily rank IC:** the Spearman rank correlation of the signal with r(t, h) across U(t). Report the mean and its
   Newey-West t.
2. **Incremental test (Fama-MacBeth):** each day, regress the rank-normal score of r(t, h) on the rank-normal scores of
   K, B1, B2, B3 and B4 (OLS). Report K's mean coefficient and its Newey-West t.
3. **Point accuracy:** a DM test of K1 against B0 on squared error. Reported only.
4. **Portfolio:**
   - Long-only top quintile, equal weight. Each day's tranche is held from the t+1 open to the t+h close; the book is
     the average of the last h tranches.
   - Excess return is measured against the equal-weight U(t) with the same tranche structure.
   - Costs: 10 bp per round trip on every tranche (5, 15 and 25 bp reported as sensitivity). A name whose t+1 open is
     limit-up is not bought.
   - Tax is reported as 20.315% of positive yearly net gains. Proportional tax leaves the Sharpe ratio unchanged, so the
     Deflated Sharpe below uses after-cost, pre-tax daily excess returns.
5. **Deflated Sharpe** (Bailey & López de Prado 2014): the number of trials N is all H1 variants evaluated so far, with
   the variance of their Sharpe ratios.
6. **PBO** via CSCV (Bailey et al. 2017): 16 blocks, across all H1 variants.
7. **Monthly consistency:** the share of calendar months whose mean rank IC is above zero.

**Power.** The minimum detectable mean IC (the IC at which Newey-West t = 3 on the dev sample) is computed from the
shuffled-date placebo's IC standard deviation. It is written into `HARNESS.md` before any real dev IC is computed. An
IC of about 0.02, the size seen on US stocks (#375), is expected to come out as "not shown".

**Dev pass, per variant (all four required):**
- Fama-MacBeth incremental t ≥ 3.0;
- Deflated Sharpe ≥ 0.95;
- PBO ≤ 0.2;
- monthly consistency ≥ 60%.

**Holdout confirmation**, run on the single best passing variant by incremental t:
- mean holdout rank IC > 0, with one-sided Newey-West p < 0.05;
- after-cost excess return > 0.

## 5. H2: volatility test (Nikkei 225; TOPIX secondary)

**Target:** RV12(t) = Σ_{i=1..12} ln(C_{t+i} / C_{t+i−1})², close-to-close, so overnight gaps count.

**Kronos forecasts**, from S2 paths (N = 100):
- **V1:** the mean over paths of Σ_{i=1..12} ln(Ĉ_{t+i} / Ĉ_{t+i−1})², with Ĉ_t = C_t.
- **V2:** the variance across paths of ln(Ĉ_{t+12} / C_t).

**Baselines** (expanding window, refit at each month start; the forecast at t uses data up to t only):
- **EWMA:** λ = 0.94. The 12-day forecast is 12 × σ²_{t+1}.
- **GARCH(1,1):** Student-t innovations, summed 12-step variance.
- **HAR-RV:** OLS of RV12(t)/12 on [1, r_t², mean(r²) over the last 5 days, mean(r²) over the last 22 days], fitted only
  on targets that end before t. The forecast is 12 × the fitted value.

**Metrics:**
- **QLIKE loss** = RV/F − ln(RV/F) − 1.
- **DM test**, one-sided, Kronos against HAR-RV, Newey-West lag 12, Holm-adjusted across V1 and V2. EWMA and GARCH are
  also reported.
- **Mincer-Zarnowitz regression** of ln RV12 on ln F, reported with HAC standard errors.

**Economic check** (reported; one no-harm rule):
- Setup: a long Nikkei 225 micro position, ¥1,000,000 capital, 10% annual volatility target.
- Contracts: n_t = clip(round(0.10/√245 × 1,000,000 / (√(F/12) × 10 × C_t)), 0, 2).
- Costs: ¥11 commission plus 1 tick (¥50) of slippage per contract per side, and a quarterly roll costing 1 tick per
  leg. Index returns stand in for futures returns (basis ignored).
- Compared: Kronos sizing, HAR sizing, and a constant 1 contract (Sharpe, maximum drawdown, worst day).
- About 460 dev days cannot separate Sharpe ratios statistically, so this check is descriptive.

**Dev pass:**
- the best of V1 and V2 beats HAR-RV with Holm-adjusted DM p < 0.05, on Nikkei 225;
- AND the maximum drawdown with Kronos sizing is ≤ 1.2 × the maximum drawdown with HAR sizing.

**Holdout confirmation**, same variant: mean QLIKE difference < 0, with one-sided DM p < 0.10.

## 6. Harness checks and placebos (Phase 2; the placebos with no data run before any dev result)

| # | Check | Pass |
|---|---|---|
| P1 | Inference mode | Every module has `training == False`; two runs with the same seed give identical forecasts |
| P2 | Upstream regression test (`tests/test_kronos_regression.py` @ 67b630e) | Passes in our environment |
| P3 | Path wrapper, n = 1 | Equals upstream `predict_batch(sample_count=1)` under the same seed |
| P4 | Random walks: 500 synthetic stocks, 40 dates, K1 under S1 | Mean rank IC within ±2 SE of 0. The pull-to-mean bias on drifting walks is recorded |
| P5 | Shuffled dates (run on dev forecasts, before any real IC) | Mean IC within ±2 SE of 0. Its IC standard deviation gives the power figure in §4 |
| P6 | Positive control: synthetic signal with a planted IC of 0.05 | Recovered IC 0.05 ± 0.01, Newey-West t > 3 |
| P7 | Reproduce upstream #375 (182 US stocks, its own setup) | IC within +0.022 ± 0.015. If its ticker list is unavailable, record "not reproducible"; this does not block |

If P1–P4 or P6 fail and can't be fixed in 2 attempts, the study stops and reports why.

## 7. Variants and logging

- **H1:** 6 variants (K1–K3 × h ∈ {5, 12}). **H2:** 2 variants (V1, V2).
- **Kronos-base** runs only the variants Kronos-small passes on dev, at most 4. The total stays ≤ 12.
- **Logging:** every forecast run writes a manifest (input hash, output hash, model revision, config, git commit,
  runtime). Every variant evaluated is reported, failures included.
- **No re-tuning:** T, top_p, lookback, horizon and costs are not changed after dev results are seen. A change means a
  new pre-registration with a new, later holdout.

## 8. Kill rules

- No H1 variant passes dev: H1 stops. No holdout, no tuning, write-up only.
- H2 fails dev: H2 stops the same way.
- Harness failures: as in §6.

## 9. Opening the holdout

The holdout is opened only after:
1. the dev results are committed; and
2. a separate commit names the single variant per hypothesis to confirm.

Then holdout data is fetched, forecasts are made once, and the one evaluation is run.
- **Pass:** go to Phase 4 (algorithm and risk design) for that hypothesis only.
- **Fail:** a documented negative result.

## 10. Cost and capital assumptions to confirm in Phase 2

- **H1:** 10 bp round trip for TOPIX500 names at a zero-commission route. This comes from unverified Phase 0 research.
  If Phase 2's broker check finds the realistic cost is above 15 bp, the cost moves to the checked figure by amendment,
  before dev results are opened.
- **H1 capital realism** (reported only): ¥3,000,000, top 15 names, 100-share lots.
- **H2:** the micro futures figures above (Phase 0, checked once).

## Amendments

### Amendment 1 (2026-11-08): H1 data from the J-Quants Free plan

Made before any stock data has been fetched. The H2 results are known by now, but they say nothing about H1's stocks.

The user chose a no-cost first run of H1. Per J-Quants' data specification (jpx-jquants.com/en/spec/data-spec), the Free
plan gives the listed-issue master and daily prices with 2 years of history and a 12-week delay, over the API only, at
5 requests per minute. The TOPIX index is not included.

1. **Dev window for H1.** Forecast dates now start at the first trading day with at least 60 trading days of Free-plan
   history, about mid-November 2024. They end, as before, at dates whose outcomes end by 2026-04-30. The exact first
   date is recorded in `HARNESS.md` when the data is fetched. All pass rules are unchanged. The power figure (§4) is
   computed on this shorter window.
2. **B3 (12-1 momentum)** needs 252 days of history, so it is missing for most of the dev window. Missing values take
   the median rank, as §4 already says. Its coverage is reported.
3. **TOPIX** was only the secondary H2 series. H2 has stopped, so this has no effect.
4. **Holdout.** The dates are unchanged. The Free plan has the full holdout (outcomes to 2026-10-31) from about
   2027-01-23. Alternatively, one month of Light can be bought if H1 passes dev.
5. **Later Light data.** If Light data is bought later, the dev pass/fail stays on this window. Extra earlier dates may
   be reported only as supplementary.
