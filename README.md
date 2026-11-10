# Kronos, tested in Japan

**A pre-registered test of whether a foundation model for price charts gives a trader anything usable on Japanese markets.**

**[Open the interactive report →](https://simetral55.github.io/kronos-zero-shot-japan/)**

[Pre-registration](PREREG.md) · [Harness checks](HARNESS.md) · [Full results](RESULTS.md) · [Market screen](FEASIBILITY.md)

---

## The verdict

**No.** On data it had never seen (July 2024 to April 2026), the open-source
[Kronos](https://github.com/shiyu-coder/Kronos) model gave us nothing a trader could use.

| Question | What we compared it with | Result |
|---|---|---|
| **How much to bet.** Can it warn when the Nikkei 225 is about to get rough? | Three textbook volatility formulas (GARCH, HAR, EWMA) | **Failed.** It was the worst of the four: QLIKE error 1.59 against 0.72 for GARCH. It kept underestimating storms. |
| **What to buy.** Can it rank ~500 large Tokyo stocks by who will do better over the next 5 or 12 days? | Simple price rules (reversal, momentum) and holding every stock | **Failed.** Ranking scores near zero (−0.001 to +0.008, none significant). Nothing beyond the simple rules. Its top picks trailed the plain average after costs in 5 of 6 versions. |

We wrote every rule down and committed it before downloading any test data. Both tests stopped at the development stage
under those rules. The final sealed months were never opened, because nothing passed.

## Why we did this

Kronos is the first open foundation model built specifically for candlesticks. It was trained on more than 12 billion
bars from 45 exchanges, and its paper reports strong forecasting gains. That is exactly the kind of tool a retail
trader is tempted to plug straight into a strategy.

Before risking any money on it, we wanted a plain answer to a plain question: **does it beat the free stuff, after
costs, on data it has never seen?**

We picked Japan because that is where we trade. The [market screen](FEASIBILITY.md) compared Tokyo stocks, Nikkei
futures, FX and US stocks on cost, capital needed, data access and tax before any modelling was done.

## What we tested

- **The model.** Kronos-small (24.7M parameters) at upstream commit `67b630e`, used zero-shot with no fine-tuning. It
  reads 40 daily candles and forecasts the next 12, as in the paper's daily setting.
- **Test 1, volatility (H2).** 435 trading days of Nikkei 225 forecasts.
  - Its 12-day volatility forecasts were compared with GARCH(1,1), HAR-RV and EWMA, using the QLIKE loss and
    Diebold-Mariano tests.
  - We also sized a ¥1,000,000 Nikkei micro-futures position by each forecast.
- **Test 2, stock picking (H1).** 343 trading days and about 497 TOPIX500 stocks a day: 170,613 forecasts, each with
  10 to 50 sampled paths.
  - Six ways of turning forecasts into rankings: average path, share of paths rising, and mean divided by spread, each
    at 5 and 12 days.
  - Each was scored by daily rank IC, by Fama-MacBeth regressions against simple price rules, and by a long-only
    top-quintile portfolio after costs.

## How we kept ourselves honest

The most common way to fool yourself in trading research is to keep adjusting the test until something works. These
are the rules we bound ourselves to.

1. **Rules before results.** [PREREG.md](PREREG.md) fixed every hypothesis, signal, baseline, pass mark and stop rule
   in commit `808d459` (8 Nov 2026, 17:29 JST), before any test data existed on disk.
   - One amendment, switching the stock data to J-Quants' free plan, was also committed before that data was fetched.
   - `git log --reverse` shows the order.
2. **Only unseen months.** Kronos's training data ends in June 2024, so nothing earlier is used for testing.
3. **Beat the simple rules after costs.** Every Kronos signal had to beat free baselines, after realistic trading costs.
   We checked the cost assumption against Tokyo's official tick-size tables before grading.
4. **Test the tester first.**
   - The grading code had to find nothing in random fake prices, and had to find a planted signal.
   - We also checked that GPU and CPU forecasts match exactly, and that two independent Nikkei data sources agree.
5. **Discount for trying several versions.** The pass rule used the Deflated Sharpe ratio and the probability of
   backtest overfitting, so six attempts couldn't produce a lucky winner.
6. **A sealed final exam.** May to October 2026 was reserved for confirming a pass, to be opened once. It never was.

One check surprised us, and we kept it in plain view. Shuffling which *day* gets which returns did not produce zero
(5 of 6 versions came out negative), while shuffling which *stock* gets which signal did. Some stocks stay strong or
weak for months, and that survives a date shuffle. We documented the cause before grading and changed no rule. The
details are in [HARNESS.md](HARNESS.md).

## Results in detail

### Test 1: Nikkei 225 volatility (435 days, Jul 2024 to Apr 2026)

| Forecaster | Mean QLIKE (lower is better) |
|---|---|
| GARCH(1,1) | **0.724** |
| HAR-RV | 0.728 |
| EWMA | 1.005 |
| Kronos, mean of path variances | 1.592 |
| Kronos, spread of path endpoints | 11.99 |

- **Accuracy.** The evidence runs the wrong way for Kronos. The one-sided p-value for "Kronos beats HAR" is 0.97.
  Tested the other way round, Kronos is worse than HAR at the 5% level (p = 0.03).
- **Calibration.** Its forecasts came in below realised volatility 63% of the time, and the spread of its own sample
  paths implied only **about a seventh** of the variance that actually showed up.
- **Sizing.** Sizing by Kronos produced the deepest drawdown (−20.5% against HAR's −15.4%). Holding a constant single
  contract had the best Sharpe ratio of the three.

### Test 2: TOPIX500 ranking (343 days, Nov 2024 to Apr 2026)

| Version | Mean rank IC (t) | Beyond simple rules, t (needs ≥ 3) | Top quintile vs all, per year after 10 bp | Deflated Sharpe (needs ≥ 0.95) |
|---|---|---|---|---|
| Average path, 5 days | +0.006 (0.89) | −0.47 | −4.4% | 0.02 |
| Share rising, 5 days | +0.005 (0.46) | −1.03 | −6.3% | 0.003 |
| Mean / spread, 5 days | +0.004 (0.38) | −1.02 | −6.1% | 0.006 |
| Average path, 12 days | +0.008 (0.51) | +0.05 | +0.04% | 0.20 |
| Share rising, 12 days | −0.000 (−0.02) | −0.58 | −5.5% | 0.01 |
| Mean / spread, 12 days | −0.001 (−0.06) | −0.55 | −4.2% | 0.03 |

- **Power.** The test could detect a rank IC down to roughly 0.01 to 0.03.
- **Consistency.** An independent test of Kronos on US stocks ([upstream issue #375](https://github.com/shiyu-coder/Kronos/issues/375))
  found IC +0.022 with t = 1.39, which is consistent with what we saw.

## What this does not show

Here is what we did *not* test:
- **Fine-tuning.** We used Kronos zero-shot. A fine-tuned or larger Kronos might behave differently.
- **Other bars and markets.** We used daily bars only, not intraday, and Japanese large caps and the Nikkei only.
- **Longer history.** About 18 months of test data, limited by the free data plan's two-year window.

The commit timestamps that order the pre-registration are self-reported, as in any git repository. They were not
anchored to an outside timestamp service.

## Lessons we'd pass on to other traders

- **A forecast isn't a signal.** Kronos's daily return forecasts were slightly closer to reality than "zero" in squared
  error, yet useless for ranking stocks. Test the decision you'll actually trade.
- **Check the model's confidence, not just its guess.** Kronos's sampled futures were far too narrow. Sizing from them
  means betting too big right before the storms.
- **Respect the training cutoff.** A model trained through June 2024 tells you nothing when you test it on 2023.
- **Get your cost floor from the exchange.** A popular daily spread estimator said 37 bp, and tick-size tables said 7
  to 14 bp. On liquid stocks the estimator is mostly noise.
- **Pick your placebo carefully.** A date-shuffle placebo is not a clean null when stocks have persistent traits. Shuffle
  across stocks within each day as well.

## Reproduce it

Everything except the licensed market data is in this repository. Expect about an afternoon of setup and a day of free
Kaggle GPU time.

### You will need

- **Tools:** Python 3.12 and [uv](https://docs.astral.sh/uv/) on Linux (tested on Ubuntu x86-64). The GPU work runs
  on Kaggle.
- **Stock data:** a free [J-Quants](https://jpx-jquants.com/) account (JPX's official data API).
  - Put your API key in `~/.config/kronos-explore/jquants.env` as `JQUANTS_API_KEY=...`, with permissions `600`.
- **GPU:** a [Kaggle](https://www.kaggle.com/) account with GPU and internet enabled (phone-verified), and the Kaggle
  CLI token in `~/.kaggle/`.

### Steps

```bash
# 1. Environment and tests (P1-P3, P6 and the statistics run here; about 1 minute on CPU)
git clone https://github.com/SIMETRAL55/kronos-zero-shot-japan
cd kronos-zero-shot-japan
uv sync && bash scripts/fetch_kronos.sh && uv run pytest

# 2. Data (Nikkei: seconds; J-Quants free plan: about 3 hours at 5 requests/minute)
uv run python scripts/fetch_dev_data.py nikkei
uv run python scripts/fetch_dev_data.py jquants --free

# 3. Placebo P4 and the H2 forecasts on Kaggle
export KAGGLE_USERNAME=your-name
uv run python scripts/kaggle_run.py push p4 '{"n_names":500,"n_days":340,"n_dates":40,"n":10,"T":0.6,"top_p":0.9,"seed":0,"data_seed":42,"batch_rows":4096}'
uv run python scripts/push_h2_dev.py
#    when COMPLETE:  uv run python scripts/kaggle_run.py fetch p4  (and h2dev)
uv run python scripts/eval_p4.py
uv run python scripts/eval_h2.py h2dev

# 4. H1 windows, private dataset, and the five GPU jobs (about 23 GPU-hours on 2x T4)
uv run python scripts/build_h1_windows.py
uv run python scripts/kaggle_dataset.py create
uv run python scripts/push_h1_jobs.py s1     # then s2a, s2b, s2c, s2d
#    fetch each with: uv run python scripts/kaggle_run.py fetch h1s1   (h1s2a ... h1s2d)

# 5. Grade H1 in the pre-registered order, then rebuild the report
uv run python scripts/eval_h1.py costs
uv run python scripts/eval_h1.py p5
uv run python scripts/eval_h1.py p5diag
uv run python scripts/eval_h1.py dev
uv run python report/make_fan_examples.py && uv run python report/build_report.py
uv run python scripts/kaggle_dataset.py delete
```

**Matching our exact dates.** The J-Quants free plan only reaches back two years, and that window moves forward every
day. To reproduce *our* dates exactly (prices from 2024-08-16, forecasts from 2024-11-14), use one month of the Light
plan (¥1,650) and run `scripts/fetch_dev_data.py jquants --start 2024-08-16` in step 2, without `--free`. A rerun on
today's free window tests the same pre-registered rules on slightly later dates. Every
data file's SHA-256 hash is listed in [`data/SNAPSHOT.json`](data/SNAPSHOT.json), so you can check your copy byte for
byte.

### What is not in this repository, and why

- **J-Quants stock data.** Its licence is for private use only, so it is never committed. Fetch your own with the
  scripts above.
- **Nikkei 225 history.** Fetch it from Yahoo Finance and Nikkei Inc. with the scripts. The only exception is 25 short
  windows inside the report's forecast chart.
- **Raw forecast files.** The `.npz` files run to hundreds of MB. Every run's input and output hashes are committed in
  `runs/*/out/manifest.json`, so a rerun can be checked against ours.

## Repository map

| Path | What it is |
|---|---|
| [`PROMPT.md`](PROMPT.md) | The original research brief that started the project |
| [`FEASIBILITY.md`](FEASIBILITY.md), [`phase0/factbase.json`](phase0/factbase.json) | Phase 0: market screen, with ~300 sourced claims |
| [`PREREG.md`](PREREG.md) | The pre-registration: hypotheses, windows, signals, baselines, pass and stop rules, and Amendment 1 |
| [`HARNESS.md`](HARNESS.md) | Every harness check (P1–P7), data provenance, the cost check, P5 and its diagnosis |
| [`RESULTS.md`](RESULTS.md) | Both tests in full, and the overall conclusion |
| `kx/` | The library: Kronos loader (`kronos_env.py`), path-keeping forecasts (`forecast.py`), signals and baselines (`signals.py`), statistics (`stats.py`), backtests (`backtest.py`), J-Quants client (`jquants.py`), universe rules (`universe.py`), Nikkei loaders (`nikkei.py`), synthetic data (`synthetic.py`) |
| `remote/jobs.py`, `scripts/kaggle_run.py` | The GPU job runner: bundles `kx/` into a private Kaggle notebook, multi-GPU, hash manifests |
| `scripts/` | Data fetch, window building, job pushing and the three evaluations (`eval_p4.py`, `eval_h2.py`, `eval_h1.py`) |
| `tests/` | The pytest suite: 46 tests, including the placebos and the no-look-ahead checks |
| `runs/` | Committed results: run manifests, result JSON and per-day series |
| `report/`, `docs/index.html` | The interactive report and the script that builds it from `runs/` |
| `docs/superpowers/plans/` | The implementation plan the code was built from |

## Credits and licence

- **Kronos** is by Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu, Changshui Zhang and Jian Li
  ([paper](https://arxiv.org/abs/2508.02739), [code](https://github.com/shiyu-coder/Kronos), MIT licence). We fetch
  their code at a pinned commit rather than copying it.
- **Data:**
  - Stock data from [J-Quants](https://jpx-jquants.com/) (JPX Market Innovation & Research), under its terms.
  - Nikkei 225 levels from Yahoo Finance, cross-checked against Nikkei Inc.'s published series.
- **This repository's code** is MIT-licensed; see [LICENSE](LICENSE). If you build on the study, [CITATION.cff](CITATION.cff)
  has the citation.

Nothing here is investment advice. It is one careful test of one model, written up so you can check every step.
