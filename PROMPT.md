# Kronos: find the edge, or prove there isn't one (deeper, not bigger)

**Question.** Does Kronos (github.com/shiyu-coder/Kronos, pinned at `67b630e`) add tradeable information beyond free
baselines, in a market other than crypto that I can trade from Japan? If yes, which decision does it improve: *what* to
hold (return ranking) or *how much* to hold (volatility forecasts)? A documented "no" is a valid result. Only if
something passes: design the algorithm, its risk management and a forward test.

**Context.**
- My method: structural edges, costs first, per-day stability, pre-registration, paper-vs-live calibration,
  Off/Measure/On. Already rejected: chart patterns, candlestick/LightGBM.
- Kronos facts:
  - Pretraining ends June 2024.
  - `KronosPredictor` never calls `.eval()`, so with the pinned huggingface_hub 0.33.1 dropout stays on (#415).
  - It averages the sample paths away (#395).
  - Normalisation leakage is probably baked into the weights and pulls forecasts toward the recent mean (#307, #422).
  - Outside tests: cross-sectional IC +0.022, t 1.39, fails after costs (#375); loses to "last price" on AAPL (#354).
- My constraints (edit):
  - Licensed brokers with an API only.
  - A small retail account (the screen judges ¥0.5M, ¥1M and ¥3M).
  - Ask before spending over ¥3,000/month on GPU or data.
  - The laptop has 14 GB RAM and no GPU.

**Phase 0: pick at most 2 markets from constraints (no modelling).** Compare JP stocks/ETFs, OSE index futures
(micro/mini), FX, and US stocks via a Japan-resident API broker on:
- capital needed for a sensible book, given lot sizes;
- round-trip cost;
- shorting;
- API access;
- point-in-time data including delisted names;
- tax (20.315% separate taxation vs crypto);
- GPU-hours for one walk-forward year.

→ `FEASIBILITY.md`: one table, the pick, and why.

**Phase 1: pre-register before opening any test data.** → `PREREG.md`, committed.
- Windows: dev 2024-07-01 → 2026-04-30; sealed holdout 2026-05-01 → 2026-10-31, opened once.
- Model: Kronos-small zero-shot. Kronos-base only if small passes. No fine-tuning this round.
- Setting: the paper's daily setting (lookback 40, horizon 12).
- Signals: mean forecast return, plus P(up) and dispersion from the individual sample paths at top_p = 1.0.
- Baselines: last price; reversal to the 40-day mean (controls for the mean-pull bias); 12-1 momentum. For
  volatility: EWMA, GARCH(1,1), HAR-RV.
- At most 12 variants, every one logged.
- Default pass rules:
  - incremental daily RankIC over the baselines with Newey-West t ≥ 3;
  - volatility QLIKE beats HAR-RV with Diebold-Mariano p < 0.05;
  - Deflated Sharpe ≥ 0.95 after costs and tax;
  - PBO ≤ 0.2.
- A kill rule.

**Phase 2: prove the harness on someone else's numbers.** → `HARNESS.md`.
- Pin versions, force `.eval()`, set seeds, and pass the repo's own regression test.
- Reproduce #375's IC (≈ +0.022) on its own setup, or one return-IC number from the paper.
- Placebos must show nothing: random walks with and without drift, and shuffled dates.
- Leakage audit: point-in-time data, survivorship, corporate actions, signal at close t → fill at the next open.

**Phase 3: evaluate.** → `RESULTS.md` with a go/no-go and the variant log.
- Run on the dev window, then open the holdout once.
- Batch inference off the laptop: Kaggle's free GPU first, Colab if needed. Estimate GPU-hours before each run.
- Report:
  - per-day IC, and Fama-MacBeth regressions with the baselines;
  - Diebold-Mariano against last price;
  - calibration (PIT, interval coverage, Brier score);
  - volatility QLIKE and Mincer-Zarnowitz;
  - a backtest at my capital with costs, tax, lot rounding and price limits.
- If nothing passes, stop and write down what evidence would reopen the question.

**Phase 4: only if something passes, the algorithm and its risk.** → `RISK.md` + a tested risk module.
- Build around whatever survived. Kronos may end up only sizing a simple strategy.
- Write the risk module test-first:
  - volatility-targeted sizing;
  - a capped fractional Kelly fed only calibrated probabilities;
  - position and gross limits;
  - daily-loss and drawdown halts;
  - gap and limit-down handling;
  - stress tests on 5 Aug 2024 and the window's worst days;
  - a model-health gate that cuts size to zero when rolling live IC or calibration falls below the pre-registered
    floor.
- Roll out Off/Measure/On.

**Phase 5: the forward-test protocol.** → `FORWARD.md`.
- Set the length from the minimum track record length.
- Hash each day's predictions and post the hash to an outside service before the open, as an independent timestamp.
- Paper first, then the smallest live size, reconciled to the broker's statements.
- Fix the promotion and kill criteria now.

**Out of scope.** Live orders, dashboards, fine-tuning, a third market, pushing to any remote.

**How to work.**
- Plan with superpowers:writing-plans. Execute with superpowers:executing-plans, one task at a time: this laptop has
  run out of memory with parallel jobs.
- Use anthropic-skills:deep-research for Phase 0 only.
- Use statistical-analysis and statsmodels for the tests.
- Use probabl-skills:audit-ml-pipeline for the leakage audit.
- Use the kaggle or colab skill for GPU runs.
- Use superpowers:test-driven-development for the backtester and the risk module.
- Use superpowers:verification-before-completion before any claim.
- Stop for my review after Phases 0, 2 and 3.
- Work in `~/kronos-explore` as its own git repo, with no remote.
- Finish with a one-page plain-language summary with numbers.
