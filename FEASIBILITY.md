# Phase 0: which markets to test Kronos in (8 Nov 2026)

Research covered four markets and three cross-market topics: tax, compute and competition. Independent fact-checking is
**partial**: the run hit the usage limit, so Tokyo-stock and US-stock facts are unchecked (see "How sure we are"). No
performance or price behaviour from July 2024 to October 2026 was looked at; the test windows stay sealed. Full
evidence: `phase0/factbase.json` (303 claims, each with a source).

## Bottom line

Carry forward **two tests**, one for each decision Kronos might improve.

1. **What to hold: Tokyo stocks, cross-sectional ranking.** TOPIX500 names, 5–12 trading-day horizon, long-only.
   - This is the market with the best chance of a real edge. Japan has no momentum premium but has documented
     short-term reversal. Individual investors there trade contrarian. A candle-image model trained on US data has
     carried over to Japan.
   - Costs are low in liquid names (≈5–15 bp round trip).
   - Research data is cheap: J-Quants Light costs ¥1,650/month and includes point-in-time listings and delisted
     names.
   - A 500-stock walk-forward year takes about 2 GPU-hours at 10 sample paths.
   - **But a tradeable book needs about ¥3M.** Stocks trade in 100-share units, and no broker offers odd lots through
     an API. At ¥500k this test is research-only.
2. **How much to hold: Nikkei 225 micro futures, with Kronos only forecasting volatility to size a simple position.**
   - This is the cheapest market to trade: ≈1.3–3.2 bp round trip.
   - Shorting works the same as buying, with no stock borrow. One contract needs ≈¥41–45k margin.
   - Volatility targeting has documented value for equity indices.
   - Practical from ¥1M (1–2 contracts). At ¥500k it is a single contract, either on or off.

**Fallback for test 2: FX through GMO Coin's API.**
- It is the easiest to run: Linux, a GMO Coin membership plus a separate FX account application, free deep
  history, and it works at ¥500k.
- It is also the only fully fact-checked market. But it has the weakest edge prior: there are only 10 currencies
  against JPY, and FX direction is the hardest thing to predict.

**Not carried forward: US stocks.**
- It is the most competitive market, and the one Kronos saw most in pretraining (about 38% of its data).
- An outside test there already found IC +0.022 (not significant).
- It is tradeable only through Webull Japan, at ≈4–9 bp plus 15 sen per USD each way for currency conversion.

**Two warnings for Phase 1:**
- At the IC seen elsewhere (≈0.02), none of these tests can reach t ≥ 3 on the ~460 development days. The Tokyo
  ranking test can detect roughly IC ≥ 0.04, so the pre-registration must state the smallest effect it can detect.
- The Tokyo-stock facts are not independently checked yet. Verify the few we rely on before opening any account.

## Comparison

| | Tokyo stocks (TOPIX500) | Nikkei 225 micro futures | FX via GMO Coin API | US stocks via Webull JP |
|---|---|---|---|---|
| Capital for a sensible book | ≈¥3M (10–15 names). ¥1M marginal (cheap stocks only). ¥500k no | ¥1M (1–2 contracts). ¥3M comfortable. ¥500k marginal (1 contract) | ≈¥300k+. Fine at ¥500k | ¥300–500k (fractional shares) |
| Round-trip cost | ≈5–15 bp liquid names. 30–100+ bp cheap small caps | ≈1.3–3.2 bp | ≈0.6–1.3 bp majors, liquid hours (incl. 0.4 bp API fee). Swap gap ≈0.15–0.2 bp/day | ≈4–9 bp, plus 15 sen/USD conversion each way |
| Shorting | Margin (信用) only. Not at moomoo or Webull | Symmetric, no borrow | Symmetric, swap cost | Margin, with conditions |
| Linux API and paper trading | 立花証券 e支店 (Linux, unrealistic demo). IBKR (Linux, real paper, 8–16 bp/side on small orders, no 特定口座) | IBKR (Linux, paper, ≈¥14–15/side, weekly manual 2FA) | GMO Coin FX API (Linux, no sandbox, so paper is self-simulated). Saxo SIM as an alternative | Webull (Linux, no private paper). moomoo (paper, 0.132%/side) |
| Research data | J-Quants Light ¥1,650/mo: point-in-time, delisted, 5 years | Index bars are enough for the volatility test. Futures bars only on J-Quants Premium (¥16,500/mo) or a one-off DataCube purchase | Free: Dukascopy (2003–), HistData (2000–), GMO bars (2023–) | Sharadar 10Y $19/mo (≈¥2,850), incl. delisted |
| Tax (2026) | 20.315%. 特定口座 源泉あり means no tax return | 20.315%, tax return needed (futures/FX group) | Same as futures | 20.315% (特定口座 at Webull and moomoo). 10% US withholding on dividends |
| Breadth | ~500 liquid names | 1–3 contracts | 10 currencies vs JPY in the test windows | Thousands |
| In Kronos pretraining? | Yes (Japanese stocks) | Japanese indices yes, OSE futures no | Yes (1,000+ pairs, OHLC only) | Yes, heavily |
| GPU per walk-forward year | 500 names: ≈2 h at N=10, ≈22 h at N=100 | Minutes | Minutes | Same as Tokyo at 500 names |
| Edge plausibility rank | 1st | 2nd (volatility sizing only) | 4th (direction) | 3rd (monthly only) |
| Fact-checked | No (limit hit) | 1 of 3 checkers | Fully | No (limit hit) |

## Market notes

**Tokyo stocks.**
- Linux-capable APIs from licensed brokers:
  - 立花証券 e支店 API: free, cash and margin including shorts. Its terms ban accumulating its market data, and
    breaking API versions are frequent.
  - IBKR: realistic paper account, but 0.08% per order with a ¥80 minimum, and no 特定口座.
  - moomoo and Webull: neither can short Japanese stocks or paper-trade them.
- kabuステーションAPI is zero-commission but rules itself out:
  - it is Windows-only;
  - it forces a logout every morning, with no automated login;
  - its terms forbid storing its data or feeding it to external AI.
- Research data should come from J-Quants, not a broker.
- Structural changes to handle:
  - the closing auction moved to 15:30 on 5 Nov 2024, inside the dev window;
  - per-stock tick sizes start 1 Mar 2027;
  - a random close starts 12 Oct 2027.

**Nikkei 225 micro futures.**
- Contract: 10× the index, ¥5 tick, ≈¥11 per side at domestic brokers.
- Day session 8:45–15:45; night session 17:00–6:00.
- 2025 average daily volume was ≈548k contracts.
- 立花's API does not support futures, so IBKR is the Linux route.
- Avoid Growth 250 futures (12–51 bp round trip).
- Kronos direction calls are untestable here: a single instrument at IC 0.02 needs ~22,500 independent observations
  to reach t = 3. Only the volatility use makes sense.

**FX.**
- Costs: GMO charges no commission, but a 0.002% API fee per filled order. It publishes no spread figures; third-party
  numbers are low-confidence.
- Spreads widen sharply around the NY close and early Tokyo hours.
- Pairs: 21 now, but only 14 existed through the test windows, so use only those.
- OANDA's API needs "Gold" status, and MT4/MT5 brokers are Windows-only, so neither is practical.

**US stocks.**
- Webull JP: zero commission since 27 Jul 2026, short selling through the API since 25 Apr 2026, 特定口座 available,
  but no private paper account.
- IBKR becomes competitive only at ¥3M (≈7–12 bp). Its US short selling is still "coming soon".

## Capital sensitivity

- **¥500k:** only FX and US stocks can hold a diversified live book. Futures means one contract, on or off. Tokyo stocks
  are research-only.
- **¥1M:** futures become practical (1–2 contracts). Tokyo stocks remain marginal.
- **¥3M:** a 10–15-name TOPIX500 book is feasible, and futures can be volatility-scaled in 3–6 steps.

## Compute

- Kronos inference has no KV cache, so each new bar re-runs the whole window: 26 GFLOP per sampled path.
- On this laptop's CPU that measured ≈250 ms per path idle, ≈450 ms under normal load.
- A 500-name daily walk-forward year: ≈2 GPU-hours at 10 paths, ≈22 h at 100 paths, on Kaggle's free T4×2 (≈30 h a
  week). Index and FX tests take minutes. All fit at ¥0.
- Kaggle retired the P100 in September 2026.
- An unmerged upstream PR adds a KV cache (≈9.5× fewer FLOPs). It's optional.

## Tax (general information, not tax advice)

- **20.315% flat rate:** listed stocks (Japanese and US), OSE futures, and FX with a registered dealer. Crypto is
  currently taxed as 雑所得 at up to ≈55.9%, with no loss carry-forward. Its 20.315% regime is enacted and expected
  from 1 Jan 2028.
- **No tax return:** only listed stocks can sit in a 特定口座 源泉徴収あり. Futures and FX need a return once your total
  non-salary income passes ¥200k.
- **Losses don't cross groups:** stocks form one group, futures/FX/CFDs another. Each carries losses forward 3 years,
  but only if you file every year.
- **2026–27 deduction cliff:** the basic deduction drops once 合計所得 exceeds ¥4.89M. On a ¥6M salary, declaring
  more than ≈¥530k of extra income costs ≈¥37.8k more. Stock gains in 源泉あり accounts avoid this.

## Ruled out or deprioritised

- **US stocks:** competition, pretraining overlap, and a prior outside null result.
- **FX direction:** the hardest prediction problem, and only 10 names.
- **Growth 250 futures:** costs too high.
- **kabuステーションAPI:** Windows-only, daily manual login, and data terms.
- **OANDA's API:** needs Gold status.
- **MT4/MT5:** Windows-only.
- **Fine-tuning:** out of scope this round anyway. J-Quants Light's history before July 2024 is entirely inside Kronos's
  pretraining period.

## Things that change the Phase 1 design

- **State the minimum detectable effect.** If Kronos's true IC in Japan is ≈0.02, the test will rightly say "not
  shown". It is designed to find IC ≈ 0.04 or more, which is also roughly what costs require.
- **Handle the 15:30 close change** (Nov 2024) when comparing closing prices across the dev window.
- **Use only the FX pairs that existed throughout** the test windows (14 of 21).
- **Paper trading in Phase 5 is self-simulated** for GMO FX, Tokyo stocks via 立花, and Webull. IBKR is the only route
  with a real paper account.

## Questions only you can answer

1. **Capital for this strategy.** ¥500k firm, ~¥1M, or ~¥3M? This decides whether the Tokyo ranking test is worth
   running now, since it is research-only below ≈¥3M.
2. **Brokers.** Are you willing to open IBKR証券 (futures, real paper account) and/or 立花証券 e支店 (Tokyo stocks)? Or
   would you rather use GMO Coin (FX)?
3. **J-Quants Light** at ¥1,650/month from Phase 2. It's within the cap; OK?
4. **A Kaggle account** with phone verification, for the free GPU in Phase 2.

## How sure we are

Each researcher's claims were meant to be checked by 2–3 independent checkers; the usage limit stopped 8 of 18 checks.

| Topic | Claims | Confirmed (2+ checks) | One check | Corrected | Unverifiable | Unchecked |
|---|---|---|---|---|---|---|
| Tokyo stocks | 45 | 0 | 0 | 0 | 0 | 45 |
| Nikkei/OSE futures | 41 | 0 | 32 | 7 | 1 | 1 |
| FX | 36 | 24 | 1 | 11 | 0 | 0 |
| US stocks | 53 | 0 | 0 | 0 | 0 | 53 |
| Tax | 39 | 35 | 0 | 4 | 0 | 0 |
| Compute | 32 | 20 | 3 | 7 | 2 | 0 |
| Competition/edge | 57 | 35 | 11 | 10 | 1 | 0 |

The checker corrections I reviewed are applied here. Every claim's sources and checker notes are in `phase0/factbase.json`.

## Key sources

- **Tokyo stocks** (81 sources): https://www.jpx.co.jp/equities/trading/domestic/03.html, https://www.jpx.co.jp/equities/listing/company-split/index.html, https://www.jpx.co.jp/corporate/research-study/small-investments/t13vrt000001jykl-att/t13vrt00000268sv.pdf, https://github.com/kabucom/kabusapi/issues/1228, https://www.moomoo.com/jp/support/topic7_217, https://www.e-shiten.jp/TorihikiRule/cost/
- **Nikkei/OSE futures** (64 sources): https://www.jpx.co.jp/english/derivatives/products/domestic/225micro-futures/01.html, https://www.jpx.co.jp/english/derivatives/products/domestic/225mini/01.html, https://www.jpx.co.jp/english/derivatives/products/domestic/225futures/01.html, https://www.jpx.co.jp/english/derivatives/products/domestic/mini-topix-futures/01.html, https://www.jpx.co.jp/english/derivatives/products/domestic/tse-growth250futures/01.html, https://www.jpx.co.jp/english/derivatives/products/domestic/jpx-nikkei400futures/01.html
- **FX** (45 sources): https://www.ffaj.or.jp/regulation/customers/, https://coin.z.com/jp/corp/about/company/, https://www.oanda.jp/, https://www.home.saxo/ja-jp, https://api.coin.z.com/fxdocs/, https://coin.z.com/jp/corp/product/info/fx/
- **US stocks** (75 sources): https://prtimes.jp/main/html/rd/p/000000077.000131836.html, https://www.webull.co.jp/products/us-stocks/service, https://www.webull.co.jp/pricing, https://developer.webull.co.jp/apis/docs/about-open-api.md, https://developer.webull.co.jp/apis/docs/authentication/individual-application.md, https://developer.webull.co.jp/apis/docs/reference/common-order-place.md
- **Tax** (39 sources): https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1463.htm, https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2260.htm, https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1476.htm, https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1522.htm, https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1474.htm, https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2250.htm
- **Compute** (27 sources): https://www.kaggle.com/docs/efficient-gpu-usage, https://www.kaggle.com/docs/notebooks, https://github.com/Kaggle/kaggle-cli/pull/1192, https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md, https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md, https://www.kaggle.com/product-feedback/255038
- **Competition/edge** (51 sources): https://www.jpx.co.jp/markets/statistics-equities/investor-type/t13vrt000000gbu4-att/stock_val_1_y25.pdf, https://www.jpx.co.jp/markets/statistics-equities/investor-type/bkk2ed0000005sm7-att/stock_val_1_y23.pdf, https://www.jpx.co.jp/markets/statistics-derivatives/sector/t13vrt000000gb5l-att/Tousi_DV_Y_2025_0106_1230.pdf, https://www.jpx.co.jp/markets/statistics-derivatives/sector/bkk2ed0000005rrk-att/Tousi_DV_Y_2023_0104_1229.pdf, https://www.fsa.go.jp/news/r8/sonota/20260930-2/HFT_ja.pdf, https://www.rieti.go.jp/jp/publications/dp/17e003.pdf
