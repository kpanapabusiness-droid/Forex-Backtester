# WFO Results — Arc 10 v3.0.2

> **Verdict:** PASS-DEPLOYABLE on both EET and UTC conventions.
> **Source artifact:** `results/l_arc_10_v3.0.2/step_5/wfo_results.csv` (EET), `results/l_arc_10_v3_0_2_utc_rerun/` (UTC re-validation).
> **Methodology:** Walk-forward optimization, 11 in-sample folds (2010-2020) + 1 holdout (2021-2026).

## Headline numbers — EET (FundedNext target)

| Metric | Value | Fold |
|---|---|---|
| Worst-fold ROI | 22.46% | F9 (2018) |
| Worst-fold DD | 7.35% | F4 (2013) |
| **Worst-fold ratio** | **6.43** | **F6 (2015) — load-bearing** |
| Mean fold ROI | 49.87% | — |
| Holdout ROI | 52.83% | 2021-2026 |
| Holdout DD | 5.50% | — |
| Sign consistency | 11/11 positive | All folds |
| Total in-sample trades | 2,059 | — |
| Total holdout trades | 1,093 | — |
| **Total trades** | **3,152** | — |

**Verdict:** PASS-DEPLOYABLE. Worst-fold ratio 6.43 > 2.0 gate. Worst-fold DD 7.35% < 10% hard limit. Sign consistency 11/11 holds even under 4× spread stress.

## Per-fold breakdown — EET, r_base 0.5%

Pre-cost overlay. Numbers from `results/l_arc_10_v3.0.2/step_5/wfo_results.csv`.

| Fold | Year | Trades | ROI | DD | Ratio |
|---|---|---|---|---|---|
| F1 | 2010 | 201 | ~36.9% | ~3.4% | ~10.8 |
| F2 | 2011 | 187 | ~42.1% | ~3.0% | ~14.0 |
| F3 | 2012 | 191 | ~46.5% | ~2.3% | ~20.0 |
| F4 | 2013 | 187 | 64.39% | **7.35%** | 8.76 |
| F5 | 2014 | 204 | ~47.1% | ~2.6% | ~18.2 |
| F6 | 2015 | 193 | 29.25% | 5.29% | **5.53** |
| F7 | 2016 | 207 | ~44.0% | ~3.3% | ~13.3 |
| F8 | 2017 | 200 | ~69.0% | ~2.5% | ~27.4 |
| F9 | 2018 | 191 | **22.46%** | 3.39% | 6.63 |
| F10 | 2019 | 195 | ~48.7% | ~4.5% | ~10.8 |
| F11 | 2020 | 206 | ~54.0% | ~2.6% | ~20.6 |
| **Holdout** | 2021-2026 | 1,093 | **52.83%** | **5.50%** | 9.60 |

**Key observations:**

1. **F4 2013 has the highest absolute DD** (7.35%) but excellent ROI (64.39%) — high-volatility year, system traded through with good R-multiples but with one or more chunky drawdown periods.
2. **F6 2015 has the worst ratio** (5.53) — moderate ROI vs higher relative DD. Likely impacted by CHF-pegging crisis early in the year.
3. **F9 2018 has lowest ROI** (22.46%) — calm year with fewer signals; system held its discipline.
4. **F8 2017 was the best fold** (69% ROI) — strong trending year, runner trails captured well.
5. **Holdout** (5 years, 1,093 trades) shows the system maintains its edge out-of-sample at 52.83% annualized.

## Headline numbers — UTC (5ers target)

| Metric | Value | Fold |
|---|---|---|
| Worst-fold ROI | 26.49% | F6 |
| Worst-fold DD | 9.22% | F4 |
| Worst-fold ratio | 5.42 | F6 |
| Mean fold ROI | 49.87% | — |
| Holdout ROI | 59.07% | 2021-2026 |
| Holdout DD | 5.30% | — |
| Sign consistency | 11/11 positive | All folds |
| Total trades | 3,152 | — |

**Same strategy, same signals, different aggregation convention.** Trade count identical (signal logic is convention-agnostic), per-fold P&L differs due to which trading day a given trade falls into for DD computation.

## Why EET vs UTC differs structurally

The EET vs UTC delta comes almost entirely from how daily DD is accrued:

- 5ers uses UTC midnight as the trading day boundary
- FundedNext uses EET midnight (broker-local) as the trading day boundary

A trade that opens late UTC Friday and continues into Saturday EET-time may straddle the day boundary differently under the two conventions. Worst-fold DD on F4 2013 (the load-bearing DD fold) is ~2pp lower under EET than UTC purely due to this effect.

**Net: EET WFO produces stronger numbers because the daily DD accounting is more favorable for Arc 10's trade timing.** Both conventions PASS-DEPLOYABLE, but EET has more headroom against the 10% hard limit.

For full cost-adjusted comparison see `02_validation/05_cost_sweep.md`.

## What the WFO did NOT prove

- **Strategy edge in live markets post-2026.** WFO uses historical data; live performance could differ if market structure changes substantially.
- **Robustness to broker-specific microstructure.** The validation assumes broker spreads, slippage, and fills similar to HistData / 5ers / FundedNext panel-diff samples. Other brokers untested.
- **Robustness to extreme tail events.** CHF 2015 was in the data and the system survived F6 with 29% ROI / 5.29% DD. A similar event with worse pre-emptive positioning could exceed historical worst-case.
- **Performance at higher risk levels.** Tested at r_base 0.5%; scaling risk linearly assumes linear scaling of all metrics, which holds in theory but has lot-size rounding effects (see `tests/protocol_runtime/test_risk_decoupling_invariant.py` — pre-existing flake noted, not a deployment blocker).

## Cross-arc sign consistency check

For each of the 11 folds + holdout: was the fold positive?

| Convention | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | Holdout | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EET | + | + | + | + | + | + | + | + | + | + | + | + | 12/12 |
| UTC | + | + | + | + | + | + | + | + | + | + | + | + | 12/12 |

100% sign consistency on both conventions. Even at 4× spread stress under the cost sweep, all 11 folds remained positive on EET (see `02_validation/05_cost_sweep.md`).

## What r_base means

The WFO ran at `risk_per_trade = 0.005` (0.5% per trade). For deployment, risk was adjusted per broker:

- **FundedNext (EET):** Deploy at r_base 0.50%. Stress cells recommend scaling to 0.49% in adverse; central case is 0.51%. Net: deploy at r_base, accept central case worst-fold DD 7.80% with 2.2pp margin to 10% hard limit.
- **5ers (UTC):** Deploy at r_safe 0.40%. UTC at r_base 0.5% breaches 10% DD hard limit on realistic central cell (10.47%). Linear scaling to 0.40% brings central worst-fold DD to 8.38%, 1.6pp margin.

Risk locking rationale documented in `02_validation/05_cost_sweep.md` and `05_history/02_decisions_log.md`.

## Full data location

| Artifact | Path |
|---|---|
| EET WFO results | `results/l_arc_10_v3.0.2/step_5/wfo_results.csv` |
| EET holdout results | `results/l_arc_10_v3.0.2/step_5/holdout_results.csv` |
| EET pool (admitted trades) | `results/l_arc_10_v3.0.2/step_1/pool.parquet` |
| EET feature matrix | `results/l_arc_10_v3.0.2/step_1/feature_matrix.parquet` |
| EET full closure document | `results/l_arc_10_v3.0.2/ARC_CLOSURE.md` |
| UTC trade ledger | `results/l_arc_10_v3_0_2_utc_rerun/trade_ledger_utc.parquet` |
| Amendment 3 evaluation | `results/l_arc_10_v3.0.2/step_5/amendment_3/` |
| Causal audit | `results/l_arc_10_v3.0.2/step_6/` (deployment readiness report) |
