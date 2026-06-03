# WFO Results — Arc 10 v3.0.2

> **Verdict:** PASS-DEPLOYABLE at the 0.40% operating tier (FundedNext/EET, from-initial basis).
> **Canonical source:** `02_validation/07_canonical_wfo.md` → `results/l_arc_10_v3.0.2_ea_faithful/` (EA-faithful, floating-equity sizing — the live-matched basis). The EET figures below are that run.
> **Legacy source (indicative only):** `results/l_arc_10_v3_0_2_utc_rerun/` (UTC, linear-overlay — not re-run on floating-equity sizing).
> **Methodology:** Walk-forward optimization, 11 in-sample folds (2010–2020) + per-year holdout (2021–2026). DD reported on both references; from-initial is FundedNext's actual MLL basis, trailing is the conservative planning anchor.

## Headline numbers — EET (FundedNext), 0.40% operating tier

| Metric | Value | Fold |
|---|---|---|
| Worst-fold ROI | 14.42% | F9 (2018) |
| Mean fold ROI | 32.48% | — |
| Worst-fold DD (from-initial) | 5.49% | F5 (2014) |
| Worst-fold DD (trailing) | 8.21% | F10 (2019) |
| Worst daily DD | 4.11% | F1 (2010) |
| Holdout ROI | per year (no CAGR) | 2021–2026 |
| Account kills | 0 | all folds + holdout |
| Sign consistency | 11/11 folds + 6/6 holdout yrs positive | — |
| Total in-sample trades | 2,059 | F1–F11 |
| Total holdout trades | 1,093 | 2021–2026 |
| **Total trades** | **3,152** | — |

**Verdict:** PASS-DEPLOYABLE at 0.40% on FundedNext's from-initial basis (5.49% < 8% target); PASS-VIABLE on the conservative trailing basis (8.21% < 10% hard limit). Worst daily 4.11% < 5%, 0 kills. 0.50% FAILS the trailing/daily basis (10.89% / 5.16%) and is a gated upgrade only.

## Per-fold breakdown — EET, 0.40% operating tier (governed)

From `results/l_arc_10_v3.0.2_ea_faithful/per_fold.csv` (governors ON). DD on both references.

| Fold | Year | Trades | ROI | Trailing DD | From-init DD | Daily DD |
|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 25.07% | 8.13% | 3.00% | 4.11% |
| F2 | 2011 | 182 | 28.09% | 7.88% | 2.53% | 3.55% |
| F3 | 2012 | 179 | 31.81% | 4.06% | 0.17% | 1.69% |
| F4 | 2013 | 195 | 46.88% | 7.45% | 0.61% | 2.14% |
| F5 | 2014 | 192 | 32.30% | 6.12% | **5.49%** | 2.50% |
| F6 | 2015 | 190 | 22.26% | 5.71% | 4.73% | 3.08% |
| F7 | 2016 | 171 | 31.96% | 5.60% | 2.35% | 2.47% |
| F8 | 2017 | 190 | 50.83% | 7.19% | 2.40% | 2.17% |
| F9 | 2018 | 176 | **14.42%** | 5.32% | 0.84% | 2.20% |
| F10 | 2019 | 195 | 34.76% | **8.21%** | 1.26% | 2.41% |
| F11 | 2020 | 188 | 38.92% | 6.21% | 4.05% | 3.28% |

Holdout, per year (never collapsed to a single CAGR):

| Year | Trades | ROI | Trailing DD | From-init DD | Daily DD |
|---|---|---|---|---|---|
| 2021 | 212 | 38.79% | 5.95% | 1.04% | 2.50% |
| 2022 | 179 | 24.11% | 7.05% | 1.66% | 2.25% |
| 2023 | 189 | 27.26% | 6.71% | 0.90% | 2.02% |
| 2024 | 191 | 42.86% | 5.57% | 2.62% | 2.46% |
| 2025 | 251 | 52.55% | 5.83% | 3.54% | 3.22% |
| 2026p | 71 | +2.72% (raw ~4-mo partial) | 4.74% | 0.61% | 1.99% |

**Key observations:**

1. **Worst-fold ROI is F9 2018 (14.42%)** — a calm, low-signal year; the system held discipline and still returned double digits.
2. **Worst trailing DD is F10 2019 (8.21%); worst from-initial DD is F5 2014 (5.49%).** FundedNext measures on from-initial, where the worst fold sits comfortably under the 8% in-system target.
3. **Worst daily DD is F1 2010 (4.11%)** — the May 2010 Flash Crash fold; the close-all governor fired but no account was killed (`governor_log.csv`).
4. **F8 2017 was the strongest fold** (50.83% ROI) — strong trending year, runner trails captured well.
5. **All 6 holdout years positive,** maintaining the edge out-of-sample at the 0.40% operating tier.

## UTC (5ers, secondary) — legacy linear-overlay WFO

> **Not re-run on the EA-faithful floating-equity basis.** The 5ers/UTC path is secondary; the canonical numbers above are EET/FundedNext (floating-equity, the live-matched basis). The earlier UTC WFO used linear-overlay sizing and is retained for reference only at `results/l_arc_10_v3_0_2_utc_rerun/`. Treat its figures as indicative, not canonical, until a floating-equity UTC run is produced.

**Same strategy, same signals, different aggregation convention.** Trade count is identical (signal logic is convention-agnostic, 3,152 total); per-fold P&L differs only in which broker-day a trade's DD falls into. Under UTC with swaps ON, the realistic central cost cell breached the 10% hard limit at r_base 0.5%, which is why 5ers deploys at 0.40% (see `05_cost_sweep.md`). No floating-equity re-validation has been run for UTC; the EA-faithful canonical applies to FundedNext/EET only.

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
- **Performance at higher risk levels.** Swept at 0.40% and 0.50%; 0.50% FAILS the conservative trailing/daily basis and is gated (`07_canonical_wfo.md`). Risk scaling has lot-size rounding effects (see `tests/protocol_runtime/test_risk_decoupling_invariant.py` — pre-existing flake noted, not a deployment blocker).

## Cross-arc sign consistency check

For each of the 11 folds + holdout: was the fold positive?

| Convention | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | Holdout | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EET | + | + | + | + | + | + | + | + | + | + | + | + | 12/12 |
| UTC | + | + | + | + | + | + | + | + | + | + | + | + | 12/12 |

100% sign consistency on both conventions. Even at 4× spread stress under the cost sweep, all 11 folds remained positive on EET (see `02_validation/05_cost_sweep.md`).

## What r_base means

The WFO was swept at `risk_per_trade` ∈ {0.40%, 0.50%}. Deployment runs at the **0.40% operating tier**:

- **FundedNext (EET):** Deploy at **0.40%** — worst-fold DD 5.49% from-initial / 8.21% trailing, daily 4.11%, 0 kills. The only level clearing both hard limits on the conservative trailing basis. 0.50% FAILS (trailing 10.89% > 10%, daily 5.16% > 5%) and is a gated upgrade only (`07_canonical_wfo.md`, `04_runbook/09_risk_and_payout_protocol.md`).
- **5ers (UTC):** Deploy at 0.40%. The legacy UTC cost sweep showed r_base 0.5% breaching the 10% hard limit on the realistic central cell; 0.40% is the secondary-path operating level (not re-run on floating-equity sizing).

Risk rationale documented in `02_validation/07_canonical_wfo.md`, `05_cost_sweep.md`, and `05_history/02_decisions_log.md`.

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
