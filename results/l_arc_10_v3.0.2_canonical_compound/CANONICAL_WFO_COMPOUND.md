# Arc 10 v3.0.2 — CANONICAL WFO, COMPOUND SIZING (FundedNext, EET, 3.5R)

> **⛔ SUPERSEDED by [`_final_canonical`](../l_arc_10_v3.0.2_final_canonical/FINAL_CANONICAL_WFO.md)** (fixed-initial sizing + `daily_ref=initial`-resetting = the deploy-faithful basis matching the live EA post-FIX-2b). This run used a **closed-equity compound (non-resetting daily)** basis, retained for reference only. The interim `_ea_faithful` pointer below is itself also superseded by `_final_canonical`.

> ⚠️ **SUPERSEDED (2026-05-31) — CLOSED-EQUITY SIZING BASIS.** This run sized off
> *closed* equity (`e_bal`, excluding floating). The live EA sizes off
> `ACCOUNT_EQUITY` **including floating open P&L** (`PositionManager.mqh:143`), so the
> basis-correct canonical is now
> [`results/l_arc_10_v3.0.2_ea_faithful/EA_FAITHFUL_WFO.md`](../l_arc_10_v3.0.2_ea_faithful/EA_FAITHFUL_WFO.md).
> Cite the EA-faithful run for every figure. Good news: the floating-vs-closed
> procyclical delta is **benign** (+0.16pp worst-fold trailing DD at 0.50% gov-on),
> so the closed-equity numbers below are close to the live basis; the **launch-at-0.40%
> decision stands** (0.40% trailing 8.21%, daily 4.11%; 0.50% trailing 10.89%, daily 5.16%).

> **Supersedes the linear canonical run** ([`results/l_arc_10_v3.0.2_canonical/`](../l_arc_10_v3.0.2_canonical/CANONICAL_WFO.md)). Same structure; corrected sizing basis. The linear run sized fixed-%-of-INITIAL; the live EA sizes **fixed-%-of-CLOSED-EQUITY at each trade's open** (concurrent opens share the snapshot; equity steps only on close; per-fold reset, compound within fold). Every ROI/DD recomputed via `simulate_continuous(compound=True)` run per fold — no re-implementation. v3.0.2 LOCKED; governors EA-faithful, not tuned. Costs ON (cell 5); r_base {0.40%,0.50%}; EET; frame sha `05dea9…9ee58a`; deterministic (two-run sha identity); PR-gated.

> **Validation gate PASSED** (frame integrity): linear zero-cost + governors-OFF reproduces the **9.22%** portfolio worst-fold trailing DD (abort-if-not).

> **NO CAGR anywhere.** Holdout scored as six independent annual-reset folds (2021–2025 full years + 2026 partial), each reset to 1.0 and compounding within its year — directly comparable to F1–F11. **2026 is a RAW partial return (~4 months), never annualised.** Full-year folds annualised over their own ~1y span (= annual return, not a multi-year CAGR).

## 1. Canonical matrix (compound; 2 risk × 2 governance; both DD refs)

| risk% | governance | worst-fold ROI% | mean-fold ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict_trailing | verdict_from_initial |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | off | 14.41 | 32.32 | 8.20 | 5.52 | 4.01 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.40 | on | 14.41 | 32.32 | 8.20 | 5.52 | 4.01 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | off | 18.21 | 41.59 | 10.15 | 6.89 | 5.00 | 0 | FAIL | PASS-DEPLOYABLE |
| 0.50 | on | 18.21 | 40.72 | 10.73 | 6.89 | 5.00 | 0 | FAIL | PASS-DEPLOYABLE |

> Per-fold = ceiling (each fold a fresh-tier reset, compounding within its year). No continuous-block row: the holdout is scored per-year (§2), not as one block. Worst-/mean-fold are over the 11 search folds F1–F11; holdout years are reported in §2/§3, not gated.

## 2. Full per-fold + per-holdout-year (compound, annual-reset — one basis)

**0.40% r_base — governors OFF** (search worst-fold ROI 14.41% / mean 32.32% / worst trailing DD 8.20% / worst from-init 5.52%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 24.59 | 8.01 | 3.01 | 4.01 | 0 | 0 |
| F2 | 2011 | 182 | 28.25 | 7.83 | 2.52 | 3.53 | 0 | 0 |
| F3 | 2012 | 179 | 31.63 | 4.00 | 0.17 | 1.67 | 0 | 0 |
| F4 | 2013 | 195 | 46.45 | 7.50 | 0.61 | 2.13 | 0 | 0 |
| F5 | 2014 | 192 | 32.26 | 6.15 | 5.52 | 2.51 | 0 | 0 |
| F6 | 2015 | 190 | 22.11 | 5.65 | 4.69 | 3.03 | 0 | 0 |
| F7 | 2016 | 171 | 31.78 | 5.60 | 2.35 | 2.48 | 0 | 0 |
| F8 | 2017 | 190 | 50.67 | 7.13 | 2.35 | 2.16 | 0 | 0 |
| F9 | 2018 | 176 | 14.41 | 5.31 | 0.80 | 2.18 | 0 | 0 |
| F10 | 2019 | 195 | 34.41 | 8.20 | 1.25 | 2.39 | 0 | 0 |
| F11 | 2020 | 188 | 38.92 | 6.16 | 4.01 | 3.31 | 0 | 0 |
| 2021 | 2021 | 212 | 38.29 | 5.89 | 1.04 | 2.52 | 0 | 0 |
| 2022 | 2022 | 179 | 24.27 | 7.09 | 1.67 | 2.25 | 0 | 0 |
| 2023 | 2023 | 189 | 27.02 | 6.64 | 0.90 | 1.99 | 0 | 0 |
| 2024 | 2024 | 191 | 42.48 | 5.58 | 2.62 | 2.45 | 0 | 0 |
| 2025 | 2025 | 251 | 51.90 | 5.82 | 3.53 | 3.22 | 0 | 0 |
| 2026p | 2026 | 71 | 2.70 | 4.70 | 0.61 | 1.99 | 0 | 0 |

**0.40% r_base — governors ON** (search worst-fold ROI 14.41% / mean 32.32% / worst trailing DD 8.20% / worst from-init 5.52%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 24.59 | 8.01 | 3.01 | 4.01 | 0 | 1 |
| F2 | 2011 | 182 | 28.25 | 7.83 | 2.52 | 3.53 | 0 | 1 |
| F3 | 2012 | 179 | 31.63 | 4.00 | 0.17 | 1.67 | 0 | 0 |
| F4 | 2013 | 195 | 46.45 | 7.50 | 0.61 | 2.13 | 0 | 0 |
| F5 | 2014 | 192 | 32.26 | 6.15 | 5.52 | 2.51 | 0 | 0 |
| F6 | 2015 | 190 | 22.11 | 5.65 | 4.69 | 3.03 | 0 | 0 |
| F7 | 2016 | 171 | 31.78 | 5.60 | 2.35 | 2.48 | 0 | 0 |
| F8 | 2017 | 190 | 50.67 | 7.13 | 2.35 | 2.16 | 0 | 0 |
| F9 | 2018 | 176 | 14.41 | 5.31 | 0.80 | 2.18 | 0 | 0 |
| F10 | 2019 | 195 | 34.41 | 8.20 | 1.25 | 2.39 | 0 | 0 |
| F11 | 2020 | 188 | 38.92 | 6.16 | 4.01 | 3.31 | 0 | 0 |
| 2021 | 2021 | 212 | 38.29 | 5.89 | 1.04 | 2.52 | 0 | 0 |
| 2022 | 2022 | 179 | 24.27 | 7.09 | 1.67 | 2.25 | 0 | 0 |
| 2023 | 2023 | 189 | 27.02 | 6.64 | 0.90 | 1.99 | 0 | 0 |
| 2024 | 2024 | 191 | 42.48 | 5.58 | 2.62 | 2.45 | 0 | 0 |
| 2025 | 2025 | 251 | 51.90 | 5.82 | 3.53 | 3.22 | 0 | 0 |
| 2026p | 2026 | 71 | 2.70 | 4.70 | 0.61 | 1.99 | 0 | 0 |

**0.50% r_base — governors OFF** (search worst-fold ROI 18.21% / mean 41.59% / worst trailing DD 10.15% / worst from-init 6.89%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 31.22 | 9.93 | 3.75 | 5.00 | 0 | 0 |
| F2 | 2011 | 182 | 36.12 | 9.75 | 3.23 | 4.41 | 0 | 0 |
| F3 | 2012 | 179 | 40.59 | 4.98 | 0.21 | 2.07 | 0 | 0 |
| F4 | 2013 | 195 | 60.23 | 9.29 | 0.76 | 2.57 | 0 | 0 |
| F5 | 2014 | 192 | 41.47 | 7.66 | 6.89 | 3.14 | 0 | 0 |
| F6 | 2015 | 190 | 28.06 | 7.02 | 5.83 | 3.77 | 0 | 0 |
| F7 | 2016 | 171 | 40.83 | 6.97 | 2.96 | 3.11 | 0 | 0 |
| F8 | 2017 | 190 | 66.21 | 8.86 | 3.00 | 2.66 | 0 | 0 |
| F9 | 2018 | 176 | 18.21 | 6.60 | 1.00 | 2.71 | 0 | 0 |
| F10 | 2019 | 195 | 44.17 | 10.15 | 1.57 | 2.98 | 0 | 0 |
| F11 | 2020 | 188 | 50.34 | 7.64 | 5.00 | 4.13 | 0 | 0 |
| 2021 | 2021 | 212 | 49.38 | 7.31 | 1.31 | 3.15 | 0 | 0 |
| 2022 | 2022 | 179 | 30.96 | 8.77 | 2.10 | 2.82 | 0 | 0 |
| 2023 | 2023 | 189 | 34.46 | 8.22 | 1.13 | 2.47 | 0 | 0 |
| 2024 | 2024 | 191 | 55.07 | 6.93 | 3.27 | 3.06 | 0 | 0 |
| 2025 | 2025 | 251 | 67.88 | 7.22 | 4.40 | 3.99 | 0 | 0 |
| 2026p | 2026 | 71 | 3.35 | 5.84 | 0.76 | 2.48 | 0 | 0 |

**0.50% r_base — governors ON** (search worst-fold ROI 18.21% / mean 40.72% / worst trailing DD 10.73% / worst from-init 6.89%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 23.61 | 10.73 | 3.75 | 5.00 | 0 | 2 |
| F2 | 2011 | 182 | 36.12 | 9.75 | 3.23 | 4.41 | 0 | 1 |
| F3 | 2012 | 179 | 40.59 | 4.98 | 0.21 | 2.07 | 0 | 0 |
| F4 | 2013 | 195 | 60.23 | 9.29 | 0.76 | 2.57 | 0 | 0 |
| F5 | 2014 | 192 | 41.47 | 7.66 | 6.89 | 3.14 | 0 | 0 |
| F6 | 2015 | 190 | 26.57 | 7.02 | 5.83 | 3.77 | 0 | 1 |
| F7 | 2016 | 171 | 40.83 | 6.97 | 2.96 | 3.11 | 0 | 0 |
| F8 | 2017 | 190 | 66.21 | 8.86 | 3.00 | 2.66 | 0 | 0 |
| F9 | 2018 | 176 | 18.21 | 6.60 | 1.00 | 2.71 | 0 | 0 |
| F10 | 2019 | 195 | 44.17 | 10.15 | 1.57 | 2.98 | 0 | 0 |
| F11 | 2020 | 188 | 49.90 | 7.64 | 5.00 | 4.04 | 0 | 1 |
| 2021 | 2021 | 212 | 49.38 | 7.31 | 1.31 | 3.15 | 0 | 0 |
| 2022 | 2022 | 179 | 30.96 | 8.77 | 2.10 | 2.82 | 0 | 0 |
| 2023 | 2023 | 189 | 34.46 | 8.22 | 1.13 | 2.47 | 0 | 0 |
| 2024 | 2024 | 191 | 55.07 | 6.93 | 3.27 | 3.06 | 0 | 0 |
| 2025 | 2025 | 251 | 67.88 | 7.22 | 4.40 | 3.99 | 0 | 1 |
| 2026p | 2026 | 71 | 3.35 | 5.84 | 0.76 | 2.48 | 0 | 0 |

> `2026p` ROI is a RAW ~4-month partial return (not annualised; see basis note).

## 3. Mean comparison — search vs holdout (same compound basis)

| risk_pct | governance | mean_search_F1_11_pct | mean_holdout_2021_2025_pct | delta_pp | roi_2026_raw_partial_pct |
|---|---|---|---|---|---|
| 0.40 | off | 32.32 | 36.79 | 4.47 | 2.70 |
| 0.40 | on | 32.32 | 36.79 | 4.47 | 2.70 |
| 0.50 | off | 41.59 | 47.55 | 5.96 | 3.35 |
| 0.50 | on | 40.72 | 47.55 | 6.83 | 3.35 |

> Holdout full-years (2021–2025) vs search folds (2010–2020) on identical compound annual-reset footing — settles 'is the holdout weaker' on one basis. 2026 raw partial shown separately (≈4 months, not comparable to a full year).

## 4. Linear → compound delta (0.50% gov-on)

| metric | linear | compound | delta_pp |
|---|---|---|---|
| worst-fold ROI % | 17.22 | 18.21 | 1.00 |
| mean-fold ROI % | 35.53 | 40.72 | 5.19 |
| worst-fold trailing DD % | 10.44 | 10.73 | 0.30 |
| worst-fold from-init DD % | 6.94 | 6.89 | -0.05 |

> Within a single ~1y fold compounding sizes up as equity grows, so ROI lifts and DD rises modestly. The binding check is whether worst-fold trailing DD at 0.50% (10.73% compound vs 10.44% linear) crosses the 10% hard limit.

## 5. Risk-level decision — re-checked on compound basis

| risk_pct | worst_fold_trailing_dd_pct | under_10pct_hard | under_8pct_deploy | worst_daily_dd_pct | under_5pct_daily | worst_fold_roi_pct | mean_fold_roi_pct |
|---|---|---|---|---|---|---|---|
| 0.40 | 8.20 | YES | NO | 4.01 | YES | 14.41 | 32.32 |
| 0.50 | 10.73 | NO | NO | 5.00 | NO | 18.21 | 40.72 |

> **Decision (compound basis): launch-at-0.40%, step-to-0.50% after a buffer banks (0.40% keeps fresh-tier worst-fold trailing DD under 10% where 0.50% does not).** Worst-fold trailing DD 8.20% at 0.40% vs 10.73% at 0.50%. Re-validates (or revises) the linear run's launch-at-0.40% call on the corrected sizing.

## 6. Governor value + kills (compound; ON vs OFF, both refs)

| risk_pct | trailing_dd_saved_pp | from_initial_dd_saved_pp | mean_roi_cost_pp | kills_from_initial |
|---|---|---|---|---|
| 0.40 | 0.00 | 0.00 | 0.00 | 0 |
| 0.50 | -0.58 | 0.00 | 0.87 | 0 |

> Total 8% kill events under from-initial firing across all compound cells: **0**. (Governors fire on the from-initial / live-EA basis; the trailing-fired sensitivity is documented in the linear run.)

## Reviewer notes / honesty constraints

- Canonical for the MODELLED system, not ground truth. Two unclosable modelled-vs-live gaps: (1) intrabar tick resolution on governor firing (H4-bar resolution of the intrabar low understates a true tick trigger); (2) close-all slippage on N concurrent positions is unmodelled (bites near the 5% daily margin).
- Compound sizing = r_base × closed equity at open; floating opens excluded from the sizing snapshot; equity steps only on close. Per-fold reset.
- Every figure traces to the committed CSVs (`matrix.csv`, `per_fold.csv`, `governor_log.csv`). No CAGR; holdout per-year; 2026 raw partial, flagged.
