# Arc 10 v3.0.2 — EA-FAITHFUL WFO (floating-equity sizing, FundedNext, EET, 3.5R)

> **The basis-correct canonical run.** Sizes every entry as the live EA does (`PositionManager.mqh:143`): `risk_amount = ACCOUNT_EQUITY × r_base`, equity **including floating open P&L**, re-read per entry. Supersedes the linear `_canonical`, the closed-equity `_canonical_compound`, and all earlier gates — first run whose sizing matches what trades live. Reconstruction (`build_schedules`, open-book marks, `_flat`) reused verbatim; only the per-entry sizing differs from the compound run (floating equity vs closed `e_bal`). v3.0.2 LOCKED; governors EA-faithful, not tuned. Costs ON (cell 5); r_base {0.40%,0.50%}; EET; frame sha `05dea9…9ee58a`; deterministic; PR-gated.

> **Validation gate PASSED** (reconstruction integrity): linear (mult≡r_base) + zero-cost reproduces the **9.22%** portfolio worst-fold trailing DD.

> **NO CAGR.** Holdout per-year (2021–2025 full + 2026 RAW partial, ~4mo, never annualised). Full-year folds annualised over their own ~1y span.

## 1. EA-faithful matrix (floating-equity; 2 risk × 2 gov; both DD refs)

| risk% | governance | worst-fold ROI% | mean-fold ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict_trailing | verdict_from_initial |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | off | 14.42 | 32.48 | 8.21 | 5.49 | 4.11 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.40 | on | 14.42 | 32.48 | 8.21 | 5.49 | 4.11 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | off | 18.23 | 41.86 | 10.16 | 6.85 | 5.16 | 0 | FAIL | PASS-DEPLOYABLE |
| 0.50 | on | 18.23 | 40.95 | 10.89 | 6.85 | 5.16 | 0 | FAIL | PASS-DEPLOYABLE |

> Worst-/mean-fold over the 11 search folds F1–F11; holdout years in §2/§3.

## 2. Full per-fold + per-holdout-year (floating-equity basis)

**0.40% r_base — governors OFF** (search worst-fold ROI 14.42% / mean 32.48% / worst trailing DD 8.21% / worst from-init 5.49%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 25.07 | 8.13 | 3.00 | 4.11 | 0 | 0 |
| F2 | 2011 | 182 | 28.09 | 7.88 | 2.53 | 3.55 | 0 | 0 |
| F3 | 2012 | 179 | 31.81 | 4.06 | 0.17 | 1.69 | 0 | 0 |
| F4 | 2013 | 195 | 46.88 | 7.45 | 0.61 | 2.14 | 0 | 0 |
| F5 | 2014 | 192 | 32.30 | 6.12 | 5.49 | 2.50 | 0 | 0 |
| F6 | 2015 | 190 | 22.26 | 5.71 | 4.73 | 3.08 | 0 | 0 |
| F7 | 2016 | 171 | 31.96 | 5.60 | 2.35 | 2.47 | 0 | 0 |
| F8 | 2017 | 190 | 50.83 | 7.19 | 2.40 | 2.17 | 0 | 0 |
| F9 | 2018 | 176 | 14.42 | 5.32 | 0.84 | 2.20 | 0 | 0 |
| F10 | 2019 | 195 | 34.76 | 8.21 | 1.26 | 2.41 | 0 | 0 |
| F11 | 2020 | 188 | 38.92 | 6.21 | 4.05 | 3.28 | 0 | 0 |
| 2021 | 2021 | 212 | 38.79 | 5.95 | 1.04 | 2.50 | 0 | 0 |
| 2022 | 2022 | 179 | 24.11 | 7.05 | 1.66 | 2.25 | 0 | 0 |
| 2023 | 2023 | 189 | 27.26 | 6.71 | 0.90 | 2.02 | 0 | 0 |
| 2024 | 2024 | 191 | 42.86 | 5.57 | 2.62 | 2.46 | 0 | 0 |
| 2025 | 2025 | 251 | 52.55 | 5.83 | 3.54 | 3.22 | 0 | 0 |
| 2026p | 2026 | 71 | 2.72 | 4.74 | 0.61 | 1.99 | 0 | 0 |

**0.40% r_base — governors ON** (search worst-fold ROI 14.42% / mean 32.48% / worst trailing DD 8.21% / worst from-init 5.49%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 25.07 | 8.13 | 3.00 | 4.11 | 0 | 1 |
| F2 | 2011 | 182 | 28.09 | 7.88 | 2.53 | 3.55 | 0 | 1 |
| F3 | 2012 | 179 | 31.81 | 4.06 | 0.17 | 1.69 | 0 | 0 |
| F4 | 2013 | 195 | 46.88 | 7.45 | 0.61 | 2.14 | 0 | 0 |
| F5 | 2014 | 192 | 32.30 | 6.12 | 5.49 | 2.50 | 0 | 0 |
| F6 | 2015 | 190 | 22.26 | 5.71 | 4.73 | 3.08 | 0 | 0 |
| F7 | 2016 | 171 | 31.96 | 5.60 | 2.35 | 2.47 | 0 | 0 |
| F8 | 2017 | 190 | 50.83 | 7.19 | 2.40 | 2.17 | 0 | 0 |
| F9 | 2018 | 176 | 14.42 | 5.32 | 0.84 | 2.20 | 0 | 0 |
| F10 | 2019 | 195 | 34.76 | 8.21 | 1.26 | 2.41 | 0 | 0 |
| F11 | 2020 | 188 | 38.92 | 6.21 | 4.05 | 3.28 | 0 | 0 |
| 2021 | 2021 | 212 | 38.79 | 5.95 | 1.04 | 2.50 | 0 | 0 |
| 2022 | 2022 | 179 | 24.11 | 7.05 | 1.66 | 2.25 | 0 | 0 |
| 2023 | 2023 | 189 | 27.26 | 6.71 | 0.90 | 2.02 | 0 | 0 |
| 2024 | 2024 | 191 | 42.86 | 5.57 | 2.62 | 2.46 | 0 | 0 |
| 2025 | 2025 | 251 | 52.55 | 5.83 | 3.54 | 3.22 | 0 | 0 |
| 2026p | 2026 | 71 | 2.72 | 4.74 | 0.61 | 1.99 | 0 | 0 |

**0.50% r_base — governors OFF** (search worst-fold ROI 18.23% / mean 41.86% / worst trailing DD 10.16% / worst from-init 6.85%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 32.01 | 10.12 | 3.74 | 5.16 | 0 | 0 |
| F2 | 2011 | 182 | 35.86 | 9.81 | 3.25 | 4.43 | 0 | 0 |
| F3 | 2012 | 179 | 40.88 | 5.06 | 0.21 | 2.12 | 0 | 0 |
| F4 | 2013 | 195 | 60.93 | 9.22 | 0.76 | 2.59 | 0 | 0 |
| F5 | 2014 | 192 | 41.53 | 7.62 | 6.85 | 3.13 | 0 | 0 |
| F6 | 2015 | 190 | 28.30 | 7.11 | 5.90 | 3.85 | 0 | 0 |
| F7 | 2016 | 171 | 41.12 | 6.97 | 2.96 | 3.09 | 0 | 0 |
| F8 | 2017 | 190 | 66.47 | 8.95 | 3.08 | 2.69 | 0 | 0 |
| F9 | 2018 | 176 | 18.23 | 6.62 | 1.07 | 2.75 | 0 | 0 |
| F10 | 2019 | 195 | 44.74 | 10.16 | 1.57 | 3.01 | 0 | 0 |
| F11 | 2020 | 188 | 50.33 | 7.71 | 5.06 | 4.09 | 0 | 0 |
| 2021 | 2021 | 212 | 50.20 | 7.40 | 1.30 | 3.13 | 0 | 0 |
| 2022 | 2022 | 179 | 30.71 | 8.71 | 2.07 | 2.81 | 0 | 0 |
| 2023 | 2023 | 189 | 34.85 | 8.33 | 1.12 | 2.53 | 0 | 0 |
| 2024 | 2024 | 191 | 55.70 | 6.92 | 3.27 | 3.08 | 0 | 0 |
| 2025 | 2025 | 251 | 68.99 | 7.23 | 4.42 | 4.00 | 0 | 0 |
| 2026p | 2026 | 71 | 3.38 | 5.89 | 0.77 | 2.49 | 0 | 0 |

**0.50% r_base — governors ON** (search worst-fold ROI 18.23% / mean 40.95% / worst trailing DD 10.89% / worst from-init 6.85%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 23.97 | 10.89 | 3.74 | 5.16 | 0 | 2 |
| F2 | 2011 | 182 | 35.86 | 9.81 | 3.25 | 4.43 | 0 | 1 |
| F3 | 2012 | 179 | 40.88 | 5.06 | 0.21 | 2.12 | 0 | 0 |
| F4 | 2013 | 195 | 60.93 | 9.22 | 0.76 | 2.59 | 0 | 0 |
| F5 | 2014 | 192 | 41.53 | 7.62 | 6.85 | 3.13 | 0 | 0 |
| F6 | 2015 | 190 | 26.79 | 7.11 | 5.90 | 3.85 | 0 | 1 |
| F7 | 2016 | 171 | 41.12 | 6.97 | 2.96 | 3.09 | 0 | 0 |
| F8 | 2017 | 190 | 66.47 | 8.95 | 3.08 | 2.69 | 0 | 0 |
| F9 | 2018 | 176 | 18.23 | 6.62 | 1.07 | 2.75 | 0 | 0 |
| F10 | 2019 | 195 | 44.74 | 10.16 | 1.57 | 3.01 | 0 | 0 |
| F11 | 2020 | 188 | 49.91 | 7.71 | 5.06 | 4.00 | 0 | 1 |
| 2021 | 2021 | 212 | 50.20 | 7.40 | 1.30 | 3.13 | 0 | 0 |
| 2022 | 2022 | 179 | 30.71 | 8.71 | 2.07 | 2.81 | 0 | 0 |
| 2023 | 2023 | 189 | 34.85 | 8.33 | 1.12 | 2.53 | 0 | 0 |
| 2024 | 2024 | 191 | 55.70 | 6.92 | 3.27 | 3.08 | 0 | 0 |
| 2025 | 2025 | 251 | 68.99 | 7.23 | 4.42 | 4.00 | 0 | 1 |
| 2026p | 2026 | 71 | 3.38 | 5.89 | 0.77 | 2.49 | 0 | 0 |

> `2026p` ROI is a RAW ~4-month partial return (not annualised).

## 3. Decision read — does launch-at-0.40% still hold on the live basis?

| risk_pct | worst_fold_trailing_dd_pct | under_10pct_hard | under_8pct_deploy | worst_daily_dd_pct | under_5pct_daily | worst_fold_roi_pct | mean_fold_roi_pct | kills |
|---|---|---|---|---|---|---|---|---|
| 0.40 | 8.21 | YES | NO | 4.11 | YES | 14.42 | 32.48 | 0 |
| 0.50 | 10.89 | NO | NO | 5.16 | NO | 18.23 | 40.95 | 0 |

> **0.40% floating-equity:** worst-fold trailing DD **8.21%** (hard 10%: CLEARS; deploy 8%: over), worst daily **4.11%** (under 5%), 0 kills. **0.50%:** trailing **10.89%**, daily **5.16%**, 0 kills.

## 4. Procyclical amplification — floating-equity vs closed-equity compound (0.50% gov-on)

| metric | closed_equity | floating_equity | delta_pp |
|---|---|---|---|
| worst-fold trailing DD % | 10.73 | 10.89 | 0.16 |
| worst-fold from-init DD % | 6.89 | 6.85 | -0.04 |
| worst daily DD % | 5.00 | 5.16 | 0.15 |
| mean-fold ROI % | 40.72 | 40.95 | 0.23 |

> Sizing off floating equity (vs closed) moves worst-fold trailing DD by **+0.16pp** at 0.50% gov-on. The closed-equity run's linear→compound step was a benign +0.30pp; this delta is the procyclical concurrency tail the basis correction exists to measure — bigger means floating sizing materially amplifies DD when the open book is up and entries size larger into a subsequent reversal.

## 5. Governor value + kills (floating basis)

| risk_pct | trailing_dd_saved_pp | from_initial_dd_saved_pp | mean_roi_cost_pp | kills_from_initial |
|---|---|---|---|---|
| 0.40 | 0.00 | 0.00 | 0.00 | 0 |
| 0.50 | -0.73 | 0.00 | 0.91 | 0 |

> **8% total-kill events (from-initial firing):** none. Procyclical sizing makes kills more likely than prior runs — reported honestly.

## The decision this feeds (stated, not acted)

> If 0.40% floating-equity worst-fold trailing DD stays under the 10% hard limit with margin AND daily under 5%, the live EA sizing is fine as-is. Here 0.40% lands at **8.21% trailing / 4.11% daily** (0 kills). If floating sizing pushes 0.40% toward/over 10% trailing or over 5% daily, the procyclical tail is material and the EA should be reconsidered (switch to closed-balance sizing — the compound run showed that costs ~nothing in ROI and keeps DD flat). The keep-vs-change-EA decision is the user's.

## Reviewer notes / honesty constraints

- Canonical for the MODELLED system, not ground truth. Two unclosable gaps: (1) intrabar tick resolution — open-book marked at H4-bar resolution of the intrabar low (a tick trigger could differ); (2) live close-all slippage on N concurrent positions unmodelled. Intra-bar entry order is the deterministic trade-id tiebreak (H4 has no finer timestamp); same-bar opens contribute ~0 floating (offset-0 mark).
- Floating-equity sizing: `mult = r_base × (realized + Σ open floating MtM)` at each entry, re-read live; equity steps continuously (floating) and on closes. Per-fold reset.
- Every figure traces to the committed CSVs (`matrix.csv`, `per_fold.csv`, `governor_log.csv`). No CAGR; holdout per-year; 2026 raw partial, flagged.
