# Arc 10 v3.0.2 — FINAL CANONICAL WFO (deploy-faithful: fixed-initial + daily_ref=initial)

> **THE deploy-faithful canonical gate run.** Matches EXACTLY how the live EA trades after FIX 2b + the backtester daily_ref alignment (PR #252/#253): **fixed-initial sizing** (1R ≡ r_base of the INITIAL balance, constant per trade — deployed EA `ArcComputeLots`) + **daily DD on `daily_ref="initial"`** (the RESETTING fixed-$/day basis: numerator = day-start equity reset each EET day, denominator = fixed initial — deployed EA `Daily_DD_Basis=INITIAL`, EquityGuards.mqh FIX 2b) + **max DD trailing (planning) + static/from-initial (FundedNext enforce)**, both reported. **Supersedes** the EA-faithful (floating) run and every prior `_canonical` / `_canonical_compound` run — retained, not deleted. v3.0.2 signal/exit/cost LOCKED; ONLY the sizing basis + daily-DD mode differ from prior runs. Governors EA-faithful, NOT tuned. Frame sha `05dea9…9ee58a` (manifest-matched). Costs ON every cell (FundedNext cell 5). EET; deterministic (two-run sha identity); PR-gated.

```
BASIS:   EET | governed (3.5/4.5 daily, 7/8 total) | 1.5x spread | $5/lot RT | swaps OFF | 0.5 slip
SIZING:  fixed-initial
DD REFS: daily = initial (fixed-$/day, RESETS daily) | max = trailing (plan) + static (enforce)
RISK:    {0.40, 0.42, 0.45}% (0.40 = canonical; 0.42/0.45 = margin-check only)
FRAME:   sha 05dea9…9ee58a | 28 pairs | 11 folds + holdout-per-year (no CAGR)
```

> **Validation gate PASSED:** linear (mult ≡ r_base) + zero-cost + governors-OFF reproduces the **9.22%** portfolio worst-fold trailing DD (abort-if-not check in driver). Reconstruction intact.

> **Two-run determinism digest:** `67cc942b0feb64de…` (full sha in driver stdout).

## F5/F6 recovery proof — the daily-reset alignment is live, the freeze is gone

> Under the OLD non-resetting `static` (now quarantined `static_noreset`) the daily governor froze **F5 2014 at −5.99%** and **F6 2015 at −5.45%** — the daily window never reset, so once the account dipped 3.5% below initial the daily-halt blocked every recovery entry for the rest of the fold. Under `daily_ref="initial"` (resetting each EET day, deploy-faithful) they recover:

| fold | roi_pct | raw_return_pct | daily_gov_fires | total_gov_fires | kills | old_static_noreset_roi_pct |
|---|---|---|---|---|---|---|
| F5 2014 | 28.91 | 28.26 | 0 | 0 | 0 | -5.99 |
| F6 2015 | 20.77 | 20.53 | 0 | 0 | 0 | -5.45 |

> **PROOF:** F5 2014 = **+28.91%** (0 daily-gov fires), F6 2015 = **+20.77%** (0 daily-gov fires). Both positive, daily governor barely binds → the resetting daily basis is live and the −5.99%/−5.45% freeze artifact is gone (vs the day_start counterfactual ~+28.9%/+20.8%). `static_noreset` is quarantined and never canonical.

## 1. Canonical matrix (3 risk × governed; fixed-initial; both DD refs)

```
BASIS:   EET | governed (3.5/4.5 daily, 7/8 total) | 1.5x spread | $5/lot RT | swaps OFF | 0.5 slip
SIZING:  fixed-initial
DD REFS: daily = initial (fixed-$/day, RESETS daily) | max = trailing (plan) + static (enforce)
RISK:    {0.40, 0.42, 0.45}% (0.40 = canonical; 0.42/0.45 = margin-check only)
FRAME:   sha 05dea9…9ee58a | 28 pairs | 11 folds + holdout-per-year (no CAGR)
```

| risk% | worst-fold ROI% | mean-fold ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict_trailing | verdict_from_initial |
|---|---|---|---|---|---|---|---|---|
| 0.40 | 13.77 | 28.85 | 7.73 | 5.55 | 4.06 | 0 | PASS-DEPLOYABLE | PASS-DEPLOYABLE |
| 0.42 | 14.46 | 30.29 | 8.09 | 5.83 | 4.27 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.45 | 15.49 | 31.97 | 9.47 | 6.24 | 4.57 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |

> Worst-/mean-fold over the 11 search folds F1–F11 (2010–2020); holdout years in §2. **0.40% is the canonical operating tier**; 0.42%/0.45% are margin-check only (NOT adoption candidates — daily DD is a fixed $/day off initial, so it only tightens in %-terms as the account grows; there is no case to raise risk). At 0.40% the total governors never fire (on == off); a few daily-halts bind harmlessly.

## 2. Full per-fold + per-holdout-year (fixed-initial, daily_ref=initial)

```
BASIS:   EET | governed (3.5/4.5 daily, 7/8 total) | 1.5x spread | $5/lot RT | swaps OFF | 0.5 slip
SIZING:  fixed-initial
DD REFS: daily = initial (fixed-$/day, RESETS daily) | max = trailing (plan) + static (enforce)
RISK:    {0.40, 0.42, 0.45}% (0.40 = canonical; 0.42/0.45 = margin-check only)
FRAME:   sha 05dea9…9ee58a | 28 pairs | 11 folds + holdout-per-year (no CAGR)
```

**0.40% r_base — governors OFF** (search worst-fold ROI 13.77% / mean 28.85% / worst trailing DD 7.73% / worst from-init 5.55% / worst daily 4.06%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | daily_fires | kills |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 23.24 | 7.73 | 3.04 | 4.06 | 0 | 0 |
| F2 | 2011 | 182 | 25.83 | 7.55 | 2.26 | 3.54 | 0 | 0 |
| F3 | 2012 | 179 | 28.47 | 3.89 | 0.17 | 1.69 | 0 | 0 |
| F4 | 2013 | 195 | 40.19 | 6.40 | 0.61 | 2.46 | 0 | 0 |
| F5 ◀F5/F6 | 2014 | 192 | 28.91 | 6.18 | 5.55 | 2.49 | 0 | 0 |
| F6 ◀F5/F6 | 2015 | 190 | 20.77 | 5.75 | 4.77 | 3.08 | 0 | 0 |
| F7 | 2016 | 171 | 28.51 | 5.52 | 2.28 | 2.44 | 0 | 0 |
| F8 | 2017 | 190 | 42.56 | 6.96 | 2.14 | 2.28 | 0 | 0 |
| F9 | 2018 | 176 | 13.77 | 5.09 | 0.80 | 2.23 | 0 | 0 |
| F10 | 2019 | 195 | 30.97 | 7.58 | 1.25 | 2.43 | 0 | 0 |
| F11 | 2020 | 188 | 34.08 | 6.21 | 4.06 | 3.32 | 0 | 0 |
| 2021 | 2021 | 212 | 33.84 | 5.36 | 1.04 | 2.51 | 0 | 0 |
| 2022 | 2022 | 179 | 22.40 | 6.87 | 1.65 | 2.24 | 0 | 0 |
| 2023 | 2023 | 189 | 24.92 | 5.34 | 0.90 | 2.04 | 0 | 0 |
| 2024 | 2024 | 191 | 36.74 | 4.88 | 2.64 | 2.46 | 0 | 0 |
| 2025 | 2025 | 251 | 43.54 | 5.86 | 3.57 | 3.31 | 0 | 0 |
| 2026p | 2026 | 71 | 2.78 | 4.59 | 0.61 | 2.00 | 0 | 0 |

**0.40% r_base — governors ON** (search worst-fold ROI 13.77% / mean 28.85% / worst trailing DD 7.73% / worst from-init 5.55% / worst daily 4.06%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | daily_fires | kills |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 23.24 | 7.73 | 3.04 | 4.06 | 1 | 0 |
| F2 | 2011 | 182 | 25.83 | 7.55 | 2.26 | 3.54 | 1 | 0 |
| F3 | 2012 | 179 | 28.47 | 3.89 | 0.17 | 1.69 | 0 | 0 |
| F4 | 2013 | 195 | 40.19 | 6.40 | 0.61 | 2.46 | 0 | 0 |
| F5 ◀F5/F6 | 2014 | 192 | 28.91 | 6.18 | 5.55 | 2.49 | 0 | 0 |
| F6 ◀F5/F6 | 2015 | 190 | 20.77 | 5.75 | 4.77 | 3.08 | 0 | 0 |
| F7 | 2016 | 171 | 28.51 | 5.52 | 2.28 | 2.44 | 0 | 0 |
| F8 | 2017 | 190 | 42.56 | 6.96 | 2.14 | 2.28 | 0 | 0 |
| F9 | 2018 | 176 | 13.77 | 5.09 | 0.80 | 2.23 | 0 | 0 |
| F10 | 2019 | 195 | 30.97 | 7.58 | 1.25 | 2.43 | 0 | 0 |
| F11 | 2020 | 188 | 34.08 | 6.21 | 4.06 | 3.32 | 0 | 0 |
| 2021 | 2021 | 212 | 33.84 | 5.36 | 1.04 | 2.51 | 0 | 0 |
| 2022 | 2022 | 179 | 22.40 | 6.87 | 1.65 | 2.24 | 0 | 0 |
| 2023 | 2023 | 189 | 24.92 | 5.34 | 0.90 | 2.04 | 0 | 0 |
| 2024 | 2024 | 191 | 36.74 | 4.88 | 2.64 | 2.46 | 0 | 0 |
| 2025 | 2025 | 251 | 43.54 | 5.86 | 3.57 | 3.31 | 0 | 0 |
| 2026p | 2026 | 71 | 2.78 | 4.59 | 0.61 | 2.00 | 0 | 0 |

**0.42% r_base — governors OFF** (search worst-fold ROI 14.46% / mean 30.29% / worst trailing DD 8.09% / worst from-init 5.83% / worst daily 4.27%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | daily_fires | kills |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 24.41 | 8.09 | 3.19 | 4.27 | 0 | 0 |
| F2 | 2011 | 182 | 27.12 | 7.91 | 2.38 | 3.72 | 0 | 0 |
| F3 | 2012 | 179 | 29.90 | 4.07 | 0.17 | 1.78 | 0 | 0 |
| F4 | 2013 | 195 | 42.21 | 6.66 | 0.64 | 2.59 | 0 | 0 |
| F5 ◀F5/F6 | 2014 | 192 | 30.36 | 6.49 | 5.83 | 2.62 | 0 | 0 |
| F6 ◀F5/F6 | 2015 | 190 | 21.81 | 6.03 | 5.01 | 3.24 | 0 | 0 |
| F7 | 2016 | 171 | 29.94 | 5.79 | 2.39 | 2.57 | 0 | 0 |
| F8 | 2017 | 190 | 44.69 | 7.29 | 2.25 | 2.39 | 0 | 0 |
| F9 | 2018 | 176 | 14.46 | 5.33 | 0.84 | 2.35 | 0 | 0 |
| F10 | 2019 | 195 | 32.52 | 7.92 | 1.32 | 2.56 | 0 | 0 |
| F11 | 2020 | 188 | 35.78 | 6.51 | 4.27 | 3.49 | 0 | 0 |
| 2021 | 2021 | 212 | 35.54 | 5.61 | 1.10 | 2.63 | 0 | 0 |
| 2022 | 2022 | 179 | 23.52 | 7.19 | 1.74 | 2.36 | 0 | 0 |
| 2023 | 2023 | 189 | 26.17 | 5.54 | 0.95 | 2.14 | 0 | 0 |
| 2024 | 2024 | 191 | 38.57 | 5.11 | 2.77 | 2.58 | 0 | 0 |
| 2025 | 2025 | 251 | 45.72 | 6.15 | 3.75 | 3.47 | 0 | 0 |
| 2026p | 2026 | 71 | 2.92 | 4.80 | 0.64 | 2.10 | 0 | 0 |

**0.42% r_base — governors ON** (search worst-fold ROI 14.46% / mean 30.29% / worst trailing DD 8.09% / worst from-init 5.83% / worst daily 4.27%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | daily_fires | kills |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 24.41 | 8.09 | 3.19 | 4.27 | 1 | 0 |
| F2 | 2011 | 182 | 27.12 | 7.91 | 2.38 | 3.72 | 1 | 0 |
| F3 | 2012 | 179 | 29.90 | 4.07 | 0.17 | 1.78 | 0 | 0 |
| F4 | 2013 | 195 | 42.21 | 6.66 | 0.64 | 2.59 | 0 | 0 |
| F5 ◀F5/F6 | 2014 | 192 | 30.36 | 6.49 | 5.83 | 2.62 | 0 | 0 |
| F6 ◀F5/F6 | 2015 | 190 | 21.81 | 6.03 | 5.01 | 3.24 | 0 | 0 |
| F7 | 2016 | 171 | 29.94 | 5.79 | 2.39 | 2.57 | 0 | 0 |
| F8 | 2017 | 190 | 44.69 | 7.29 | 2.25 | 2.39 | 0 | 0 |
| F9 | 2018 | 176 | 14.46 | 5.33 | 0.84 | 2.35 | 0 | 0 |
| F10 | 2019 | 195 | 32.52 | 7.92 | 1.32 | 2.56 | 0 | 0 |
| F11 | 2020 | 188 | 35.78 | 6.51 | 4.27 | 3.49 | 0 | 0 |
| 2021 | 2021 | 212 | 35.54 | 5.61 | 1.10 | 2.63 | 0 | 0 |
| 2022 | 2022 | 179 | 23.52 | 7.19 | 1.74 | 2.36 | 0 | 0 |
| 2023 | 2023 | 189 | 26.17 | 5.54 | 0.95 | 2.14 | 0 | 0 |
| 2024 | 2024 | 191 | 38.57 | 5.11 | 2.77 | 2.58 | 0 | 0 |
| 2025 | 2025 | 251 | 45.72 | 6.15 | 3.75 | 3.47 | 0 | 0 |
| 2026p | 2026 | 71 | 2.92 | 4.80 | 0.64 | 2.10 | 0 | 0 |

**0.45% r_base — governors OFF** (search worst-fold ROI 15.49% / mean 32.46% / worst trailing DD 8.63% / worst from-init 6.24% / worst daily 4.57%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | daily_fires | kills |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 26.17 | 8.63 | 3.42 | 4.57 | 0 | 0 |
| F2 | 2011 | 182 | 29.07 | 8.44 | 2.55 | 3.98 | 0 | 0 |
| F3 | 2012 | 179 | 32.03 | 4.34 | 0.19 | 1.90 | 0 | 0 |
| F4 | 2013 | 195 | 45.23 | 7.04 | 0.68 | 2.77 | 0 | 0 |
| F5 ◀F5/F6 | 2014 | 192 | 32.53 | 6.95 | 6.24 | 2.81 | 0 | 0 |
| F6 ◀F5/F6 | 2015 | 190 | 23.37 | 6.45 | 5.37 | 3.47 | 0 | 0 |
| F7 | 2016 | 171 | 32.08 | 6.19 | 2.56 | 2.75 | 0 | 0 |
| F8 | 2017 | 190 | 47.89 | 7.79 | 2.41 | 2.56 | 0 | 0 |
| F9 | 2018 | 176 | 15.49 | 5.69 | 0.90 | 2.51 | 0 | 0 |
| F10 | 2019 | 195 | 34.85 | 8.41 | 1.41 | 2.74 | 0 | 0 |
| F11 | 2020 | 188 | 38.35 | 6.97 | 4.57 | 3.74 | 0 | 0 |
| 2021 | 2021 | 212 | 38.08 | 5.97 | 1.17 | 2.82 | 0 | 0 |
| 2022 | 2022 | 179 | 25.20 | 7.66 | 1.86 | 2.53 | 0 | 0 |
| 2023 | 2023 | 189 | 28.04 | 5.84 | 1.01 | 2.30 | 0 | 0 |
| 2024 | 2024 | 191 | 41.33 | 5.45 | 2.97 | 2.76 | 0 | 0 |
| 2025 | 2025 | 251 | 49.01 | 6.58 | 4.01 | 3.72 | 0 | 0 |
| 2026p | 2026 | 71 | 3.13 | 5.12 | 0.69 | 2.25 | 0 | 0 |

**0.45% r_base — governors ON** (search worst-fold ROI 15.49% / mean 31.97% / worst trailing DD 9.47% / worst from-init 6.24% / worst daily 4.57%):

| fold | yr | n | roi_pct | trailing_dd_pct | from_init_dd_pct | daily_dd_pct | daily_fires | kills |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 20.73 | 9.47 | 3.42 | 4.57 | 1 | 0 |
| F2 | 2011 | 182 | 29.07 | 8.44 | 2.55 | 3.98 | 1 | 0 |
| F3 | 2012 | 179 | 32.03 | 4.34 | 0.19 | 1.90 | 0 | 0 |
| F4 | 2013 | 195 | 45.23 | 7.04 | 0.68 | 2.77 | 0 | 0 |
| F5 ◀F5/F6 | 2014 | 192 | 32.53 | 6.95 | 6.24 | 2.81 | 0 | 0 |
| F6 ◀F5/F6 | 2015 | 190 | 23.37 | 6.45 | 5.37 | 3.47 | 0 | 0 |
| F7 | 2016 | 171 | 32.08 | 6.19 | 2.56 | 2.75 | 0 | 0 |
| F8 | 2017 | 190 | 47.89 | 7.79 | 2.41 | 2.56 | 0 | 0 |
| F9 | 2018 | 176 | 15.49 | 5.69 | 0.90 | 2.51 | 0 | 0 |
| F10 | 2019 | 195 | 34.85 | 8.41 | 1.41 | 2.74 | 0 | 0 |
| F11 | 2020 | 188 | 38.35 | 6.97 | 4.57 | 3.74 | 1 | 0 |
| 2021 | 2021 | 212 | 38.08 | 5.97 | 1.17 | 2.82 | 0 | 0 |
| 2022 | 2022 | 179 | 25.20 | 7.66 | 1.86 | 2.53 | 0 | 0 |
| 2023 | 2023 | 189 | 28.04 | 5.84 | 1.01 | 2.30 | 0 | 0 |
| 2024 | 2024 | 191 | 41.33 | 5.45 | 2.97 | 2.76 | 0 | 0 |
| 2025 | 2025 | 251 | 49.01 | 6.58 | 4.01 | 3.72 | 1 | 0 |
| 2026p | 2026 | 71 | 3.13 | 5.12 | 0.69 | 2.25 | 0 | 0 |

> `F5 2014` / `F6 2015` are flagged ◀F5/F6 — both positive (recovery proof). `2026p` ROI is a RAW ~4-month partial return (NOT annualised, NO CAGR). Full-year folds annualised over their own ~1y span.

## 3. Risk-decision read — does 0.40% clear both DD limits with margin?

```
BASIS:   EET | governed (3.5/4.5 daily, 7/8 total) | 1.5x spread | $5/lot RT | swaps OFF | 0.5 slip
SIZING:  fixed-initial
DD REFS: daily = initial (fixed-$/day, RESETS daily) | max = trailing (plan) + static (enforce)
RISK:    {0.40, 0.42, 0.45}% (0.40 = canonical; 0.42/0.45 = margin-check only)
FRAME:   sha 05dea9…9ee58a | 28 pairs | 11 folds + holdout-per-year (no CAGR)
```

| risk_pct | tier | worst_fold_trailing_dd_pct | under_10pct_hard | under_8pct_deploy | worst_daily_dd_pct | under_5pct_daily | worst_fold_roi_pct | mean_fold_roi_pct | kills |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | CANONICAL | 7.73 | YES | YES | 4.06 | YES | 13.77 | 28.85 | 0 |
| 0.42 | margin-check | 8.09 | YES | NO | 4.27 | YES | 14.46 | 30.29 | 0 |
| 0.45 | margin-check | 9.47 | YES | NO | 4.57 | YES | 15.49 | 31.97 | 0 |

> **0.40% (canonical):** worst-fold trailing DD **7.73%** (hard 10%: CLEARS; deploy 8%: clears), worst daily **4.06%** (under 5%), worst-fold ROI **13.77%** / mean **28.85%**, 0 kills → **0.40% CONFIRMED as the operating tier — clears both DD limits with margin.**

> **0.42% (margin-check):** trailing **8.09%**, daily **4.27%**, 0 kills. **0.45% (margin-check):** trailing **9.47%**, daily **4.57%**, 0 kills. These QUANTIFY proximity to the limits and are NOT recommended — the headline stays 0.40%.

## 4. vs superseded EA-faithful (floating) — why fixed-initial is now canonical

**0.40% gov-on — fixed-initial (canonical) vs floating (EA-faithful, superseded):**

| metric | fixed_initial | floating_ea_faithful | delta_pp |
|---|---|---|---|
| worst-fold ROI % | 13.77 | 14.42 | -0.65 |
| mean-fold ROI % | 28.85 | 32.48 | -3.64 |
| worst-fold trailing DD % | 7.73 | 8.21 | -0.47 |
| worst from-init DD % | 5.55 | 5.49 | 0.06 |
| worst daily DD % | 4.06 | 4.11 | -0.05 |

> **Why fixed-initial is now canonical:** the deployed EA sizes every entry off the FIXED INITIAL balance (`ArcComputeLots` post-FIX-1), NOT floating equity. The floating-equity run was a procyclical-tail *measurement* (it sized larger when the open book was up, into subsequent reversals) — useful to confirm the EA should stay on fixed-initial sizing, but it does NOT match how the EA trades. Fixed-initial is the deploy-faithful basis and therefore THE canonical gate. The EA-faithful run is retained, labelled SUPERSEDED, for the procyclical comparison only.

## Reviewer notes / honesty constraints

- Canonical for the MODELLED system, not ground truth. Two unclosable gaps: (1) intrabar tick resolution — the open book is marked at H4-bar resolution of the intrabar low (a true tick trigger could differ); (2) live close-all slippage on N concurrent positions is unmodelled.
- Sizing: fixed-initial — 1R ≡ r_base of the INITIAL balance, constant per trade, per-fold reset to 1.0 (the linear `governed_wfo.simulate_fold` path). Matches the deployed EA `ArcComputeLots` fixed-initial sizing.
- Daily DD: `daily_ref="initial"` (resetting fixed-$/day off initial = deployed EA `Daily_DD_Basis=INITIAL`, EquityGuards.mqh FIX 2b). NOT `static_noreset` (the quarantined non-resetting freeze-bug mode).
- Every figure traces to the committed CSVs (`matrix.csv`, `per_fold.csv`, `governor_log.csv`). No CAGR; holdout per-year; 2026 raw partial, flagged. Prior fabrication disclosure stands.
