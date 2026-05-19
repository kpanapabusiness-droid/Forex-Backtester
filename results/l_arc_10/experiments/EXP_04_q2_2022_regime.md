# EXP-04 — Q2 2022 fold regime characterisation

**Status:** experimental (not a Step 5 gate).

## Question
Does the AUC drop in fold 2 correlate with a measurable, entry-time-knowable regime descriptor?

## Per-fold summary

| fold | n_test | date range | base success | E AUC | D1 AUC |
|---:|---:|---|---:|---:|---:|
| 0 | 38 | 2021-08-26 → 2022-08-30 | 0.5000 | 0.6247 | 0.6690 |
| 1 | 38 | 2022-09-05 → 2023-07-06 | 0.4474 | 0.6695 | 0.6246 |
| 2 | 38 | 2023-07-13 → 2024-06-05 | 0.5000 | 0.5512 | 0.5180 |
| 3 | 38 | 2024-06-14 → 2025-04-01 | 0.2632 | 0.6464 | 0.4607 |
| 4 | 38 | 2025-04-10 → 2025-12-19 | 0.5789 | 0.6562 | 0.6761 |

## Fold-2 regime outliers (|z-score| ≥ 1.5 within 5-fold distribution)

_No descriptor places fold 2 ≥ 1.5σ from the cohort mean._

## All descriptors — sorted by correlation with E AUC

| descriptor | fold2 z | corr w/ E AUC | corr w/ D1 AUC |
|---|---:|---:|---:|
| spread_pips_used | -1.24 | +0.722 | +0.388 |
| ema50_4h_dist_atr | +0.66 | -0.669 | -0.139 |
| atr14_4h_at_entry | +1.39 | -0.535 | -0.571 |
| atr14_d1_at_entry_pct_of_close | -0.61 | +0.515 | +0.361 |
| ema200_d1_dist_atr | +1.33 | -0.471 | -0.258 |
| ema20_d1_dist_atr | +1.23 | -0.401 | -0.400 |
| ema50_d1_dist_atr | +1.14 | -0.353 | -0.191 |
| ema200_4h_dist_atr | +1.12 | -0.344 | +0.003 |
| ema20_4h_dist_atr | -0.01 | -0.343 | +0.091 |
| atr14_d1_at_entry | +1.13 | -0.332 | -0.491 |
| ema50_slope_d1 | +1.10 | -0.323 | -0.043 |
| ema20_slope_d1 | +0.98 | -0.271 | +0.088 |
| atr14_4h_at_entry_pct_of_close | -0.16 | +0.268 | +0.009 |
| range_pct_atr_last_20_bars | +0.29 | -0.138 | +0.250 |
| ema50_slope_4h | -0.26 | -0.093 | -0.125 |
| range_pct_atr_last_5_bars | +0.21 | +0.091 | -0.773 |
| ema20_slope_4h | -0.53 | +0.003 | +0.077 |

## Caveats and interpretation
- n=5 folds is too few for statistical inference on correlations; Pearson values are ranking only.
- Fold size is the protocol-mandated 5-fold TimeSeriesSplit; cannot increase n without altering Step 4 wiring.
- A descriptor flagged as fold-2 outlier (|z| ≥ 1.5) AND with |corr| ≥ 0.5 to E AUC is the candidate 
  for cross-arc entry-time regime filtering (Open-04 informational support).

No fold-2 outliers found; fold-2 AUC drop is not explained by these regime descriptors.

## Artefacts
- `raw/exp_04_fold_summary.csv` (sha256 `aac67eb426e111c6…`)
- `raw/exp_04_regime_correlations.csv` (sha256 `1bc2c10c202032ca…`)

