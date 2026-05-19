# Arc 10 — WFO ORACLE c1 result (experimental upper bound)

> ⚠️ EXPERIMENTAL ORACLE — runs over §16a HALT. Cluster-ID lookahead permitted, all other constraints hold. Arc 10 dispatch only. Upper bound; not deployable. Do not promote.

**Pair file:** [`WFO_BASE_RESULT.md`](../wfo_base/WFO_BASE_RESULT.md)
**Dispatch:** `ARC_10_STEP_5_WFO_ORACLE_C1.md`

## Oracle definition (narrow)
- Cluster ID is available at trade-entry time. Pool restricted to c1 (V-shape recovery) trades.
- No realised P&L visibility. No future-bar price. No fold-2 regime knowledge. No centroid drift correction.
- Implementation: filter Arc 10 Step 1 pool to c1 trades (228), then run identical WFO machinery as base.

## WFO design (mirrors base)
- Pool: Arc 10 c1 subset (V-shape recovery), n=228.
- Window: anchored expanding training window.
- Initial train: first 50% of c1 by entry_time = 114 trades.
- Test windows: 8 sequential temporal blocks.
- Reoptimisation per fold: inner 3-fold TimeSeriesSplit on training to pick combo with highest mean inner-CV Sharpe.
- Parameter grid: mirrored from base WFO exactly for direct gap comparison.

## Aggregate metrics (8 folds, point estimate)

| metric | mean | median | std | min | max | 95% CI (bootstrap n=2000) |
|---|---:|---:|---:|---:|---:|---|
| sharpe_annual | 4.61 | 4.532 | 1.978 | 1.616 | 6.802 | [3.132, 6.088] |
| calmar | 71.2 | 80.04 | 39.79 | 21.81 | 111.1 | [39.4, 99.14] |
| expectancy_r | 1.554 | 1.828 | 1.235 | -1 | 2.812 | [0.6987, 2.231] |
| max_drawdown_pct | 0.4997 | 0.5 | 0.2666 | -0 | 0.9975 | — |
| total_return_pct | 4.03 | 5.036 | 3.213 | -0.5 | 7.708 | — |
| cagr_pct | 37.92 | 40.02 | 17.54 | 10.9 | 55.57 | — |
| win_rate | 0.65 | 0.7333 | 0.3045 | 0 | 1 | — |
| profit_factor | 8.097 | 8.531 | 5.373 | 0 | 15.06 | — |
| n_admit | 3.875 | 5 | 2.167 | 1 | 6 | — |

## Per-fold breakdown

| fold | date range | n_test | n_admit | mode | t | SL | feat | Sharpe | MaxDD% | Calmar | WinRate | ProfFact | Expectancy_R |
|---:|---|---:|---:|:---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2023-07-13 → 2023-12-13 | 15 | 1 | E | 0.65 | 3.00 | base+L1_minus_L0_atr_only | nan | -0 | nan | 1 | nan | 0.9056 |
| 1 | 2023-12-28 → 2024-04-01 | 15 | 5 | E | 0.65 | 3.00 | base+HTF | 3.474 | 0.9975 | 27.96 | 0.6 | 6.176 | 2.07 |
| 2 | 2024-04-03 → 2024-09-04 | 14 | 1 | E | 0.65 | 2.00 | base+HTF | nan | 0.5 | nan | 0 | 0 | -1 |
| 3 | 2024-09-10 → 2024-12-10 | 14 | 6 | E | 0.55 | 3.00 | base+HTF | 6.802 | 0.5 | 111.1 | 0.8333 | 10.51 | 1.585 |
| 4 | 2025-01-02 → 2025-03-10 | 14 | 2 | E | 0.65 | 3.00 | base+HTF | 1.616 | 0.5 | 21.81 | 0.5 | 3.229 | 1.114 |
| 5 | 2025-03-21 → 2025-05-29 | 14 | 5 | E | 0.55 | 3.00 | base+HTF | 6.703 | 0.5 | 97.76 | 0.8 | 15.06 | 2.812 |
| 6 | 2025-05-30 → 2025-09-16 | 14 | 5 | E | 0.55 | 3.00 | base+HTF | 4.305 | 0.5 | 62.32 | 0.8 | 13.17 | 2.435 |
| 7 | 2025-09-18 → 2025-12-19 | 14 | 6 | E | 0.55 | 3.00 | base+HTF | 4.758 | 0.5 | 106.2 | 0.6667 | 8.531 | 2.51 |

## Effective trade-count per fold

c1 pool is smaller (228 vs 802) so per-fold N is correspondingly thinner. Bootstrap CIs above
widen accordingly — interpret with the smaller-N caveat.

## Selected-parameter trajectory

- Modes selected across folds: ['E']
- SLs selected: [2.0, 3.0]
- Thresholds selected: [0.55, 0.65]
- Feature sets selected: ['base+HTF', 'base+L1_minus_L0_atr_only']

## Files
- `folds.csv`, `params_history.csv`, `oos_trades.csv`, `inner_cv_scores.csv` — mirror base WFO schema.

## sha256
```
folds_csv: b43614deeeb2d798f074d0fba24404ee8ac03430d92cb53fa9010e03deb738ef
params_history_csv: 89a01303a2ef10a078663bf1001311db3f89f01ce20043bd736b0973ed5f6ae5
oos_trades_csv: cc6c3e2c11776e5e1fed788f1f30baf11e99d8900facdd7c3ec7e8a73db22bdc
inner_cv_scores_csv: 60baf20eb9dcd3acbfe16eb8f1cc7edfcfa992c5d0b2aa12104ae509a9f350b9
```

## What this is NOT
- Not a production WFO — cluster ID is not available at trade-entry time in any live system today.
- Not a deployment evaluation.
- Not a classifier evaluation — assumes perfect classification.
