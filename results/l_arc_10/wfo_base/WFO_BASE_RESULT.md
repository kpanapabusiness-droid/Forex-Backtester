# Arc 10 — WFO BASE result (experimental)

> ⚠️ EXPERIMENTAL — runs over §16a HALT. Arc 10 dispatch only. Do not deploy. Do not promote.

**Pair file:** [`WFO_ORACLE_C1_RESULT.md`](../wfo_oracle_c1/WFO_ORACLE_C1_RESULT.md)
**Dispatch:** `ARC_10_STEP_5_WFO_BASE.md`
**Reads:** `ARC_10_LIVE.md`, `ARC_10_RESULT.md`, `results/l_arc_10/experiments/`

## WFO design
- Pool: Arc 10 Step 1 full pool, 802 trades (no cluster filtering).
- Window: anchored expanding training window.
- Initial train: first 50% of pool by entry_time = 401 trades.
- Test windows: 8 sequential temporal blocks; each trade tested OOS exactly once.
- Reoptimisation per fold: inner 3-fold TimeSeriesSplit on training to pick parameter combo
  with highest mean inner-CV Sharpe (annualised). Then re-fit on full training, apply to OOS.

## Parameter grid (mirrored on oracle WFO for direct gap comparison)

| Parameter | Values |
|---|---|
| mode | { E, D1 } |
| Pipeline threshold | [0.55, 0.65] |
| SL multiplier (×ATR) | [2.0, 3.0] |
| feature_set (E only) | ['base+HTF', 'base+L1_minus_L0_atr_only'] |
| confidence-weighted sizing | omitted (not implemented in Arc 10 codebase) |

**Sharpe annualisation:** per-fold trades-per-year computed from fold OOS entry-time span; Sharpe_annual = mean(R)/std(R) × sqrt(trades_per_year).

**Window-type justification:** anchored expanding. Rolling rejected because Arc 10's data window
starts only 5y ago — discarding early training would waste training mass; the production
Pipeline E/D1 classifier in the dispatch also uses anchored expanding (5-fold TimeSeriesSplit at Step 4).

## Aggregate metrics (across 8 folds, point estimate)

| metric | mean | median | std | min | max | 95% CI (bootstrap n=2000) |
|---|---:|---:|---:|---:|---:|---|
| sharpe_annual | -1.292 | -1.292 | 0 | -1.292 | -1.292 | — |
| calmar | -14.19 | -14.19 | 13.9 | -24.02 | -4.361 | — |
| expectancy_r | 0.4044 | -0.6441 | 2.358 | -1 | 3.906 | [-1, 2.679] |
| max_drawdown_pct | 1.115 | 0.7487 | 1.297 | -0 | 2.963 | — |
| total_return_pct | -0.2432 | 0 | 1.339 | -2.963 | 1.953 | — |
| cagr_pct | -37.76 | -37.76 | 47.25 | -71.17 | -4.35 | — |
| win_rate | 0.3333 | 0.1667 | 0.4714 | 0 | 1 | — |
| profit_factor | 0.1893 | 0 | 0.3278 | 0 | 0.5678 | — |
| n_admit | 1.375 | 0.5 | 2.134 | 0 | 6 | — |

## Per-fold breakdown

| fold | date range | n_test | n_admit | mode | t | SL | feat | Sharpe | MaxDD% | Calmar | WinRate | ProfFact | Expectancy_R |
|---:|---|---:|---:|:---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2023-06-22 → 2023-10-13 | 51 | 0 | D1 | 0.55 | 2.00 | (D1_fixed) | nan | nan | nan | nan | nan | nan |
| 1 | 2023-10-19 → 2024-02-02 | 50 | 0 | D1 | 0.55 | 2.00 | (D1_fixed) | nan | nan | nan | nan | nan | nan |
| 2 | 2024-02-06 → 2024-06-17 | 50 | 1 | E | 0.55 | 3.00 | base+L1_minus_L0_atr_only | nan | -0 | nan | 1 | nan | 3.906 |
| 3 | 2024-06-18 → 2024-11-01 | 50 | 0 | D1 | 0.55 | 3.00 | (D1_fixed) | nan | nan | nan | nan | nan | nan |
| 4 | 2024-11-05 → 2025-02-03 | 50 | 6 | D1 | 0.55 | 3.00 | (D1_fixed) | nan | 2.963 | -24.02 | 0 | 0 | -1 |
| 5 | 2025-02-03 → 2025-05-22 | 50 | 0 | E | 0.55 | 3.00 | base+L1_minus_L0_atr_only | nan | nan | nan | nan | nan | nan |
| 6 | 2025-05-23 → 2025-09-08 | 50 | 3 | D1 | 0.55 | 3.00 | (D1_fixed) | -1.292 | 0.9975 | -4.361 | 0.3333 | 0.5678 | -0.2881 |
| 7 | 2025-09-09 → 2025-12-31 | 50 | 1 | D1 | 0.55 | 2.00 | (D1_fixed) | nan | 0.5 | nan | 0 | 0 | -1 |

## Selected-parameter trajectory

Stability check — does the optimiser converge on a single regime, or drift?

- Modes selected across folds: ['D1', 'E']
- SLs selected: [2.0, 3.0]
- Thresholds selected: [0.55]
- Feature sets selected: ['(D1_fixed)', 'base+L1_minus_L0_atr_only']

## Files
- `folds.csv` — per-fold raw metrics.
- `params_history.csv` — selected parameters per fold + inner-CV Sharpe.
- `oos_trades.csv` — per-OOS-trade (admit flag, final_r, mode, SL).
- `inner_cv_scores.csv` — full inner-CV score sweep (for parameter-grid diagnostics).

## sha256
```
folds_csv: 40d8a3347d32dd3aa9bf74d98de203b73f70b62e6a5a98e5a8c7429f3dfdbded
params_history_csv: 60e419f9ea13a433d220c41825e79b3cadb9e267ada9fd91d8695ef2c9a39e11
oos_trades_csv: 0f4b30d093e6cf3d268acb507f053a02c097994477ea6bbece205de7b73c72de
inner_cv_scores_csv: 145d4a81756f3adaf3662d76f40f5a7517b2b6e1e6e525fadd5308f71c79224d
```

## What this is NOT
- Not a production Step 5 WFO. Arc 10 disposition remains STEP_4_HALT.
- Not a deployment evaluation. No production decision flows from these numbers.
- Not a v2.4 calibration input on its own — pair with `WFO_ORACLE_C1_RESULT.md` and EXP-01-06.
