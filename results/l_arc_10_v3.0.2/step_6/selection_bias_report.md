# Step 6 — §6 selection_bias report

- **Category:** `selection_bias`
- **Passed:** True  (critical: 0/3 fails, warnings: 1, info: 0)

## Diagnostic

```json
{
  "configs_evaluated_step5": 48,
  "holdout_start": "2021-01-01 00:00:00+00:00"
}
```

## Checks

| # | Name | Status | Message |
|---:|---|---|---|
| 1 | `configs_evaluated_recorded` | PASS | configs_evaluated_step5 = 48; scope flag = 'thin' (derived from N: thin) |
| 2 | `ratio_clears_bonferroni_noise_floor` | PASS | worst-fold ratio 6.427 vs Bonferroni-equivalent floor 2.056 (N_effective=48, recorded=48, sub_protocol_trials=0) |
| 3 | `holdout_never_touched_during_is` | PASS | WFO search artefacts all bounded before holdout_start (2021-01-01T00:00:00+00:00) |
| 4 | `sub_protocol_trial_counts_included` | PASS | no sub-protocol manifest on disk — vanilla arc |
| 5 | `cluster_selection_recorded` | WARNING | capturability.csv missing 'is_candidate' column |
| 6 | `sl_multiplier_step3_step5_independence` | PASS | sl_atr_multiplier absent on one side (cap=False, wfo=False) — re-tuning not detectable |
| 7 | `feature_set_selection_recorded` | PASS | n_features_considered=0, n_features_kept=0, selection pressure ≈ 0.00× (higher = more pressure on the chosen subset) |
| 8 | `search_scope_flag_matches_n` | PASS | recorded='thin', derived='thin' (N=48); match |

## Evidence

### `configs_evaluated_recorded` (PASS)

```json
{
  "configs_evaluated_step5": 48,
  "derived_scope": "thin",
  "recorded_scope": "thin"
}
```

### `ratio_clears_bonferroni_noise_floor` (PASS)

```json
{
  "configs_evaluated_step5": 48,
  "n_effective": 48,
  "noise_floor": 2.0558496250072116,
  "sub_protocol_trials": 0,
  "worst_fold_ratio": 6.4273
}
```

### `holdout_never_touched_during_is` (PASS)

```json
{
  "date_columns_checked": [],
  "holdout_start": "2021-01-01 00:00:00+00:00"
}
```

### `sub_protocol_trial_counts_included` (PASS)

```json
{
  "heavy_ml_manifest_present": false
}
```

### `cluster_selection_recorded` (WARNING)

```json
{
  "columns": [
    "n",
    "reach_1r",
    "reach_2r",
    "reach_3r",
    "mfe_p25",
    "mfe_p50",
    "mfe_p75",
    "mfe_p90",
    "mae_p50",
    "ww_pp",
    "ttp_p50",
    "ttp_p75",
    "bars_held_p50",
    "mean_r",
    "p25_r",
    "p50_r",
    "cluster_id",
    "archetype",
    "composite",
    "candidate_at_sl_2x",
    "best_sl_multiplier",
    "best_sl_composite",
    "candidate_at_best_sl"
  ]
}
```

### `sl_multiplier_step3_step5_independence` (PASS)

```json
{
  "cap_has_col": false,
  "wfo_has_col": false
}
```

### `feature_set_selection_recorded` (PASS)

```json
{
  "n_features_considered": 0,
  "n_features_kept": 0,
  "selection_pressure_ratio": 0.0
}
```

### `search_scope_flag_matches_n` (PASS)

```json
{
  "derived": "thin",
  "n": 48,
  "recorded": "thin"
}
```

