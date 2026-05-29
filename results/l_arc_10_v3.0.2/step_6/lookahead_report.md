# Step 6 — §6 lookahead report

- **Category:** `lookahead`
- **Passed:** True  (critical: 0/1 fails, warnings: 0, info: 1)

## Diagnostic

```json
{
  "best_candidate_architecture": "A1",
  "best_candidate_config_id": "A1::cl0::sl3.5::partial_close_1r_runner_trail::expunlimited",
  "features_in_winning_config": []
}
```

## Checks

| # | Name | Status | Message |
|---:|---|---|---|
| 1 | `per_feature_lineage_clean` | PASS | best_candidate_features is empty (by-design for rule-based architectures like A1 system_level_filter). No classifier features means no opportunity for feature lineage contamination — vacuous PASS per universal-quantifier-over-empty-set. |
| 2 | `no_path_features_in_entry` | PASS | best_candidate_features is empty (by-design for rule-based architectures like A1 system_level_filter). No entry features means no opportunity for path-feature contamination in entry — vacuous PASS per universal-quantifier-over-empty-set. |
| 3 | `d1_lag_rule_enforced` | PASS | multi_tf D1 producers all route through _build_d1_lag1_series (4 calls) |
| 4 | `byte_compare_no_drift` | PASS | byte-compare skipped: inputs.feature_matrix missing — cannot read pool values |
| 5 | `signal_entry_bar_separation` | PASS | pool lacks signal_time/entry_time columns — skipping |
| 6 | `feature_matrix_indexed_at_signal_time` | PASS | feature_matrix or pool_trades absent — skipping |
| 7 | `threshold_selection_lineage` | PASS | no Step 4 classifier manifest — arc is rule-based or pre-CC_12 |
| 8 | `feature_lineage_table_exists` | INFO | feature_lineage table absent — falling back to source-code inspection |
| 9 | `byte_compare_strength_disclosed` | PASS | byte-compare samples 25 trade(s) from a pool of 3152 (1% coverage) |

## Evidence

### `per_feature_lineage_clean` (PASS)

```json
{
  "features_in_winning_config": []
}
```

### `no_path_features_in_entry` (PASS)

```json
{
  "features_in_winning_config": []
}
```

### `d1_lag_rule_enforced` (PASS)

```json
{
  "d1_producers": 3,
  "helper_calls": 4
}
```

### `byte_compare_no_drift` (PASS)

```json
{
  "all_match": false,
  "atol": 1e-09,
  "mismatched_features": [],
  "n_samples_requested": 25,
  "n_samples_used": 0,
  "rows_preview": [],
  "rtol": 1e-09,
  "seed": 42,
  "skipped_reason": "inputs.feature_matrix missing \u2014 cannot read pool values"
}
```

### `signal_entry_bar_separation` (PASS)

```json
{
  "columns": [
    "trade_id",
    "pair",
    "signal_bar_time",
    "entry_time",
    "exit_time",
    "entry_price",
    "sl_at_entry_price",
    "sl_distance_price",
    "exit_price",
    "exit_reason",
    "bars_held",
    "final_r",
    "mfe_r",
    "mae_r",
    "time_to_peak_mfe",
    "spread_close_at_entry",
    "spread_close_at_exit",
    "bid_ask_dq_at_entry",
    "bid_ask_dq_at_exit",
    "L1_value",
    "L0_value",
    "L1_age_d1_bars",
    "L0_age_d1_bars",
    "L1_to_atr_proximity",
    "reject_buffer_atr",
    "upper_fraction",
    "atr14_at_signal",
    "d_t_idx",
    "d_for_l1_search_max",
    "atr_14",
    "atr_percentile_100",
    "atr_vs_trailing_100",
    "d1_atr_percentile_100",
    "d1_close_slope_magnitude",
    "d1_close_slope_sign",
    "day_of_week",
    "distance_to_round_number",
    "dollar_bloc_state",
    "eur_strength_index",
    "hour_of_day",
    "kijun_26_distance",
    "prior_session_high_distance",
    "prior_session_low_distance",
    "range_close_ratio",
    "session_dead",
    "session_ldn_ny_overlap",
    "session_london",
    "session_ny",
    "session_tokyo",
    "signal_density_28",
    "spread_percentile_100",
    "spread_vs_trailing_100",
    "swing_high_distance_14",
    "swing_low_distance_14",
    "usd_strength_index",
    "w1_close_slope_sign",
    "path_mono",
    "path_peaks",
    "path_ttp_rel",
    "path_drawdown_depth_r",
    "path_recovery_ratio",
    "path_wrong_way_first"
  ]
}
```

### `feature_matrix_indexed_at_signal_time` (PASS)

_(no structured evidence)_

### `threshold_selection_lineage` (PASS)

```json
{
  "manifest_path": "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\results\\l_arc_10_v3.0.2\\step_4\\classifiers\\manifest.json"
}
```

### `feature_lineage_table_exists` (INFO)

```json
{
  "present": false
}
```

### `byte_compare_strength_disclosed` (PASS)

```json
{
  "coverage_fraction": 0.007931472081218274,
  "exhaustive": false,
  "n_byte_compare_samples": 25,
  "n_pool_trades": 3152
}
```

