# Step 6 — §6 statistical report

- **Category:** `statistical`
- **Passed:** True  (critical: 0/2 fails, warnings: 2, info: 0)

## Diagnostic

```json
{
  "n_pool_trades": 3152
}
```

## Checks

| # | Name | Status | Message |
|---:|---|---|---|
| 1 | `sample_size_adequate` | PASS | pool has 3152 trades; ≥ 100 required for Lo-corrected Sharpe meaning. Per-fold ≥ 25 verified by §3 gate (sign_pos_folds='11/11') |
| 2 | `pair_set_survivorship` | PASS | 28 pair(s) present in pool, 0 missing from expected pair_set |
| 3 | `regime_coverage_diverse` | PASS | pool spans 17 calendar year(s) (2010-2026); ≥ 3 expected for vol-regime diversity |
| 4 | `cross_pair_correlation_acceptable` | PASS | max abs cross-pair daily-bucket correlation = 0.408; warn at 0.70 |
| 5 | `trade_clustering_acceptable` | WARNING | top 39/198 buckets carry 84% of total positive R; threshold = 50% |
| 6 | `per_pair_edge_homogeneity` | WARNING | worst-pair mean R = -0.670 (18/28 pairs negative); top pair share of positive R = 23% |
| 7 | `outlier_influence_acceptable` | PASS | full mean R = -0.037; trimmed (top-5% removed) mean R = -0.547 (PASS — trimmed-vs-full ratio 14.99; edge robust to outlier removal) |
| 8 | `session_edge_concentration` | PASS | top session = ny carries 39% of total positive R; threshold = 60% |
| 9 | `weekday_edge_concentration` | PASS | top weekday = Tue carries 23% of total positive R; threshold = 35% |
| 10 | `lo_corrected_sharpe_recorded` | PASS | raw Sharpe ≈ -0.746, Lo-corrected ≈ -0.657 (N=3152, lag-1 ρ = 0.145) |

## Evidence

### `sample_size_adequate` (PASS)

```json
{
  "lo_corrected_min_trades": 100,
  "n_total_trades": 3152,
  "sign_pos_folds": "11/11"
}
```

### `pair_set_survivorship` (PASS)

```json
{
  "missing": [],
  "n_pairs_expected": 28,
  "n_pairs_in_pool": 28
}
```

### `regime_coverage_diverse` (PASS)

```json
{
  "n_years": 17,
  "years": [
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026
  ]
}
```

### `cross_pair_correlation_acceptable` (PASS)

```json
{
  "max_abs_correlation": 0.4083623737547617,
  "n_pairs": 28,
  "threshold": 0.7
}
```

### `trade_clustering_acceptable` (WARNING)

```json
{
  "concentration_threshold": 0.5,
  "n_30day_buckets": 198,
  "top_quintile_share_of_positive_r": 0.8402260014586566
}
```

### `per_pair_edge_homogeneity` (WARNING)

```json
{
  "n_negative_pairs": 18,
  "n_pairs": 28,
  "top_pair_share_of_positive_r": 0.23041978181649578,
  "worst_pair_mean_r": -0.6703813849706538
}
```

### `outlier_influence_acceptable` (PASS)

```json
{
  "full_mean_r": -0.036502318695982104,
  "p95_cutoff_r": 6.191573849712962,
  "trimmed_mean_r": -0.547189337411749,
  "trimmed_to_full_ratio": 14.990536408635856
}
```

### `session_edge_concentration` (PASS)

```json
{
  "per_session_share": {
    "asian": 0.25119880218096635,
    "london": 0.36221199450666586,
    "ny": 0.38658920331236785
  },
  "top_session": "ny",
  "top_session_share": 0.38658920331236785
}
```

### `weekday_edge_concentration` (PASS)

```json
{
  "per_dow_share": {
    "Fri": 0.16020077432628668,
    "Mon": 0.21549851957018631,
    "Sun": 0.022555413831790726,
    "Thu": 0.17598982126695645,
    "Tue": 0.22706868029691082,
    "Wed": 0.19868679070786907
  },
  "top_dow": "Tue",
  "top_dow_share": 0.22706868029691082
}
```

### `lo_corrected_sharpe_recorded` (PASS)

```json
{
  "lag1_autocorr": 0.14480222607655632,
  "lo_corrected_sharpe": -0.6572882070046926,
  "mean_r": -0.036502318695982104,
  "n_trades": 3152,
  "raw_sharpe": -0.7464214343492772,
  "std_r": 2.745550616904111
}
```

