# Step 6 — §6 execution_realism report

- **Category:** `execution_realism`
- **Passed:** True  (critical: 0/3 fails, warnings: 2, info: 0)

## Diagnostic

```json
{
  "primary_tf": "H4",
  "r_safe_pct": 0.005439315858450406,
  "sizing_convention": "reset_floor"
}
```

## Checks

| # | Name | Status | Message |
|---:|---|---|---|
| 1 | `real_spread_source_present` | PASS | HistData spread source present at data\histdata (0 pair dir(s)) |
| 2 | `spread_regime_within_tolerance` | PASS | no recognised spread column in pool — skipping |
| 3 | `next_bar_open_fill_realistic` | PASS | pool lacks signal_time/entry_time — cannot verify fill spacing |
| 4 | `lot_rounding_at_r_safe` | PASS | pool lacks sl_distance column — cannot infer lot sizing |
| 5 | `post_fill_sl_anchor` | PASS | SL anchor uses realised fill price (entry_price) |
| 6 | `boundary_convention_propagation` | PASS | panel convention='5ers_eet', closure recorded='5ers_eet': match |
| 7 | `mid_price_refactor_active` | PASS | multipair_backtester has 2 mid-price references, 0 bid-only references |
| 8 | `histdata_vs_venue_spread_differential` | WARNING | HistData↔5ers spread comparison MISSING — recommend live spread sample on 5ers MT5 before deployment go/no-go (baseline files: []; spread P&L decomp artefact: False) |
| 9 | `news_filter_assumption_declared` | WARNING | news calendar config present: False; closure references news filter: False — closure must declare whether the EA filters news (silent divergence is the Arc 10 mechanism) |
| 10 | `zero_spread_bar_fraction` | PASS | no spread / data-quality column in pool — skipping |
| 11 | `weekend_gap_handling_declared` | PASS | 464 of 3152 trade(s) held over weekend (Fri entry, post-weekend exit) |
| 12 | `utc_bar_boundary` | PASS | entry_time parses with UTC tz on sample |
| 13 | `spread_pnl_decomposition` | PASS | skipped — no top-1 trade ledger supplied to Step 6 (pre-PR closure or orchestrator wiring pending) |

## Evidence

### `real_spread_source_present` (PASS)

```json
{
  "n_pair_dirs": 0,
  "path": "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\trusting-brahmagupta-15aa77\\data\\histdata"
}
```

### `spread_regime_within_tolerance` (PASS)

```json
{
  "columns_seen": [
    "spread_close_at_entry",
    "spread_close_at_exit",
    "spread_percentile_100",
    "spread_vs_trailing_100"
  ]
}
```

### `next_bar_open_fill_realistic` (PASS)

_(no structured evidence)_

### `lot_rounding_at_r_safe` (PASS)

```json
{
  "columns_seen": [
    "sl_at_entry_price",
    "sl_distance_price",
    "d1_close_slope_magnitude",
    "d1_close_slope_sign",
    "w1_close_slope_sign"
  ]
}
```

### `post_fill_sl_anchor` (PASS)

```json
{
  "sl_uses_entry_price": true,
  "sl_uses_signal_close": false
}
```

### `boundary_convention_propagation` (PASS)

```json
{
  "declared": "5ers_eet",
  "recorded": "5ers_eet"
}
```

### `mid_price_refactor_active` (PASS)

```json
{
  "bid_references": 0,
  "mid_references": 2
}
```

### `histdata_vs_venue_spread_differential` (WARNING)

```json
{
  "spread_decomposition_present": false,
  "venue_baseline_files": []
}
```

### `news_filter_assumption_declared` (WARNING)

```json
{
  "closure_mentions_news": false,
  "news_calendar_files": []
}
```

### `zero_spread_bar_fraction` (PASS)

_(no structured evidence)_

### `weekend_gap_handling_declared` (PASS)

```json
{
  "has_weekend_exposure": true,
  "n_total_trades": 3152,
  "n_weekend_held": 464
}
```

### `utc_bar_boundary` (PASS)

```json
{
  "sample_count": 5,
  "tz_present": true
}
```

### `spread_pnl_decomposition` (PASS)

```json
{
  "top_1_trade_ledger_present": false
}
```

