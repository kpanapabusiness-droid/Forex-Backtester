# Arc 11 v3.0 — Step 4 swing-feature audit

Per dispatch §'Special producer-level audit at Step 4'.

## Cluster 0
- Swing-derived features in top-10: ['break_magnitude_atr', 'swing_low_distance_14']
- Producer-level causality re-verified at intent doc §3 (PASS).
  - `h_ref`, `h_ref_bar_offset`, `break_magnitude_atr`, `trend_filter_swing_low`:
    enforced by `RIGHT_EDGE_OFFSET=4` in `signals/lchar_swing_high_breakout_trend.py`.
    No bar with k > t-4 is consumed at trigger time.
  - `swing_high_distance_14` / `swing_low_distance_14`: one-sided 14-bar trailing,
    shifted by 1 bar in `core/features/price_geometry.py`. Lineage = clean.
