# Audit 4 — Feature list review

## Entry-time feature classification (28 features)

| Feature | Classification |
|---|---|
| body_to_range_ratio | entry-time |
| upper_wick_ratio | entry-time |
| lower_wick_ratio | entry-time |
| range_to_atr_14 | entry-time |
| ret_5bar_atr | entry-time |
| ret_20bar_atr | entry-time |
| pos_in_20bar_range | entry-time |
| rsi_14 | entry-time |
| n_swing_lows | entry-time |
| most_recent_sl_lag | entry-time |
| swing_low_dist_atr | entry-time |
| mother_bar_range_atr | entry-time |
| inside_bar_range_atr | entry-time |
| ib_range_ratio | entry-time |
| break_bar_body_atr | entry-time |
| break_close_above_high_atr | entry-time |
| d1_trend_state | entry-time |
| d1_atr_ratio_to_4h | entry-time |
| d1_pos_in_20d_range | entry-time |
| d1_ret_5d_atr | entry-time |
| d1_rsi_14 | entry-time |
| d1_close_above_kijun | entry-time |
| d1_bars_since_swing_high | entry-time |
| d1_bars_since_swing_low | entry-time |
| session_london | entry-time |
| session_ny_overlap | entry-time |
| hour_sin | entry-time |
| hour_cos | entry-time |

## Column-name overlap with path-shape feature matrix

Path-shape columns scanned: ['local_peaks_count', 'monotonicity_ratio_in_profit', 'pullback_magnitude_median', 'time_to_peak_mfe_relative']
Overlap with entry-time features: []

## Max abs Pearson correlation (entry-time × path-shape, same trade pool)

- Max |r|: 0.3285 between `d1_bars_since_swing_low` (entry-time) and `local_peaks_count` (path-shape) (r=-0.3285)
- Pairs with |r| > 0.85: 0

Full correlation table: `audit_4_correlation_matrix.csv` (sorted by |r| desc)
