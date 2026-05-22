# Arc 11 v3.0 — Step 1 Integrity Report

Per L_PROTOCOL §2 Step 1 + dispatch §'Integrity checks'.

## 1. Pool size
- n = **7149** (min = 500) → **PASS**

## 2. Per-pair n distribution
- min=214, max=287, median=256
- pairs with n<30: []
- pairs with n=0: []

## 3. Coverage / gap report (per pair)
- AUDCAD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26164 H4 bars)
- AUDCHF: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26169 H4 bars)
- AUDJPY: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26168 H4 bars)
- AUDNZD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26168 H4 bars)
- AUDUSD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26163 H4 bars)
- CADCHF: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26165 H4 bars)
- CADJPY: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26168 H4 bars)
- CHFJPY: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26167 H4 bars)
- EURAUD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26164 H4 bars)
- EURCAD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26168 H4 bars)
- EURCHF: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26170 H4 bars)
- EURGBP: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26166 H4 bars)
- EURJPY: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26161 H4 bars)
- EURNZD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26166 H4 bars)
- EURUSD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26170 H4 bars)
- GBPAUD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26076 H4 bars)
- GBPCAD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26147 H4 bars)
- GBPCHF: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26160 H4 bars)
- GBPJPY: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26165 H4 bars)
- GBPNZD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26154 H4 bars)
- GBPUSD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26169 H4 bars)
- NZDCAD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26164 H4 bars)
- NZDCHF: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26168 H4 bars)
- NZDJPY: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26164 H4 bars)
- NZDUSD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26162 H4 bars)
- USDCAD: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26164 H4 bars)
- USDCHF: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26165 H4 bars)
- USDJPY: 2010-01-03 20:00:00+00:00 → 2026-04-10 20:00:00+00:00 (26165 H4 bars)

## 4. Spread regime (mean spread_close per pair — informational)
- AUDCAD: 0.000325508
- AUDCHF: 0.000303628
- AUDJPY: 0.0210101
- AUDNZD: 0.000379397
- AUDUSD: 0.00015927
- CADCHF: 0.000321312
- CADJPY: 0.0232754
- CHFJPY: 0.0280697
- EURAUD: 0.000348519
- EURCAD: 0.000340819
- EURCHF: 0.000172917
- EURGBP: 0.000158046
- EURJPY: 0.0177408
- EURNZD: 0.000618765
- EURUSD: 9.9e-05
- GBPAUD: 0.000492545
- GBPCAD: 0.000488993
- GBPCHF: 0.000392235
- GBPJPY: 0.0330568
- GBPNZD: 0.000994198
- GBPUSD: 0.000193591
- NZDCAD: 0.000407213
- NZDCHF: 0.000422127
- NZDJPY: 0.0282477
- NZDUSD: 0.00019988
- USDCAD: 0.000204914
- USDCHF: 0.000191354
- USDJPY: 0.0121227

## 5. D1-lag NaN perturbation (3 trades)
- n_nan_d1_close_slope_sign: 0
- n_nan_d1_atr_percentile_100: 0
- verdict: **PASS**

## 6. Lookahead spot-check (dispatch elevated: 10 trades w/ swing-detection emphasis)
- sampled: 10, pass: 10, fail: 0
- verdict: **PASS**
  - EURCAD 2018-04-16 08:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - AUDJPY 2023-08-09 00:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - AUDUSD 2010-08-27 12:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - USDCHF 2019-06-11 12:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - GBPAUD 2020-09-16 12:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - EURUSD 2017-08-21 12:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - GBPCAD 2010-04-07 16:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - CADJPY 2014-07-23 16:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - USDCAD 2016-07-08 12:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)
  - GBPNZD 2019-08-02 16:00:00+00:00 → PASS (right_edge_offset_ge_4=True, entry_after_signal=True, break_magnitude_positive=True, trend_filter_swing_low_finite=True, recompute_h_ref_matches=True)

## 7. Right-edge swing audit (Arc 9 lesson; producer-level)
- min h_ref_bar_offset = 4.0 (required ≥ 4)
- trades with offset < 4 = 0
- verdict: **PASS**

## 8. KH-24 co-fire (informational, dispatch §'Integrity checks')
- DEFERRED — KH-24 strategy not loaded at Step 1; informational only per dispatch.

## 9. Feature lineage summary
- total features = 27
  - clean: 23, suspect: 4, unverified: 0

## 10. Determinism (two-run sha256)
- verdict: **SKIPPED**
