# Top-10 raw performers — arc_discovery_02

> Ranking metric: mean R per rule (dispatch Override 1).
> Bonferroni threshold (primary, alpha/N_evaluated): 2.143e-05 (N_evaluated=2333)
> Bonferroni threshold (budget, alpha/10000): 5.000e-06
> Follow-up arcs spawned: top-3 only (chat methodology constraint).
> Ranks 4-10: analysis-only; do NOT enter the deployment-track pipeline.

| Rank | Rule ID | Mean R | Pool size | p-value | Bonf. primary | Bonf. budget | Time-exit % | Follow-up | Rule spec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 39857 | +0.1562 | 5778 | 1.229e-10 | PASS | PASS | 12.3% | YES (deployment-track) | `(NOT(atr_percentile_100 < p25) AND NOT(atr_vs_trailing_100 >= p10))` |
| 2 | 19935 | +0.0990 | 6685 | 1.238e-07 | PASS | PASS | 24.6% | YES (deployment-track) | `(NOT(range_close_ratio != p25) OR (atr_vs_trailing_100 > p90 AND (atr_14 > p50 AND prior_session_high_distance >= p75)))` |
| 3 | 10383 | +0.0866 | 8099 | 2.260e-06 | PASS | PASS | 16.2% | YES (deployment-track) | `(distance_to_round_number != p90 AND (prior_session_low_distance < p50 AND (NOT(kijun_26_distance <= p75) AND NOT(atr_vs_trailing_100 < p50))))` |
| 4 | 6967 | +0.0852 | 15564 | 2.981e-11 | PASS | PASS | 19.5% | no (analysis-only) | `((prior_session_low_distance >= p90 OR NOT(session_ldn_ny_overlap == p25)) AND (NOT(atr_percentile_100 < p75) AND kijun_26_distance > p50))` |
| 5 | 38142 | +0.0825 | 5758 | 1.092e-04 | fail | fail | 21.2% | no (analysis-only) | `(atr_percentile_100 > p90 AND (NOT(atr_14 >= p90) AND NOT(swing_high_distance_14 < p75)))` |
| 6 | 31519 | +0.0699 | 15079 | 1.589e-08 | PASS | PASS | 22.8% | no (analysis-only) | `(atr_vs_trailing_100 >= p90 AND (kijun_26_distance >= p75 AND atr_percentile_100 > p25))` |
| 7 | 45049 | +0.0678 | 806 | 2.454e-01 | fail | fail | 13.9% | no (analysis-only) | `((atr_percentile_100 == p50 AND NOT(kijun_26_distance > p25)) AND (swing_low_distance_14 <= p90 AND spread_percentile_100 >= p10))` |
| 8 | 22156 | +0.0588 | 6037 | 3.253e-03 | fail | fail | 20.9% | no (analysis-only) | `(((NOT(swing_low_distance_14 <= p25) AND NOT(atr_vs_trailing_100 < p90)) AND NOT(atr_vs_trailing_100 <= p25)) AND (session_tokyo == p75 AND NOT(swing_high_distance_14 > p50)))` |
| 9 | 10475 | +0.0541 | 2607 | 8.736e-02 | fail | fail | 15.5% | no (analysis-only) | `((NOT(atr_percentile_100 > p50) AND NOT(range_close_ratio <= p90)) AND day_of_week == p10)` |
| 10 | 41976 | +0.0529 | 7793 | 8.206e-03 | fail | fail | 9.3% | no (analysis-only) | `(swing_low_distance_14 <= p10 AND (atr_percentile_100 < p25 AND (distance_to_round_number < p75 AND (NOT(distance_to_round_number < p90) OR NOT(session_ldn_ny_overlap > p10)))))` |
