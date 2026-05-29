# Top-10 raw performers — arc_discovery_01

> Ranking metric: mean R per rule (dispatch Override 1).
> Bonferroni threshold (primary, alpha/N_evaluated): 1.171e-04 (N_evaluated=427)
> Bonferroni threshold (budget, alpha/10000): 9.785e-05
> Follow-up arcs spawned: top-3 only (chat methodology constraint).
> Ranks 4-10: analysis-only; do NOT enter the deployment-track pipeline.

| Rank | Rule ID | Mean R | Pool size | p-value | Bonf. primary | Bonf. budget | Follow-up | Rule spec |
|---|---|---|---|---|---|---|---|---|
| 1 | 420 | +0.0992 | 203 | 4.704e-01 | fail | fail | YES (deployment-track) | `(NOT(spread_vs_trailing_100 > p50) AND NOT(swing_high_distance_14 != p50))` |
| 2 | 72 | +0.0209 | 119885 | 6.608e-05 | PASS | PASS | YES (deployment-track) | `((NOT(prior_session_high_distance >= p25) OR NOT(day_of_week == p90)) AND (NOT(distance_to_round_number < p10) AND NOT(kijun_26_distance < p90)))` |
| 3 | 366 | -0.0122 | 1063 | 8.278e-01 | fail | fail | YES (deployment-track) | `(((distance_to_round_number < p25 AND NOT(spread_vs_trailing_100 <= p75)) AND atr_percentile_100 == p25) AND (session_ldn_ny_overlap <= p10 OR kijun_26_distance > p90))` |
| 4 | 238 | -0.0155 | 2742 | 6.501e-01 | fail | fail | no (analysis-only) | `(atr_14 >= p75 AND distance_to_round_number > p25)` |
| 5 | 392 | -0.0324 | 191135 | 6.784e-16 | PASS | PASS | no (analysis-only) | `NOT(kijun_26_distance < p90)` |
| 6 | 327 | -0.0372 | 458168 | 6.143e-46 | PASS | PASS | no (analysis-only) | `(atr_14 >= p75 AND NOT(hour_of_day == p90))` |
| 7 | 69 | -0.0429 | 30330 | 2.362e-05 | PASS | PASS | no (analysis-only) | `(NOT(prior_session_high_distance <= p90) AND (range_close_ratio < p25 OR NOT(range_close_ratio >= p10)))` |
| 8 | 219 | -0.0430 | 464386 | 1.354e-61 | PASS | PASS | no (analysis-only) | `(swing_high_distance_14 >= p75 AND (atr_14 >= p75 AND (distance_to_round_number <= p10 OR day_of_week >= p10)))` |
| 9 | 289 | -0.0443 | 480663 | 8.167e-68 | PASS | PASS | no (analysis-only) | `(swing_low_distance_14 > p75 OR atr_14 > p90)` |
| 10 | 158 | -0.0453 | 187664 | 8.079e-30 | PASS | PASS | no (analysis-only) | `(NOT(prior_session_low_distance < p90) AND (swing_low_distance_14 != p75 OR NOT(day_of_week == p10)))` |
