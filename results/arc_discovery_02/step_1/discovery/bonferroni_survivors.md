# Bonferroni survivors — arc_discovery_02

> Primary threshold (alpha=0.05, denominator=N_evaluated=2333): p < 2.143e-05
> Budget threshold (alpha/10000, transparency only): p < 5.000e-06

| Rank | Rule ID | Mean R | Pool size | p-value | t-stat | Rule spec |
|---|---|---|---|---|---|---|
| 1 | 27835 | -0.0690 | 186132 | 4.186e-80 | -18.962 | `(NOT(atr_14 >= p75) AND spread_percentile_100 < p50)` |
| 2 | 29851 | -0.0676 | 182633 | 1.323e-76 | -18.533 | `(NOT(swing_low_distance_14 >= p75) AND (spread_vs_trailing_100 < p50 AND NOT(prior_session_high_distance == p25)))` |
| 3 | 39736 | -0.0719 | 157233 | 1.761e-75 | -18.394 | `(spread_vs_trailing_100 < p50 AND (swing_low_distance_14 >= p10 AND swing_low_distance_14 <= p75))` |
| 4 | 37111 | -0.0795 | 122890 | 6.119e-72 | -17.948 | `(((atr_vs_trailing_100 != p25 OR NOT(atr_percentile_100 > p50)) AND NOT(spread_vs_trailing_100 < p25)) AND (session_tokyo > p75 OR NOT(spread_vs_trailing_100 >= p50)))` |
| 5 | 47602 | -0.0630 | 195383 | 2.015e-68 | -17.487 | `(atr_14 < p25 OR (NOT(spread_vs_trailing_100 > p50) AND day_of_week <= p25))` |
| 6 | 26399 | -0.0784 | 111449 | 4.586e-64 | -16.910 | `((NOT(session_london > p75) AND atr_14 <= p50) AND atr_percentile_100 > p50)` |
| 7 | 21623 | -0.0800 | 102557 | 1.743e-61 | -16.556 | `(spread_vs_trailing_100 > p10 AND (NOT(atr_percentile_100 < p50) AND atr_14 < p50))` |
| 8 | 33936 | -0.0689 | 137699 | 1.658e-60 | -16.417 | `((NOT(spread_vs_trailing_100 >= p50) AND NOT(prior_session_high_distance > p90)) AND (NOT(spread_percentile_100 <= p25) AND (NOT(day_of_week < p10) OR hour_of_day >= p25)))` |
| 9 | 7449 | -0.0600 | 183768 | 5.225e-60 | -16.345 | `(session_dead < p90 AND spread_vs_trailing_100 <= p50)` |
| 10 | 38309 | -0.0728 | 123457 | 5.422e-60 | -16.345 | `((spread_vs_trailing_100 > p10 OR NOT(prior_session_high_distance <= p10)) AND (spread_percentile_100 < p50 AND distance_to_round_number > p50))` |
| 11 | 45985 | -0.0600 | 188738 | 5.722e-60 | -16.339 | `(NOT(spread_vs_trailing_100 >= p50) AND atr_percentile_100 < p75)` |
| 12 | 16343 | -0.0600 | 183761 | 6.181e-60 | -16.335 | `(session_dead < p75 AND (NOT(hour_of_day == p25) AND NOT(spread_vs_trailing_100 >= p50)))` |
| 13 | 11850 | -0.0580 | 195996 | 1.788e-59 | -16.269 | `(((kijun_26_distance == p75 OR spread_vs_trailing_100 <= p50) AND (swing_high_distance_14 < p90 AND spread_percentile_100 < p75)) AND NOT(prior_session_low_distance == p75))` |
| 14 | 29178 | -0.0664 | 145860 | 2.252e-58 | -16.115 | `((day_of_week >= p50 AND NOT(atr_14 == p10)) AND spread_vs_trailing_100 <= p50)` |
| 15 | 48732 | -0.0713 | 124243 | 2.699e-58 | -16.105 | `(NOT(spread_vs_trailing_100 >= p50) AND NOT(swing_high_distance_14 > p50))` |
| 16 | 34025 | -0.0646 | 153361 | 4.911e-58 | -16.066 | `((session_london != p90 AND NOT(session_ldn_ny_overlap > p25)) AND NOT(spread_vs_trailing_100 > p50))` |
| 17 | 18354 | -0.0646 | 153355 | 5.734e-58 | -16.057 | `(session_london == p25 AND spread_vs_trailing_100 < p50)` |
| 18 | 27064 | -0.0715 | 126679 | 3.578e-57 | -15.944 | `(NOT(atr_percentile_100 != p75) OR NOT(atr_14 >= p25))` |
| 19 | 4993 | -0.0649 | 147661 | 5.354e-57 | -15.917 | `((hour_of_day >= p50 OR swing_high_distance_14 <= p10) AND (NOT(spread_vs_trailing_100 >= p50) AND atr_percentile_100 != p90))` |
| 20 | 5928 | -0.0608 | 170388 | 6.547e-56 | -15.759 | `(((NOT(atr_14 >= p25) OR swing_high_distance_14 == p50) OR session_ny == p10) AND (spread_vs_trailing_100 <= p50 OR session_ny >= p90))` |
| 21 | 19823 | -0.0549 | 200021 | 2.599e-53 | -15.375 | `((spread_vs_trailing_100 < p50 AND distance_to_round_number <= p90) AND atr_vs_trailing_100 <= p90)` |
| 22 | 28324 | -0.0982 | 63627 | 5.322e-52 | -15.187 | `((NOT(day_of_week <= p90) AND spread_percentile_100 > p10) OR ((NOT(distance_to_round_number != p50) OR NOT(atr_14 > p10)) AND NOT(atr_vs_trailing_100 == p50)))` |
| 23 | 13365 | -0.0633 | 140400 | 1.802e-51 | -15.099 | `(((NOT(atr_14 <= p75) AND atr_14 <= p25) OR spread_percentile_100 < p50) AND (swing_high_distance_14 <= p50 OR range_close_ratio > p90))` |
| 24 | 15891 | -0.0603 | 163295 | 1.831e-51 | -15.097 | `(session_ldn_ny_overlap == p10 AND (NOT(kijun_26_distance >= p10) OR (NOT(range_close_ratio > p10) OR atr_14 <= p25)))` |
| 25 | 41017 | -0.0536 | 198674 | 1.397e-50 | -14.961 | `(session_ldn_ny_overlap == p75 AND NOT(spread_percentile_100 >= p50))` |
| 26 | 40969 | -0.0862 | 81488 | 3.580e-50 | -14.905 | `(NOT(spread_vs_trailing_100 == p10) AND (range_close_ratio < p10 OR NOT(atr_14 > p10)))` |
| 27 | 521 | -0.0628 | 137766 | 5.664e-50 | -14.870 | `(NOT(hour_of_day <= p10) AND (spread_vs_trailing_100 <= p50 AND NOT(hour_of_day < p50)))` |
| 28 | 15850 | -0.1061 | 53113 | 6.497e-50 | -14.870 | `(atr_percentile_100 == p50 OR atr_14 <= p10)` |
| 29 | 44800 | -0.1061 | 53113 | 6.497e-50 | -14.870 | `((NOT(atr_14 > p10) OR NOT(atr_percentile_100 != p50)) OR (session_tokyo < p10 AND range_close_ratio != p75))` |
| 30 | 23674 | -0.0627 | 137760 | 6.595e-50 | -14.860 | `(spread_vs_trailing_100 < p50 AND hour_of_day >= p50)` |
| 31 | 28618 | -0.0627 | 137760 | 6.595e-50 | -14.860 | `((NOT(session_tokyo > p10) OR session_dead < p25) AND spread_vs_trailing_100 < p50)` |
| 32 | 39612 | -0.0627 | 137760 | 6.595e-50 | -14.860 | `(session_ny == p50 AND (((session_ldn_ny_overlap < p75 OR session_ny != p90) AND NOT(day_of_week < p90)) OR spread_vs_trailing_100 < p50))` |
| 33 | 27913 | -0.0630 | 136410 | 8.570e-50 | -14.842 | `((spread_percentile_100 != p50 AND spread_vs_trailing_100 <= p50) AND session_ny == p50)` |
| 34 | 17727 | -0.0628 | 142375 | 3.349e-49 | -14.750 | `((NOT(kijun_26_distance == p75) AND swing_low_distance_14 < p50) AND (swing_high_distance_14 < p50 AND NOT(atr_vs_trailing_100 >= p90)))` |
| 35 | 6411 | -0.0621 | 137547 | 6.334e-49 | -14.707 | `((session_tokyo != p75 AND NOT(prior_session_high_distance == p10)) AND NOT(spread_vs_trailing_100 >= p50))` |
| 36 | 13680 | -0.0548 | 181785 | 1.726e-48 | -14.637 | `(prior_session_high_distance <= p75 AND NOT(spread_percentile_100 > p50))` |
| 37 | 29545 | -0.0527 | 187940 | 1.437e-46 | -14.333 | `((((NOT(swing_high_distance_14 < p10) AND NOT(hour_of_day < p50)) OR spread_percentile_100 < p25) OR NOT(prior_session_low_distance < p90)) AND NOT(spread_percentile_100 >= p50))` |
| 38 | 40831 | -0.0668 | 110663 | 3.336e-45 | -14.116 | `(NOT(spread_percentile_100 > p25) AND NOT(kijun_26_distance >= p90))` |
| 39 | 1982 | -0.0570 | 151829 | 5.472e-45 | -14.079 | `(spread_percentile_100 <= p50 AND (NOT(session_ldn_ny_overlap != p75) AND NOT(atr_vs_trailing_100 <= p25)))` |
| 40 | 40626 | -0.0557 | 161977 | 8.389e-45 | -14.048 | `(NOT(session_london >= p75) AND NOT(spread_percentile_100 >= p50))` |
| 41 | 33757 | -0.0719 | 91252 | 7.349e-44 | -13.897 | `(atr_14 != p90 AND (spread_percentile_100 <= p25 AND NOT(swing_high_distance_14 >= p75)))` |
| 42 | 46215 | -0.0584 | 143877 | 8.024e-44 | -13.888 | `((spread_vs_trailing_100 <= p50 AND NOT(range_close_ratio > p90)) AND NOT(hour_of_day > p50))` |
| 43 | 448 | -0.0545 | 168860 | 1.100e-43 | -13.864 | `((NOT(prior_session_low_distance >= p75) AND NOT(prior_session_high_distance < p90)) OR (session_ny <= p75 AND (atr_14 <= p25 OR NOT(spread_vs_trailing_100 > p10))))` |
| 44 | 3598 | -0.0694 | 101102 | 7.993e-43 | -13.724 | `((session_ny == p25 AND atr_14 <= p25) OR distance_to_round_number < p10)` |
| 45 | 42621 | -0.0530 | 173450 | 2.162e-42 | -13.649 | `((NOT(spread_percentile_100 > p50) AND (NOT(session_ny < p10) AND (NOT(session_tokyo <= p25) OR atr_percentile_100 < p50))) OR day_of_week < p10)` |
| 46 | 16144 | -0.0612 | 122660 | 4.358e-42 | -13.599 | `(((atr_percentile_100 < p90 AND session_ny == p90) OR atr_vs_trailing_100 != p10) AND (NOT(prior_session_low_distance < p50) AND spread_vs_trailing_100 <= p50))` |
| 47 | 39894 | -0.0494 | 194086 | 7.114e-42 | -13.561 | `(((NOT(day_of_week > p75) OR prior_session_low_distance == p50) AND spread_percentile_100 <= p50) OR (session_ny != p90 AND NOT(session_ny <= p75)))` |
| 48 | 23393 | -0.0602 | 121899 | 2.500e-40 | -13.299 | `(((NOT(spread_vs_trailing_100 >= p90) AND NOT(spread_percentile_100 == p25)) AND spread_vs_trailing_100 < p50) AND NOT(prior_session_low_distance >= p50))` |
| 49 | 32939 | -0.0971 | 48679 | 3.325e-40 | -13.285 | `((NOT(swing_high_distance_14 < p10) OR atr_14 == p75) AND (session_dead < p50 OR (atr_14 <= p10 OR day_of_week > p90)))` |
| 50 | 47242 | -0.0700 | 95786 | 9.990e-40 | -13.196 | `(session_ny >= p25 AND (spread_percentile_100 <= p10 OR NOT(atr_14 >= p10)))` |
| 51 | 28120 | -0.0669 | 99614 | 1.066e-39 | -13.191 | `((NOT(swing_low_distance_14 > p50) AND NOT(swing_high_distance_14 > p50)) AND NOT(session_london >= p90))` |
| 52 | 2157 | -0.0832 | 60379 | 1.386e-39 | -13.175 | `((spread_percentile_100 <= p25 AND session_ldn_ny_overlap <= p90) AND distance_to_round_number > p50)` |
| 53 | 12521 | -0.0588 | 124444 | 2.393e-39 | -13.129 | `((NOT(swing_high_distance_14 == p90) AND spread_vs_trailing_100 < p50) AND swing_low_distance_14 < p50)` |
| 54 | 48558 | -0.0478 | 196140 | 4.896e-39 | -13.073 | `((distance_to_round_number < p25 AND day_of_week > p25) OR atr_14 <= p25)` |
| 55 | 17148 | -0.0544 | 144489 | 6.236e-39 | -13.055 | `((NOT(spread_vs_trailing_100 >= p50) AND NOT(session_ldn_ny_overlap != p10)) AND NOT(atr_14 <= p25))` |
| 56 | 3921 | -0.0547 | 140540 | 1.059e-38 | -13.015 | `((spread_percentile_100 < p50 AND (atr_vs_trailing_100 >= p25 AND session_ldn_ny_overlap == p50)) AND (NOT(kijun_26_distance < p25) OR NOT(atr_14 > p90)))` |
| 57 | 18975 | -0.0688 | 91595 | 1.404e-38 | -12.995 | `((spread_vs_trailing_100 >= p25 AND (atr_14 <= p25 AND session_tokyo <= p75)) OR (spread_vs_trailing_100 >= p25 AND atr_percentile_100 == p90))` |
| 58 | 38994 | -0.0661 | 93707 | 2.837e-38 | -12.941 | `(((NOT(session_ldn_ny_overlap != p50) AND NOT(hour_of_day <= p50)) OR (NOT(atr_percentile_100 != p10) AND distance_to_round_number != p75)) AND spread_vs_trailing_100 < p50)` |
| 59 | 3095 | -0.0509 | 162875 | 7.618e-38 | -12.863 | `(spread_percentile_100 <= p25 OR (session_london != p90 AND spread_vs_trailing_100 < p25))` |
| 60 | 6996 | -0.0566 | 120563 | 8.547e-37 | -12.675 | `(atr_vs_trailing_100 > p50 AND ((spread_percentile_100 == p25 OR session_dead <= p90) AND NOT(spread_percentile_100 >= p50)))` |
| 61 | 33508 | -0.0607 | 109450 | 9.853e-37 | -12.665 | `(spread_percentile_100 < p25 AND ((swing_low_distance_14 < p75 AND NOT(swing_low_distance_14 == p90)) OR NOT(session_ny <= p25)))` |
| 62 | 40488 | -0.0656 | 86350 | 1.756e-36 | -12.620 | `((spread_vs_trailing_100 <= p50 AND (NOT(swing_low_distance_14 > p10) OR session_london < p75)) AND range_close_ratio > p50)` |
| 63 | 20297 | -0.0592 | 112133 | 2.167e-36 | -12.602 | `(NOT(atr_vs_trailing_100 < p25) AND (spread_vs_trailing_100 <= p25 OR (NOT(swing_high_distance_14 > p10) AND atr_vs_trailing_100 <= p90)))` |
| 64 | 40913 | -0.0746 | 66109 | 8.302e-35 | -12.314 | `((NOT(spread_percentile_100 > p75) AND NOT(swing_high_distance_14 > p50)) AND ((swing_low_distance_14 <= p75 AND session_london < p90) AND NOT(spread_vs_trailing_100 > p50)))` |
| 65 | 22305 | -0.0581 | 118116 | 1.615e-34 | -12.257 | `((NOT(spread_percentile_100 != p75) OR NOT(spread_vs_trailing_100 > p50)) AND (kijun_26_distance != p90 AND range_close_ratio <= p50))` |
| 66 | 31121 | -0.0532 | 133590 | 3.364e-34 | -12.197 | `(((session_tokyo <= p50 AND (NOT(spread_percentile_100 < p25) AND NOT(session_ldn_ny_overlap != p90))) OR kijun_26_distance > p50) AND NOT(spread_percentile_100 >= p50))` |
| 67 | 34038 | -0.0554 | 121701 | 1.722e-33 | -12.063 | `(((NOT(session_ny == p50) OR NOT(atr_percentile_100 > p10)) AND (kijun_26_distance > p90 AND session_ny >= p90)) OR NOT(spread_percentile_100 >= p25))` |
| 68 | 36605 | -0.0476 | 163493 | 2.507e-33 | -12.032 | `(NOT(atr_vs_trailing_100 == p90) AND ((NOT(session_london != p90) OR swing_low_distance_14 >= p50) AND NOT(spread_percentile_100 >= p50)))` |
| 69 | 1788 | -0.0582 | 108347 | 2.957e-33 | -12.019 | `(session_ny == p10 AND spread_vs_trailing_100 <= p50)` |
| 70 | 12456 | -0.0582 | 108347 | 2.957e-33 | -12.019 | `(NOT(spread_vs_trailing_100 > p50) AND session_ny < p50)` |
| 71 | 39864 | -0.0582 | 108347 | 2.957e-33 | -12.019 | `((NOT(spread_vs_trailing_100 > p50) AND session_ny < p75) AND NOT(session_london > p75))` |
| 72 | 8286 | -0.0784 | 59022 | 4.503e-33 | -11.988 | `(atr_14 < p50 AND NOT(spread_percentile_100 < p75))` |
| 73 | 33316 | -0.0627 | 88132 | 4.936e-33 | -11.978 | `(NOT(swing_high_distance_14 >= p75) AND spread_vs_trailing_100 < p25)` |
| 74 | 14633 | -0.0567 | 112876 | 7.623e-33 | -11.940 | `(NOT(spread_percentile_100 != p50) OR ((spread_vs_trailing_100 < p50 OR session_ny == p50) AND NOT(session_ny >= p50)))` |
| 75 | 10629 | -0.0602 | 97620 | 2.511e-32 | -11.841 | `(((NOT(prior_session_low_distance == p90) AND NOT(prior_session_low_distance > p90)) AND spread_vs_trailing_100 < p50) AND session_tokyo != p50)` |
| 76 | 28877 | -0.1129 | 25843 | 6.256e-32 | -11.776 | `(NOT(session_ny < p25) AND (NOT(day_of_week <= p75) AND swing_high_distance_14 <= p25))` |
| 77 | 13885 | -0.0534 | 122413 | 8.434e-32 | -11.738 | `(NOT(prior_session_high_distance >= p50) AND spread_vs_trailing_100 <= p50)` |
| 78 | 12627 | -0.0557 | 110110 | 1.735e-31 | -11.677 | `((prior_session_high_distance <= p90 AND (kijun_26_distance > p90 OR spread_vs_trailing_100 <= p50)) AND (range_close_ratio != p90 AND NOT(prior_session_low_distance <= p50)))` |
| 79 | 20292 | -0.0883 | 43134 | 1.805e-31 | -11.680 | `((swing_high_distance_14 <= p10 AND ((NOT(swing_low_distance_14 != p50) OR NOT(atr_percentile_100 == p75)) OR NOT(day_of_week > p90))) AND atr_percentile_100 >= p10)` |
| 80 | 12886 | -0.0873 | 45712 | 2.104e-31 | -11.666 | `((NOT(range_close_ratio >= p90) AND NOT(swing_high_distance_14 > p10)) AND NOT(session_dead < p10))` |
| 81 | 11703 | -0.0547 | 116488 | 2.776e-31 | -11.637 | `(NOT(spread_percentile_100 >= p25) AND NOT(day_of_week > p90))` |
| 82 | 8008 | -0.0546 | 116404 | 3.490e-31 | -11.618 | `((NOT(session_tokyo < p25) AND prior_session_high_distance != p10) AND ((kijun_26_distance < p50 OR swing_low_distance_14 != p75) AND NOT(spread_percentile_100 >= p25)))` |
| 83 | 31266 | -0.0875 | 42973 | 9.190e-31 | -11.540 | `(NOT(swing_high_distance_14 >= p10) AND (NOT(atr_percentile_100 < p10) OR range_close_ratio == p75))` |
| 84 | 46360 | -0.0875 | 42973 | 9.190e-31 | -11.540 | `(NOT(atr_percentile_100 < p10) AND swing_high_distance_14 < p10)` |
| 85 | 930 | -0.0557 | 109200 | 9.289e-31 | -11.534 | `(NOT(atr_vs_trailing_100 < p10) AND NOT(spread_percentile_100 > p25))` |
| 86 | 33870 | -0.0557 | 109200 | 9.289e-31 | -11.534 | `(spread_percentile_100 <= p25 AND atr_vs_trailing_100 > p10)` |
| 87 | 12863 | -0.0534 | 119811 | 9.800e-31 | -11.529 | `(NOT(spread_percentile_100 >= p25) OR NOT(prior_session_high_distance != p50))` |
| 88 | 3802 | -0.0534 | 119773 | 1.041e-30 | -11.524 | `((spread_percentile_100 < p25 OR NOT(atr_vs_trailing_100 != p50)) AND (NOT(session_tokyo < p10) OR prior_session_high_distance < p10))` |
| 89 | 16675 | -0.0534 | 119767 | 1.053e-30 | -11.523 | `((spread_vs_trailing_100 != p75 OR kijun_26_distance == p10) AND NOT(spread_percentile_100 >= p25))` |
| 90 | 130 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `NOT(spread_percentile_100 >= p25)` |
| 91 | 2940 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `spread_percentile_100 < p25` |
| 92 | 6775 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `((NOT(atr_vs_trailing_100 >= p25) AND NOT(session_ny <= p90)) OR NOT(spread_percentile_100 >= p25))` |
| 93 | 8183 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `spread_percentile_100 < p25` |
| 94 | 9699 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `spread_percentile_100 < p25` |
| 95 | 11049 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `spread_percentile_100 < p25` |
| 96 | 12394 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(((NOT(hour_of_day == p50) OR hour_of_day >= p50) OR NOT(atr_14 <= p50)) AND NOT(spread_percentile_100 >= p25))` |
| 97 | 22698 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `spread_percentile_100 < p25` |
| 98 | 25217 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(NOT(session_dead < p25) AND spread_percentile_100 < p25)` |
| 99 | 29957 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `((NOT(session_ny > p90) OR NOT(session_ldn_ny_overlap <= p90)) AND spread_percentile_100 < p25)` |
| 100 | 29981 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(NOT(session_ny < p10) AND spread_percentile_100 < p25)` |
| 101 | 31987 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `((spread_percentile_100 < p25 AND (NOT(kijun_26_distance == p10) OR NOT(range_close_ratio <= p75))) AND session_london <= p75)` |
| 102 | 37967 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `NOT(spread_percentile_100 >= p25)` |
| 103 | 41428 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `NOT(spread_percentile_100 >= p25)` |
| 104 | 42476 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(NOT(session_dead <= p75) OR ((swing_high_distance_14 < p25 OR NOT(session_tokyo < p25)) AND NOT(spread_percentile_100 >= p25)))` |
| 105 | 42974 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `NOT(spread_percentile_100 >= p25)` |
| 106 | 43588 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `NOT(spread_percentile_100 >= p25)` |
| 107 | 45048 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(NOT(session_dead <= p90) OR NOT(spread_percentile_100 >= p25))` |
| 108 | 45718 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(NOT(spread_percentile_100 >= p25) OR (session_dead >= p90 AND session_ldn_ny_overlap != p50))` |
| 109 | 46000 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(spread_percentile_100 < p25 OR (session_ldn_ny_overlap < p50 AND NOT(distance_to_round_number > p75)))` |
| 110 | 47624 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `(((NOT(distance_to_round_number >= p50) AND hour_of_day >= p90) OR (NOT(spread_percentile_100 > p25) OR distance_to_round_number >= p10)) AND NOT(spread_percentile_100 >= p25))` |
| 111 | 49785 | -0.0534 | 119772 | 1.063e-30 | -11.522 | `spread_percentile_100 < p25` |
| 112 | 38160 | -0.0534 | 119771 | 1.085e-30 | -11.520 | `(spread_percentile_100 < p25 AND (kijun_26_distance >= p25 OR NOT(spread_vs_trailing_100 == p75)))` |
| 113 | 47854 | -0.0535 | 118965 | 1.170e-30 | -11.514 | `((NOT(swing_low_distance_14 != p75) AND swing_high_distance_14 > p10) OR (spread_percentile_100 < p25 AND (NOT(atr_14 <= p25) OR NOT(prior_session_high_distance <= p10))))` |
| 114 | 11957 | -0.0534 | 119777 | 1.182e-30 | -11.513 | `(NOT(spread_percentile_100 >= p25) OR (NOT(day_of_week >= p50) AND NOT(prior_session_high_distance != p10)))` |
| 115 | 27572 | -0.0580 | 100013 | 1.360e-30 | -11.501 | `(spread_percentile_100 < p25 AND ((NOT(prior_session_low_distance < p25) OR session_tokyo > p90) OR (NOT(distance_to_round_number >= p90) AND prior_session_low_distance >= p10)))` |
| 116 | 11654 | -0.0532 | 119807 | 1.531e-30 | -11.490 | `(((swing_high_distance_14 == p90 OR range_close_ratio == p10) OR prior_session_low_distance == p75) OR NOT(spread_percentile_100 >= p25))` |
| 117 | 11744 | -0.0464 | 157802 | 1.932e-30 | -11.469 | `((NOT(prior_session_high_distance >= p10) OR NOT(spread_percentile_100 > p25)) AND NOT(session_dead < p25))` |
| 118 | 22781 | -0.0463 | 157818 | 2.057e-30 | -11.464 | `(NOT(atr_14 != p50) OR (prior_session_high_distance <= p10 OR spread_percentile_100 <= p25))` |
| 119 | 34360 | -0.0624 | 82501 | 3.451e-30 | -11.421 | `((NOT(session_ny < p90) AND spread_vs_trailing_100 <= p50) AND (day_of_week == p10 OR (kijun_26_distance >= p25 AND NOT(session_ldn_ny_overlap >= p90))))` |
| 120 | 5490 | -0.0656 | 75843 | 4.316e-30 | -11.402 | `((((session_london == p75 OR NOT(session_ldn_ny_overlap >= p10)) OR spread_percentile_100 <= p25) AND NOT(swing_low_distance_14 >= p90)) AND NOT(session_london != p10))` |
| 121 | 110 | -0.0571 | 102247 | 4.893e-30 | -11.390 | `((spread_percentile_100 < p25 AND NOT(prior_session_low_distance == p50)) AND NOT(prior_session_high_distance > p90))` |
| 122 | 22496 | -0.0550 | 108957 | 6.593e-30 | -11.364 | `(NOT(distance_to_round_number <= p10) AND (((session_london < p25 AND kijun_26_distance < p10) AND NOT(atr_14 == p90)) OR spread_percentile_100 < p25))` |
| 123 | 48382 | -0.0518 | 123509 | 7.507e-30 | -11.352 | `(NOT(spread_percentile_100 >= p25) OR NOT(atr_percentile_100 != p90))` |
| 124 | 43980 | -0.0819 | 48836 | 7.809e-30 | -11.353 | `((atr_percentile_100 != p75 AND NOT(swing_high_distance_14 >= p10)) AND (NOT(prior_session_high_distance == p10) OR spread_vs_trailing_100 != p90))` |
| 125 | 49271 | -0.0818 | 48774 | 8.453e-30 | -11.346 | `((spread_percentile_100 != p75 OR prior_session_low_distance > p75) AND (NOT(swing_high_distance_14 >= p10) OR NOT(swing_high_distance_14 != p75)))` |
| 126 | 45317 | -0.0498 | 131767 | 1.564e-29 | -11.287 | `(spread_percentile_100 < p25 OR (((swing_low_distance_14 < p25 OR distance_to_round_number <= p25) AND swing_low_distance_14 > p90) AND NOT(hour_of_day < p75)))` |
| 127 | 41600 | -0.0668 | 67101 | 2.466e-29 | -11.250 | `(spread_vs_trailing_100 < p50 AND (day_of_week > p75 OR NOT(range_close_ratio <= p90)))` |
| 128 | 37980 | -0.0622 | 82917 | 2.728e-29 | -11.240 | `(((session_tokyo == p10 OR session_london == p10) AND (NOT(atr_percentile_100 >= p25) OR NOT(hour_of_day != p50))) AND NOT(spread_vs_trailing_100 > p50))` |
| 129 | 7579 | -0.0533 | 112202 | 5.367e-29 | -11.179 | `(range_close_ratio > p10 AND spread_percentile_100 < p25)` |
| 130 | 29365 | -0.0533 | 112202 | 5.367e-29 | -11.179 | `((spread_percentile_100 < p25 AND NOT(range_close_ratio <= p10)) AND session_tokyo >= p25)` |
| 131 | 5691 | -0.0518 | 119977 | 6.495e-29 | -11.162 | `(NOT(spread_percentile_100 > p25) AND (spread_vs_trailing_100 <= p25 OR NOT(prior_session_high_distance == p10)))` |
| 132 | 47999 | -0.0572 | 92130 | 2.137e-28 | -11.056 | `((spread_vs_trailing_100 <= p25 AND NOT(kijun_26_distance >= p75)) AND atr_vs_trailing_100 != p90)` |
| 133 | 12770 | -0.0944 | 32077 | 2.385e-28 | -11.053 | `(swing_high_distance_14 < p10 AND range_close_ratio >= p25)` |
| 134 | 46662 | -0.1398 | 14468 | 2.510e-28 | -11.061 | `((NOT(session_tokyo != p10) AND NOT(atr_percentile_100 == p75)) AND (NOT(day_of_week < p90) AND swing_high_distance_14 < p25))` |
| 135 | 43080 | -0.0549 | 107819 | 3.481e-28 | -11.012 | `(((NOT(session_ny >= p90) AND (spread_percentile_100 < p50 AND NOT(spread_vs_trailing_100 <= p25))) AND NOT(spread_percentile_100 > p50)) OR range_close_ratio < p10)` |
| 136 | 4831 | -0.0448 | 156318 | 4.029e-28 | -10.997 | `((kijun_26_distance <= p10 AND NOT(hour_of_day > p90)) OR NOT(spread_percentile_100 >= p25))` |
| 137 | 40656 | -0.0879 | 39132 | 4.466e-28 | -10.994 | `(NOT(swing_high_distance_14 > p10) AND day_of_week != p50)` |
| 138 | 461 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(NOT(session_ldn_ny_overlap >= p50) OR NOT(spread_percentile_100 > p25))` |
| 139 | 804 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `spread_percentile_100 <= p25` |
| 140 | 5425 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 141 | 5824 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(NOT(spread_percentile_100 > p25) AND spread_vs_trailing_100 != p50)` |
| 142 | 10314 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 143 | 13782 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 144 | 16311 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(NOT(spread_percentile_100 > p25) OR (NOT(swing_low_distance_14 == p50) AND (hour_of_day == p50 AND NOT(session_tokyo <= p10))))` |
| 145 | 21465 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(NOT(session_london >= p10) OR NOT(spread_percentile_100 > p25))` |
| 146 | 25313 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `spread_percentile_100 <= p25` |
| 147 | 25978 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 148 | 26570 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(spread_percentile_100 <= p25 OR NOT(session_dead <= p75))` |
| 149 | 30060 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 150 | 31082 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `spread_percentile_100 <= p25` |
| 151 | 34047 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(spread_percentile_100 <= p25 AND (range_close_ratio != p25 OR session_dead >= p10))` |
| 152 | 36017 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `spread_percentile_100 <= p25` |
| 153 | 36805 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(((session_tokyo != p10 AND atr_14 != p75) OR (session_ny == p90 OR session_tokyo <= p25)) AND NOT(spread_percentile_100 > p25))` |
| 154 | 37837 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `((spread_percentile_100 == p25 OR (NOT(atr_14 == p10) OR (session_ldn_ny_overlap >= p10 AND NOT(session_ldn_ny_overlap > p90)))) AND spread_percentile_100 <= p25)` |
| 155 | 37878 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `spread_percentile_100 <= p25` |
| 156 | 38041 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `(spread_percentile_100 <= p25 AND session_ny <= p90)` |
| 157 | 38528 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 158 | 41823 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `spread_percentile_100 <= p25` |
| 159 | 42744 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `spread_percentile_100 <= p25` |
| 160 | 42820 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `((session_ldn_ny_overlap > p10 AND NOT(range_close_ratio != p90)) OR spread_percentile_100 <= p25)` |
| 161 | 46791 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 162 | 47165 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 163 | 47538 | -0.0502 | 123442 | 5.345e-28 | -10.972 | `NOT(spread_percentile_100 > p25)` |
| 164 | 11081 | -0.0502 | 123437 | 5.468e-28 | -10.970 | `(spread_percentile_100 <= p25 AND swing_high_distance_14 != p90)` |
| 165 | 13284 | -0.0502 | 123440 | 5.556e-28 | -10.969 | `((NOT(session_london >= p50) OR NOT(spread_vs_trailing_100 == p10)) AND (NOT(atr_14 == p10) AND spread_percentile_100 <= p25))` |
| 166 | 30092 | -0.0500 | 123478 | 7.887e-28 | -10.937 | `((spread_percentile_100 <= p25 OR NOT(kijun_26_distance != p90)) OR NOT(prior_session_low_distance != p75))` |
| 167 | 540 | -0.0876 | 38976 | 8.621e-28 | -10.935 | `(NOT(swing_high_distance_14 >= p10) AND NOT(day_of_week == p50))` |
| 168 | 32258 | -0.0464 | 141299 | 1.279e-27 | -10.893 | `((NOT(spread_percentile_100 >= p50) AND (NOT(day_of_week >= p90) AND NOT(atr_14 <= p25))) OR (NOT(swing_high_distance_14 > p10) AND swing_high_distance_14 > p50))` |
| 169 | 43240 | -0.0831 | 44253 | 1.322e-27 | -10.895 | `((NOT(atr_percentile_100 > p50) AND (kijun_26_distance >= p75 OR NOT(distance_to_round_number >= p25))) AND NOT(session_tokyo >= p75))` |
| 170 | 48137 | -0.0563 | 95676 | 1.965e-27 | -10.855 | `(spread_percentile_100 < p25 AND (NOT(prior_session_high_distance != p90) OR (session_london <= p50 OR NOT(session_ldn_ny_overlap != p25))))` |
| 171 | 4313 | -0.1342 | 13749 | 2.203e-27 | -10.864 | `((atr_vs_trailing_100 >= p90 AND NOT(atr_vs_trailing_100 < p50)) AND (session_ldn_ny_overlap < p90 AND NOT(swing_low_distance_14 >= p50)))` |
| 172 | 21160 | -0.0899 | 34554 | 3.397e-27 | -10.810 | `(atr_vs_trailing_100 > p25 AND NOT(swing_high_distance_14 >= p10))` |
| 173 | 11675 | -0.0529 | 106460 | 3.771e-27 | -10.795 | `((NOT(atr_percentile_100 <= p10) AND spread_percentile_100 < p25) OR range_close_ratio == p75)` |
| 174 | 19871 | -0.0691 | 59139 | 5.512e-27 | -10.762 | `((NOT(prior_session_high_distance <= p75) AND NOT(swing_low_distance_14 < p25)) AND spread_vs_trailing_100 < p50)` |
| 175 | 42533 | -0.0783 | 47064 | 7.426e-27 | -10.736 | `(NOT(swing_high_distance_14 >= p90) AND ((session_ldn_ny_overlap == p10 AND NOT(spread_percentile_100 < p90)) AND swing_high_distance_14 <= p75))` |
| 176 | 41013 | -0.0432 | 159580 | 1.049e-26 | -10.699 | `(prior_session_low_distance == p75 OR (kijun_26_distance <= p10 OR NOT(spread_percentile_100 > p25)))` |
| 177 | 25201 | -0.0566 | 90005 | 1.505e-26 | -10.667 | `(((NOT(spread_vs_trailing_100 > p50) AND NOT(kijun_26_distance >= p50)) AND NOT(session_ldn_ny_overlap > p10)) AND NOT(range_close_ratio > p90))` |
| 178 | 40269 | -0.0445 | 146452 | 2.691e-26 | -10.612 | `(spread_percentile_100 < p25 OR spread_vs_trailing_100 < p10)` |
| 179 | 3037 | -0.0578 | 86134 | 2.830e-26 | -10.608 | `((NOT(atr_vs_trailing_100 != p25) OR NOT(prior_session_low_distance < p25)) AND NOT(spread_percentile_100 >= p25))` |
| 180 | 3767 | -0.0541 | 96290 | 3.321e-26 | -10.593 | `(spread_percentile_100 < p25 AND (NOT(range_close_ratio <= p25) OR NOT(range_close_ratio < p90)))` |
| 181 | 12799 | -0.0504 | 109437 | 8.792e-26 | -10.501 | `(NOT(prior_session_low_distance > p90) AND spread_vs_trailing_100 < p25)` |
| 182 | 31203 | -0.0850 | 37588 | 1.072e-25 | -10.487 | `(((NOT(swing_high_distance_14 > p50) AND NOT(hour_of_day <= p10)) AND spread_vs_trailing_100 > p75) AND distance_to_round_number > p25)` |
| 183 | 17258 | -0.0544 | 92750 | 1.417e-25 | -10.456 | `((NOT(spread_vs_trailing_100 > p50) AND kijun_26_distance != p90) AND (session_london != p10 OR NOT(spread_vs_trailing_100 != p25)))` |
| 184 | 6508 | -0.0564 | 87196 | 1.499e-25 | -10.451 | `((NOT(spread_vs_trailing_100 >= p50) AND hour_of_day > p10) AND (session_ny == p10 OR NOT(session_ny == p75)))` |
| 185 | 29107 | -0.0543 | 92750 | 1.611e-25 | -10.444 | `(spread_vs_trailing_100 < p50 AND NOT(session_london != p90))` |
| 186 | 6165 | -0.0543 | 92749 | 1.646e-25 | -10.442 | `(spread_vs_trailing_100 < p50 AND (session_london > p10 AND atr_14 != p10))` |
| 187 | 43549 | -0.0768 | 47015 | 2.615e-25 | -10.401 | `(swing_high_distance_14 < p10 AND (NOT(range_close_ratio >= p25) OR (NOT(spread_vs_trailing_100 >= p90) AND distance_to_round_number >= p10)))` |
| 188 | 37769 | -0.0809 | 41952 | 2.859e-25 | -10.393 | `((NOT(atr_14 > p25) OR NOT(atr_percentile_100 != p10)) AND session_dead >= p75)` |
| 189 | 43107 | -0.0567 | 85582 | 2.929e-25 | -10.387 | `((spread_vs_trailing_100 <= p50 AND (NOT(distance_to_round_number == p75) AND NOT(session_tokyo < p75))) AND prior_session_high_distance <= p75)` |
| 190 | 11741 | -0.0492 | 115048 | 3.806e-25 | -10.362 | `(NOT(session_london < p10) AND (NOT(spread_percentile_100 > p25) AND (NOT(hour_of_day <= p10) OR kijun_26_distance > p50)))` |
| 191 | 16541 | -0.0431 | 149594 | 3.952e-25 | -10.357 | `(spread_vs_trailing_100 <= p10 OR NOT(spread_percentile_100 > p25))` |
| 192 | 5711 | -0.0650 | 63509 | 6.336e-25 | -10.315 | `(prior_session_low_distance > p10 AND (NOT(spread_percentile_100 >= p25) AND day_of_week >= p50))` |
| 193 | 42838 | -0.0528 | 98632 | 6.937e-25 | -10.304 | `((NOT(kijun_26_distance == p10) AND NOT(session_ldn_ny_overlap == p90)) AND NOT(spread_percentile_100 > p25))` |
| 194 | 5004 | -0.0574 | 82009 | 8.528e-25 | -10.285 | `(NOT(session_london > p25) AND spread_percentile_100 < p25)` |
| 195 | 31406 | -0.0712 | 47053 | 9.164e-25 | -10.281 | `((NOT(range_close_ratio < p75) AND NOT(spread_percentile_100 > p50)) AND ((atr_percentile_100 != p50 OR session_london < p90) AND NOT(swing_low_distance_14 >= p75)))` |
| 196 | 28213 | -0.0445 | 138106 | 9.608e-25 | -10.272 | `((kijun_26_distance >= p75 AND (NOT(session_london != p75) AND distance_to_round_number < p25)) OR NOT(spread_percentile_100 > p25))` |
| 197 | 32327 | -0.0871 | 32542 | 2.065e-24 | -10.204 | `((day_of_week > p50 OR NOT(atr_percentile_100 > p10)) AND ((atr_14 < p25 OR distance_to_round_number >= p90) AND atr_percentile_100 >= p50))` |
| 198 | 7119 | -0.0799 | 41225 | 2.430e-24 | -10.187 | `((NOT(swing_low_distance_14 != p90) OR NOT(session_tokyo < p25)) AND (hour_of_day != p25 AND swing_high_distance_14 < p10))` |
| 199 | 46988 | -0.0407 | 160018 | 2.912e-24 | -10.164 | `(NOT(spread_percentile_100 > p25) OR swing_low_distance_14 >= p90)` |
| 200 | 13159 | -0.0887 | 33013 | 3.150e-24 | -10.163 | `(swing_high_distance_14 <= p10 AND NOT(session_london >= p90))` |
| 201 | 42616 | -0.0886 | 32869 | 4.248e-24 | -10.134 | `((session_london != p90 OR spread_vs_trailing_100 == p90) AND swing_high_distance_14 < p10)` |
| 202 | 33697 | -0.1118 | 18614 | 6.301e-24 | -10.101 | `((atr_14 <= p75 AND kijun_26_distance > p50) AND (distance_to_round_number != p90 AND NOT(spread_vs_trailing_100 <= p90)))` |
| 203 | 26057 | -0.0991 | 24621 | 8.517e-24 | -10.068 | `(atr_14 < p50 AND kijun_26_distance >= p75)` |
| 204 | 3165 | -0.0569 | 85254 | 1.199e-23 | -10.027 | `((session_dead >= p90 AND NOT(swing_high_distance_14 < p75)) OR ((distance_to_round_number <= p90 OR day_of_week >= p75) AND atr_14 <= p10))` |
| 205 | 43162 | -0.0490 | 107808 | 1.560e-23 | -10.000 | `(spread_percentile_100 < p25 AND (NOT(session_dead <= p90) OR NOT(distance_to_round_number >= p90)))` |
| 206 | 38750 | -0.0672 | 54309 | 2.163e-23 | -9.970 | `(NOT(swing_low_distance_14 < p50) AND ((NOT(prior_session_high_distance != p50) OR atr_14 < p50) AND NOT(atr_14 > p90)))` |
| 207 | 7432 | -0.0547 | 84402 | 3.345e-23 | -9.925 | `(spread_percentile_100 <= p25 AND (session_london < p75 AND spread_percentile_100 < p75))` |
| 208 | 43287 | -0.0558 | 81114 | 4.595e-23 | -9.893 | `(((distance_to_round_number > p75 OR NOT(atr_percentile_100 < p75)) OR session_ldn_ny_overlap == p50) AND (spread_percentile_100 < p25 AND NOT(day_of_week >= p90)))` |
| 209 | 113 | -0.0589 | 67391 | 9.548e-23 | -9.820 | `(spread_percentile_100 <= p25 AND range_close_ratio >= p50)` |
| 210 | 27464 | -0.0541 | 84547 | 1.800e-22 | -9.755 | `((NOT(day_of_week <= p90) OR (NOT(session_dead <= p75) OR (spread_percentile_100 <= p50 AND NOT(session_tokyo <= p50)))) AND prior_session_low_distance < p90)` |
| 211 | 12907 | -0.0766 | 41633 | 2.631e-22 | -9.719 | `(prior_session_low_distance < p75 AND (NOT(range_close_ratio >= p75) AND spread_percentile_100 >= p90))` |
| 212 | 6489 | -0.0550 | 77972 | 4.572e-22 | -9.660 | `(NOT(spread_percentile_100 > p25) AND (atr_percentile_100 >= p25 AND NOT(prior_session_high_distance >= p90)))` |
| 213 | 33008 | -0.0766 | 40583 | 6.811e-22 | -9.622 | `(day_of_week != p90 AND NOT(swing_high_distance_14 >= p10))` |
| 214 | 22331 | -0.0974 | 23271 | 9.085e-22 | -9.596 | `((hour_of_day == p90 AND NOT(kijun_26_distance < p25)) AND (NOT(session_ny != p90) AND (day_of_week > p75 AND session_ny >= p50)))` |
| 215 | 12658 | -0.0609 | 57182 | 9.348e-22 | -9.588 | `(atr_percentile_100 >= p75 AND (NOT(spread_vs_trailing_100 >= p50) AND hour_of_day <= p90))` |
| 216 | 27463 | -0.0682 | 47807 | 1.461e-21 | -9.542 | `(NOT(day_of_week != p90) AND NOT(spread_vs_trailing_100 >= p50))` |
| 217 | 11998 | -0.0857 | 31662 | 1.795e-21 | -9.523 | `((session_london != p25 OR NOT(hour_of_day >= p75)) AND swing_high_distance_14 < p10)` |
| 218 | 40426 | -0.0829 | 33745 | 2.292e-21 | -9.497 | `(((NOT(session_london == p25) AND NOT(atr_vs_trailing_100 >= p75)) AND NOT(swing_high_distance_14 >= p25)) AND atr_percentile_100 <= p90)` |
| 219 | 29370 | -0.0395 | 149413 | 2.371e-21 | -9.489 | `(spread_percentile_100 < p25 OR ((day_of_week != p75 AND kijun_26_distance >= p90) AND NOT(session_ny > p90)))` |
| 220 | 31671 | -0.0580 | 64776 | 2.566e-21 | -9.482 | `((range_close_ratio > p50 AND session_tokyo < p75) AND atr_14 < p50)` |
| 221 | 19155 | -0.0488 | 97813 | 3.026e-21 | -9.464 | `(NOT(spread_percentile_100 > p25) AND (NOT(session_dead != p75) OR kijun_26_distance > p25))` |
| 222 | 44691 | -0.0428 | 124781 | 3.985e-21 | -9.435 | `((spread_vs_trailing_100 <= p25 AND NOT(prior_session_high_distance >= p90)) OR ((prior_session_high_distance < p10 AND session_tokyo != p25) AND atr_14 > p25))` |
| 223 | 37000 | -0.1025 | 20911 | 6.566e-21 | -9.390 | `(NOT(atr_vs_trailing_100 >= p50) AND ((NOT(session_dead < p10) AND NOT(kijun_26_distance <= p50)) AND session_ldn_ny_overlap > p50))` |
| 224 | 48352 | -0.0778 | 36181 | 1.093e-20 | -9.332 | `(range_close_ratio != p25 AND ((NOT(day_of_week <= p25) AND prior_session_high_distance > p50) AND (swing_high_distance_14 < p25 AND NOT(session_ny != p10))))` |
| 225 | 1350 | -0.0918 | 24340 | 1.266e-20 | -9.319 | `(NOT(distance_to_round_number < p90) AND (NOT(spread_vs_trailing_100 > p50) OR distance_to_round_number <= p10))` |
| 226 | 28115 | -0.0450 | 111324 | 1.376e-20 | -9.304 | `(NOT(kijun_26_distance != p25) OR (NOT(spread_percentile_100 > p25) AND distance_to_round_number <= p90))` |
| 227 | 33923 | -0.0918 | 24100 | 1.891e-20 | -9.277 | `(distance_to_round_number > p90 AND (kijun_26_distance == p50 OR spread_vs_trailing_100 < p50))` |
| 228 | 16362 | -0.0626 | 54971 | 2.663e-20 | -9.235 | `(prior_session_high_distance >= p10 AND ((session_tokyo == p90 AND (spread_vs_trailing_100 < p50 AND swing_low_distance_14 <= p50)) AND NOT(atr_vs_trailing_100 == p25)))` |
| 229 | 4301 | -0.1127 | 16703 | 5.386e-20 | -9.168 | `((NOT(atr_vs_trailing_100 > p75) AND (NOT(prior_session_low_distance == p75) AND NOT(prior_session_low_distance < p75))) AND (day_of_week == p90 AND atr_percentile_100 <= p90))` |
| 230 | 28580 | -0.0604 | 59056 | 5.412e-20 | -9.159 | `(NOT(kijun_26_distance > p90) AND (session_ldn_ny_overlap <= p50 AND NOT(spread_percentile_100 != p90)))` |
| 231 | 40997 | -0.0923 | 23384 | 6.120e-20 | -9.150 | `((swing_high_distance_14 <= p10 AND prior_session_low_distance > p50) AND (NOT(atr_14 >= p10) OR NOT(session_london != p50)))` |
| 232 | 26482 | -0.0486 | 88439 | 6.652e-20 | -9.135 | `(NOT(atr_vs_trailing_100 <= p25) AND spread_vs_trailing_100 <= p25)` |
| 233 | 22679 | -0.0525 | 76167 | 9.725e-20 | -9.094 | `((NOT(swing_high_distance_14 < p50) AND (spread_vs_trailing_100 < p50 OR range_close_ratio == p50)) AND session_london < p90)` |
| 234 | 15282 | -0.0944 | 23725 | 1.212e-19 | -9.076 | `((NOT(swing_high_distance_14 != p90) OR (swing_high_distance_14 <= p10 AND swing_low_distance_14 != p75)) AND NOT(session_tokyo < p75))` |
| 235 | 20078 | -0.0564 | 64665 | 1.231e-19 | -9.069 | `((atr_14 <= p90 AND spread_vs_trailing_100 < p25) AND session_london != p90)` |
| 236 | 25651 | -0.0942 | 23721 | 1.373e-19 | -9.062 | `(swing_high_distance_14 <= p10 AND (NOT(session_ny == p75) AND NOT(swing_high_distance_14 >= p75)))` |
| 237 | 43857 | -0.0546 | 69755 | 1.606e-19 | -9.040 | `(NOT(session_ny == p50) AND (NOT(spread_percentile_100 > p50) AND NOT(prior_session_high_distance <= p25)))` |
| 238 | 19161 | -0.0554 | 69772 | 1.619e-19 | -9.039 | `(((NOT(spread_percentile_100 > p25) AND atr_vs_trailing_100 != p10) AND day_of_week >= p75) OR (atr_vs_trailing_100 < p25 AND spread_percentile_100 == p90))` |
| 239 | 16597 | -0.0412 | 123089 | 2.057e-19 | -9.012 | `(kijun_26_distance == p50 OR spread_vs_trailing_100 < p25)` |
| 240 | 1615 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 241 | 1719 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(((session_ldn_ny_overlap > p90 AND (NOT(session_ny >= p75) OR day_of_week < p10)) OR NOT(spread_vs_trailing_100 != p25)) OR NOT(spread_vs_trailing_100 >= p25))` |
| 242 | 6200 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `((NOT(session_tokyo >= p50) OR (session_tokyo < p90 AND (session_london < p10 OR session_ny > p90))) OR NOT(spread_vs_trailing_100 >= p25))` |
| 243 | 6979 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 244 | 7754 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 245 | 8054 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(spread_vs_trailing_100 <= p25 AND (NOT(kijun_26_distance == p75) OR ((NOT(prior_session_low_distance < p75) OR distance_to_round_number <= p90) OR NOT(session_tokyo != p25))))` |
| 246 | 8808 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 247 | 9036 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(NOT(spread_vs_trailing_100 > p25) OR ((NOT(atr_vs_trailing_100 != p75) AND session_ny >= p10) AND NOT(prior_session_high_distance > p10)))` |
| 248 | 10453 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 249 | 13381 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 250 | 13429 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 251 | 13808 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 252 | 14093 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 253 | 14269 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 254 | 14496 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 255 | 17173 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `((NOT(session_ny == p50) OR atr_14 != p10) AND NOT(spread_vs_trailing_100 > p25))` |
| 256 | 17670 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 257 | 17900 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 258 | 18253 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 259 | 18430 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 260 | 18510 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 261 | 18673 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 262 | 19523 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 263 | 21471 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 264 | 21582 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 265 | 23286 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 266 | 25031 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 267 | 25362 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `((NOT(swing_high_distance_14 == p50) OR atr_vs_trailing_100 <= p50) AND NOT(spread_vs_trailing_100 > p25))` |
| 268 | 26766 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 269 | 26805 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 270 | 26867 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 271 | 26879 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 272 | 26918 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 273 | 27108 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 274 | 27158 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 275 | 27314 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 276 | 28529 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 277 | 30279 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 278 | 30416 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 279 | 32620 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 280 | 32892 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 281 | 33130 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `((session_dead >= p50 AND spread_vs_trailing_100 < p25) AND (NOT(atr_vs_trailing_100 == p90) OR NOT(prior_session_low_distance >= p75)))` |
| 282 | 34283 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 283 | 35612 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 284 | 35955 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 285 | 36006 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(spread_vs_trailing_100 <= p25 OR NOT(session_tokyo >= p50))` |
| 286 | 37798 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(NOT(session_tokyo >= p25) OR spread_vs_trailing_100 <= p25)` |
| 287 | 38816 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 288 | 40801 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 289 | 41280 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 290 | 42642 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 291 | 43219 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 >= p25)` |
| 292 | 43663 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `NOT(spread_vs_trailing_100 > p25)` |
| 293 | 44258 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 294 | 44364 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(spread_vs_trailing_100 <= p25 OR session_ldn_ny_overlap < p75)` |
| 295 | 45707 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 < p25` |
| 296 | 46685 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(spread_vs_trailing_100 < p25 OR ((distance_to_round_number == p25 AND session_tokyo < p25) AND NOT(hour_of_day == p25)))` |
| 297 | 47085 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(session_ldn_ny_overlap > p90 OR spread_vs_trailing_100 <= p25)` |
| 298 | 47961 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 299 | 48111 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `(NOT(spread_vs_trailing_100 >= p25) AND (kijun_26_distance <= p90 OR atr_14 != p50))` |
| 300 | 49625 | -0.0411 | 123050 | 2.349e-19 | -8.997 | `spread_vs_trailing_100 <= p25` |
| 301 | 40818 | -0.0411 | 123046 | 2.505e-19 | -8.990 | `((spread_vs_trailing_100 < p25 OR atr_vs_trailing_100 == p75) AND kijun_26_distance != p75)` |
| 302 | 36209 | -0.0411 | 123038 | 2.509e-19 | -8.990 | `(NOT(swing_high_distance_14 == p25) AND NOT(spread_vs_trailing_100 >= p25))` |
| 303 | 42772 | -0.0411 | 123039 | 2.613e-19 | -8.985 | `((session_ny == p90 OR prior_session_high_distance != p90) AND spread_vs_trailing_100 < p25)` |
| 304 | 43342 | -0.0413 | 121616 | 2.635e-19 | -8.984 | `(spread_vs_trailing_100 <= p25 AND NOT(spread_percentile_100 == p50))` |
| 305 | 22967 | -0.0869 | 26977 | 2.758e-19 | -8.985 | `(swing_high_distance_14 < p10 AND (hour_of_day < p75 AND spread_percentile_100 != p90))` |
| 306 | 25690 | -0.0474 | 94017 | 3.002e-19 | -8.971 | `(NOT(atr_14 != p50) OR (NOT(spread_percentile_100 > p50) AND session_tokyo == p90))` |
| 307 | 31900 | -0.0936 | 23614 | 3.052e-19 | -8.975 | `((NOT(session_ny > p75) AND session_tokyo == p90) AND NOT(swing_high_distance_14 >= p10))` |
| 308 | 46416 | -0.0936 | 23614 | 3.052e-19 | -8.975 | `((swing_high_distance_14 < p10 AND (NOT(swing_high_distance_14 >= p25) AND NOT(session_tokyo == p10))) AND NOT(session_tokyo < p50))` |
| 309 | 36061 | -0.0640 | 49286 | 3.370e-19 | -8.960 | `(day_of_week == p75 AND spread_percentile_100 <= p50)` |
| 310 | 38283 | -0.0611 | 54199 | 3.449e-19 | -8.957 | `((atr_14 < p50 OR NOT(session_london <= p75)) AND day_of_week >= p90)` |
| 311 | 19605 | -0.0640 | 49285 | 3.456e-19 | -8.957 | `(NOT(spread_percentile_100 > p50) AND ((NOT(atr_14 == p25) AND day_of_week == p75) AND (swing_low_distance_14 >= p50 OR kijun_26_distance <= p90)))` |
| 312 | 27395 | -0.0408 | 123003 | 4.103e-19 | -8.936 | `(NOT(spread_vs_trailing_100 >= p25) AND (NOT(prior_session_high_distance == p10) OR NOT(distance_to_round_number > p25)))` |
| 313 | 41825 | -0.0409 | 122667 | 4.129e-19 | -8.935 | `(spread_vs_trailing_100 <= p25 AND (atr_percentile_100 != p75 OR NOT(session_tokyo != p50)))` |
| 314 | 2380 | -0.0408 | 123008 | 4.261e-19 | -8.931 | `((NOT(prior_session_low_distance == p10) OR NOT(atr_vs_trailing_100 <= p50)) AND (spread_vs_trailing_100 < p25 OR session_london > p90))` |
| 315 | 11967 | -0.0882 | 26420 | 4.769e-19 | -8.924 | `(NOT(session_london >= p75) AND (NOT(range_close_ratio >= p75) AND swing_high_distance_14 <= p10))` |
| 316 | 25116 | -0.0407 | 122975 | 5.490e-19 | -8.903 | `((NOT(spread_vs_trailing_100 >= p25) AND NOT(prior_session_high_distance == p90)) OR session_ldn_ny_overlap < p50)` |
| 317 | 29731 | -0.0523 | 71260 | 5.562e-19 | -8.903 | `(((spread_vs_trailing_100 <= p50 AND NOT(prior_session_high_distance < p50)) AND NOT(session_ldn_ny_overlap < p75)) AND NOT(atr_14 <= p25))` |
| 318 | 8760 | -0.0480 | 89174 | 6.789e-19 | -8.880 | `(NOT(spread_percentile_100 > p25) AND (hour_of_day < p90 OR (atr_14 >= p50 AND NOT(spread_percentile_100 <= p90))))` |
| 319 | 9518 | -0.0480 | 89174 | 6.789e-19 | -8.880 | `(spread_percentile_100 <= p25 AND ((NOT(spread_percentile_100 > p50) OR atr_14 >= p10) AND hour_of_day <= p75))` |
| 320 | 30844 | -0.0609 | 48901 | 7.111e-19 | -8.877 | `((NOT(session_ldn_ny_overlap > p50) AND NOT(spread_percentile_100 > p50)) AND NOT(range_close_ratio <= p75))` |
| 321 | 31692 | -0.0407 | 121934 | 8.060e-19 | -8.861 | `((NOT(spread_percentile_100 == p50) OR day_of_week == p10) AND (spread_vs_trailing_100 <= p25 OR atr_14 == p10))` |
| 322 | 45044 | -0.0429 | 111031 | 8.267e-19 | -8.858 | `(NOT(swing_high_distance_14 <= p10) AND (NOT(session_ny > p90) AND spread_percentile_100 <= p25))` |
| 323 | 3313 | -0.0519 | 76168 | 1.056e-18 | -8.831 | `(((swing_high_distance_14 > p90 OR prior_session_low_distance > p90) OR (range_close_ratio <= p10 OR hour_of_day <= p10)) AND spread_vs_trailing_100 <= p50)` |
| 324 | 11110 | -0.0470 | 90980 | 1.828e-18 | -8.769 | `((range_close_ratio == p75 AND NOT(session_ny >= p90)) OR (session_ldn_ny_overlap <= p25 AND (spread_percentile_100 < p50 AND NOT(session_tokyo < p75))))` |
| 325 | 12195 | -0.0470 | 90980 | 1.828e-18 | -8.769 | `(session_ny != p90 AND NOT(spread_percentile_100 >= p50))` |
| 326 | 13491 | -0.0470 | 90980 | 1.828e-18 | -8.769 | `(NOT(session_ny == p90) AND (spread_percentile_100 < p50 AND NOT(session_dead < p10)))` |
| 327 | 30748 | -0.0470 | 90980 | 1.828e-18 | -8.769 | `((NOT(session_ldn_ny_overlap < p25) AND NOT(spread_percentile_100 >= p50)) AND NOT(session_tokyo < p90))` |
| 328 | 13753 | -0.0544 | 62080 | 1.939e-18 | -8.764 | `(atr_14 < p50 AND (NOT(range_close_ratio < p50) AND session_london < p90))` |
| 329 | 9839 | -0.0846 | 28619 | 2.069e-18 | -8.759 | `((NOT(swing_low_distance_14 == p50) AND ((range_close_ratio < p50 AND atr_percentile_100 == p90) OR swing_high_distance_14 < p10)) AND NOT(atr_percentile_100 > p50))` |
| 330 | 21314 | -0.0633 | 48413 | 3.733e-18 | -8.690 | `((NOT(atr_14 >= p25) OR NOT(swing_high_distance_14 >= p10)) AND (kijun_26_distance < p10 OR NOT(session_london < p75)))` |
| 331 | 34150 | -0.0711 | 36835 | 4.304e-18 | -8.675 | `(NOT(spread_vs_trailing_100 >= p50) AND (((NOT(atr_vs_trailing_100 != p25) OR NOT(hour_of_day != p25)) AND NOT(kijun_26_distance >= p90)) AND prior_session_low_distance != p75))` |
| 332 | 44786 | -0.0962 | 19434 | 5.751e-18 | -8.646 | `((swing_high_distance_14 < p10 AND NOT(day_of_week < p75)) OR NOT(atr_vs_trailing_100 != p90))` |
| 333 | 47858 | -0.0962 | 19434 | 5.751e-18 | -8.646 | `(swing_high_distance_14 < p10 AND NOT(day_of_week < p75))` |
| 334 | 35317 | -0.0441 | 98062 | 5.909e-18 | -8.636 | `(day_of_week != p25 AND NOT(spread_vs_trailing_100 > p25))` |
| 335 | 18331 | -0.0400 | 118537 | 7.077e-18 | -8.615 | `(((range_close_ratio > p50 OR (NOT(hour_of_day <= p10) AND NOT(prior_session_low_distance == p50))) AND NOT(range_close_ratio == p25)) AND spread_vs_trailing_100 <= p25)` |
| 336 | 45841 | -0.0438 | 98405 | 8.190e-18 | -8.599 | `(((NOT(session_london == p90) OR NOT(session_tokyo == p75)) AND NOT(spread_vs_trailing_100 >= p25)) OR (NOT(distance_to_round_number != p50) AND NOT(spread_vs_trailing_100 != p50)))` |
| 337 | 26091 | -0.0400 | 117151 | 1.151e-17 | -8.559 | `((distance_to_round_number == p10 OR NOT(kijun_26_distance == p50)) AND (hour_of_day >= p25 AND NOT(spread_vs_trailing_100 > p25)))` |
| 338 | 19733 | -0.0403 | 114697 | 1.213e-17 | -8.553 | `(NOT(kijun_26_distance == p25) AND (spread_vs_trailing_100 <= p25 AND NOT(range_close_ratio < p10)))` |
| 339 | 41106 | -0.0843 | 27589 | 1.259e-17 | -8.553 | `(swing_high_distance_14 <= p10 AND ((NOT(session_tokyo < p10) AND (spread_percentile_100 != p10 AND NOT(session_tokyo > p75))) AND atr_vs_trailing_100 < p50))` |
| 340 | 47911 | -0.0462 | 89116 | 1.425e-17 | -8.535 | `(prior_session_high_distance < p75 AND spread_percentile_100 <= p25)` |
| 341 | 29318 | -0.0714 | 37141 | 1.467e-17 | -8.534 | `(NOT(swing_high_distance_14 > p10) AND (atr_percentile_100 >= p90 OR spread_vs_trailing_100 > p25))` |
| 342 | 46170 | -0.0741 | 31681 | 1.547e-17 | -8.528 | `((NOT(spread_vs_trailing_100 >= p25) OR NOT(spread_vs_trailing_100 <= p75)) AND (distance_to_round_number > p75 AND (distance_to_round_number == p90 OR session_ny >= p50)))` |
| 343 | 23619 | -0.1064 | 14396 | 1.906e-17 | -8.510 | `(((NOT(atr_percentile_100 < p10) AND NOT(spread_vs_trailing_100 <= p90)) AND session_ny != p50) AND (NOT(spread_vs_trailing_100 > p75) OR swing_low_distance_14 >= p50))` |
| 344 | 48292 | -0.0471 | 86068 | 2.062e-17 | -8.492 | `((NOT(atr_percentile_100 > p90) AND session_tokyo != p10) AND NOT(spread_percentile_100 > p50))` |
| 345 | 20137 | -0.0335 | 172497 | 3.927e-17 | -8.416 | `(swing_high_distance_14 >= p75 OR ((NOT(session_ldn_ny_overlap > p90) AND swing_high_distance_14 == p75) OR (swing_low_distance_14 != p50 AND atr_14 < p10)))` |
| 346 | 12245 | -0.0956 | 18667 | 4.592e-17 | -8.405 | `(((spread_vs_trailing_100 > p90 AND (NOT(range_close_ratio == p10) OR spread_percentile_100 < p90)) AND day_of_week >= p75) AND atr_vs_trailing_100 <= p75)` |
| 347 | 16506 | -0.0502 | 69829 | 4.874e-17 | -8.392 | `(((NOT(range_close_ratio != p50) OR NOT(spread_vs_trailing_100 > p25)) AND atr_vs_trailing_100 >= p25) AND (NOT(session_dead >= p50) OR hour_of_day != p50))` |
| 348 | 9541 | -0.0684 | 36838 | 7.023e-17 | -8.351 | `(((spread_percentile_100 != p75 AND NOT(spread_vs_trailing_100 <= p75)) AND NOT(spread_percentile_100 > p75)) AND session_tokyo == p50)` |
| 349 | 37069 | -0.1039 | 14706 | 8.292e-17 | -8.337 | `(((day_of_week >= p75 AND NOT(prior_session_low_distance <= p25)) AND spread_vs_trailing_100 > p90) AND (NOT(prior_session_low_distance == p50) OR day_of_week > p90))` |
| 350 | 29235 | -0.0749 | 29971 | 1.058e-16 | -8.303 | `(((session_dead >= p90 OR NOT(session_ldn_ny_overlap > p50)) AND hour_of_day == p25) AND (spread_vs_trailing_100 <= p50 AND distance_to_round_number > p25))` |
| 351 | 49088 | -0.0515 | 62938 | 1.846e-16 | -8.234 | `(NOT(range_close_ratio <= p25) AND (NOT(day_of_week > p25) AND NOT(atr_14 > p50)))` |
| 352 | 48860 | -0.0776 | 27642 | 1.864e-16 | -8.236 | `((spread_vs_trailing_100 > p75 AND ((NOT(range_close_ratio > p50) OR NOT(kijun_26_distance >= p90)) AND spread_vs_trailing_100 >= p10)) AND hour_of_day > p75)` |
| 353 | 37441 | -0.0640 | 41163 | 1.884e-16 | -8.233 | `(((NOT(kijun_26_distance < p25) AND NOT(session_dead <= p10)) OR prior_session_high_distance == p75) AND NOT(spread_vs_trailing_100 < p75))` |
| 354 | 627 | -0.0424 | 96570 | 2.282e-16 | -8.208 | `((NOT(spread_vs_trailing_100 > p25) AND session_ldn_ny_overlap <= p75) AND NOT(session_london < p10))` |
| 355 | 19612 | -0.0424 | 96570 | 2.282e-16 | -8.208 | `(NOT(session_ldn_ny_overlap != p50) AND NOT(spread_vs_trailing_100 > p25))` |
| 356 | 41448 | -0.0471 | 78874 | 2.347e-16 | -8.205 | `(NOT(day_of_week <= p10) AND ((session_tokyo >= p75 AND kijun_26_distance != p10) AND spread_percentile_100 < p50))` |
| 357 | 20449 | -0.0423 | 96580 | 2.373e-16 | -8.203 | `((spread_percentile_100 == p90 OR session_ldn_ny_overlap < p90) AND (spread_vs_trailing_100 < p25 OR range_close_ratio == p75))` |
| 358 | 23831 | -0.1728 | 4262 | 2.455e-16 | -8.230 | `((NOT(kijun_26_distance <= p50) AND (atr_vs_trailing_100 >= p10 AND NOT(hour_of_day < p10))) AND (NOT(range_close_ratio < p75) AND spread_vs_trailing_100 >= p90))` |
| 359 | 42292 | -0.0731 | 32892 | 2.469e-16 | -8.201 | `(NOT(atr_vs_trailing_100 != p50) OR (swing_high_distance_14 < p10 AND (NOT(distance_to_round_number >= p75) OR NOT(swing_high_distance_14 != p50))))` |
| 360 | 6515 | -0.1044 | 14872 | 2.564e-16 | -8.202 | `(prior_session_low_distance < p90 AND (spread_vs_trailing_100 >= p75 AND hour_of_day == p90))` |
| 361 | 25418 | -0.0994 | 15900 | 4.184e-16 | -8.142 | `(NOT(day_of_week != p90) AND hour_of_day == p10)` |
| 362 | 33057 | -0.0522 | 59678 | 4.648e-16 | -8.123 | `(NOT(atr_percentile_100 < p50) AND spread_percentile_100 <= p25)` |
| 363 | 30158 | -0.0721 | 32388 | 4.751e-16 | -8.122 | `(((NOT(session_london != p75) OR NOT(prior_session_low_distance > p90)) AND NOT(session_dead != p10)) AND NOT(swing_high_distance_14 > p10))` |
| 364 | 2749 | -0.0939 | 19376 | 5.173e-16 | -8.114 | `((NOT(session_dead == p50) AND NOT(kijun_26_distance < p75)) AND NOT(atr_vs_trailing_100 > p50))` |
| 365 | 41548 | -0.0692 | 34940 | 6.859e-16 | -8.077 | `(NOT(hour_of_day < p25) AND (prior_session_high_distance >= p75 AND NOT(swing_high_distance_14 > p10)))` |
| 366 | 32217 | -0.0584 | 49426 | 7.418e-16 | -8.066 | `((session_dead != p10 AND distance_to_round_number <= p90) AND ((spread_vs_trailing_100 < p50 AND atr_vs_trailing_100 < p90) AND prior_session_high_distance <= p90))` |
| 367 | 46187 | -0.0964 | 16903 | 8.077e-16 | -8.061 | `((distance_to_round_number <= p90 AND NOT(day_of_week < p75)) AND NOT(swing_high_distance_14 > p10))` |
| 368 | 45139 | -0.1396 | 7431 | 8.544e-16 | -8.064 | `((NOT(prior_session_low_distance >= p10) AND NOT(day_of_week != p75)) AND (range_close_ratio <= p50 OR (NOT(spread_vs_trailing_100 <= p10) AND prior_session_high_distance != p10)))` |
| 369 | 39328 | -0.0861 | 21127 | 9.373e-16 | -8.041 | `(swing_low_distance_14 <= p10 AND NOT(atr_percentile_100 <= p50))` |
| 370 | 42471 | -0.0894 | 20896 | 9.602e-16 | -8.038 | `((NOT(swing_high_distance_14 > p10) AND ((session_tokyo != p10 AND NOT(session_dead < p25)) OR NOT(distance_to_round_number >= p25))) AND NOT(distance_to_round_number >= p90))` |
| 371 | 27283 | -0.0426 | 97048 | 1.250e-15 | -8.001 | `(NOT(swing_high_distance_14 <= p90) OR range_close_ratio <= p10)` |
| 372 | 11964 | -0.0417 | 94572 | 1.404e-15 | -7.986 | `(NOT(spread_vs_trailing_100 > p25) AND (day_of_week >= p25 AND NOT(session_ldn_ny_overlap < p10)))` |
| 373 | 34374 | -0.1017 | 15121 | 1.569e-15 | -7.980 | `(session_tokyo != p25 AND ((NOT(session_dead > p50) OR NOT(atr_percentile_100 <= p50)) AND swing_high_distance_14 <= p10))` |
| 374 | 48144 | -0.1406 | 7061 | 1.659e-15 | -7.983 | `((NOT(swing_low_distance_14 > p25) AND atr_vs_trailing_100 >= p75) AND (atr_vs_trailing_100 == p10 OR session_dead != p50))` |
| 375 | 19479 | -0.1024 | 15297 | 2.003e-15 | -7.949 | `(((session_dead == p90 AND swing_high_distance_14 < p10) AND atr_14 != p10) AND (atr_14 != p90 OR session_london != p90))` |
| 376 | 41783 | -0.1020 | 15377 | 2.126e-15 | -7.942 | `(swing_high_distance_14 <= p10 AND NOT(session_dead == p10))` |
| 377 | 8698 | -0.0569 | 46749 | 5.500e-15 | -7.818 | `(session_ldn_ny_overlap > p50 AND (session_ny != p50 OR NOT(spread_vs_trailing_100 >= p50)))` |
| 378 | 7724 | -0.0465 | 71925 | 6.199e-15 | -7.802 | `(NOT(spread_vs_trailing_100 >= p25) AND NOT(session_london > p10))` |
| 379 | 14397 | -0.1182 | 10219 | 7.126e-15 | -7.794 | `(day_of_week > p75 AND NOT(swing_high_distance_14 > p10))` |
| 380 | 8849 | -0.0632 | 37479 | 8.608e-15 | -7.761 | `((spread_vs_trailing_100 > p75 AND NOT(hour_of_day > p10)) AND (session_london <= p50 OR NOT(spread_vs_trailing_100 <= p50)))` |
| 381 | 43979 | -0.0567 | 46255 | 8.769e-15 | -7.759 | `(((atr_percentile_100 != p25 AND NOT(range_close_ratio == p50)) AND NOT(session_ldn_ny_overlap <= p10)) AND spread_vs_trailing_100 <= p50)` |
| 382 | 43482 | -0.0747 | 27776 | 9.670e-15 | -7.748 | `(session_dead != p10 AND (prior_session_high_distance > p75 AND prior_session_low_distance != p75))` |
| 383 | 12209 | -0.0764 | 26085 | 1.010e-14 | -7.742 | `((NOT(swing_high_distance_14 > p10) AND (NOT(atr_percentile_100 <= p10) AND kijun_26_distance < p75)) AND NOT(kijun_26_distance <= p25))` |
| 384 | 14155 | -0.0460 | 73923 | 1.030e-14 | -7.737 | `((spread_percentile_100 <= p50 AND NOT(session_ny > p25)) AND ((atr_vs_trailing_100 >= p25 OR NOT(distance_to_round_number > p25)) OR NOT(swing_high_distance_14 < p90)))` |
| 385 | 2587 | -0.0745 | 27778 | 1.174e-14 | -7.723 | `(prior_session_high_distance > p75 AND session_dead > p50)` |
| 386 | 39295 | -0.0311 | 158410 | 1.358e-14 | -7.701 | `(NOT(kijun_26_distance <= p90) OR (spread_vs_trailing_100 <= p25 OR session_tokyo < p50))` |
| 387 | 924 | -0.0371 | 111476 | 1.388e-14 | -7.698 | `((spread_vs_trailing_100 <= p25 OR (atr_vs_trailing_100 == p50 AND (NOT(session_ny <= p75) AND NOT(atr_14 >= p75)))) AND distance_to_round_number <= p90)` |
| 388 | 45103 | -0.0491 | 62589 | 1.471e-14 | -7.692 | `((spread_vs_trailing_100 <= p50 OR prior_session_high_distance == p25) AND swing_low_distance_14 <= p25)` |
| 389 | 46542 | -0.0616 | 39818 | 1.560e-14 | -7.685 | `((NOT(atr_14 > p50) OR session_ldn_ny_overlap > p10) AND NOT(hour_of_day >= p25))` |
| 390 | 16875 | -0.0483 | 63728 | 1.855e-14 | -7.662 | `(NOT(prior_session_low_distance > p25) AND NOT(spread_vs_trailing_100 > p50))` |
| 391 | 7450 | -0.0496 | 61839 | 2.331e-14 | -7.633 | `(NOT(prior_session_low_distance < p50) AND NOT(spread_percentile_100 > p25))` |
| 392 | 8842 | -0.0735 | 28286 | 2.425e-14 | -7.630 | `(((spread_vs_trailing_100 > p50 OR session_ldn_ny_overlap != p25) AND swing_high_distance_14 < p10) AND (NOT(spread_vs_trailing_100 == p75) AND NOT(distance_to_round_number == p75)))` |
| 393 | 41258 | -0.0930 | 16366 | 2.490e-14 | -7.629 | `(NOT(swing_high_distance_14 > p25) AND NOT(distance_to_round_number <= p90))` |
| 394 | 25990 | -0.1618 | 4508 | 3.341e-14 | -7.609 | `((NOT(session_tokyo > p25) AND (range_close_ratio != p25 AND NOT(distance_to_round_number == p90))) AND (NOT(atr_vs_trailing_100 <= p90) AND NOT(swing_low_distance_14 >= p25)))` |
| 395 | 46480 | -0.0500 | 63549 | 3.426e-14 | -7.583 | `(NOT(session_dead != p10) AND range_close_ratio <= p25)` |
| 396 | 26792 | -0.0669 | 32876 | 3.444e-14 | -7.584 | `(((NOT(session_london < p50) OR atr_percentile_100 <= p90) AND NOT(swing_low_distance_14 >= p50)) AND (NOT(atr_percentile_100 < p50) AND range_close_ratio < p25))` |
| 397 | 22546 | -0.0500 | 63546 | 3.450e-14 | -7.582 | `((range_close_ratio <= p25 AND session_dead < p90) AND (NOT(session_ldn_ny_overlap >= p75) OR NOT(kijun_26_distance == p50)))` |
| 398 | 23024 | -0.1103 | 11032 | 3.990e-14 | -7.571 | `(NOT(range_close_ratio == p10) AND ((NOT(swing_high_distance_14 > p25) AND NOT(range_close_ratio < p10)) AND (NOT(hour_of_day > p10) AND atr_percentile_100 <= p75)))` |
| 399 | 33397 | -0.0886 | 17852 | 4.278e-14 | -7.558 | `(NOT(day_of_week <= p50) AND ((NOT(spread_percentile_100 < p25) AND NOT(swing_high_distance_14 > p25)) AND NOT(session_tokyo > p25)))` |
| 400 | 27258 | -0.0600 | 38751 | 4.635e-14 | -7.545 | `(swing_low_distance_14 <= p75 AND spread_percentile_100 <= p10)` |
| 401 | 9814 | -0.0401 | 92636 | 4.998e-14 | -7.533 | `(spread_percentile_100 <= p25 AND NOT(atr_14 <= p25))` |
| 402 | 38180 | -0.0547 | 48107 | 6.062e-14 | -7.509 | `(NOT(atr_14 >= p50) AND ((kijun_26_distance <= p10 OR NOT(session_ny < p25)) AND day_of_week <= p10))` |
| 403 | 41142 | -0.0493 | 56858 | 6.097e-14 | -7.508 | `((NOT(prior_session_low_distance > p50) AND (session_dead != p75 AND NOT(spread_vs_trailing_100 > p50))) AND (hour_of_day < p90 AND range_close_ratio >= p25))` |
| 404 | 20165 | -0.0640 | 38535 | 6.465e-14 | -7.501 | `(((session_ny != p75 OR atr_percentile_100 >= p25) AND range_close_ratio < p10) AND (distance_to_round_number >= p50 OR NOT(distance_to_round_number < p25)))` |
| 405 | 33160 | -0.0727 | 27575 | 6.722e-14 | -7.497 | `(((prior_session_high_distance >= p75 AND session_dead != p50) AND (NOT(session_london > p75) OR NOT(spread_percentile_100 < p50))) AND NOT(atr_percentile_100 == p75))` |
| 406 | 33969 | -0.0543 | 49134 | 7.410e-14 | -7.483 | `((session_ny <= p50 OR NOT(spread_vs_trailing_100 <= p25)) AND ((NOT(atr_14 > p50) AND NOT(session_london == p25)) AND day_of_week < p75))` |
| 407 | 14611 | -0.0536 | 49829 | 8.027e-14 | -7.472 | `(distance_to_round_number <= p90 AND ((NOT(spread_vs_trailing_100 > p90) OR prior_session_high_distance >= p50) AND (spread_vs_trailing_100 > p75 AND day_of_week >= p50)))` |
| 408 | 38222 | -0.0748 | 24670 | 9.556e-14 | -7.451 | `((atr_percentile_100 >= p10 OR (NOT(kijun_26_distance > p50) OR NOT(prior_session_low_distance <= p50))) AND (NOT(session_ny <= p25) AND NOT(swing_high_distance_14 > p10)))` |
| 409 | 49204 | -0.0481 | 58412 | 9.786e-14 | -7.446 | `(spread_vs_trailing_100 < p25 AND atr_vs_trailing_100 > p50)` |
| 410 | 25711 | -0.0749 | 25860 | 9.945e-14 | -7.446 | `(hour_of_day == p25 AND (prior_session_high_distance >= p50 AND range_close_ratio <= p50))` |
| 411 | 3134 | -0.0710 | 27351 | 1.341e-13 | -7.406 | `(swing_high_distance_14 != p25 AND (swing_high_distance_14 < p10 AND (session_ny != p25 OR prior_session_high_distance > p90)))` |
| 412 | 5549 | -0.1140 | 8806 | 1.416e-13 | -7.407 | `(NOT(kijun_26_distance > p50) AND ((kijun_26_distance > p10 AND day_of_week <= p50) AND atr_vs_trailing_100 >= p90))` |
| 413 | 18665 | -0.0474 | 64000 | 1.749e-13 | -7.368 | `((prior_session_high_distance == p50 OR (NOT(prior_session_high_distance > p50) OR NOT(spread_vs_trailing_100 <= p90))) AND (NOT(spread_percentile_100 >= p25) AND NOT(prior_session_low_distance == p25)))` |
| 414 | 14065 | -0.0784 | 22720 | 1.805e-13 | -7.367 | `((spread_percentile_100 > p50 OR NOT(session_dead <= p90)) AND (NOT(prior_session_high_distance <= p50) AND NOT(swing_high_distance_14 >= p10)))` |
| 415 | 8012 | -0.0410 | 87686 | 2.064e-13 | -7.346 | `((range_close_ratio <= p10 OR atr_14 >= p90) AND (NOT(session_ldn_ny_overlap != p10) OR session_ldn_ny_overlap == p75))` |
| 416 | 45990 | -0.0609 | 39061 | 2.718e-13 | -7.310 | `(spread_percentile_100 >= p90 AND (atr_percentile_100 < p50 OR ((NOT(atr_14 != p50) OR spread_percentile_100 == p50) OR NOT(distance_to_round_number != p25))))` |
| 417 | 6673 | -0.0722 | 25620 | 3.117e-13 | -7.293 | `(((NOT(kijun_26_distance >= p10) OR NOT(session_ny != p90)) AND NOT(swing_high_distance_14 >= p10)) OR NOT(session_ny >= p10))` |
| 418 | 26554 | -0.0722 | 25620 | 3.117e-13 | -7.293 | `((NOT(swing_high_distance_14 >= p10) AND session_tokyo != p75) OR session_tokyo > p90)` |
| 419 | 30468 | -0.0722 | 25620 | 3.117e-13 | -7.293 | `((NOT(swing_high_distance_14 >= p10) OR (NOT(day_of_week != p75) AND swing_high_distance_14 < p10)) AND session_ny >= p50)` |
| 420 | 31151 | -0.0722 | 25620 | 3.117e-13 | -7.293 | `((NOT(swing_high_distance_14 < p75) OR NOT(session_tokyo == p90)) AND NOT(swing_high_distance_14 >= p10))` |
| 421 | 320 | -0.0698 | 29624 | 3.151e-13 | -7.291 | `((NOT(atr_percentile_100 < p25) AND (atr_percentile_100 < p10 OR NOT(range_close_ratio >= p10))) AND (range_close_ratio == p90 OR swing_high_distance_14 < p75))` |
| 422 | 24977 | -0.0601 | 40137 | 3.676e-13 | -7.269 | `(atr_vs_trailing_100 >= p10 AND range_close_ratio <= p10)` |
| 423 | 15487 | -0.0729 | 23716 | 4.555e-13 | -7.242 | `(((session_ldn_ny_overlap > p25 AND NOT(session_dead <= p90)) OR day_of_week >= p50) AND (NOT(kijun_26_distance <= p50) AND hour_of_day == p75))` |
| 424 | 48524 | -0.0894 | 17557 | 4.641e-13 | -7.241 | `(NOT(atr_vs_trailing_100 >= p50) AND (session_london == p25 AND NOT(swing_high_distance_14 >= p10)))` |
| 425 | 19899 | -0.0703 | 25680 | 5.210e-13 | -7.223 | `((NOT(spread_percentile_100 > p10) AND distance_to_round_number >= p50) AND (NOT(swing_high_distance_14 == p10) OR hour_of_day >= p75))` |
| 426 | 21007 | -0.0635 | 34483 | 5.615e-13 | -7.212 | `((prior_session_low_distance <= p50 AND NOT(atr_percentile_100 >= p90)) AND (NOT(spread_percentile_100 != p90) OR NOT(atr_14 != p50)))` |
| 427 | 17613 | -0.0732 | 23192 | 8.070e-13 | -7.164 | `((spread_vs_trailing_100 > p75 AND day_of_week == p90) AND NOT(session_ldn_ny_overlap > p90))` |
| 428 | 46970 | -0.0545 | 45355 | 8.174e-13 | -7.160 | `(NOT(distance_to_round_number >= p10) OR spread_percentile_100 == p50)` |
| 429 | 33682 | -0.0657 | 34505 | 9.762e-13 | -7.137 | `(atr_percentile_100 < p25 AND (session_tokyo < p50 OR range_close_ratio <= p25))` |
| 430 | 47343 | -0.0507 | 50856 | 1.029e-12 | -7.128 | `((NOT(spread_vs_trailing_100 > p50) AND (hour_of_day != p25 AND hour_of_day < p50)) AND (NOT(distance_to_round_number >= p75) OR NOT(spread_vs_trailing_100 <= p75)))` |
| 431 | 49883 | -0.0918 | 13991 | 1.035e-12 | -7.132 | `((NOT(hour_of_day != p75) AND (NOT(prior_session_low_distance >= p90) OR (NOT(spread_percentile_100 != p75) OR NOT(session_ny > p25)))) AND day_of_week == p75)` |
| 432 | 1033 | -0.0876 | 15376 | 1.076e-12 | -7.126 | `(NOT(atr_vs_trailing_100 <= p50) AND (swing_high_distance_14 <= p10 AND session_london == p25))` |
| 433 | 39374 | -0.0814 | 16171 | 1.371e-12 | -7.093 | `(day_of_week <= p25 AND NOT(atr_vs_trailing_100 < p90))` |
| 434 | 12917 | -0.0574 | 39902 | 1.563e-12 | -7.071 | `(distance_to_round_number < p10 OR NOT(swing_low_distance_14 != p25))` |
| 435 | 48230 | -0.0770 | 20955 | 1.625e-12 | -7.068 | `(prior_session_low_distance < p25 AND (day_of_week > p90 OR (NOT(swing_high_distance_14 >= p50) AND NOT(day_of_week > p10))))` |
| 436 | 18029 | -0.0466 | 59916 | 2.131e-12 | -7.027 | `(NOT(spread_percentile_100 >= p25) AND NOT(hour_of_day >= p75))` |
| 437 | 4834 | -0.0387 | 86932 | 2.233e-12 | -7.020 | `((NOT(spread_percentile_100 > p25) AND (session_ldn_ny_overlap == p90 OR (hour_of_day >= p50 OR swing_low_distance_14 >= p25))) AND NOT(distance_to_round_number >= p75))` |
| 438 | 4522 | -0.1138 | 9730 | 2.447e-12 | -7.015 | `((atr_percentile_100 < p10 AND session_london <= p90) AND (NOT(session_ny == p10) AND NOT(atr_14 <= p50)))` |
| 439 | 38090 | -0.0749 | 21128 | 2.682e-12 | -6.998 | `((NOT(session_tokyo != p50) AND distance_to_round_number >= p90) AND NOT(spread_percentile_100 >= p75))` |
| 440 | 1934 | -0.0637 | 32628 | 2.774e-12 | -6.991 | `((range_close_ratio <= p10 AND (atr_vs_trailing_100 > p10 AND swing_low_distance_14 > p10)) AND NOT(spread_percentile_100 == p50))` |
| 441 | 18797 | -0.0755 | 23884 | 2.877e-12 | -6.987 | `(range_close_ratio <= p10 AND NOT(kijun_26_distance <= p50))` |
| 442 | 9314 | -0.0568 | 39758 | 2.891e-12 | -6.985 | `(NOT(atr_vs_trailing_100 != p50) OR distance_to_round_number < p10)` |
| 443 | 3176 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 444 | 3281 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(NOT(distance_to_round_number >= p10) OR session_tokyo > p90)` |
| 445 | 4271 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 446 | 6963 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 447 | 7191 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `distance_to_round_number < p10` |
| 448 | 8296 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(NOT(distance_to_round_number >= p10) AND NOT(session_tokyo > p75))` |
| 449 | 10996 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(distance_to_round_number < p10 AND distance_to_round_number != p10)` |
| 450 | 13740 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `((session_ny > p75 AND spread_vs_trailing_100 == p90) OR distance_to_round_number < p10)` |
| 451 | 14118 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(((session_ny == p25 OR NOT(distance_to_round_number == p10)) OR swing_low_distance_14 >= p10) AND distance_to_round_number < p10)` |
| 452 | 14962 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(distance_to_round_number < p10 AND (NOT(session_dead <= p50) OR (day_of_week >= p10 OR swing_low_distance_14 <= p25)))` |
| 453 | 17630 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 454 | 17938 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 455 | 19685 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(session_tokyo > p75 OR distance_to_round_number < p10)` |
| 456 | 22575 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `distance_to_round_number < p10` |
| 457 | 24023 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(spread_vs_trailing_100 == p25 OR NOT(distance_to_round_number >= p10))` |
| 458 | 24300 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `distance_to_round_number < p10` |
| 459 | 24312 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 460 | 24466 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 461 | 24529 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(session_ny > p50 OR NOT(distance_to_round_number >= p10))` |
| 462 | 26401 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(NOT(session_dead <= p75) OR NOT(distance_to_round_number >= p10))` |
| 463 | 28613 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `distance_to_round_number < p10` |
| 464 | 30966 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `((atr_14 <= p10 AND session_dead < p10) OR NOT(distance_to_round_number >= p10))` |
| 465 | 30973 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 466 | 31194 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 467 | 33029 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(distance_to_round_number < p10 OR range_close_ratio == p10)` |
| 468 | 33405 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 469 | 34934 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(NOT(distance_to_round_number >= p10) OR session_tokyo < p10)` |
| 470 | 37161 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(((day_of_week >= p50 OR NOT(prior_session_low_distance == p25)) OR NOT(atr_percentile_100 == p25)) AND distance_to_round_number < p10)` |
| 471 | 37539 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(NOT(distance_to_round_number >= p10) OR session_tokyo > p90)` |
| 472 | 38931 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(atr_14 != p75 AND distance_to_round_number < p10)` |
| 473 | 40150 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(NOT(distance_to_round_number >= p10) OR (prior_session_high_distance == p25 AND atr_14 == p25))` |
| 474 | 41558 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(distance_to_round_number < p10 AND (session_tokyo >= p50 OR NOT(day_of_week <= p10)))` |
| 475 | 44467 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `(session_tokyo > p75 OR NOT(distance_to_round_number >= p10))` |
| 476 | 45696 | -0.0568 | 39757 | 2.954e-12 | -6.982 | `NOT(distance_to_round_number >= p10)` |
| 477 | 34541 | -0.0567 | 39691 | 3.310e-12 | -6.966 | `(distance_to_round_number < p10 AND (session_london != p90 OR swing_high_distance_14 > p10))` |
| 478 | 11167 | -0.0762 | 19947 | 4.257e-12 | -6.933 | `(NOT(session_dead != p50) AND (swing_low_distance_14 > p50 AND NOT(swing_high_distance_14 > p10)))` |
| 479 | 31764 | -0.0511 | 52490 | 5.253e-12 | -6.900 | `(range_close_ratio < p10 OR (day_of_week > p90 AND distance_to_round_number < p25))` |
| 480 | 35066 | -0.0560 | 43934 | 5.282e-12 | -6.900 | `(NOT(prior_session_low_distance > p90) AND (NOT(prior_session_high_distance == p75) AND (session_dead <= p75 AND range_close_ratio < p10)))` |
| 481 | 1811 | -0.0464 | 58884 | 5.301e-12 | -6.899 | `(NOT(spread_percentile_100 != p90) AND NOT(range_close_ratio > p75))` |
| 482 | 338 | -0.0600 | 37418 | 5.469e-12 | -6.895 | `(range_close_ratio <= p10 AND ((NOT(spread_percentile_100 != p10) AND distance_to_round_number == p10) OR NOT(session_tokyo == p50)))` |
| 483 | 10059 | -0.0600 | 37417 | 5.587e-12 | -6.892 | `((session_tokyo != p25 AND (NOT(range_close_ratio >= p10) OR session_ldn_ny_overlap == p90)) AND range_close_ratio < p50)` |
| 484 | 24969 | -0.0600 | 37417 | 5.587e-12 | -6.892 | `(session_ny == p25 AND range_close_ratio <= p10)` |
| 485 | 36824 | -0.0600 | 37417 | 5.587e-12 | -6.892 | `(range_close_ratio <= p10 AND (((NOT(atr_vs_trailing_100 != p75) OR NOT(kijun_26_distance == p25)) OR NOT(range_close_ratio > p25)) AND session_ny == p10))` |
| 486 | 46093 | -0.0600 | 37417 | 5.587e-12 | -6.892 | `(range_close_ratio < p10 AND session_ny < p90)` |
| 487 | 30761 | -0.0647 | 29339 | 5.787e-12 | -6.888 | `(NOT(distance_to_round_number >= p10) AND ((swing_low_distance_14 > p50 AND (distance_to_round_number == p90 AND session_london > p50)) OR NOT(spread_percentile_100 >= p90)))` |
| 488 | 39983 | -0.0570 | 38293 | 5.804e-12 | -6.887 | `(NOT(distance_to_round_number >= p10) AND prior_session_low_distance != p50)` |
| 489 | 39249 | -0.0558 | 43947 | 6.225e-12 | -6.876 | `((prior_session_low_distance <= p90 AND NOT(range_close_ratio > p10)) AND (session_ny == p50 OR hour_of_day <= p75))` |
| 490 | 34112 | -0.0548 | 40869 | 7.096e-12 | -6.858 | `(range_close_ratio != p75 AND ((spread_vs_trailing_100 <= p10 OR atr_14 < p50) AND NOT(hour_of_day > p10)))` |
| 491 | 14945 | -0.0562 | 39177 | 7.203e-12 | -6.856 | `((distance_to_round_number < p10 AND NOT(swing_high_distance_14 <= p25)) AND (kijun_26_distance <= p50 OR NOT(atr_14 < p50)))` |
| 492 | 42236 | -0.0626 | 30594 | 7.362e-12 | -6.853 | `(range_close_ratio < p50 AND NOT(spread_vs_trailing_100 <= p90))` |
| 493 | 43085 | -0.0728 | 23543 | 7.786e-12 | -6.846 | `(NOT(swing_high_distance_14 >= p10) AND NOT(spread_vs_trailing_100 <= p50))` |
| 494 | 41187 | -0.0714 | 22391 | 7.846e-12 | -6.845 | `((NOT(session_london < p25) OR swing_high_distance_14 >= p10) AND (session_ny < p90 AND (spread_vs_trailing_100 >= p90 AND kijun_26_distance >= p25)))` |
| 495 | 19945 | -0.1223 | 7225 | 8.331e-12 | -6.844 | `((swing_high_distance_14 == p50 OR prior_session_high_distance >= p90) AND NOT(day_of_week < p90))` |
| 496 | 43635 | -0.1221 | 7223 | 9.255e-12 | -6.829 | `(NOT(day_of_week != p90) AND prior_session_high_distance > p90)` |
| 497 | 49142 | -0.1221 | 7223 | 9.255e-12 | -6.829 | `(NOT(prior_session_high_distance <= p90) AND day_of_week > p75)` |
| 498 | 47043 | -0.0519 | 44205 | 1.009e-11 | -6.807 | `(prior_session_high_distance == p90 OR (NOT(spread_vs_trailing_100 >= p50) AND (atr_14 > p25 AND NOT(session_dead <= p25))))` |
| 499 | 37458 | -0.0610 | 33395 | 1.038e-11 | -6.804 | `((NOT(day_of_week <= p75) OR hour_of_day == p75) AND (spread_vs_trailing_100 > p10 AND NOT(range_close_ratio > p25)))` |
| 500 | 41379 | -0.0546 | 40091 | 1.054e-11 | -6.801 | `(NOT(session_ny >= p75) AND (atr_14 <= p50 AND (atr_14 <= p75 AND distance_to_round_number < p50)))` |
| 501 | 32877 | -0.1001 | 10151 | 1.143e-11 | -6.795 | `((prior_session_high_distance < p25 OR NOT(kijun_26_distance != p75)) AND NOT(spread_vs_trailing_100 < p90))` |
| 502 | 30505 | -0.0518 | 49082 | 1.580e-11 | -6.742 | `(NOT(range_close_ratio >= p10) AND ((NOT(swing_high_distance_14 > p90) OR NOT(atr_vs_trailing_100 > p50)) OR NOT(prior_session_high_distance < p10)))` |
| 503 | 21277 | -0.0851 | 17150 | 1.639e-11 | -6.740 | `((NOT(kijun_26_distance <= p50) AND range_close_ratio < p10) AND ((NOT(atr_14 == p90) AND hour_of_day != p50) AND prior_session_low_distance < p75))` |
| 504 | 4339 | -0.0525 | 47209 | 1.760e-11 | -6.726 | `(((session_dead <= p50 OR day_of_week < p10) OR session_london <= p10) AND (NOT(range_close_ratio >= p10) AND session_ldn_ny_overlap <= p10))` |
| 505 | 20150 | -0.0525 | 47209 | 1.760e-11 | -6.726 | `(NOT(range_close_ratio > p10) AND (NOT(spread_vs_trailing_100 == p50) AND NOT(session_ldn_ny_overlap > p25)))` |
| 506 | 28804 | -0.1105 | 8282 | 1.959e-11 | -6.718 | `(spread_vs_trailing_100 > p75 AND (day_of_week >= p90 AND (session_ny == p25 AND NOT(atr_14 < p50))))` |
| 507 | 24526 | -0.0535 | 45510 | 2.193e-11 | -6.694 | `((prior_session_low_distance != p75 OR (NOT(prior_session_low_distance != p50) OR session_london == p75)) AND NOT(range_close_ratio > p10))` |
| 508 | 44993 | -0.0632 | 24198 | 2.212e-11 | -6.694 | `((NOT(spread_vs_trailing_100 >= p50) OR NOT(atr_14 != p50)) AND NOT(range_close_ratio < p90))` |
| 509 | 23895 | -0.0908 | 13085 | 2.262e-11 | -6.694 | `(day_of_week == p90 AND (NOT(session_dead <= p75) OR (swing_low_distance_14 >= p10 AND (spread_vs_trailing_100 <= p90 AND NOT(hour_of_day != p90)))))` |
| 510 | 48194 | -0.0508 | 49919 | 2.644e-11 | -6.667 | `(range_close_ratio < p10 OR NOT(distance_to_round_number != p90))` |
| 511 | 45930 | -0.0519 | 47509 | 2.759e-11 | -6.660 | `((NOT(range_close_ratio >= p10) AND session_ldn_ny_overlap >= p10) AND (NOT(session_london != p25) OR NOT(spread_percentile_100 == p90)))` |
| 512 | 49771 | -0.0532 | 45373 | 2.902e-11 | -6.653 | `(NOT(prior_session_low_distance == p50) AND NOT(range_close_ratio > p10))` |
| 513 | 35069 | -0.0558 | 36911 | 2.909e-11 | -6.653 | `(NOT(distance_to_round_number >= p10) AND NOT(spread_percentile_100 < p10))` |
| 514 | 6967 | +0.0852 | 15564 | 2.981e-11 | 6.652 | `((prior_session_low_distance >= p90 OR NOT(session_ldn_ny_overlap == p25)) AND (NOT(atr_percentile_100 < p75) AND kijun_26_distance > p50))` |
| 515 | 11546 | -0.0514 | 48212 | 3.070e-11 | -6.645 | `((NOT(hour_of_day != p75) OR NOT(range_close_ratio >= p10)) AND hour_of_day != p75)` |
| 516 | 32244 | -0.0517 | 47769 | 3.081e-11 | -6.644 | `((session_ldn_ny_overlap > p90 OR range_close_ratio <= p10) AND (NOT(session_london > p10) OR (spread_percentile_100 >= p75 OR NOT(session_ny == p75))))` |
| 517 | 39053 | -0.0505 | 49866 | 3.479e-11 | -6.626 | `(NOT(range_close_ratio > p10) OR (distance_to_round_number == p90 AND atr_vs_trailing_100 <= p90))` |
| 518 | 30531 | -0.0625 | 28574 | 3.696e-11 | -6.618 | `((spread_vs_trailing_100 >= p90 AND range_close_ratio < p50) AND ((NOT(day_of_week > p10) AND atr_percentile_100 != p50) OR kijun_26_distance < p90))` |
| 519 | 38319 | -0.0508 | 49247 | 3.702e-11 | -6.617 | `(NOT(spread_percentile_100 == p10) AND NOT(range_close_ratio > p10))` |
| 520 | 44306 | -0.0399 | 73037 | 3.796e-11 | -6.613 | `(NOT(kijun_26_distance > p10) OR (NOT(swing_low_distance_14 == p75) AND distance_to_round_number < p10))` |
| 521 | 21271 | -0.0446 | 57131 | 4.050e-11 | -6.604 | `(NOT(prior_session_high_distance > p75) AND NOT(spread_percentile_100 < p90))` |
| 522 | 12694 | -0.0586 | 33316 | 4.183e-11 | -6.600 | `(distance_to_round_number < p10 AND (NOT(hour_of_day == p25) OR NOT(session_tokyo >= p50)))` |
| 523 | 27102 | -0.0505 | 49549 | 4.248e-11 | -6.597 | `(range_close_ratio < p10 OR ((NOT(prior_session_high_distance > p25) AND range_close_ratio >= p50) AND atr_14 < p25))` |
| 524 | 1331 | -0.0413 | 66179 | 4.283e-11 | -6.595 | `(NOT(kijun_26_distance >= p10) OR (spread_vs_trailing_100 >= p90 AND ((distance_to_round_number < p50 OR NOT(kijun_26_distance >= p10)) AND session_london == p25)))` |
| 525 | 46061 | -0.2070 | 2080 | 4.686e-11 | -6.616 | `((swing_low_distance_14 <= p25 AND ((session_tokyo < p50 OR NOT(hour_of_day >= p75)) AND NOT(atr_percentile_100 < p90))) AND session_london > p10)` |
| 526 | 27897 | -0.0503 | 49475 | 4.958e-11 | -6.574 | `(range_close_ratio <= p10 OR ((prior_session_low_distance >= p75 AND atr_percentile_100 < p90) AND (swing_high_distance_14 <= p50 AND NOT(distance_to_round_number != p25))))` |
| 527 | 46637 | -0.0502 | 49442 | 5.873e-11 | -6.548 | `((prior_session_low_distance == p10 AND session_ldn_ny_overlap != p50) OR range_close_ratio <= p10)` |
| 528 | 40391 | -0.0501 | 49505 | 5.883e-11 | -6.548 | `(NOT(range_close_ratio >= p10) OR (NOT(kijun_26_distance != p75) OR swing_low_distance_14 == p10))` |
| 529 | 48368 | -0.0612 | 28056 | 5.918e-11 | -6.548 | `((NOT(distance_to_round_number >= p10) AND distance_to_round_number < p90) AND atr_vs_trailing_100 > p25)` |
| 530 | 2245 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `((NOT(session_dead != p90) AND atr_14 == p50) OR NOT(range_close_ratio >= p10))` |
| 531 | 2647 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 532 | 4323 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(range_close_ratio <= p10 AND range_close_ratio <= p75)` |
| 533 | 5322 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(((swing_high_distance_14 < p25 OR swing_high_distance_14 <= p90) OR (NOT(swing_high_distance_14 < p50) OR NOT(atr_percentile_100 > p75))) AND NOT(range_close_ratio > p10))` |
| 534 | 8181 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `((session_london < p25 AND (swing_low_distance_14 >= p50 AND NOT(atr_14 > p25))) OR (NOT(range_close_ratio >= p10) AND NOT(session_ny > p50)))` |
| 535 | 8429 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio > p10)` |
| 536 | 12642 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 537 | 13202 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio > p10)` |
| 538 | 13724 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio <= p10` |
| 539 | 14627 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(range_close_ratio < p10 AND NOT(range_close_ratio >= p10))` |
| 540 | 15862 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(range_close_ratio < p10 AND NOT(session_ldn_ny_overlap < p10))` |
| 541 | 18305 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio > p10)` |
| 542 | 20210 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio > p10)` |
| 543 | 20573 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 544 | 20740 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 545 | 20853 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 546 | 21496 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 547 | 22006 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(NOT(atr_14 == p10) AND (NOT(range_close_ratio >= p10) OR NOT(session_ny <= p90)))` |
| 548 | 22150 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 549 | 23748 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(range_close_ratio < p10 OR NOT(session_tokyo <= p90))` |
| 550 | 25657 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio <= p10` |
| 551 | 26703 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(NOT(range_close_ratio >= p10) OR range_close_ratio == p90)` |
| 552 | 27691 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio <= p10` |
| 553 | 28099 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio <= p10` |
| 554 | 28363 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(session_ldn_ny_overlap >= p10 AND NOT(range_close_ratio > p10))` |
| 555 | 28853 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 556 | 29391 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio > p10)` |
| 557 | 30397 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `((NOT(kijun_26_distance > p10) AND range_close_ratio == p25) OR NOT(range_close_ratio > p10))` |
| 558 | 30989 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio <= p10` |
| 559 | 31237 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 560 | 31417 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 561 | 31854 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 562 | 32527 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 563 | 32679 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 564 | 37148 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio <= p10` |
| 565 | 37261 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio > p10)` |
| 566 | 37554 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 567 | 39143 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(NOT(range_close_ratio >= p10) AND (NOT(session_dead != p75) OR session_dead <= p10))` |
| 568 | 39460 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 569 | 40526 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 570 | 41474 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `(NOT(session_london < p25) AND (range_close_ratio < p10 AND NOT(session_ny < p25)))` |
| 571 | 42004 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio <= p10` |
| 572 | 42227 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 573 | 42407 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio >= p10)` |
| 574 | 42447 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 575 | 48002 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `((NOT(hour_of_day <= p50) AND (NOT(session_ldn_ny_overlap == p75) AND (atr_vs_trailing_100 < p50 AND prior_session_low_distance != p10))) OR NOT(range_close_ratio >= p10))` |
| 576 | 49538 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `range_close_ratio < p10` |
| 577 | 49900 | -0.0501 | 49438 | 5.920e-11 | -6.547 | `NOT(range_close_ratio > p10)` |
| 578 | 14292 | -0.1069 | 9279 | 6.142e-11 | -6.548 | `(((NOT(spread_vs_trailing_100 >= p75) AND session_dead > p50) AND spread_percentile_100 > p25) AND NOT(swing_high_distance_14 > p10))` |
| 579 | 2977 | -0.0933 | 11936 | 6.457e-11 | -6.539 | `((NOT(swing_low_distance_14 >= p10) AND NOT(distance_to_round_number <= p75)) AND atr_vs_trailing_100 >= p25)` |
| 580 | 30869 | -0.0493 | 45461 | 6.461e-11 | -6.534 | `(spread_percentile_100 < p10 AND (swing_low_distance_14 < p90 OR NOT(session_tokyo < p75)))` |
| 581 | 42299 | -0.0503 | 48973 | 6.560e-11 | -6.532 | `(atr_percentile_100 != p50 AND range_close_ratio <= p10)` |
| 582 | 6143 | -0.0500 | 49426 | 6.789e-11 | -6.527 | `(NOT(range_close_ratio >= p10) AND atr_vs_trailing_100 != p50)` |
| 583 | 8450 | -0.0500 | 49426 | 6.789e-11 | -6.527 | `(NOT(range_close_ratio > p10) AND (atr_vs_trailing_100 != p50 OR NOT(swing_high_distance_14 < p75)))` |
| 584 | 29548 | -0.0500 | 49324 | 6.862e-11 | -6.525 | `(range_close_ratio < p10 AND (NOT(spread_vs_trailing_100 <= p10) OR ((distance_to_round_number >= p50 OR atr_vs_trailing_100 >= p90) OR NOT(atr_vs_trailing_100 >= p75))))` |
| 585 | 37583 | -0.0499 | 49463 | 7.426e-11 | -6.513 | `(NOT(range_close_ratio > p10) OR ((NOT(range_close_ratio < p50) AND prior_session_low_distance == p25) AND (kijun_26_distance >= p10 OR NOT(session_ny > p10))))` |
| 586 | 20442 | -0.0663 | 26969 | 7.431e-11 | -6.514 | `(((NOT(range_close_ratio >= p10) AND session_dead != p50) AND swing_high_distance_14 != p25) AND (session_tokyo <= p90 OR distance_to_round_number > p10))` |
| 587 | 22415 | -0.0437 | 59906 | 7.574e-11 | -6.510 | `(NOT(spread_percentile_100 < p90) AND atr_vs_trailing_100 < p75)` |
| 588 | 16312 | -0.0684 | 23954 | 7.663e-11 | -6.510 | `(NOT(distance_to_round_number >= p10) AND NOT(kijun_26_distance < p10))` |
| 589 | 23142 | -0.0542 | 41511 | 8.209e-11 | -6.498 | `(NOT(distance_to_round_number < p25) AND (range_close_ratio <= p10 OR ((NOT(atr_vs_trailing_100 <= p75) OR prior_session_high_distance == p50) AND NOT(session_ny >= p10))))` |
| 590 | 30855 | -0.0661 | 26971 | 8.468e-11 | -6.495 | `((range_close_ratio <= p10 AND hour_of_day <= p25) OR NOT(session_ldn_ny_overlap >= p10))` |
| 591 | 36560 | -0.0661 | 26971 | 8.468e-11 | -6.495 | `((NOT(session_london < p50) OR (atr_14 == p75 OR distance_to_round_number <= p25)) AND (NOT(range_close_ratio > p10) AND session_dead == p75))` |
| 592 | 29160 | -0.0770 | 17636 | 9.351e-11 | -6.481 | `((session_london < p90 AND NOT(session_ny < p75)) AND swing_high_distance_14 <= p10)` |
| 593 | 12390 | -0.1199 | 6703 | 9.423e-11 | -6.486 | `((distance_to_round_number >= p90 AND spread_percentile_100 <= p75) AND (prior_session_low_distance != p50 AND NOT(hour_of_day <= p75)))` |
| 594 | 13797 | -0.0574 | 33360 | 1.040e-10 | -6.463 | `(NOT(distance_to_round_number >= p10) AND ((NOT(prior_session_low_distance == p25) OR NOT(atr_14 == p10)) AND session_ldn_ny_overlap < p90))` |
| 595 | 40733 | -0.0401 | 67778 | 1.119e-10 | -6.451 | `((NOT(session_ldn_ny_overlap != p50) AND NOT(spread_percentile_100 < p90)) AND ((NOT(session_dead > p10) OR kijun_26_distance <= p90) OR NOT(swing_high_distance_14 < p50)))` |
| 596 | 21911 | -0.0401 | 67817 | 1.207e-10 | -6.439 | `(NOT(session_ldn_ny_overlap != p75) AND (spread_percentile_100 == p90 AND ((NOT(swing_high_distance_14 == p10) AND NOT(day_of_week > p90)) OR hour_of_day >= p10)))` |
| 597 | 28459 | -0.0401 | 67817 | 1.207e-10 | -6.439 | `(NOT(session_ldn_ny_overlap != p25) AND NOT(spread_percentile_100 < p90))` |
| 598 | 39857 | +0.1562 | 5778 | 1.229e-10 | 6.448 | `(NOT(atr_percentile_100 < p25) AND NOT(atr_vs_trailing_100 >= p10))` |
| 599 | 39088 | -0.0493 | 49104 | 1.439e-10 | -6.413 | `((atr_percentile_100 != p75 OR NOT(session_tokyo >= p10)) AND NOT(range_close_ratio >= p10))` |
| 600 | 49185 | -0.0944 | 11383 | 1.577e-10 | -6.404 | `(NOT(session_dead < p25) AND (NOT(spread_vs_trailing_100 < p90) AND swing_high_distance_14 <= p25))` |
| 601 | 32558 | -0.0483 | 50100 | 1.893e-10 | -6.371 | `((NOT(prior_session_low_distance > p90) AND range_close_ratio < p10) OR distance_to_round_number == p25)` |
| 602 | 22201 | -0.0641 | 23746 | 1.953e-10 | -6.368 | `((atr_vs_trailing_100 != p10 AND (session_tokyo > p10 AND spread_vs_trailing_100 > p90)) AND (session_tokyo > p25 AND NOT(atr_percentile_100 <= p25)))` |
| 603 | 48238 | -0.0526 | 42126 | 2.093e-10 | -6.356 | `((prior_session_high_distance == p25 OR NOT(range_close_ratio >= p10)) AND (NOT(session_london != p50) OR day_of_week <= p25))` |
| 604 | 9452 | -0.0547 | 35491 | 2.146e-10 | -6.352 | `((NOT(spread_percentile_100 < p75) OR NOT(session_ldn_ny_overlap > p10)) AND ((session_london >= p10 OR prior_session_low_distance == p50) AND distance_to_round_number < p10))` |
| 605 | 46155 | -0.0538 | 40639 | 2.255e-10 | -6.345 | `((NOT(atr_vs_trailing_100 > p25) OR NOT(prior_session_low_distance == p75)) AND (range_close_ratio <= p10 AND (atr_14 <= p50 OR NOT(session_dead > p25))))` |
| 606 | 23790 | -0.0791 | 16422 | 2.357e-10 | -6.340 | `((distance_to_round_number != p50 AND swing_low_distance_14 <= p10) AND distance_to_round_number >= p75)` |
| 607 | 25419 | -0.0547 | 38683 | 2.636e-10 | -6.320 | `(prior_session_low_distance <= p75 AND (range_close_ratio < p10 AND (session_ny <= p10 OR NOT(hour_of_day < p25))))` |
| 608 | 25491 | -0.0528 | 37237 | 2.729e-10 | -6.315 | `(session_ny > p90 OR (NOT(spread_percentile_100 == p90) AND NOT(distance_to_round_number > p10)))` |
| 609 | 17530 | -0.0580 | 31595 | 2.962e-10 | -6.303 | `(day_of_week >= p90 AND (NOT(atr_14 != p25) OR NOT(range_close_ratio > p25)))` |
| 610 | 21211 | -0.0580 | 31595 | 2.962e-10 | -6.303 | `(day_of_week > p75 AND (session_ny <= p90 AND NOT(range_close_ratio >= p25)))` |
| 611 | 40798 | -0.0539 | 33332 | 3.220e-10 | -6.290 | `(kijun_26_distance >= p75 AND (kijun_26_distance != p90 AND NOT(spread_vs_trailing_100 < p75)))` |
| 612 | 43933 | -0.0612 | 26094 | 3.495e-10 | -6.278 | `(day_of_week >= p90 AND prior_session_low_distance >= p75)` |
| 613 | 38071 | -0.0822 | 14201 | 3.886e-10 | -6.263 | `(((NOT(session_london < p25) AND atr_vs_trailing_100 != p90) AND (NOT(prior_session_low_distance >= p10) AND NOT(prior_session_high_distance < p25))) AND session_dead < p75)` |
| 614 | 27453 | -0.0381 | 71546 | 4.135e-10 | -6.250 | `((session_dead < p50 OR (atr_14 != p90 AND range_close_ratio <= p90)) AND NOT(spread_percentile_100 < p90))` |
| 615 | 43231 | -0.0576 | 34499 | 4.179e-10 | -6.249 | `(range_close_ratio < p10 AND swing_high_distance_14 < p50)` |
| 616 | 5394 | -0.0567 | 31997 | 4.179e-10 | -6.249 | `((NOT(prior_session_low_distance != p90) OR NOT(hour_of_day == p90)) AND ((NOT(session_tokyo >= p50) OR NOT(distance_to_round_number >= p10)) AND hour_of_day != p90))` |
| 617 | 45101 | -0.0578 | 30594 | 4.820e-10 | -6.227 | `(distance_to_round_number < p10 AND prior_session_low_distance > p50)` |
| 618 | 2554 | -0.0486 | 47276 | 4.995e-10 | -6.221 | `(range_close_ratio <= p10 AND (hour_of_day < p50 OR NOT(spread_percentile_100 == p90)))` |
| 619 | 11053 | -0.0659 | 24387 | 5.018e-10 | -6.221 | `(((prior_session_high_distance != p75 AND hour_of_day == p25) OR NOT(spread_percentile_100 != p25)) AND (range_close_ratio <= p25 AND NOT(kijun_26_distance < p10)))` |
| 620 | 11908 | -0.0807 | 12819 | 5.582e-10 | -6.207 | `((NOT(range_close_ratio != p25) OR (NOT(kijun_26_distance >= p90) AND NOT(kijun_26_distance <= p25))) AND (NOT(session_dead <= p10) AND NOT(range_close_ratio < p75)))` |
| 621 | 17027 | -0.0979 | 9417 | 7.544e-10 | -6.161 | `((prior_session_low_distance < p10 AND NOT(session_ny < p90)) AND NOT(swing_low_distance_14 >= p10))` |
| 622 | 12438 | -0.1124 | 7007 | 7.550e-10 | -6.163 | `(NOT(day_of_week != p90) AND ((distance_to_round_number >= p50 AND NOT(hour_of_day < p90)) AND prior_session_low_distance >= p10))` |
| 623 | 23626 | -0.0642 | 22416 | 8.703e-10 | -6.134 | `(atr_percentile_100 < p50 AND atr_vs_trailing_100 > p50)` |
| 624 | 15066 | -0.0419 | 56456 | 8.792e-10 | -6.131 | `(((NOT(swing_low_distance_14 > p25) AND session_ny <= p10) AND spread_percentile_100 < p25) OR NOT(spread_percentile_100 > p10))` |
| 625 | 26821 | -0.0455 | 47594 | 8.958e-10 | -6.128 | `(((spread_vs_trailing_100 >= p10 OR atr_vs_trailing_100 >= p25) AND distance_to_round_number <= p75) AND NOT(distance_to_round_number > p10))` |
| 626 | 31425 | -0.0421 | 58460 | 1.101e-09 | -6.095 | `((NOT(range_close_ratio >= p10) OR prior_session_high_distance < p10) AND session_ny == p10)` |
| 627 | 39189 | -0.0373 | 69326 | 1.179e-09 | -6.084 | `((NOT(session_dead > p50) OR NOT(prior_session_low_distance == p25)) AND (NOT(spread_percentile_100 < p90) AND NOT(prior_session_low_distance <= p10)))` |
| 628 | 17327 | -0.0702 | 19090 | 1.316e-09 | -6.068 | `(((NOT(day_of_week != p90) AND NOT(atr_percentile_100 > p25)) AND day_of_week >= p75) AND atr_vs_trailing_100 != p10)` |
| 629 | 5409 | -0.0650 | 22175 | 1.333e-09 | -6.066 | `((distance_to_round_number < p90 AND swing_high_distance_14 < p10) AND session_tokyo == p25)` |
| 630 | 15304 | -0.2482 | 1247 | 1.510e-09 | -6.089 | `(kijun_26_distance > p10 AND (NOT(kijun_26_distance > p75) AND (NOT(spread_vs_trailing_100 <= p50) AND (spread_percentile_100 == p75 AND NOT(hour_of_day == p90)))))` |
| 631 | 38978 | -0.0850 | 13744 | 1.634e-09 | -6.035 | `((NOT(session_ny > p10) OR spread_vs_trailing_100 == p10) AND (NOT(swing_high_distance_14 >= p10) AND NOT(atr_percentile_100 > p50)))` |
| 632 | 35217 | -0.0836 | 13262 | 1.680e-09 | -6.030 | `(spread_percentile_100 > p75 AND ((session_dead == p25 OR NOT(prior_session_high_distance <= p75)) AND session_dead != p25))` |
| 633 | 13437 | -0.0323 | 91147 | 1.749e-09 | -6.020 | `(prior_session_high_distance <= p10 OR (session_ldn_ny_overlap >= p10 AND NOT(spread_percentile_100 >= p10)))` |
| 634 | 649 | -0.0729 | 17458 | 1.871e-09 | -6.012 | `((swing_low_distance_14 <= p10 OR NOT(atr_percentile_100 != p90)) AND NOT(distance_to_round_number <= p75))` |
| 635 | 1078 | -0.0289 | 110050 | 2.159e-09 | -5.986 | `(spread_vs_trailing_100 < p25 AND (atr_14 > p10 OR NOT(session_london <= p75)))` |
| 636 | 47840 | -0.0555 | 33067 | 2.236e-09 | -5.981 | `(prior_session_high_distance != p90 AND (NOT(distance_to_round_number > p90) AND (session_tokyo > p10 AND NOT(range_close_ratio >= p10))))` |
| 637 | 46067 | -0.0660 | 20106 | 2.336e-09 | -5.975 | `((hour_of_day < p25 AND spread_percentile_100 >= p10) AND ((NOT(session_tokyo > p90) AND range_close_ratio == p50) OR spread_vs_trailing_100 > p90))` |
| 638 | 29439 | -0.0409 | 55506 | 2.442e-09 | -5.966 | `((spread_percentile_100 == p90 OR session_dead > p90) AND NOT(session_london >= p75))` |
| 639 | 1060 | -0.0618 | 24503 | 2.455e-09 | -5.967 | `(distance_to_round_number < p10 AND NOT(day_of_week < p50))` |
| 640 | 29324 | -0.0453 | 44780 | 2.491e-09 | -5.963 | `((prior_session_high_distance > p10 AND NOT(spread_percentile_100 > p10)) AND ((swing_high_distance_14 == p90 OR NOT(session_dead <= p75)) OR NOT(swing_high_distance_14 == p10)))` |
| 641 | 4582 | -0.0752 | 16626 | 2.674e-09 | -5.954 | `(atr_percentile_100 < p25 AND hour_of_day >= p90)` |
| 642 | 24492 | -0.0312 | 96784 | 2.926e-09 | -5.936 | `(NOT(spread_percentile_100 >= p10) OR (kijun_26_distance < p10 OR atr_percentile_100 == p25))` |
| 643 | 7704 | -0.0364 | 71322 | 3.009e-09 | -5.932 | `(NOT(session_dead > p25) AND (NOT(spread_percentile_100 != p90) OR (NOT(range_close_ratio >= p25) AND NOT(day_of_week >= p50))))` |
| 644 | 14112 | -0.0511 | 37100 | 3.116e-09 | -5.927 | `(range_close_ratio < p25 AND session_tokyo < p90)` |
| 645 | 9875 | -0.0538 | 34360 | 3.173e-09 | -5.924 | `((NOT(range_close_ratio > p10) AND NOT(swing_high_distance_14 >= p25)) OR (atr_percentile_100 != p75 AND day_of_week > p90))` |
| 646 | 29199 | -0.0692 | 19242 | 3.226e-09 | -5.922 | `((NOT(atr_percentile_100 >= p50) AND ((prior_session_low_distance < p50 OR NOT(atr_percentile_100 != p10)) OR NOT(atr_vs_trailing_100 != p10))) AND NOT(hour_of_day > p10))` |
| 647 | 36052 | -0.0913 | 11077 | 3.354e-09 | -5.918 | `(swing_high_distance_14 < p25 AND (hour_of_day <= p10 AND ((range_close_ratio <= p25 AND range_close_ratio != p10) OR NOT(prior_session_high_distance != p10))))` |
| 648 | 19263 | -0.0510 | 38972 | 3.510e-09 | -5.907 | `((prior_session_high_distance > p25 AND (range_close_ratio <= p10 AND prior_session_high_distance > p25)) AND NOT(session_london > p75))` |
| 649 | 34954 | -0.0727 | 17480 | 3.708e-09 | -5.900 | `(NOT(atr_percentile_100 > p25) AND ((NOT(distance_to_round_number == p10) OR NOT(session_tokyo < p25)) AND NOT(hour_of_day <= p75)))` |
| 650 | 14234 | -0.0641 | 20708 | 4.027e-09 | -5.886 | `(NOT(session_dead == p10) AND ((prior_session_high_distance > p25 AND spread_vs_trailing_100 >= p10) AND NOT(spread_vs_trailing_100 <= p90)))` |
| 651 | 23716 | -0.0467 | 39807 | 4.319e-09 | -5.873 | `((NOT(session_london != p50) AND (NOT(atr_14 <= p25) AND NOT(spread_percentile_100 > p25))) AND NOT(prior_session_high_distance <= p25))` |
| 652 | 25691 | -0.0657 | 21193 | 4.410e-09 | -5.870 | `((NOT(kijun_26_distance < p25) OR NOT(distance_to_round_number != p10)) AND NOT(distance_to_round_number >= p10))` |
| 653 | 16548 | -0.0366 | 67728 | 4.625e-09 | -5.861 | `(NOT(hour_of_day == p75) AND spread_percentile_100 >= p90)` |
| 654 | 36664 | -0.0518 | 37409 | 4.775e-09 | -5.856 | `((hour_of_day == p25 OR NOT(swing_low_distance_14 >= p50)) AND range_close_ratio <= p10)` |
| 655 | 10873 | -0.0514 | 34876 | 4.811e-09 | -5.855 | `(NOT(distance_to_round_number >= p10) AND (session_london >= p10 AND NOT(atr_vs_trailing_100 > p90)))` |
| 656 | 25483 | -0.0424 | 50434 | 5.119e-09 | -5.844 | `((session_london != p50 AND atr_percentile_100 == p25) OR distance_to_round_number <= p10)` |
| 657 | 36843 | +0.0401 | 59769 | 5.624e-09 | 5.828 | `(spread_vs_trailing_100 > p50 AND swing_high_distance_14 > p75)` |
| 658 | 21498 | -0.0697 | 16280 | 5.796e-09 | -5.826 | `((NOT(distance_to_round_number <= p50) AND NOT(prior_session_high_distance < p90)) OR (NOT(spread_vs_trailing_100 >= p25) AND (NOT(atr_14 > p25) AND NOT(atr_14 <= p25))))` |
| 659 | 5458 | +0.0400 | 59784 | 5.851e-09 | 5.822 | `(NOT(spread_vs_trailing_100 <= p50) AND (swing_high_distance_14 >= p75 AND range_close_ratio != p90))` |
| 660 | 20632 | -0.0495 | 40584 | 6.379e-09 | -5.808 | `(NOT(atr_percentile_100 > p75) AND NOT(range_close_ratio >= p10))` |
| 661 | 22049 | -0.0417 | 50930 | 6.545e-09 | -5.803 | `((NOT(distance_to_round_number > p10) OR NOT(hour_of_day >= p25)) AND NOT(prior_session_high_distance > p25))` |
| 662 | 29650 | -0.0484 | 41612 | 6.707e-09 | -5.799 | `(range_close_ratio < p10 AND (NOT(day_of_week == p75) OR ((atr_vs_trailing_100 < p25 OR atr_vs_trailing_100 < p25) AND atr_vs_trailing_100 >= p25)))` |
| 663 | 5043 | -0.0444 | 44157 | 7.249e-09 | -5.786 | `(kijun_26_distance == p50 OR (atr_14 <= p90 AND spread_percentile_100 < p10))` |
| 664 | 42470 | -0.0405 | 53022 | 7.884e-09 | -5.772 | `(((NOT(session_london != p10) AND prior_session_high_distance != p25) OR (NOT(hour_of_day >= p25) OR session_london > p75)) AND spread_percentile_100 >= p90)` |
| 665 | 2601 | -0.1038 | 7931 | 8.260e-09 | -5.769 | `((NOT(spread_percentile_100 == p25) OR NOT(day_of_week <= p90)) AND atr_percentile_100 == p10)` |
| 666 | 14164 | -0.0921 | 9250 | 8.518e-09 | -5.763 | `((swing_high_distance_14 < p10 AND atr_vs_trailing_100 >= p50) AND (NOT(session_tokyo <= p50) OR prior_session_low_distance <= p10))` |
| 667 | 9852 | -0.0497 | 35820 | 8.802e-09 | -5.754 | `((NOT(spread_percentile_100 >= p25) AND NOT(session_tokyo <= p50)) AND NOT(range_close_ratio == p25))` |
| 668 | 24481 | -0.0497 | 35820 | 8.802e-09 | -5.754 | `(((NOT(atr_14 >= p90) OR session_tokyo > p50) OR NOT(prior_session_low_distance >= p75)) AND (session_tokyo == p75 AND NOT(spread_percentile_100 >= p25)))` |
| 669 | 41413 | -0.0497 | 35820 | 8.802e-09 | -5.754 | `(session_tokyo == p90 AND NOT(spread_percentile_100 >= p25))` |
| 670 | 11191 | -0.0722 | 15197 | 8.884e-09 | -5.754 | `((NOT(swing_low_distance_14 <= p90) OR NOT(session_ldn_ny_overlap < p10)) AND (NOT(hour_of_day != p90) AND spread_vs_trailing_100 > p90))` |
| 671 | 41048 | -0.0721 | 16421 | 9.489e-09 | -5.743 | `((((NOT(session_ldn_ny_overlap > p50) OR hour_of_day == p25) OR swing_high_distance_14 != p90) AND NOT(session_london == p25)) AND swing_high_distance_14 <= p10)` |
| 672 | 34295 | -0.1081 | 7218 | 1.032e-08 | -5.732 | `(atr_vs_trailing_100 < p50 AND (NOT(spread_vs_trailing_100 > p90) AND atr_percentile_100 == p10))` |
| 673 | 13397 | -0.0422 | 48822 | 1.079e-08 | -5.719 | `((atr_percentile_100 != p25 OR NOT(spread_percentile_100 <= p25)) AND NOT(distance_to_round_number > p10))` |
| 674 | 42218 | -0.0454 | 41566 | 1.120e-08 | -5.713 | `((session_tokyo > p25 AND (spread_percentile_100 == p90 AND (NOT(session_ldn_ny_overlap < p25) OR NOT(prior_session_high_distance < p50)))) OR prior_session_high_distance == p90)` |
| 675 | 5850 | -0.0407 | 52048 | 1.155e-08 | -5.707 | `(distance_to_round_number <= p10 OR (atr_percentile_100 == p90 AND NOT(prior_session_low_distance >= p75)))` |
| 676 | 15579 | -0.0509 | 34856 | 1.226e-08 | -5.697 | `(range_close_ratio < p25 AND (NOT(session_london < p75) AND (NOT(prior_session_high_distance == p25) OR (NOT(session_london >= p10) OR distance_to_round_number > p50))))` |
| 677 | 26343 | -0.0453 | 41554 | 1.229e-08 | -5.697 | `(NOT(spread_percentile_100 != p90) AND session_ny != p90)` |
| 678 | 34840 | -0.0453 | 41554 | 1.229e-08 | -5.697 | `((session_ny != p90 AND spread_percentile_100 >= p90) AND (hour_of_day != p90 OR NOT(swing_high_distance_14 == p10)))` |
| 679 | 3108 | -0.0557 | 30733 | 1.242e-08 | -5.695 | `((range_close_ratio < p10 OR atr_vs_trailing_100 == p10) AND NOT(prior_session_high_distance < p50))` |
| 680 | 41689 | -0.1002 | 7854 | 1.473e-08 | -5.671 | `(NOT(hour_of_day >= p25) AND (distance_to_round_number >= p50 AND spread_percentile_100 == p90))` |
| 681 | 35880 | -0.0417 | 48929 | 1.499e-08 | -5.663 | `((NOT(atr_vs_trailing_100 != p50) AND NOT(session_tokyo == p75)) OR distance_to_round_number <= p10)` |
| 682 | 2960 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `(((NOT(range_close_ratio <= p50) OR NOT(day_of_week >= p90)) OR session_london >= p10) AND (distance_to_round_number <= p10 OR session_dead < p50))` |
| 683 | 11290 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `NOT(distance_to_round_number > p10)` |
| 684 | 13799 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 685 | 17031 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `NOT(distance_to_round_number > p10)` |
| 686 | 17898 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 687 | 19689 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 688 | 21363 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 689 | 22314 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 690 | 24124 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `(range_close_ratio == p75 OR distance_to_round_number <= p10)` |
| 691 | 24422 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 692 | 27146 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `NOT(distance_to_round_number > p10)` |
| 693 | 29518 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `(distance_to_round_number <= p10 OR ((atr_14 == p25 AND (NOT(day_of_week == p75) OR kijun_26_distance > p10)) AND NOT(prior_session_low_distance < p90)))` |
| 694 | 31435 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 695 | 31846 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 696 | 32479 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `(atr_vs_trailing_100 == p10 OR distance_to_round_number <= p10)` |
| 697 | 38906 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 698 | 39079 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `NOT(distance_to_round_number > p10)` |
| 699 | 39583 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 700 | 40245 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `(NOT(session_dead > p75) AND (NOT(swing_high_distance_14 == p25) AND distance_to_round_number <= p10))` |
| 701 | 44293 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 702 | 44333 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 703 | 45367 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 704 | 48376 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `distance_to_round_number <= p10` |
| 705 | 49274 | -0.0417 | 48928 | 1.523e-08 | -5.660 | `((NOT(session_ldn_ny_overlap >= p10) AND NOT(kijun_26_distance >= p10)) OR NOT(distance_to_round_number > p10))` |
| 706 | 31519 | +0.0699 | 15079 | 1.589e-08 | 5.655 | `(atr_vs_trailing_100 >= p90 AND (kijun_26_distance >= p75 AND atr_percentile_100 > p25))` |
| 707 | 45237 | -0.0632 | 22278 | 1.592e-08 | -5.653 | `((NOT(swing_high_distance_14 < p50) AND spread_percentile_100 < p75) AND ((range_close_ratio == p10 OR NOT(session_london != p10)) AND NOT(atr_percentile_100 >= p25)))` |
| 708 | 45435 | -0.0478 | 36067 | 1.651e-08 | -5.646 | `((NOT(range_close_ratio >= p25) AND NOT(session_ldn_ny_overlap != p10)) AND (prior_session_low_distance > p10 AND NOT(atr_vs_trailing_100 <= p50)))` |
| 709 | 22134 | -0.0418 | 48314 | 1.680e-08 | -5.643 | `(NOT(atr_14 <= p50) AND NOT(distance_to_round_number > p10))` |
| 710 | 19245 | -0.0409 | 48259 | 1.751e-08 | -5.636 | `(NOT(spread_vs_trailing_100 <= p25) AND ((day_of_week <= p10 AND NOT(hour_of_day > p10)) OR NOT(prior_session_high_distance >= p10)))` |
| 711 | 628 | -0.0416 | 48732 | 1.826e-08 | -5.629 | `(distance_to_round_number <= p10 AND (NOT(atr_14 <= p25) OR NOT(spread_percentile_100 != p90)))` |
| 712 | 39823 | -0.0321 | 84699 | 1.827e-08 | -5.628 | `((swing_high_distance_14 >= p75 OR atr_14 < p10) AND session_ny < p75)` |
| 713 | 17272 | -0.0581 | 24573 | 1.926e-08 | -5.620 | `(atr_14 <= p50 AND (day_of_week == p50 AND (NOT(session_tokyo < p75) AND day_of_week != p25)))` |
| 714 | 41983 | -0.0900 | 9735 | 2.050e-08 | -5.612 | `(NOT(distance_to_round_number != p75) OR ((NOT(spread_percentile_100 <= p50) OR NOT(hour_of_day > p50)) AND (atr_percentile_100 == p75 OR atr_percentile_100 == p10)))` |
| 715 | 43859 | -0.0413 | 48992 | 2.105e-08 | -5.604 | `(NOT(kijun_26_distance != p25) OR distance_to_round_number <= p10)` |
| 716 | 28727 | -0.0411 | 48790 | 2.128e-08 | -5.602 | `(NOT(swing_high_distance_14 < p75) AND (NOT(distance_to_round_number <= p50) OR NOT(distance_to_round_number > p10)))` |
| 717 | 15351 | -0.0568 | 27476 | 2.137e-08 | -5.602 | `(swing_high_distance_14 < p25 AND (range_close_ratio < p10 OR range_close_ratio >= p90))` |
| 718 | 5143 | -0.0751 | 14090 | 2.139e-08 | -5.604 | `((NOT(hour_of_day != p10) OR NOT(hour_of_day != p90)) AND (NOT(prior_session_low_distance > p10) AND NOT(session_tokyo != p25)))` |
| 719 | 39353 | -0.0333 | 75589 | 2.150e-08 | -5.600 | `(NOT(distance_to_round_number > p10) OR (NOT(kijun_26_distance >= p10) AND prior_session_high_distance <= p25))` |
| 720 | 20214 | -0.1001 | 8040 | 2.260e-08 | -5.596 | `(((NOT(atr_percentile_100 != p10) OR swing_low_distance_14 == p50) AND (NOT(atr_14 > p10) OR NOT(atr_percentile_100 > p75))) AND NOT(kijun_26_distance == p10))` |
| 721 | 19334 | -0.1003 | 7282 | 2.306e-08 | -5.594 | `((spread_percentile_100 == p90 AND distance_to_round_number >= p90) OR session_ldn_ny_overlap < p75)` |
| 722 | 2880 | -0.0426 | 45170 | 2.403e-08 | -5.581 | `(((NOT(spread_percentile_100 > p10) AND NOT(session_london > p75)) AND spread_percentile_100 <= p90) AND atr_vs_trailing_100 >= p10)` |
| 723 | 29891 | -0.1008 | 7867 | 2.422e-08 | -5.584 | `(((swing_low_distance_14 >= p90 OR prior_session_high_distance != p50) OR spread_percentile_100 < p10) AND (NOT(swing_high_distance_14 == p50) AND atr_percentile_100 == p10))` |
| 724 | 9708 | -0.0701 | 16365 | 2.519e-08 | -5.575 | `(NOT(swing_high_distance_14 >= p10) AND NOT(session_london <= p25))` |
| 725 | 1274 | -0.0335 | 72129 | 2.546e-08 | -5.571 | `((swing_low_distance_14 == p90 OR atr_percentile_100 > p10) AND NOT(spread_percentile_100 < p90))` |
| 726 | 37851 | -0.0653 | 17968 | 2.622e-08 | -5.567 | `((prior_session_low_distance >= p90 OR NOT(swing_high_distance_14 >= p50)) AND (NOT(kijun_26_distance >= p25) OR swing_low_distance_14 == p75))` |
| 727 | 9031 | -0.0501 | 33671 | 2.715e-08 | -5.560 | `(NOT(prior_session_high_distance < p75) AND (NOT(swing_low_distance_14 >= p25) OR hour_of_day <= p25))` |
| 728 | 48963 | -0.0420 | 46215 | 2.755e-08 | -5.557 | `(spread_percentile_100 < p10 AND (session_dead < p25 OR (NOT(atr_percentile_100 <= p10) OR swing_high_distance_14 <= p50)))` |
| 729 | 32435 | -0.0997 | 7999 | 2.778e-08 | -5.560 | `(atr_percentile_100 == p10 AND NOT(distance_to_round_number == p50))` |
| 730 | 8490 | -0.0992 | 8008 | 3.197e-08 | -5.536 | `(NOT(atr_percentile_100 != p10) AND ((session_dead >= p75 AND kijun_26_distance == p75) OR NOT(kijun_26_distance == p50)))` |
| 731 | 38372 | -0.0409 | 48519 | 3.236e-08 | -5.529 | `((distance_to_round_number <= p10 AND NOT(session_london < p50)) AND NOT(spread_percentile_100 == p75))` |
| 732 | 218 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 733 | 812 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `(atr_percentile_100 == p10 AND ((day_of_week >= p25 AND NOT(hour_of_day > p75)) OR session_dead <= p90))` |
| 734 | 1152 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 735 | 2708 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `(NOT(atr_percentile_100 != p10) OR ((range_close_ratio <= p10 AND session_ny > p50) AND NOT(swing_low_distance_14 > p10)))` |
| 736 | 4890 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 737 | 9588 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 738 | 9983 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 739 | 10781 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 740 | 11467 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `(atr_percentile_100 < p25 AND atr_percentile_100 == p10)` |
| 741 | 13282 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 742 | 13667 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 743 | 17176 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `(atr_percentile_100 == p10 OR (atr_vs_trailing_100 > p75 AND session_tokyo > p75))` |
| 744 | 17588 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 745 | 18332 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 746 | 24164 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 747 | 27829 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `(NOT(atr_percentile_100 > p10) AND atr_percentile_100 == p10)` |
| 748 | 28159 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 749 | 28326 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 750 | 28660 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 751 | 28784 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 752 | 29639 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 753 | 30568 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 754 | 32111 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 755 | 33585 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 756 | 37810 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 757 | 38764 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 758 | 41530 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 759 | 43375 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `atr_percentile_100 == p10` |
| 760 | 44870 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 761 | 45876 | -0.0991 | 8010 | 3.327e-08 | -5.529 | `NOT(atr_percentile_100 != p10)` |
| 762 | 31043 | -0.0991 | 8012 | 3.336e-08 | -5.528 | `((NOT(atr_vs_trailing_100 < p90) AND (NOT(atr_vs_trailing_100 == p10) AND swing_high_distance_14 == p10)) OR atr_percentile_100 == p10)` |
| 763 | 13330 | -0.1143 | 5724 | 3.346e-08 | -5.530 | `(atr_percentile_100 == p10 AND range_close_ratio >= p25)` |
| 764 | 36400 | -0.0354 | 62272 | 3.382e-08 | -5.521 | `((NOT(session_london >= p75) AND (NOT(session_ny > p10) AND NOT(atr_percentile_100 <= p90))) OR NOT(swing_high_distance_14 < p90))` |
| 765 | 25676 | -0.1162 | 5701 | 3.419e-08 | -5.526 | `(NOT(atr_percentile_100 != p10) AND (NOT(session_dead > p50) OR prior_session_high_distance == p90))` |
| 766 | 29071 | -0.0983 | 8060 | 3.878e-08 | -5.502 | `(NOT(atr_percentile_100 != p10) OR prior_session_high_distance == p75)` |
| 767 | 44872 | -0.0985 | 8028 | 3.891e-08 | -5.501 | `(atr_percentile_100 == p10 OR (NOT(prior_session_high_distance != p10) AND (NOT(day_of_week > p75) OR distance_to_round_number != p25)))` |
| 768 | 13836 | -0.0624 | 19676 | 3.936e-08 | -5.496 | `((NOT(swing_high_distance_14 >= p25) AND session_ldn_ny_overlap != p10) AND (hour_of_day > p25 OR NOT(prior_session_low_distance <= p75)))` |
| 769 | 17822 | -0.0624 | 19673 | 3.939e-08 | -5.496 | `(((prior_session_low_distance < p90 OR NOT(spread_percentile_100 < p10)) AND session_ldn_ny_overlap >= p90) AND swing_high_distance_14 < p25)` |
| 770 | 31271 | -0.0970 | 7744 | 4.103e-08 | -5.492 | `(NOT(range_close_ratio < p75) AND (NOT(atr_percentile_100 > p10) AND (session_tokyo <= p50 AND atr_14 != p50)))` |
| 771 | 2858 | -0.0987 | 7964 | 4.175e-08 | -5.489 | `(atr_percentile_100 == p10 AND (spread_vs_trailing_100 == p25 OR (swing_high_distance_14 > p10 OR NOT(swing_low_distance_14 < p10))))` |
| 772 | 42165 | -0.1066 | 6804 | 4.194e-08 | -5.489 | `(atr_percentile_100 == p10 AND day_of_week != p90)` |
| 773 | 27614 | -0.1312 | 4329 | 4.200e-08 | -5.492 | `((atr_percentile_100 == p10 AND (NOT(atr_percentile_100 == p75) OR session_ny > p75)) AND (distance_to_round_number > p25 AND NOT(spread_percentile_100 < p25)))` |
| 774 | 14424 | -0.0420 | 45517 | 4.356e-08 | -5.477 | `(((NOT(distance_to_round_number != p90) OR NOT(range_close_ratio >= p50)) OR swing_low_distance_14 >= p25) AND spread_percentile_100 <= p10)` |
| 775 | 25811 | -0.0411 | 47230 | 4.443e-08 | -5.473 | `(distance_to_round_number <= p10 AND NOT(prior_session_low_distance == p90))` |
| 776 | 41275 | -0.0411 | 47232 | 4.479e-08 | -5.472 | `(NOT(prior_session_low_distance == p75) AND distance_to_round_number <= p10)` |
| 777 | 20167 | -0.0889 | 9386 | 4.537e-08 | -5.473 | `(((NOT(day_of_week > p10) AND NOT(distance_to_round_number <= p90)) AND session_ldn_ny_overlap >= p50) AND NOT(day_of_week == p25))` |
| 778 | 42862 | -0.0313 | 82710 | 4.623e-08 | -5.466 | `(atr_vs_trailing_100 <= p75 AND NOT(prior_session_low_distance <= p75))` |
| 779 | 14219 | -0.0416 | 45588 | 4.882e-08 | -5.456 | `(spread_percentile_100 < p10 AND (NOT(day_of_week != p25) OR ((NOT(session_ldn_ny_overlap == p90) OR NOT(atr_vs_trailing_100 <= p50)) OR day_of_week < p50)))` |
| 780 | 49905 | -0.1022 | 7246 | 4.883e-08 | -5.461 | `(NOT(atr_percentile_100 != p10) AND ((hour_of_day <= p90 AND session_london == p10) OR swing_high_distance_14 < p75))` |
| 781 | 16664 | -0.0978 | 8025 | 5.024e-08 | -5.456 | `(NOT(atr_percentile_100 != p10) OR (NOT(session_ny >= p90) AND (NOT(session_tokyo != p75) AND NOT(prior_session_low_distance != p75))))` |
| 782 | 15665 | -0.0468 | 37157 | 5.190e-08 | -5.446 | `(NOT(atr_percentile_100 > p75) AND (NOT(session_dead >= p10) OR distance_to_round_number <= p10))` |
| 783 | 35383 | -0.0787 | 11403 | 5.516e-08 | -5.437 | `((NOT(distance_to_round_number <= p90) AND (prior_session_low_distance < p25 AND swing_high_distance_14 >= p25)) AND NOT(swing_high_distance_14 == p25))` |
| 784 | 522 | -0.0700 | 15103 | 5.664e-08 | -5.432 | `(spread_vs_trailing_100 > p75 AND (hour_of_day < p10 OR (prior_session_low_distance <= p75 AND NOT(day_of_week != p25))))` |
| 785 | 29955 | -0.0390 | 51574 | 5.723e-08 | -5.428 | `(NOT(swing_high_distance_14 != p10) OR (spread_percentile_100 <= p10 OR NOT(swing_low_distance_14 != p10)))` |
| 786 | 16056 | -0.0508 | 28094 | 5.879e-08 | -5.424 | `(range_close_ratio >= p50 AND NOT(spread_percentile_100 > p10))` |
| 787 | 45272 | -0.0508 | 28094 | 5.879e-08 | -5.424 | `(spread_percentile_100 <= p10 AND NOT(range_close_ratio < p50))` |
| 788 | 32976 | -0.0291 | 92866 | 5.970e-08 | -5.420 | `(NOT(kijun_26_distance > p10) OR (NOT(spread_percentile_100 >= p10) OR (NOT(prior_session_low_distance != p75) AND (NOT(swing_low_distance_14 >= p90) AND NOT(atr_14 >= p90)))))` |
| 789 | 17638 | -0.0491 | 36389 | 6.014e-08 | -5.420 | `(NOT(range_close_ratio > p10) AND (atr_vs_trailing_100 <= p50 OR swing_low_distance_14 <= p10))` |
| 790 | 36168 | -0.0400 | 48607 | 6.255e-08 | -5.412 | `((kijun_26_distance == p25 AND session_london >= p75) OR NOT(spread_percentile_100 >= p10))` |
| 791 | 15564 | -0.0400 | 48589 | 6.274e-08 | -5.412 | `(NOT(session_tokyo >= p50) OR (spread_percentile_100 < p10 OR NOT(spread_vs_trailing_100 != p90)))` |
| 792 | 25176 | -0.0629 | 18132 | 6.299e-08 | -5.412 | `(NOT(distance_to_round_number <= p75) AND (prior_session_low_distance > p10 AND NOT(session_ldn_ny_overlap <= p75)))` |
| 793 | 49833 | -0.2590 | 835 | 6.303e-08 | -5.459 | `((NOT(spread_vs_trailing_100 <= p10) AND ((NOT(distance_to_round_number <= p75) AND NOT(distance_to_round_number < p50)) AND NOT(hour_of_day > p10))) AND prior_session_high_distance <= p25)` |
| 794 | 355 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 795 | 2556 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `(((NOT(session_ldn_ny_overlap > p90) AND swing_high_distance_14 > p75) AND NOT(day_of_week >= p10)) OR spread_percentile_100 < p10)` |
| 796 | 2782 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 797 | 2855 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `(NOT(spread_percentile_100 >= p10) AND (hour_of_day >= p75 OR (spread_percentile_100 != p50 OR session_london < p10)))` |
| 798 | 3074 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 799 | 3403 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `(((NOT(swing_high_distance_14 > p90) OR NOT(swing_high_distance_14 < p50)) OR NOT(atr_percentile_100 >= p10)) AND spread_percentile_100 < p10)` |
| 800 | 3586 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `((NOT(atr_14 < p10) OR NOT(hour_of_day < p10)) AND (spread_percentile_100 < p10 AND session_ldn_ny_overlap >= p50))` |
| 801 | 8561 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `((NOT(session_dead < p10) OR NOT(distance_to_round_number > p50)) AND NOT(spread_percentile_100 >= p10))` |
| 802 | 9438 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `((NOT(session_ldn_ny_overlap > p90) OR atr_14 == p50) AND spread_percentile_100 < p10)` |
| 803 | 10899 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `NOT(spread_percentile_100 >= p10)` |
| 804 | 11209 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 805 | 13300 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `(NOT(session_ny < p10) AND spread_percentile_100 < p10)` |
| 806 | 13644 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `(spread_percentile_100 < p10 OR (NOT(spread_vs_trailing_100 > p10) AND (atr_14 < p25 AND (spread_vs_trailing_100 == p50 AND session_london != p50))))` |
| 807 | 14100 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 808 | 14601 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `(NOT(spread_percentile_100 >= p10) AND (NOT(atr_percentile_100 == p10) OR swing_low_distance_14 != p75))` |
| 809 | 14775 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `NOT(spread_percentile_100 >= p10)` |
| 810 | 18363 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 811 | 21275 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `NOT(spread_percentile_100 >= p10)` |
| 812 | 27455 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `((atr_14 == p90 AND (swing_low_distance_14 < p90 AND NOT(session_dead <= p90))) OR NOT(spread_percentile_100 >= p10))` |
| 813 | 31676 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 814 | 32974 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `NOT(spread_percentile_100 >= p10)` |
| 815 | 34337 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `NOT(spread_percentile_100 >= p10)` |
| 816 | 38154 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `NOT(spread_percentile_100 >= p10)` |
| 817 | 47334 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `spread_percentile_100 < p10` |
| 818 | 49860 | -0.0400 | 48584 | 6.369e-08 | -5.409 | `NOT(spread_percentile_100 >= p10)` |
| 819 | 48236 | -0.0400 | 48568 | 6.463e-08 | -5.406 | `(((hour_of_day <= p90 AND swing_low_distance_14 != p10) OR atr_percentile_100 < p10) AND spread_percentile_100 < p10)` |
| 820 | 5430 | -0.0406 | 47155 | 6.739e-08 | -5.399 | `(((distance_to_round_number <= p10 OR session_dead > p90) AND (NOT(prior_session_high_distance >= p90) OR atr_14 >= p75)) AND prior_session_high_distance != p90)` |
| 821 | 24095 | -0.0419 | 43621 | 6.868e-08 | -5.396 | `((NOT(prior_session_low_distance == p50) AND NOT(spread_percentile_100 >= p10)) AND ((NOT(atr_vs_trailing_100 < p10) OR NOT(atr_percentile_100 <= p25)) OR hour_of_day == p90))` |
| 822 | 43051 | -0.0402 | 47776 | 6.930e-08 | -5.394 | `(((NOT(session_london >= p10) OR NOT(distance_to_round_number >= p90)) OR (NOT(session_dead != p50) AND NOT(kijun_26_distance == p50))) AND spread_percentile_100 < p10)` |
| 823 | 4030 | -0.0960 | 8119 | 7.026e-08 | -5.395 | `(atr_percentile_100 == p10 OR ((NOT(prior_session_high_distance <= p90) AND (NOT(atr_percentile_100 >= p75) AND NOT(prior_session_low_distance >= p50))) AND NOT(session_ldn_ny_overlap <= p75)))` |
| 824 | 40925 | -0.1298 | 4281 | 7.105e-08 | -5.398 | `(session_tokyo != p90 AND NOT(atr_percentile_100 != p10))` |
| 825 | 43176 | -0.0956 | 7223 | 7.117e-08 | -5.394 | `(atr_vs_trailing_100 == p10 OR (NOT(swing_high_distance_14 >= p10) AND ((NOT(atr_vs_trailing_100 <= p50) OR NOT(prior_session_low_distance != p90)) AND distance_to_round_number >= p75)))` |
| 826 | 27248 | -0.0853 | 10223 | 7.265e-08 | -5.388 | `(((NOT(spread_percentile_100 != p10) AND (swing_low_distance_14 != p25 AND session_dead <= p10)) OR atr_percentile_100 == p10) OR atr_vs_trailing_100 == p50)` |
| 827 | 44940 | -0.0961 | 8075 | 7.722e-08 | -5.378 | `(((NOT(session_ny > p90) OR NOT(swing_low_distance_14 != p75)) OR NOT(prior_session_high_distance > p50)) AND (kijun_26_distance == p25 OR NOT(atr_percentile_100 != p10)))` |
| 828 | 19810 | -0.0461 | 34932 | 7.913e-08 | -5.370 | `(((NOT(atr_vs_trailing_100 < p90) OR hour_of_day < p90) AND NOT(range_close_ratio >= p25)) AND atr_vs_trailing_100 >= p50)` |
| 829 | 1493 | -0.0964 | 6884 | 8.107e-08 | -5.371 | `(prior_session_low_distance < p10 AND atr_vs_trailing_100 >= p90)` |
| 830 | 38844 | -0.1008 | 7100 | 8.337e-08 | -5.365 | `((NOT(swing_low_distance_14 > p10) AND atr_vs_trailing_100 >= p25) AND day_of_week == p75)` |
| 831 | 37732 | -0.0369 | 55864 | 8.678e-08 | -5.353 | `(((session_dead == p90 AND prior_session_high_distance > p90) OR NOT(spread_percentile_100 > p10)) AND NOT(spread_vs_trailing_100 == p10))` |
| 832 | 4421 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 833 | 6744 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `(NOT(spread_vs_trailing_100 == p25) AND NOT(spread_percentile_100 > p10))` |
| 834 | 8836 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `spread_percentile_100 <= p10` |
| 835 | 10212 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `(spread_percentile_100 <= p10 AND atr_14 != p90)` |
| 836 | 11170 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 837 | 12287 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `spread_percentile_100 <= p10` |
| 838 | 12983 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `(NOT(spread_percentile_100 > p10) OR NOT(atr_14 != p10))` |
| 839 | 13906 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `(spread_percentile_100 <= p10 AND ((NOT(kijun_26_distance == p25) AND prior_session_low_distance == p50) OR session_london <= p90))` |
| 840 | 16971 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `(NOT(spread_percentile_100 > p10) OR (NOT(range_close_ratio != p10) AND spread_vs_trailing_100 >= p50))` |
| 841 | 20515 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `spread_percentile_100 <= p10` |
| 842 | 24050 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `spread_percentile_100 <= p10` |
| 843 | 25271 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `spread_percentile_100 <= p10` |
| 844 | 25410 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 845 | 29437 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `spread_percentile_100 <= p10` |
| 846 | 30617 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 847 | 31175 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 848 | 34378 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `((atr_14 < p75 OR ((hour_of_day != p75 OR NOT(prior_session_high_distance > p90)) OR NOT(atr_vs_trailing_100 == p90))) AND spread_percentile_100 <= p10)` |
| 849 | 35002 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `(session_ny <= p90 AND NOT(spread_percentile_100 > p10))` |
| 850 | 35183 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `spread_percentile_100 <= p10` |
| 851 | 35700 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 852 | 36182 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `((session_tokyo <= p75 OR NOT(distance_to_round_number != p90)) AND spread_percentile_100 <= p10)` |
| 853 | 37239 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 854 | 40976 | -0.0384 | 51336 | 9.197e-08 | -5.343 | `NOT(spread_percentile_100 > p10)` |
| 855 | 24420 | -0.0290 | 87958 | 1.005e-07 | -5.326 | `(NOT(atr_14 < p25) AND spread_vs_trailing_100 <= p25)` |
| 856 | 42242 | -0.0502 | 29308 | 1.041e-07 | -5.321 | `((NOT(day_of_week <= p75) OR kijun_26_distance <= p50) AND (NOT(distance_to_round_number > p10) AND NOT(distance_to_round_number == p90)))` |
| 857 | 18605 | -0.0858 | 9718 | 1.059e-07 | -5.320 | `(((swing_high_distance_14 >= p50 OR session_dead != p90) AND distance_to_round_number > p75) AND (range_close_ratio == p25 OR NOT(atr_percentile_100 > p10)))` |
| 858 | 11138 | -0.0857 | 9408 | 1.172e-07 | -5.302 | `((session_dead > p25 AND (distance_to_round_number > p90 AND day_of_week > p25)) AND NOT(session_dead != p90))` |
| 859 | 29977 | -0.0510 | 28888 | 1.216e-07 | -5.292 | `((((swing_high_distance_14 < p50 AND NOT(atr_percentile_100 >= p25)) AND swing_low_distance_14 != p75) OR prior_session_high_distance <= p10) AND spread_percentile_100 >= p75)` |
| 860 | 19935 | +0.0990 | 6685 | 1.238e-07 | 5.294 | `(NOT(range_close_ratio != p25) OR (atr_vs_trailing_100 > p90 AND (atr_14 > p50 AND prior_session_high_distance >= p75)))` |
| 861 | 36426 | -0.0674 | 15581 | 1.240e-07 | -5.290 | `((NOT(swing_high_distance_14 != p75) OR atr_14 <= p50) AND (hour_of_day <= p10 AND (session_ny < p75 AND day_of_week > p50)))` |
| 862 | 20112 | -0.1542 | 2869 | 1.284e-07 | -5.294 | `(NOT(atr_percentile_100 != p10) AND (NOT(day_of_week >= p75) AND session_tokyo == p25))` |
| 863 | 14992 | -0.0389 | 48710 | 1.363e-07 | -5.271 | `((NOT(atr_percentile_100 > p90) AND prior_session_low_distance == p25) OR spread_percentile_100 < p10)` |
| 864 | 43864 | -0.0392 | 47959 | 1.485e-07 | -5.255 | `((atr_percentile_100 != p25 AND NOT(distance_to_round_number > p10)) AND (NOT(distance_to_round_number >= p25) AND NOT(spread_percentile_100 == p75)))` |
| 865 | 35046 | -0.0521 | 25506 | 1.516e-07 | -5.252 | `(NOT(hour_of_day != p90) AND NOT(prior_session_high_distance <= p75))` |
| 866 | 44818 | -0.0520 | 25511 | 1.600e-07 | -5.242 | `(NOT(hour_of_day < p90) AND (prior_session_high_distance >= p75 OR NOT(session_dead <= p50)))` |
| 867 | 19576 | -0.0344 | 60963 | 1.658e-07 | -5.235 | `((NOT(spread_percentile_100 < p90) AND NOT(session_ny < p10)) AND day_of_week != p10)` |
| 868 | 32444 | -0.1298 | 3599 | 1.796e-07 | -5.230 | `(NOT(spread_vs_trailing_100 < p90) AND NOT(spread_percentile_100 < p90))` |
| 869 | 27015 | -0.1072 | 5474 | 1.813e-07 | -5.224 | `((NOT(swing_high_distance_14 < p10) AND prior_session_high_distance <= p50) AND (NOT(kijun_26_distance < p50) AND (NOT(atr_percentile_100 != p25) OR distance_to_round_number >= p90)))` |
| 870 | 41063 | -0.0400 | 44620 | 1.819e-07 | -5.218 | `(spread_percentile_100 < p10 AND distance_to_round_number > p10)` |
| 871 | 11952 | -0.0444 | 37145 | 1.871e-07 | -5.213 | `((spread_percentile_100 <= p25 AND NOT(distance_to_round_number == p90)) AND NOT(session_ny == p90))` |
| 872 | 31270 | -0.0609 | 20235 | 1.877e-07 | -5.213 | `((NOT(spread_percentile_100 == p75) AND range_close_ratio < p25) AND day_of_week == p75)` |
| 873 | 49279 | -0.0389 | 47645 | 1.881e-07 | -5.211 | `((((spread_vs_trailing_100 < p25 OR NOT(atr_14 <= p25)) OR NOT(session_ldn_ny_overlap > p75)) OR atr_vs_trailing_100 == p10) AND NOT(spread_percentile_100 >= p10))` |
| 874 | 6780 | -0.0491 | 29243 | 1.923e-07 | -5.208 | `(session_dead > p50 AND NOT(spread_percentile_100 < p90))` |
| 875 | 14938 | -0.0491 | 29243 | 1.923e-07 | -5.208 | `(NOT(session_dead == p25) AND ((kijun_26_distance != p25 AND NOT(session_london <= p75)) OR NOT(spread_percentile_100 < p90)))` |
| 876 | 42258 | -0.0491 | 29243 | 1.923e-07 | -5.208 | `((session_dead >= p75 AND NOT(spread_percentile_100 != p90)) OR (NOT(spread_percentile_100 != p90) AND (atr_percentile_100 <= p50 AND NOT(hour_of_day > p10))))` |
| 877 | 33569 | -0.0650 | 17169 | 1.994e-07 | -5.202 | `(NOT(prior_session_low_distance >= p90) AND (session_ldn_ny_overlap <= p90 AND (kijun_26_distance > p50 AND NOT(swing_low_distance_14 > p25))))` |
| 878 | 19610 | -0.0442 | 37184 | 2.077e-07 | -5.193 | `(spread_percentile_100 <= p25 AND (NOT(session_ny == p75) OR NOT(hour_of_day >= p10)))` |
| 879 | 38751 | -0.0442 | 37184 | 2.077e-07 | -5.193 | `(NOT(session_tokyo <= p50) AND (NOT(session_tokyo > p10) OR spread_percentile_100 <= p25))` |
| 880 | 98 | -0.0490 | 29245 | 2.114e-07 | -5.190 | `(NOT(spread_percentile_100 != p90) AND ((NOT(prior_session_high_distance != p75) OR NOT(atr_vs_trailing_100 != p10)) OR (session_tokyo < p50 OR session_dead != p50)))` |
| 881 | 20888 | -0.0567 | 21859 | 2.156e-07 | -5.187 | `(((NOT(atr_14 < p50) AND NOT(atr_vs_trailing_100 < p90)) OR session_tokyo > p25) AND NOT(distance_to_round_number >= p10))` |
| 882 | 48928 | -0.0904 | 7932 | 2.212e-07 | -5.185 | `(distance_to_round_number > p90 AND NOT(session_ldn_ny_overlap == p25))` |
| 883 | 20912 | -0.0799 | 11500 | 2.224e-07 | -5.183 | `((NOT(atr_vs_trailing_100 != p10) OR kijun_26_distance >= p50) AND ((atr_vs_trailing_100 < p10 AND NOT(swing_high_distance_14 == p90)) AND session_ny != p25))` |
| 884 | 17087 | -0.0395 | 45322 | 2.269e-07 | -5.177 | `(spread_percentile_100 <= p10 AND (prior_session_low_distance > p10 OR atr_percentile_100 < p25))` |
| 885 | 17067 | -0.0225 | 140667 | 2.295e-07 | -5.174 | `((NOT(atr_14 < p75) AND (session_dead > p75 OR prior_session_high_distance <= p50)) OR (NOT(swing_high_distance_14 <= p90) OR spread_percentile_100 <= p10))` |
| 886 | 8767 | -0.0669 | 14202 | 2.295e-07 | -5.176 | `(((session_ldn_ny_overlap > p10 AND NOT(atr_percentile_100 < p10)) AND (NOT(hour_of_day <= p90) OR swing_low_distance_14 >= p50)) AND NOT(swing_high_distance_14 >= p50))` |
| 887 | 37091 | -0.0591 | 20521 | 2.449e-07 | -5.163 | `(NOT(atr_percentile_100 <= p50) AND NOT(range_close_ratio >= p10))` |
| 888 | 45883 | -0.0425 | 39242 | 2.487e-07 | -5.160 | `((NOT(session_ldn_ny_overlap > p50) AND spread_percentile_100 < p10) OR (NOT(swing_high_distance_14 != p90) OR session_dead < p10))` |
| 889 | 5295 | -0.1052 | 5981 | 2.718e-07 | -5.148 | `((swing_low_distance_14 < p75 AND (NOT(session_ny > p75) AND NOT(atr_percentile_100 != p10))) AND NOT(atr_percentile_100 < p10))` |
| 890 | 15942 | -0.0501 | 27853 | 2.867e-07 | -5.133 | `(swing_low_distance_14 <= p25 AND spread_vs_trailing_100 >= p75)` |
| 891 | 15167 | -0.1633 | 2097 | 2.875e-07 | -5.148 | `(((NOT(prior_session_low_distance <= p25) AND NOT(prior_session_high_distance > p25)) AND NOT(spread_percentile_100 == p90)) AND NOT(swing_low_distance_14 > p25))` |
| 892 | 5477 | -0.0381 | 48396 | 2.972e-07 | -5.126 | `(spread_percentile_100 <= p10 AND (hour_of_day >= p25 OR NOT(prior_session_low_distance < p50)))` |
| 893 | 4740 | -0.0286 | 86392 | 3.093e-07 | -5.118 | `(NOT(spread_vs_trailing_100 > p10) OR NOT(distance_to_round_number >= p10))` |
| 894 | 7784 | -0.0349 | 57085 | 3.099e-07 | -5.118 | `((((atr_vs_trailing_100 < p90 AND NOT(swing_low_distance_14 < p90)) AND spread_vs_trailing_100 > p25) AND session_tokyo != p50) OR NOT(distance_to_round_number > p10))` |
| 895 | 46667 | -0.0608 | 17977 | 3.310e-07 | -5.107 | `(((hour_of_day > p75 AND NOT(spread_vs_trailing_100 < p75)) OR NOT(prior_session_high_distance != p90)) AND (NOT(range_close_ratio > p50) OR NOT(distance_to_round_number < p90)))` |
| 896 | 23749 | -0.0265 | 95422 | 3.465e-07 | -5.097 | `((swing_low_distance_14 != p90 AND prior_session_low_distance > p90) OR (spread_vs_trailing_100 < p50 AND NOT(day_of_week >= p25)))` |
| 897 | 27921 | +0.0512 | 22972 | 3.543e-07 | 5.094 | `(NOT(atr_vs_trailing_100 <= p90) AND kijun_26_distance > p50)` |
| 898 | 14133 | +0.0510 | 22976 | 3.885e-07 | 5.076 | `(NOT(kijun_26_distance < p50) AND NOT(atr_vs_trailing_100 <= p90))` |
| 899 | 17245 | -0.0347 | 56960 | 3.928e-07 | -5.073 | `(NOT(spread_percentile_100 > p10) OR distance_to_round_number == p25)` |
| 900 | 24519 | -0.0381 | 46341 | 4.329e-07 | -5.055 | `(swing_high_distance_14 <= p90 AND NOT(spread_percentile_100 > p10))` |
| 901 | 23609 | -0.0467 | 31548 | 4.429e-07 | -5.051 | `(NOT(day_of_week != p25) AND (range_close_ratio <= p25 OR swing_high_distance_14 >= p90))` |
| 902 | 7770 | -0.1256 | 3770 | 4.432e-07 | -5.058 | `((NOT(kijun_26_distance < p25) AND (spread_percentile_100 == p75 AND session_tokyo <= p90)) OR NOT(atr_vs_trailing_100 != p10))` |
| 903 | 41618 | -0.0553 | 22964 | 4.484e-07 | -5.049 | `(spread_vs_trailing_100 <= p90 AND (range_close_ratio < p25 AND NOT(day_of_week > p10)))` |
| 904 | 40080 | -0.1189 | 4530 | 4.548e-07 | -5.052 | `((atr_percentile_100 == p10 AND NOT(session_dead >= p90)) AND (prior_session_high_distance < p10 OR hour_of_day != p90))` |
| 905 | 25863 | -0.0325 | 63885 | 4.558e-07 | -5.045 | `((NOT(swing_high_distance_14 > p10) AND NOT(session_london <= p10)) OR prior_session_high_distance < p10)` |
| 906 | 44358 | +0.0417 | 43041 | 4.750e-07 | 5.037 | `(NOT(spread_vs_trailing_100 != p50) OR (NOT(range_close_ratio > p90) AND NOT(kijun_26_distance < p90)))` |
| 907 | 28771 | -0.0383 | 45640 | 5.039e-07 | -5.026 | `((NOT(spread_percentile_100 > p10) AND NOT(atr_percentile_100 <= p10)) AND NOT(range_close_ratio == p50))` |
| 908 | 16213 | -0.0618 | 15927 | 6.357e-07 | -4.982 | `((NOT(session_ldn_ny_overlap == p10) AND (NOT(swing_low_distance_14 == p50) OR swing_low_distance_14 == p50)) AND (NOT(day_of_week != p75) AND session_dead == p50))` |
| 909 | 43386 | -0.0393 | 42473 | 6.954e-07 | -4.963 | `(range_close_ratio != p75 AND ((NOT(hour_of_day != p75) OR NOT(prior_session_low_distance == p10)) AND (NOT(spread_percentile_100 >= p10) AND swing_low_distance_14 > p10)))` |
| 910 | 48257 | -0.0583 | 19199 | 6.974e-07 | -4.964 | `(session_ny < p50 AND NOT(distance_to_round_number >= p10))` |
| 911 | 2283 | -0.0572 | 20275 | 7.130e-07 | -4.959 | `((range_close_ratio < p10 AND (session_ny != p25 OR atr_percentile_100 != p75)) AND atr_percentile_100 > p50)` |
| 912 | 1512 | -0.0390 | 41471 | 7.980e-07 | -4.937 | `(range_close_ratio > p25 AND ((spread_percentile_100 <= p10 OR prior_session_low_distance == p25) AND (atr_14 != p25 AND NOT(distance_to_round_number == p75))))` |
| 913 | 44281 | -0.0320 | 68145 | 8.284e-07 | -4.929 | `(NOT(range_close_ratio >= p10) OR (session_ldn_ny_overlap != p10 AND (NOT(session_tokyo >= p25) OR (NOT(distance_to_round_number >= p25) OR NOT(session_dead == p25)))))` |
| 914 | 35935 | -0.1047 | 5690 | 8.770e-07 | -4.923 | `(atr_percentile_100 == p10 AND prior_session_low_distance > p25)` |
| 915 | 39303 | -0.0554 | 20558 | 9.092e-07 | -4.912 | `((NOT(distance_to_round_number == p50) AND (NOT(atr_vs_trailing_100 != p50) OR NOT(hour_of_day < p50))) AND distance_to_round_number < p10)` |
| 916 | 25107 | -0.0733 | 10680 | 9.163e-07 | -4.912 | `(((swing_high_distance_14 < p10 AND session_dead < p90) AND NOT(swing_low_distance_14 <= p50)) AND NOT(session_london > p10))` |
| 917 | 45177 | -0.0612 | 16889 | 9.470e-07 | -4.904 | `(day_of_week > p50 AND NOT(distance_to_round_number >= p10))` |
| 918 | 47372 | -0.1196 | 3958 | 1.013e-06 | -4.897 | `(NOT(day_of_week <= p90) AND kijun_26_distance > p75)` |
| 919 | 30414 | -0.0459 | 31046 | 1.069e-06 | -4.880 | `(range_close_ratio < p10 AND (atr_14 > p10 AND NOT(session_ldn_ny_overlap != p10)))` |
| 920 | 37370 | -0.1595 | 2036 | 1.074e-06 | -4.892 | `(NOT(atr_percentile_100 != p75) AND kijun_26_distance <= p50)` |
| 921 | 43098 | -0.0697 | 12200 | 1.125e-06 | -4.871 | `((swing_high_distance_14 > p10 AND NOT(prior_session_low_distance >= p50)) AND ((kijun_26_distance != p10 AND hour_of_day == p25) AND kijun_26_distance >= p50))` |
| 922 | 9171 | -0.0413 | 35782 | 1.152e-06 | -4.865 | `((prior_session_high_distance > p90 OR kijun_26_distance < p10) AND day_of_week > p50)` |
| 923 | 17653 | -0.0587 | 16534 | 1.184e-06 | -4.860 | `(hour_of_day >= p50 AND (prior_session_low_distance > p50 AND (NOT(swing_low_distance_14 < p50) AND (NOT(spread_vs_trailing_100 < p75) AND day_of_week != p10))))` |
| 924 | 38184 | -0.0293 | 72698 | 1.325e-06 | -4.836 | `((NOT(swing_high_distance_14 < p75) AND NOT(swing_high_distance_14 > p75)) OR (spread_percentile_100 >= p90 AND NOT(distance_to_round_number > p90)))` |
| 925 | 23435 | -0.0291 | 71412 | 1.404e-06 | -4.825 | `(NOT(spread_percentile_100 >= p10) OR (NOT(atr_14 < p90) AND (atr_percentile_100 > p50 AND NOT(day_of_week == p25))))` |
| 926 | 11789 | -0.0340 | 52843 | 1.505e-06 | -4.811 | `(kijun_26_distance != p75 AND (((prior_session_low_distance > p90 OR NOT(session_ldn_ny_overlap != p25)) AND day_of_week > p10) AND NOT(spread_percentile_100 != p90)))` |
| 927 | 43346 | -0.0808 | 9040 | 1.546e-06 | -4.808 | `((distance_to_round_number >= p75 AND NOT(hour_of_day <= p25)) AND atr_percentile_100 <= p10)` |
| 928 | 7340 | -0.0682 | 14065 | 1.666e-06 | -4.792 | `((spread_percentile_100 > p75 AND NOT(range_close_ratio >= p10)) AND (spread_vs_trailing_100 != p75 OR session_ldn_ny_overlap != p50))` |
| 929 | 19697 | -0.0542 | 18024 | 1.687e-06 | -4.789 | `(NOT(atr_vs_trailing_100 <= p90) AND (NOT(range_close_ratio >= p50) OR day_of_week == p25))` |
| 930 | 35931 | -0.0573 | 19910 | 1.712e-06 | -4.786 | `(session_ldn_ny_overlap <= p50 AND (NOT(day_of_week < p75) AND range_close_ratio <= p10))` |
| 931 | 40140 | -0.0706 | 12260 | 1.858e-06 | -4.771 | `(spread_vs_trailing_100 >= p75 AND (NOT(swing_low_distance_14 < p10) AND range_close_ratio < p10))` |
| 932 | 15998 | -0.0599 | 17671 | 1.929e-06 | -4.762 | `((NOT(hour_of_day <= p50) AND range_close_ratio <= p25) AND (prior_session_high_distance < p90 OR NOT(prior_session_high_distance > p75)))` |
| 933 | 1518 | -0.1470 | 2286 | 1.992e-06 | -4.767 | `((NOT(spread_percentile_100 != p75) AND prior_session_low_distance > p50) AND (swing_high_distance_14 >= p10 OR (session_dead > p25 OR prior_session_low_distance >= p50)))` |
| 934 | 2854 | -0.0543 | 18374 | 2.030e-06 | -4.752 | `((NOT(prior_session_high_distance <= p90) AND NOT(swing_high_distance_14 > p25)) OR atr_14 == p50)` |
| 935 | 31286 | -0.0287 | 72583 | 2.114e-06 | -4.743 | `(((session_dead != p25 OR session_dead != p75) AND spread_percentile_100 == p90) AND NOT(distance_to_round_number >= p90))` |
| 936 | 5838 | -0.0651 | 12055 | 2.136e-06 | -4.742 | `(atr_percentile_100 > p75 AND NOT(distance_to_round_number <= p90))` |
| 937 | 49818 | -0.0589 | 16629 | 2.147e-06 | -4.741 | `(NOT(hour_of_day != p10) AND (NOT(spread_percentile_100 < p90) OR swing_low_distance_14 == p50))` |
| 938 | 800 | -0.0431 | 35356 | 2.194e-06 | -4.735 | `(range_close_ratio < p10 AND spread_percentile_100 <= p75)` |
| 939 | 11759 | -0.0431 | 35356 | 2.194e-06 | -4.735 | `((range_close_ratio <= p10 AND spread_percentile_100 <= p75) OR (session_ldn_ny_overlap < p75 AND atr_percentile_100 > p50))` |
| 940 | 10383 | +0.0866 | 8099 | 2.260e-06 | 4.732 | `(distance_to_round_number != p90 AND (prior_session_low_distance < p50 AND (NOT(kijun_26_distance <= p75) AND NOT(atr_vs_trailing_100 < p50))))` |
| 941 | 26794 | -0.0661 | 13733 | 2.334e-06 | -4.724 | `((atr_14 == p25 OR NOT(atr_vs_trailing_100 >= p25)) AND (NOT(hour_of_day <= p75) AND atr_percentile_100 < p25))` |
| 942 | 27086 | -0.0403 | 31894 | 2.426e-06 | -4.715 | `((range_close_ratio >= p75 OR atr_14 == p75) AND (NOT(kijun_26_distance == p90) AND (spread_vs_trailing_100 <= p25 OR session_tokyo > p90)))` |
| 943 | 40630 | -0.0901 | 7068 | 2.512e-06 | -4.711 | `(session_dead > p75 OR ((NOT(atr_percentile_100 != p10) AND atr_14 <= p90) AND (NOT(swing_high_distance_14 >= p75) OR NOT(day_of_week == p25))))` |
| 944 | 46878 | -0.0517 | 22472 | 2.578e-06 | -4.703 | `(session_ny > p25 AND atr_percentile_100 < p10)` |
| 945 | 48755 | -0.0385 | 39534 | 2.703e-06 | -4.693 | `(NOT(day_of_week < p25) AND NOT(distance_to_round_number > p10))` |
| 946 | 44784 | -0.0953 | 5835 | 2.794e-06 | -4.690 | `((NOT(swing_high_distance_14 >= p25) AND (day_of_week > p75 AND (spread_percentile_100 >= p75 OR NOT(atr_vs_trailing_100 != p75)))) AND session_ldn_ny_overlap >= p75)` |
| 947 | 19305 | -0.0512 | 21211 | 2.960e-06 | -4.675 | `((session_ldn_ny_overlap > p25 OR NOT(spread_vs_trailing_100 < p50)) AND (NOT(atr_percentile_100 != p75) OR NOT(spread_percentile_100 > p10)))` |
| 948 | 21470 | -0.0926 | 6606 | 3.077e-06 | -4.670 | `((NOT(kijun_26_distance != p50) OR swing_low_distance_14 <= p10) AND (NOT(session_dead <= p75) OR distance_to_round_number >= p90))` |
| 949 | 49879 | -0.0851 | 6918 | 3.135e-06 | -4.666 | `(NOT(kijun_26_distance >= p25) AND (spread_vs_trailing_100 > p90 AND NOT(session_dead == p50)))` |
| 950 | 22164 | -0.0584 | 16890 | 3.245e-06 | -4.656 | `(prior_session_high_distance < p90 AND (distance_to_round_number < p10 AND session_tokyo == p90))` |
| 951 | 12486 | -0.0667 | 12277 | 3.364e-06 | -4.649 | `((atr_vs_trailing_100 > p10 AND NOT(distance_to_round_number > p10)) AND (session_tokyo >= p90 AND kijun_26_distance < p90))` |
| 952 | 15910 | -0.1204 | 3362 | 3.634e-06 | -4.639 | `(NOT(atr_percentile_100 != p75) AND (kijun_26_distance <= p75 OR distance_to_round_number >= p50))` |
| 953 | 38204 | -0.0610 | 14109 | 4.369e-06 | -4.595 | `((NOT(kijun_26_distance < p10) AND day_of_week > p90) AND session_ny <= p90)` |
| 954 | 23576 | -0.0515 | 20205 | 4.497e-06 | -4.588 | `(session_dead > p25 AND ((NOT(atr_percentile_100 <= p90) OR NOT(spread_vs_trailing_100 > p25)) AND (session_ldn_ny_overlap != p10 OR day_of_week <= p50)))` |
| 955 | 45004 | -0.0446 | 29470 | 4.628e-06 | -4.582 | `((range_close_ratio < p10 OR day_of_week < p10) AND swing_high_distance_14 > p25)` |
| 956 | 10082 | -0.0409 | 33176 | 4.682e-06 | -4.579 | `(distance_to_round_number <= p10 AND ((NOT(prior_session_low_distance <= p90) AND NOT(hour_of_day < p50)) OR NOT(session_dead > p10)))` |
| 957 | 4689 | -0.0499 | 21614 | 4.764e-06 | -4.576 | `(NOT(swing_high_distance_14 == p75) AND (NOT(day_of_week >= p25) AND spread_vs_trailing_100 > p75))` |
| 958 | 32369 | -0.0519 | 19880 | 5.167e-06 | -4.559 | `(day_of_week >= p75 AND NOT(spread_percentile_100 >= p10))` |
| 959 | 16043 | -0.0386 | 37058 | 5.455e-06 | -4.547 | `(NOT(prior_session_high_distance >= p75) AND spread_percentile_100 <= p10)` |
| 960 | 48509 | -0.0446 | 30889 | 5.714e-06 | -4.537 | `(NOT(range_close_ratio > p10) AND prior_session_low_distance < p50)` |
| 961 | 49708 | -0.0446 | 30889 | 5.714e-06 | -4.537 | `(NOT(range_close_ratio > p10) AND prior_session_low_distance < p50)` |
| 962 | 5202 | -0.0648 | 12246 | 5.796e-06 | -4.536 | `(NOT(kijun_26_distance < p50) AND (NOT(session_ldn_ny_overlap <= p75) AND NOT(prior_session_low_distance > p50)))` |
| 963 | 9103 | -0.0349 | 44153 | 6.183e-06 | -4.521 | `((NOT(spread_percentile_100 != p90) AND session_dead < p90) AND (hour_of_day != p10 AND NOT(swing_high_distance_14 >= p90)))` |
| 964 | 14536 | -0.0799 | 8344 | 6.484e-06 | -4.513 | `(kijun_26_distance != p50 AND (session_ny <= p10 AND (hour_of_day > p25 AND swing_high_distance_14 <= p10)))` |
| 965 | 37353 | -0.0338 | 47539 | 6.632e-06 | -4.506 | `((distance_to_round_number <= p90 OR session_ny < p75) AND NOT(spread_percentile_100 > p10))` |
| 966 | 28883 | -0.0565 | 14367 | 6.675e-06 | -4.505 | `(prior_session_high_distance < p50 AND (NOT(range_close_ratio < p75) AND NOT(kijun_26_distance <= p75)))` |
| 967 | 2903 | -0.0406 | 31700 | 6.961e-06 | -4.496 | `((((NOT(prior_session_high_distance == p25) AND NOT(session_tokyo <= p10)) AND NOT(kijun_26_distance >= p90)) AND NOT(distance_to_round_number >= p50)) AND session_london > p25)` |
| 968 | 1195 | -0.0686 | 10657 | 6.980e-06 | -4.497 | `((NOT(session_dead <= p50) AND NOT(kijun_26_distance < p25)) AND (NOT(day_of_week == p50) AND distance_to_round_number >= p90))` |
| 969 | 7055 | -0.0463 | 24016 | 6.993e-06 | -4.495 | `(((atr_14 > p90 OR spread_vs_trailing_100 >= p75) AND NOT(prior_session_high_distance <= p50)) AND NOT(session_ny == p10))` |
| 970 | 3935 | -0.0384 | 33412 | 7.484e-06 | -4.480 | `(NOT(range_close_ratio <= p50) AND ((NOT(swing_low_distance_14 > p25) AND NOT(atr_14 > p50)) OR NOT(atr_vs_trailing_100 != p90)))` |
| 971 | 4608 | -0.0898 | 6439 | 7.558e-06 | -4.481 | `(NOT(atr_percentile_100 != p10) AND NOT(hour_of_day == p50))` |
| 972 | 47725 | -0.0898 | 6439 | 7.558e-06 | -4.481 | `((NOT(session_ldn_ny_overlap != p75) OR (NOT(prior_session_high_distance >= p25) AND prior_session_low_distance == p10)) AND atr_percentile_100 == p10)` |
| 973 | 20785 | -0.0742 | 8204 | 7.560e-06 | -4.480 | `(((atr_vs_trailing_100 != p10 AND NOT(atr_percentile_100 < p90)) AND prior_session_low_distance <= p75) AND NOT(session_london < p90))` |
| 974 | 10616 | -0.0414 | 33322 | 7.716e-06 | -4.474 | `((NOT(prior_session_low_distance != p10) AND NOT(prior_session_low_distance < p75)) OR ((range_close_ratio <= p10 AND day_of_week < p75) OR NOT(atr_percentile_100 != p90)))` |
| 975 | 41553 | -0.0284 | 67802 | 7.774e-06 | -4.472 | `((NOT(distance_to_round_number >= p10) AND NOT(distance_to_round_number >= p75)) OR (NOT(session_tokyo >= p75) AND (prior_session_low_distance >= p90 OR NOT(range_close_ratio > p10))))` |
| 976 | 475 | -0.0371 | 38272 | 7.822e-06 | -4.471 | `(atr_vs_trailing_100 == p50 OR (NOT(prior_session_high_distance <= p50) AND NOT(spread_percentile_100 != p90)))` |
| 977 | 11074 | -0.0312 | 53176 | 8.099e-06 | -4.463 | `((prior_session_high_distance < p50 AND (NOT(swing_low_distance_14 > p90) AND spread_vs_trailing_100 <= p25)) AND NOT(spread_vs_trailing_100 >= p90))` |
| 978 | 39852 | -0.0509 | 20428 | 8.304e-06 | -4.458 | `(swing_high_distance_14 <= p75 AND (range_close_ratio <= p50 AND NOT(session_ldn_ny_overlap < p90)))` |
| 979 | 77 | -0.0657 | 9712 | 8.868e-06 | -4.445 | `((NOT(session_ldn_ny_overlap >= p50) OR NOT(session_dead != p90)) AND (session_dead > p90 OR range_close_ratio >= p90))` |
| 980 | 21485 | -0.0657 | 9712 | 8.868e-06 | -4.445 | `(session_dead >= p75 AND NOT(range_close_ratio < p90))` |
| 981 | 26123 | -0.0655 | 9714 | 9.497e-06 | -4.431 | `((NOT(range_close_ratio < p10) AND session_dead != p10) AND ((NOT(session_tokyo >= p75) OR kijun_26_distance == p90) OR NOT(range_close_ratio <= p90)))` |
| 982 | 13733 | -0.0617 | 12832 | 9.547e-06 | -4.429 | `((NOT(swing_high_distance_14 >= p10) OR NOT(atr_14 < p25)) AND NOT(day_of_week <= p90))` |
| 983 | 27787 | -0.0441 | 25655 | 1.106e-05 | -4.396 | `(NOT(swing_low_distance_14 >= p10) AND (spread_vs_trailing_100 <= p50 OR NOT(spread_percentile_100 != p10)))` |
| 984 | 36224 | -0.0357 | 41498 | 1.205e-05 | -4.377 | `(NOT(atr_vs_trailing_100 >= p50) AND (NOT(distance_to_round_number >= p50) AND session_dead >= p75))` |
| 985 | 11121 | -0.0285 | 62507 | 1.234e-05 | -4.372 | `(spread_vs_trailing_100 <= p25 AND (swing_low_distance_14 >= p50 OR NOT(swing_high_distance_14 < p75)))` |
| 986 | 7380 | -0.0921 | 5059 | 1.294e-05 | -4.366 | `(NOT(session_london == p10) AND NOT(spread_vs_trailing_100 < p90))` |
| 987 | 34253 | -0.0921 | 5059 | 1.294e-05 | -4.366 | `(session_london != p25 AND spread_vs_trailing_100 >= p90)` |
| 988 | 49092 | -0.0921 | 5059 | 1.294e-05 | -4.366 | `(NOT(session_london == p50) AND (session_london > p75 OR NOT(spread_vs_trailing_100 < p90)))` |
| 989 | 36 | -0.0330 | 46721 | 1.322e-05 | -4.357 | `((NOT(session_dead < p10) AND NOT(spread_percentile_100 > p10)) AND (distance_to_round_number < p90 OR atr_vs_trailing_100 < p10))` |
| 990 | 41718 | -0.0466 | 23842 | 1.342e-05 | -4.354 | `(range_close_ratio <= p25 AND (hour_of_day > p75 OR session_dead > p75))` |
| 991 | 34159 | -0.0394 | 30832 | 1.343e-05 | -4.354 | `((NOT(spread_vs_trailing_100 >= p10) AND kijun_26_distance < p90) AND atr_percentile_100 > p25)` |
| 992 | 37065 | -0.0488 | 20816 | 1.400e-05 | -4.345 | `((session_tokyo <= p90 AND NOT(prior_session_high_distance < p90)) AND (NOT(atr_vs_trailing_100 > p50) AND NOT(range_close_ratio == p90)))` |
| 993 | 34215 | -0.0915 | 5056 | 1.467e-05 | -4.338 | `(NOT(spread_vs_trailing_100 <= p90) AND NOT(session_london < p90))` |
| 994 | 44583 | -0.0915 | 5056 | 1.467e-05 | -4.338 | `((NOT(spread_vs_trailing_100 <= p90) AND session_london != p25) OR NOT(spread_vs_trailing_100 != p25))` |
| 995 | 17074 | -0.0631 | 11213 | 1.467e-05 | -4.335 | `(spread_percentile_100 < p10 AND NOT(atr_percentile_100 < p75))` |
| 996 | 8230 | -0.1160 | 3263 | 1.485e-05 | -4.337 | `(NOT(session_ny != p10) AND (spread_percentile_100 == p75 AND kijun_26_distance < p90))` |
| 997 | 46599 | -0.0607 | 14167 | 1.486e-05 | -4.332 | `((NOT(atr_percentile_100 >= p90) AND prior_session_low_distance <= p25) AND (spread_percentile_100 >= p90 AND swing_high_distance_14 < p75))` |
| 998 | 32442 | -0.0675 | 9418 | 1.558e-05 | -4.323 | `(NOT(atr_percentile_100 <= p75) AND (((prior_session_low_distance < p75 AND NOT(day_of_week != p75)) AND session_london == p25) AND prior_session_low_distance > p25))` |
| 999 | 19905 | -0.0945 | 5317 | 1.664e-05 | -4.310 | `(atr_vs_trailing_100 != p50 AND ((NOT(prior_session_low_distance >= p75) AND (NOT(spread_vs_trailing_100 <= p90) OR session_ldn_ny_overlap < p50)) AND prior_session_high_distance > p75))` |
| 1000 | 4429 | -0.0677 | 11132 | 1.698e-05 | -4.303 | `((NOT(kijun_26_distance != p25) OR ((NOT(session_dead != p25) OR swing_low_distance_14 == p90) AND NOT(spread_percentile_100 < p90))) AND swing_low_distance_14 <= p25)` |
| 1001 | 23039 | -0.0423 | 30327 | 1.730e-05 | -4.298 | `((NOT(range_close_ratio > p10) AND session_london <= p90) AND NOT(spread_vs_trailing_100 <= p50))` |
| 1002 | 430 | -0.0439 | 25814 | 1.805e-05 | -4.289 | `(((NOT(swing_high_distance_14 > p90) OR hour_of_day == p75) OR distance_to_round_number > p50) AND NOT(distance_to_round_number >= p10))` |
| 1003 | 13481 | -0.0350 | 40008 | 1.907e-05 | -4.276 | `(NOT(hour_of_day == p75) AND (NOT(spread_percentile_100 > p10) OR (prior_session_low_distance == p25 AND NOT(spread_vs_trailing_100 == p75))))` |
| 1004 | 43284 | -0.0229 | 97011 | 1.925e-05 | -4.274 | `((NOT(swing_low_distance_14 <= p90) AND session_ldn_ny_overlap >= p75) OR range_close_ratio <= p10)` |
| 1005 | 28358 | -0.0560 | 13192 | 2.035e-05 | -4.263 | `(((NOT(session_tokyo < p75) AND NOT(atr_percentile_100 < p90)) AND day_of_week != p75) AND (NOT(session_ny != p10) AND session_london == p50))` |
| 1006 | 6558 | -0.1103 | 3830 | 2.041e-05 | -4.266 | `((((NOT(day_of_week >= p90) OR atr_vs_trailing_100 != p90) AND NOT(atr_percentile_100 != p10)) AND NOT(spread_percentile_100 < p50)) AND atr_14 != p10)` |
| 1007 | 39211 | -0.1969 | 1036 | 2.055e-05 | -4.279 | `(spread_percentile_100 == p75 AND (NOT(atr_14 != p50) OR NOT(spread_vs_trailing_100 <= p75)))` |
| 1008 | 11635 | -0.0434 | 25306 | 2.093e-05 | -4.256 | `((distance_to_round_number <= p10 AND session_ny > p25) OR ((swing_low_distance_14 == p25 AND distance_to_round_number > p90) AND NOT(kijun_26_distance < p75)))` |
| 1009 | 17225 | -0.0434 | 25306 | 2.093e-05 | -4.256 | `(session_ny > p25 AND NOT(distance_to_round_number > p10))` |
| 1010 | 41340 | -0.0434 | 25306 | 2.093e-05 | -4.256 | `(session_ny == p75 AND distance_to_round_number <= p10)` |
| 1011 | 13826 | -0.0270 | 66284 | 2.103e-05 | -4.254 | `(((NOT(atr_14 > p50) OR NOT(kijun_26_distance >= p10)) AND (NOT(session_london != p90) OR kijun_26_distance <= p10)) AND NOT(swing_high_distance_14 < p50))` |
