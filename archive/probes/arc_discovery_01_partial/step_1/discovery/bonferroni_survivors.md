# Bonferroni survivors — arc_discovery_01

> Primary threshold (alpha=0.05, denominator=N_evaluated=427): p < 1.171e-04
> Budget threshold (alpha/10000, transparency only): p < 9.785e-05

| Rank | Rule ID | Mean R | Pool size | p-value | t-stat | Rule spec |
|---|---|---|---|---|---|---|
| 1 | 1 | -0.1021 | 478586 | 0.000e+00 | -41.782 | `(NOT(swing_high_distance_14 > p90) AND NOT(atr_14 > p25))` |
| 2 | 2 | -0.0847 | 1147663 | 0.000e+00 | -53.177 | `((NOT(prior_session_high_distance <= p90) AND NOT(atr_14 < p75)) OR (NOT(session_dead > p10) AND (session_ldn_ny_overlap == p50 OR session_tokyo == p10)))` |
| 3 | 3 | -0.0998 | 925907 | 0.000e+00 | -57.050 | `((atr_vs_trailing_100 > p25 AND (NOT(prior_session_high_distance > p75) OR NOT(session_ldn_ny_overlap < p50))) AND (NOT(swing_high_distance_14 != p25) OR NOT(session_london >= p90)))` |
| 4 | 4 | -0.0995 | 796913 | 0.000e+00 | -52.208 | `(NOT(session_dead <= p50) OR NOT(swing_low_distance_14 != p50))` |
| 5 | 5 | -0.0939 | 1419663 | 0.000e+00 | -66.103 | `(((prior_session_low_distance < p25 OR NOT(swing_high_distance_14 >= p90)) AND session_ldn_ny_overlap < p25) OR atr_percentile_100 > p25)` |
| 6 | 6 | -0.1038 | 887983 | 0.000e+00 | -57.385 | `(session_london <= p90 AND (((day_of_week != p90 OR NOT(session_dead > p50)) AND atr_14 != p50) AND spread_vs_trailing_100 > p50))` |
| 7 | 8 | -0.1039 | 470851 | 0.000e+00 | -42.295 | `(swing_high_distance_14 < p25 AND (NOT(prior_session_low_distance == p90) AND (swing_high_distance_14 != p25 OR NOT(session_london != p50))))` |
| 8 | 11 | -0.0942 | 1428574 | 0.000e+00 | -66.152 | `((NOT(spread_percentile_100 <= p25) AND ((session_ny <= p75 OR NOT(atr_percentile_100 != p10)) OR NOT(day_of_week < p25))) AND NOT(session_ny < p50))` |
| 9 | 12 | -0.0914 | 1903409 | 0.000e+00 | -74.047 | `((atr_percentile_100 != p90 OR (NOT(prior_session_high_distance < p75) AND NOT(kijun_26_distance <= p10))) OR NOT(prior_session_low_distance <= p50))` |
| 10 | 13 | -0.1012 | 1014573 | 0.000e+00 | -60.210 | `(range_close_ratio < p10 OR NOT(swing_high_distance_14 > p50))` |
| 11 | 14 | -0.0879 | 837733 | 0.000e+00 | -46.934 | `((NOT(spread_vs_trailing_100 > p25) AND NOT(atr_vs_trailing_100 != p25)) OR (atr_vs_trailing_100 <= p90 AND (NOT(prior_session_low_distance < p50) AND NOT(session_london < p50))))` |
| 12 | 16 | -0.0912 | 1877010 | 0.000e+00 | -73.377 | `(NOT(prior_session_low_distance == p75) AND (atr_vs_trailing_100 != p75 OR atr_percentile_100 > p50))` |
| 13 | 17 | -0.1057 | 529887 | 0.000e+00 | -46.200 | `(((day_of_week > p25 AND NOT(session_ldn_ny_overlap == p90)) AND atr_percentile_100 >= p50) OR session_ldn_ny_overlap < p25)` |
| 14 | 19 | -0.0913 | 1878129 | 0.000e+00 | -73.459 | `((atr_vs_trailing_100 >= p50 AND NOT(session_london != p90)) OR (prior_session_low_distance < p90 OR prior_session_low_distance > p10))` |
| 15 | 20 | -0.0927 | 1850682 | 0.000e+00 | -74.172 | `((NOT(session_london == p25) OR (atr_14 == p90 OR NOT(kijun_26_distance >= p75))) OR day_of_week != p50)` |
| 16 | 21 | -0.0803 | 717077 | 0.000e+00 | -39.701 | `((NOT(swing_high_distance_14 < p25) OR session_london != p25) AND NOT(session_london == p10))` |
| 17 | 22 | -0.0970 | 838749 | 0.000e+00 | -52.761 | `(((session_tokyo <= p25 OR NOT(hour_of_day != p10)) AND NOT(atr_vs_trailing_100 < p50)) OR NOT(distance_to_round_number > p10))` |
| 18 | 23 | -0.0956 | 1201437 | 0.000e+00 | -61.819 | `(NOT(session_tokyo <= p90) OR (((session_london < p25 OR NOT(atr_14 > p75)) AND atr_percentile_100 == p25) OR session_tokyo < p75))` |
| 19 | 25 | -0.0799 | 1212599 | 0.000e+00 | -51.211 | `((NOT(swing_high_distance_14 <= p75) AND session_ny <= p90) OR (NOT(spread_percentile_100 > p50) AND session_london <= p90))` |
| 20 | 28 | -0.1021 | 478586 | 0.000e+00 | -41.782 | `atr_14 < p25` |
| 21 | 30 | -0.0917 | 1542195 | 0.000e+00 | -66.955 | `(swing_low_distance_14 > p50 OR (((NOT(spread_vs_trailing_100 != p50) OR NOT(session_ny <= p10)) OR NOT(session_tokyo != p25)) OR NOT(swing_high_distance_14 <= p75)))` |
| 22 | 31 | -0.0914 | 1912126 | 0.000e+00 | -74.180 | `(prior_session_low_distance <= p25 OR NOT(swing_low_distance_14 == p75))` |
| 23 | 32 | -0.0921 | 1870628 | 0.000e+00 | -73.932 | `(spread_percentile_100 < p50 OR (prior_session_low_distance == p50 OR NOT(spread_vs_trailing_100 < p10)))` |
| 24 | 35 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(session_ldn_ny_overlap < p50) OR (prior_session_low_distance > p10 AND NOT(distance_to_round_number != p75)))` |
| 25 | 37 | -0.0917 | 1837320 | 0.000e+00 | -72.955 | `(((NOT(range_close_ratio == p75) AND atr_percentile_100 == p10) OR (NOT(prior_session_low_distance != p10) OR NOT(hour_of_day == p10))) OR NOT(spread_percentile_100 >= p10))` |
| 26 | 38 | -0.0884 | 1689741 | 0.000e+00 | -67.426 | `(prior_session_high_distance >= p90 OR (NOT(session_ny > p90) AND (NOT(prior_session_low_distance < p10) OR range_close_ratio == p90)))` |
| 27 | 39 | -0.0925 | 959425 | 0.000e+00 | -52.464 | `((NOT(spread_vs_trailing_100 == p75) AND atr_percentile_100 <= p50) AND atr_vs_trailing_100 < p90)` |
| 28 | 40 | -0.0910 | 1833427 | 0.000e+00 | -72.283 | `(hour_of_day <= p90 OR (hour_of_day == p75 OR NOT(spread_percentile_100 > p75)))` |
| 29 | 41 | -0.1154 | 478136 | 0.000e+00 | -47.558 | `hour_of_day >= p75` |
| 30 | 42 | -0.0875 | 1349577 | 0.000e+00 | -59.587 | `((NOT(session_tokyo > p50) AND day_of_week <= p90) OR NOT(swing_high_distance_14 <= p75))` |
| 31 | 43 | -0.0887 | 1724503 | 0.000e+00 | -68.357 | `((((NOT(kijun_26_distance >= p90) AND NOT(range_close_ratio <= p75)) OR NOT(hour_of_day >= p90)) AND NOT(atr_vs_trailing_100 < p10)) OR distance_to_round_number <= p50)` |
| 32 | 44 | -0.0962 | 1034015 | 0.000e+00 | -57.558 | `((NOT(prior_session_high_distance <= p90) AND kijun_26_distance == p90) OR (prior_session_low_distance <= p50 OR NOT(spread_vs_trailing_100 >= p10)))` |
| 33 | 45 | -0.1130 | 477578 | 0.000e+00 | -46.581 | `(NOT(swing_high_distance_14 <= p50) AND swing_high_distance_14 <= p75)` |
| 34 | 47 | -0.0914 | 1909940 | 0.000e+00 | -74.161 | `(NOT(distance_to_round_number == p90) OR (session_tokyo > p90 OR (NOT(distance_to_round_number <= p25) AND prior_session_low_distance >= p90)))` |
| 35 | 49 | -0.0932 | 933305 | 0.000e+00 | -51.916 | `(NOT(range_close_ratio >= p50) AND (spread_vs_trailing_100 <= p90 OR NOT(hour_of_day < p25)))` |
| 36 | 50 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((NOT(prior_session_high_distance <= p50) OR (NOT(session_tokyo >= p90) OR NOT(session_tokyo >= p90))) OR session_tokyo >= p25)` |
| 37 | 52 | -0.0913 | 1877292 | 0.000e+00 | -73.428 | `((session_london >= p10 OR atr_percentile_100 == p75) AND NOT(prior_session_high_distance == p50))` |
| 38 | 53 | -0.0930 | 1054446 | 0.000e+00 | -56.081 | `((NOT(day_of_week != p10) AND (atr_percentile_100 == p50 AND atr_vs_trailing_100 >= p50)) OR (swing_high_distance_14 >= p90 OR prior_session_low_distance < p50))` |
| 39 | 54 | -0.0833 | 1274659 | 0.000e+00 | -54.896 | `(session_tokyo > p25 OR (NOT(session_london == p50) OR session_london > p50))` |
| 40 | 56 | -0.0913 | 1899557 | 0.000e+00 | -73.846 | `(NOT(atr_percentile_100 >= p75) OR NOT(prior_session_high_distance == p25))` |
| 41 | 62 | -0.0973 | 1242364 | 0.000e+00 | -63.883 | `(spread_vs_trailing_100 >= p75 OR (NOT(atr_percentile_100 <= p75) OR (NOT(spread_percentile_100 != p75) OR (NOT(session_dead == p25) OR prior_session_low_distance == p90))))` |
| 42 | 63 | -0.0798 | 999324 | 0.000e+00 | -46.377 | `((day_of_week != p90 AND (NOT(session_ny <= p25) OR (NOT(session_london <= p10) AND NOT(session_dead != p50)))) OR NOT(atr_vs_trailing_100 > p10))` |
| 43 | 64 | -0.0914 | 1912155 | 0.000e+00 | -74.190 | `(NOT(swing_low_distance_14 == p90) OR day_of_week <= p50)` |
| 44 | 65 | -0.0914 | 1912190 | 0.000e+00 | -74.188 | `NOT(range_close_ratio == p50)` |
| 45 | 67 | -0.0843 | 1014281 | 0.000e+00 | -49.667 | `((NOT(session_ny <= p50) AND range_close_ratio != p50) OR (NOT(distance_to_round_number < p90) OR (NOT(prior_session_low_distance <= p75) AND NOT(session_tokyo <= p50))))` |
| 46 | 71 | -0.0849 | 713343 | 0.000e+00 | -41.757 | `(session_tokyo > p50 AND ((NOT(range_close_ratio > p75) OR NOT(session_dead < p90)) OR NOT(atr_vs_trailing_100 <= p10)))` |
| 47 | 73 | -0.0936 | 1605952 | 0.000e+00 | -69.506 | `((NOT(swing_low_distance_14 != p90) OR NOT(session_dead < p75)) OR (NOT(prior_session_low_distance == p10) AND NOT(atr_vs_trailing_100 >= p75)))` |
| 48 | 75 | -0.0845 | 716923 | 0.000e+00 | -41.649 | `session_tokyo > p50` |
| 49 | 79 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(spread_vs_trailing_100 == p90) OR (NOT(distance_to_round_number == p75) AND hour_of_day != p90))` |
| 50 | 80 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `session_ny >= p25` |
| 51 | 82 | -0.1154 | 478136 | 0.000e+00 | -47.558 | `(hour_of_day >= p75 AND (session_ny > p25 OR NOT(session_ldn_ny_overlap > p75)))` |
| 52 | 83 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(session_ldn_ny_overlap < p10) AND (NOT(session_dead < p25) OR (NOT(distance_to_round_number != p75) AND NOT(atr_14 > p10))))` |
| 53 | 84 | -0.0898 | 647944 | 0.000e+00 | -42.515 | `(NOT(range_close_ratio > p10) OR (atr_vs_trailing_100 > p75 AND atr_14 != p90))` |
| 54 | 87 | -0.0921 | 1786461 | 0.000e+00 | -72.412 | `((session_tokyo != p75 AND NOT(distance_to_round_number > p75)) OR NOT(atr_percentile_100 < p10))` |
| 55 | 88 | -0.0934 | 1720985 | 0.000e+00 | -71.916 | `((((NOT(atr_percentile_100 != p25) OR NOT(atr_percentile_100 != p50)) OR session_ny <= p75) OR session_dead < p75) AND spread_vs_trailing_100 >= p10)` |
| 56 | 89 | -0.1009 | 1106811 | 0.000e+00 | -63.026 | `((NOT(swing_high_distance_14 >= p50) OR (range_close_ratio >= p75 AND NOT(session_ny < p75))) OR NOT(session_ny >= p10))` |
| 57 | 90 | -0.0969 | 1545809 | 0.000e+00 | -71.150 | `(atr_14 < p90 AND (NOT(atr_percentile_100 < p10) AND session_ldn_ny_overlap >= p75))` |
| 58 | 91 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((NOT(session_tokyo > p90) AND session_ldn_ny_overlap <= p50) OR (NOT(atr_14 != p25) OR NOT(session_london == p10)))` |
| 59 | 92 | -0.0917 | 1769046 | 0.000e+00 | -71.669 | `(NOT(spread_percentile_100 < p10) OR atr_vs_trailing_100 <= p25)` |
| 60 | 93 | -0.0994 | 796686 | 0.000e+00 | -52.182 | `session_dead != p50` |
| 61 | 97 | -0.0891 | 1625607 | 0.000e+00 | -66.642 | `(NOT(swing_low_distance_14 > p10) OR swing_low_distance_14 >= p25)` |
| 62 | 99 | -0.0988 | 1151247 | 0.000e+00 | -62.490 | `(NOT(session_dead < p75) OR (NOT(prior_session_high_distance < p10) AND NOT(session_london == p75)))` |
| 63 | 100 | -0.0880 | 1479858 | 0.000e+00 | -62.673 | `((NOT(hour_of_day <= p90) AND (spread_percentile_100 > p10 OR NOT(spread_vs_trailing_100 <= p25))) OR (range_close_ratio <= p90 AND NOT(spread_percentile_100 >= p90)))` |
| 64 | 101 | -0.0914 | 1895463 | 0.000e+00 | -73.838 | `(NOT(session_ldn_ny_overlap <= p50) OR NOT(atr_percentile_100 == p90))` |
| 65 | 102 | -0.0803 | 717077 | 0.000e+00 | -39.701 | `session_london == p75` |
| 66 | 104 | -0.0914 | 1911229 | 0.000e+00 | -74.183 | `((NOT(distance_to_round_number == p90) OR session_dead < p75) OR swing_low_distance_14 >= p90)` |
| 67 | 105 | -0.0914 | 1912183 | 0.000e+00 | -74.188 | `(((session_ldn_ny_overlap >= p10 OR swing_low_distance_14 >= p10) OR swing_low_distance_14 == p25) AND (spread_vs_trailing_100 != p90 AND NOT(session_london < p25)))` |
| 68 | 106 | -0.0955 | 1195268 | 0.000e+00 | -61.621 | `NOT(session_tokyo >= p75)` |
| 69 | 107 | -0.0914 | 1892880 | 0.000e+00 | -73.836 | `(spread_vs_trailing_100 == p75 OR atr_percentile_100 != p25)` |
| 70 | 108 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(((NOT(hour_of_day <= p90) AND session_ldn_ny_overlap != p50) OR NOT(prior_session_high_distance >= p90)) OR range_close_ratio != p25)` |
| 71 | 111 | -0.0748 | 1145847 | 0.000e+00 | -46.320 | `day_of_week <= p50` |
| 72 | 114 | -0.0889 | 858410 | 0.000e+00 | -48.413 | `((hour_of_day <= p25 OR NOT(atr_percentile_100 <= p75)) AND (atr_14 >= p90 OR (range_close_ratio <= p90 AND NOT(distance_to_round_number == p75))))` |
| 73 | 116 | -0.0886 | 1407666 | 0.000e+00 | -61.734 | `NOT(prior_session_low_distance <= p25)` |
| 74 | 118 | -0.0932 | 1721140 | 0.000e+00 | -71.554 | `NOT(range_close_ratio >= p90)` |
| 75 | 120 | -0.0908 | 1720694 | 0.000e+00 | -69.773 | `(spread_vs_trailing_100 <= p90 AND NOT(atr_vs_trailing_100 == p10))` |
| 76 | 121 | -0.0889 | 1398960 | 0.000e+00 | -61.663 | `((NOT(session_dead == p75) OR NOT(kijun_26_distance != p90)) OR (NOT(session_tokyo < p90) AND (swing_high_distance_14 <= p50 AND session_london >= p10)))` |
| 77 | 122 | -0.0914 | 1879662 | 0.000e+00 | -73.505 | `(swing_high_distance_14 > p10 OR (NOT(atr_percentile_100 >= p75) OR NOT(distance_to_round_number != p75)))` |
| 78 | 125 | -0.0914 | 1911425 | 0.000e+00 | -74.133 | `NOT(distance_to_round_number == p50)` |
| 79 | 126 | -0.0891 | 1025111 | 0.000e+00 | -52.817 | `(((NOT(range_close_ratio < p90) AND range_close_ratio >= p10) OR NOT(session_ny == p75)) AND (NOT(spread_vs_trailing_100 >= p75) OR range_close_ratio <= p25))` |
| 80 | 128 | -0.0994 | 796546 | 0.000e+00 | -52.154 | `(session_dead != p10 AND (NOT(atr_vs_trailing_100 == p25) AND session_london != p90))` |
| 81 | 129 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((kijun_26_distance >= p50 AND atr_14 < p25) OR ((kijun_26_distance <= p25 OR range_close_ratio <= p75) OR range_close_ratio >= p50))` |
| 82 | 131 | -0.0883 | 1331413 | 0.000e+00 | -60.075 | `(NOT(swing_low_distance_14 < p25) AND ((NOT(distance_to_round_number < p50) OR swing_low_distance_14 < p75) OR NOT(prior_session_low_distance <= p75)))` |
| 83 | 132 | -0.0914 | 1911855 | 0.000e+00 | -74.142 | `(hour_of_day == p10 OR atr_vs_trailing_100 != p10)` |
| 84 | 133 | -0.1087 | 956109 | 0.000e+00 | -62.612 | `(NOT(atr_vs_trailing_100 != p25) OR NOT(spread_vs_trailing_100 <= p50))` |
| 85 | 134 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(atr_14 == p90) OR (spread_vs_trailing_100 != p50 AND (atr_14 == p75 AND hour_of_day > p10)))` |
| 86 | 135 | -0.0918 | 1437876 | 0.000e+00 | -64.143 | `((NOT(atr_vs_trailing_100 >= p75) AND NOT(prior_session_high_distance != p25)) OR NOT(atr_percentile_100 > p75))` |
| 87 | 137 | -0.0994 | 796707 | 0.000e+00 | -52.188 | `(((NOT(spread_vs_trailing_100 > p25) AND NOT(swing_low_distance_14 <= p50)) AND prior_session_low_distance == p90) OR session_dead > p25)` |
| 88 | 139 | -0.0944 | 1578514 | 0.000e+00 | -69.701 | `(((distance_to_round_number != p10 AND NOT(session_ldn_ny_overlap > p10)) OR session_dead == p90) OR session_dead > p50)` |
| 89 | 140 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(session_ldn_ny_overlap == p75) OR ((NOT(range_close_ratio < p10) OR distance_to_round_number <= p25) OR NOT(session_ldn_ny_overlap > p10)))` |
| 90 | 141 | -0.0913 | 1775017 | 0.000e+00 | -71.184 | `((NOT(distance_to_round_number > p25) AND NOT(swing_high_distance_14 == p90)) OR NOT(atr_vs_trailing_100 >= p90))` |
| 91 | 142 | -0.0892 | 1691987 | 0.000e+00 | -67.805 | `(range_close_ratio <= p75 OR (swing_low_distance_14 == p25 OR (NOT(distance_to_round_number >= p50) OR NOT(distance_to_round_number > p25))))` |
| 92 | 143 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `session_ldn_ny_overlap >= p25` |
| 93 | 144 | -0.0994 | 796686 | 0.000e+00 | -52.182 | `((NOT(session_tokyo > p25) OR (session_dead == p75 AND NOT(session_dead <= p50))) AND NOT(session_dead == p25))` |
| 94 | 145 | -0.0850 | 1036133 | 0.000e+00 | -50.621 | `((NOT(spread_percentile_100 == p25) AND NOT(prior_session_low_distance <= p50)) OR (swing_high_distance_14 > p75 AND kijun_26_distance <= p50))` |
| 95 | 148 | -0.0914 | 1912183 | 0.000e+00 | -74.188 | `(spread_vs_trailing_100 != p90 OR (NOT(session_tokyo >= p75) AND NOT(swing_low_distance_14 != p90)))` |
| 96 | 149 | -0.0901 | 1720734 | 0.000e+00 | -69.685 | `NOT(range_close_ratio <= p10)` |
| 97 | 150 | -0.1026 | 1191872 | 0.000e+00 | -66.266 | `(range_close_ratio >= p75 OR NOT(spread_vs_trailing_100 <= p50))` |
| 98 | 153 | -0.0800 | 956034 | 0.000e+00 | -45.452 | `(atr_vs_trailing_100 == p10 OR NOT(swing_high_distance_14 < p50))` |
| 99 | 154 | -0.0913 | 1877374 | 0.000e+00 | -73.426 | `prior_session_low_distance != p75` |
| 100 | 155 | -0.1043 | 478371 | 0.000e+00 | -42.203 | `((NOT(swing_low_distance_14 > p50) OR session_tokyo == p25) AND NOT(swing_low_distance_14 > p25))` |
| 101 | 156 | -0.0889 | 1192877 | 0.000e+00 | -57.208 | `(hour_of_day >= p25 AND (NOT(distance_to_round_number == p75) AND spread_percentile_100 != p90))` |
| 102 | 159 | -0.0914 | 1911828 | 0.000e+00 | -74.158 | `((NOT(session_ny > p25) AND (session_tokyo >= p50 AND atr_vs_trailing_100 <= p10)) OR swing_low_distance_14 != p50)` |
| 103 | 163 | -0.0903 | 1549795 | 0.000e+00 | -65.988 | `(((swing_high_distance_14 <= p25 OR NOT(kijun_26_distance <= p50)) OR NOT(session_dead == p90)) OR NOT(session_dead != p10))` |
| 104 | 164 | -0.1162 | 766344 | 0.000e+00 | -61.108 | `NOT(day_of_week < p75)` |
| 105 | 166 | -0.0932 | 1597656 | 0.000e+00 | -69.214 | `(distance_to_round_number == p25 OR ((atr_14 != p50 AND NOT(session_ny != p25)) OR NOT(session_london > p10)))` |
| 106 | 168 | -0.0907 | 1865747 | 0.000e+00 | -72.694 | `(distance_to_round_number <= p90 OR (atr_vs_trailing_100 > p25 OR NOT(swing_low_distance_14 != p75)))` |
| 107 | 169 | -0.0914 | 1912190 | 0.000e+00 | -74.188 | `(NOT(prior_session_low_distance != p10) OR range_close_ratio != p10)` |
| 108 | 170 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(session_london >= p10 OR NOT(prior_session_high_distance <= p75))` |
| 109 | 171 | -0.1088 | 945892 | 0.000e+00 | -62.287 | `(spread_vs_trailing_100 > p50 AND (NOT(session_london == p75) OR (spread_percentile_100 > p10 OR (prior_session_low_distance < p10 OR session_london == p50))))` |
| 110 | 173 | -0.0845 | 1097585 | 0.000e+00 | -51.824 | `(session_ldn_ny_overlap == p90 OR NOT(prior_session_low_distance < p50))` |
| 111 | 174 | -0.1079 | 1434143 | 0.000e+00 | -76.870 | `NOT(atr_14 >= p75)` |
| 112 | 175 | -0.0913 | 1877282 | 0.000e+00 | -73.428 | `prior_session_low_distance != p90` |
| 113 | 179 | -0.0900 | 1808258 | 0.000e+00 | -70.985 | `((NOT(atr_14 <= p10) AND session_ldn_ny_overlap != p10) OR ((day_of_week < p90 AND session_ny <= p10) OR prior_session_low_distance > p10))` |
| 114 | 180 | -0.0856 | 1115488 | 0.000e+00 | -53.036 | `((NOT(swing_low_distance_14 > p90) OR (NOT(atr_vs_trailing_100 == p90) OR NOT(distance_to_round_number < p50))) AND NOT(session_dead >= p75))` |
| 115 | 181 | -0.0926 | 1007936 | 0.000e+00 | -54.381 | `(NOT(swing_low_distance_14 > p25) OR NOT(session_tokyo <= p50))` |
| 116 | 182 | -0.0915 | 578907 | 0.000e+00 | -39.738 | `(range_close_ratio <= p50 AND atr_vs_trailing_100 < p50)` |
| 117 | 183 | -0.0986 | 1728655 | 0.000e+00 | -76.534 | `(distance_to_round_number > p25 OR session_london == p10)` |
| 118 | 184 | -0.0899 | 1408158 | 0.000e+00 | -62.335 | `(session_london <= p90 AND NOT(prior_session_high_distance > p75))` |
| 119 | 185 | -0.1021 | 478586 | 0.000e+00 | -41.782 | `NOT(atr_14 > p25)` |
| 120 | 186 | -0.0822 | 799436 | 0.000e+00 | -42.617 | `((swing_low_distance_14 == p50 OR NOT(atr_vs_trailing_100 > p10)) OR (NOT(atr_vs_trailing_100 >= p10) OR session_tokyo > p25))` |
| 121 | 188 | -0.0916 | 1900245 | 0.000e+00 | -74.126 | `((NOT(session_ny < p90) OR swing_high_distance_14 > p75) OR NOT(spread_percentile_100 == p75))` |
| 122 | 189 | -0.0924 | 1886725 | 0.000e+00 | -74.467 | `((NOT(atr_14 >= p90) AND prior_session_low_distance > p75) OR (NOT(day_of_week < p25) OR atr_percentile_100 < p90))` |
| 123 | 190 | -0.0935 | 1593420 | 0.000e+00 | -69.302 | `(prior_session_high_distance == p50 OR session_ldn_ny_overlap == p10)` |
| 124 | 191 | -0.1014 | 867644 | 0.000e+00 | -55.691 | `((NOT(spread_percentile_100 == p50) AND NOT(atr_percentile_100 > p10)) OR (NOT(spread_vs_trailing_100 <= p10) AND (distance_to_round_number >= p25 AND NOT(session_dead > p10))))` |
| 125 | 193 | -0.0913 | 1903440 | 0.000e+00 | -73.963 | `((distance_to_round_number < p75 OR NOT(spread_percentile_100 != p50)) OR prior_session_high_distance != p90)` |
| 126 | 194 | -0.0915 | 1840844 | 0.000e+00 | -72.898 | `((hour_of_day != p10 OR NOT(swing_low_distance_14 <= p90)) OR NOT(prior_session_high_distance != p90))` |
| 127 | 195 | -0.0917 | 1876664 | 0.000e+00 | -73.703 | `(atr_vs_trailing_100 < p50 OR hour_of_day != p10)` |
| 128 | 196 | -0.0926 | 985151 | 0.000e+00 | -54.116 | `(prior_session_low_distance < p25 OR ((NOT(day_of_week != p10) OR session_tokyo != p75) AND spread_percentile_100 <= p50))` |
| 129 | 198 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(((NOT(range_close_ratio < p10) OR NOT(swing_low_distance_14 == p75)) OR NOT(atr_14 <= p10)) OR NOT(kijun_26_distance > p10))` |
| 130 | 199 | -0.0934 | 945857 | 0.000e+00 | -53.345 | `spread_percentile_100 > p50` |
| 131 | 202 | -0.0939 | 1720262 | 0.000e+00 | -72.505 | `((session_tokyo < p25 OR NOT(kijun_26_distance < p10)) AND (NOT(distance_to_round_number == p50) OR (NOT(atr_vs_trailing_100 <= p90) AND NOT(session_dead >= p90))))` |
| 132 | 204 | -0.0914 | 1034294 | 0.000e+00 | -54.573 | `((NOT(session_london >= p25) OR ((session_ldn_ny_overlap > p50 AND range_close_ratio >= p50) OR prior_session_high_distance < p50)) AND prior_session_low_distance != p75)` |
| 133 | 206 | -0.0930 | 1675191 | 0.000e+00 | -70.618 | `((NOT(spread_vs_trailing_100 == p10) AND NOT(kijun_26_distance > p25)) OR NOT(session_ldn_ny_overlap != p25))` |
| 134 | 207 | -0.0915 | 1895901 | 0.000e+00 | -73.996 | `(((NOT(atr_14 < p50) OR (hour_of_day <= p90 AND session_ldn_ny_overlap < p50)) AND NOT(atr_14 <= p10)) OR prior_session_high_distance != p25)` |
| 135 | 208 | -0.0932 | 657171 | 0.000e+00 | -44.649 | `((atr_14 <= p25 OR (prior_session_high_distance >= p90 OR NOT(range_close_ratio != p50))) OR NOT(session_tokyo >= p25))` |
| 136 | 210 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((session_ny <= p75 OR swing_low_distance_14 != p90) OR (swing_high_distance_14 != p75 AND session_ldn_ny_overlap == p50))` |
| 137 | 211 | -0.0898 | 943667 | 0.000e+00 | -51.175 | `spread_percentile_100 < p50` |
| 138 | 213 | -0.0841 | 955870 | 0.000e+00 | -47.926 | `(NOT(session_ny == p75) AND NOT(hour_of_day > p75))` |
| 139 | 214 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(atr_14 <= p50) OR (swing_low_distance_14 < p75 OR NOT(atr_vs_trailing_100 < p90)))` |
| 140 | 216 | -0.0924 | 943965 | 0.000e+00 | -51.944 | `NOT(atr_percentile_100 >= p50)` |
| 141 | 217 | -0.0875 | 795203 | 0.000e+00 | -45.791 | `(((session_dead <= p50 AND NOT(hour_of_day <= p10)) AND kijun_26_distance != p10) AND (NOT(session_ldn_ny_overlap != p10) AND NOT(prior_session_low_distance == p90)))` |
| 142 | 222 | -0.1050 | 1407986 | 0.000e+00 | -73.651 | `((NOT(prior_session_low_distance != p10) OR NOT(session_dead < p10)) AND prior_session_low_distance <= p75)` |
| 143 | 223 | -0.0930 | 1876415 | 0.000e+00 | -74.877 | `distance_to_round_number != p10` |
| 144 | 224 | -0.0867 | 1434140 | 0.000e+00 | -60.782 | `(session_ny == p75 OR NOT(session_tokyo < p90))` |
| 145 | 227 | -0.0954 | 1742447 | 0.000e+00 | -74.027 | `(NOT(spread_percentile_100 >= p25) OR NOT(prior_session_low_distance > p90))` |
| 146 | 228 | -0.0923 | 1666868 | 0.000e+00 | -69.828 | `(atr_vs_trailing_100 <= p75 OR ((NOT(spread_percentile_100 <= p50) OR NOT(spread_percentile_100 != p10)) AND NOT(atr_percentile_100 == p90)))` |
| 147 | 229 | -0.0711 | 955136 | 0.000e+00 | -40.318 | `((session_dead >= p10 OR session_tokyo >= p90) AND ((NOT(day_of_week > p25) OR session_ldn_ny_overlap != p50) OR NOT(day_of_week > p25)))` |
| 148 | 230 | -0.1010 | 1654460 | 0.000e+00 | -76.694 | `(NOT(kijun_26_distance > p75) OR (NOT(session_dead > p75) AND NOT(swing_low_distance_14 > p75)))` |
| 149 | 231 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `spread_vs_trailing_100 != p10` |
| 150 | 232 | -0.0900 | 1816095 | 0.000e+00 | -71.083 | `(((NOT(swing_low_distance_14 < p25) AND NOT(session_dead == p75)) OR prior_session_low_distance <= p50) OR distance_to_round_number < p75)` |
| 151 | 233 | -0.0916 | 1906347 | 0.000e+00 | -74.282 | `((kijun_26_distance < p90 OR (distance_to_round_number != p25 OR session_dead != p10)) OR (NOT(session_ny > p90) AND session_london < p10))` |
| 152 | 236 | -0.0897 | 1754282 | 0.000e+00 | -69.569 | `((session_london <= p75 AND (swing_high_distance_14 > p10 OR kijun_26_distance == p75)) OR (range_close_ratio <= p25 AND atr_vs_trailing_100 < p25))` |
| 153 | 237 | -0.0928 | 1194974 | 0.000e+00 | -59.505 | `NOT(session_ny >= p90)` |
| 154 | 240 | -0.0854 | 1565277 | 0.000e+00 | -62.436 | `((NOT(prior_session_high_distance > p50) OR day_of_week <= p50) OR atr_14 == p50)` |
| 155 | 242 | -0.0955 | 1195268 | 0.000e+00 | -61.621 | `NOT(session_tokyo != p50)` |
| 156 | 243 | -0.0924 | 943965 | 0.000e+00 | -51.944 | `atr_percentile_100 < p50` |
| 157 | 244 | -0.0914 | 1912190 | 0.000e+00 | -74.188 | `((session_ny <= p75 AND NOT(session_ny >= p10)) OR (NOT(atr_vs_trailing_100 != p90) OR (NOT(session_ldn_ny_overlap == p10) OR NOT(range_close_ratio == p10))))` |
| 158 | 245 | -0.1175 | 319773 | 0.000e+00 | -38.852 | `NOT(spread_percentile_100 < p90)` |
| 159 | 246 | -0.0905 | 955399 | 0.000e+00 | -52.882 | `(((range_close_ratio != p50 AND NOT(atr_14 == p90)) AND NOT(atr_14 == p90)) AND NOT(range_close_ratio < p50))` |
| 160 | 247 | -0.0914 | 1911777 | 0.000e+00 | -74.175 | `(NOT(prior_session_high_distance == p50) OR hour_of_day != p90)` |
| 161 | 251 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `spread_vs_trailing_100 != p10` |
| 162 | 253 | -0.0913 | 1861735 | 0.000e+00 | -73.097 | `((NOT(prior_session_high_distance <= p25) OR session_ldn_ny_overlap < p10) OR (NOT(spread_percentile_100 <= p10) OR session_dead > p90))` |
| 163 | 256 | -0.0918 | 1898126 | 0.000e+00 | -74.296 | `(NOT(swing_low_distance_14 < p10) OR (spread_vs_trailing_100 <= p90 OR (NOT(range_close_ratio > p25) AND (spread_percentile_100 < p25 OR distance_to_round_number == p25))))` |
| 164 | 257 | -0.0956 | 469187 | 0.000e+00 | -38.933 | `(NOT(prior_session_high_distance <= p75) AND (kijun_26_distance == p90 OR (day_of_week < p75 OR (NOT(swing_high_distance_14 == p10) OR distance_to_round_number == p90))))` |
| 165 | 258 | -0.0803 | 717077 | 0.000e+00 | -39.701 | `((spread_percentile_100 <= p50 AND (NOT(session_ny < p50) AND (atr_vs_trailing_100 >= p90 AND NOT(session_ny >= p25)))) OR session_london != p25)` |
| 166 | 259 | -0.0941 | 1826112 | 0.000e+00 | -74.807 | `((NOT(distance_to_round_number <= p90) OR NOT(swing_high_distance_14 <= p90)) OR (kijun_26_distance <= p90 OR spread_percentile_100 < p50))` |
| 167 | 260 | -0.1009 | 786237 | 0.000e+00 | -52.642 | `((distance_to_round_number != p10 OR NOT(session_tokyo != p25)) AND session_dead == p90)` |
| 168 | 261 | -0.0941 | 1354764 | 0.000e+00 | -64.275 | `(NOT(atr_14 > p25) OR NOT(session_ny > p10))` |
| 169 | 262 | -0.0916 | 1895308 | 0.000e+00 | -74.001 | `atr_percentile_100 != p75` |
| 170 | 264 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((NOT(session_london <= p90) OR atr_14 != p10) OR NOT(swing_high_distance_14 == p50))` |
| 171 | 266 | -0.0913 | 1877327 | 0.000e+00 | -73.424 | `(((spread_vs_trailing_100 >= p90 OR swing_low_distance_14 >= p75) AND prior_session_high_distance >= p75) OR (NOT(day_of_week >= p10) OR prior_session_high_distance != p25))` |
| 172 | 267 | -0.0842 | 938682 | 0.000e+00 | -47.806 | `(NOT(prior_session_low_distance < p50) OR (((NOT(swing_high_distance_14 < p75) AND NOT(kijun_26_distance == p50)) AND NOT(spread_percentile_100 != p25)) AND NOT(prior_session_low_distance < p25)))` |
| 173 | 268 | -0.1048 | 1530427 | 0.000e+00 | -76.951 | `(day_of_week > p75 OR NOT(swing_high_distance_14 > p75))` |
| 174 | 269 | -0.0926 | 1842358 | 0.000e+00 | -73.765 | `(((NOT(hour_of_day == p50) OR swing_low_distance_14 < p10) OR spread_percentile_100 == p10) OR hour_of_day <= p10)` |
| 175 | 270 | -0.0924 | 1832493 | 0.000e+00 | -73.477 | `(NOT(atr_14 != p25) OR (session_dead >= p90 OR NOT(hour_of_day == p50)))` |
| 176 | 271 | -0.0929 | 1537265 | 0.000e+00 | -67.626 | `(NOT(spread_percentile_100 <= p25) OR (NOT(session_ldn_ny_overlap <= p50) AND swing_low_distance_14 < p90))` |
| 177 | 272 | -0.0905 | 1720790 | 0.000e+00 | -69.666 | `(session_london < p25 OR NOT(distance_to_round_number <= p10))` |
| 178 | 273 | -0.1118 | 717007 | 0.000e+00 | -56.419 | `(spread_percentile_100 >= p50 AND NOT(swing_low_distance_14 > p75))` |
| 179 | 275 | -0.0955 | 1195268 | 0.000e+00 | -61.621 | `NOT(session_tokyo != p50)` |
| 180 | 276 | -0.0905 | 955443 | 0.000e+00 | -52.885 | `((NOT(atr_14 <= p10) AND kijun_26_distance == p90) OR range_close_ratio >= p50)` |
| 181 | 277 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(session_london == p50) OR (NOT(hour_of_day != p50) OR (NOT(atr_vs_trailing_100 > p25) OR session_dead >= p50)))` |
| 182 | 279 | -0.0872 | 1680271 | 0.000e+00 | -66.209 | `((atr_14 <= p90 OR (NOT(day_of_week >= p90) OR NOT(session_london >= p25))) AND distance_to_round_number < p90)` |
| 183 | 281 | -0.0912 | 1243170 | 0.000e+00 | -59.414 | `(((swing_high_distance_14 < p90 OR swing_low_distance_14 <= p90) AND NOT(session_ldn_ny_overlap <= p90)) OR (NOT(range_close_ratio >= p75) AND range_close_ratio > p10))` |
| 184 | 282 | -0.0951 | 973801 | 0.000e+00 | -55.231 | `((NOT(spread_percentile_100 <= p50) AND NOT(session_ny >= p90)) OR NOT(swing_high_distance_14 >= p25))` |
| 185 | 283 | -0.0909 | 1852377 | 0.000e+00 | -72.612 | `(hour_of_day != p75 OR (prior_session_low_distance > p75 AND (range_close_ratio != p90 OR NOT(session_ny < p10))))` |
| 186 | 284 | -0.0887 | 1603241 | 0.000e+00 | -65.799 | `(((NOT(kijun_26_distance >= p10) OR swing_high_distance_14 > p25) OR (distance_to_round_number != p75 AND NOT(prior_session_high_distance > p25))) OR session_ny >= p75)` |
| 187 | 285 | -0.0857 | 1115505 | 0.000e+00 | -53.045 | `NOT(session_dead > p25)` |
| 188 | 286 | -0.0892 | 1691533 | 0.000e+00 | -67.926 | `(NOT(swing_high_distance_14 <= p25) OR ((NOT(atr_14 > p25) AND NOT(session_tokyo < p10)) OR NOT(spread_vs_trailing_100 != p25)))` |
| 189 | 287 | -0.0923 | 956674 | 0.000e+00 | -52.056 | `(NOT(swing_high_distance_14 == p10) AND range_close_ratio < p50)` |
| 190 | 290 | -0.0977 | 1639821 | 0.000e+00 | -73.928 | `((NOT(atr_vs_trailing_100 < p25) AND NOT(session_dead != p10)) OR (NOT(distance_to_round_number <= p25) OR (NOT(session_tokyo != p25) AND NOT(session_london >= p25))))` |
| 191 | 291 | -0.0914 | 1911836 | 0.000e+00 | -74.143 | `(((NOT(swing_low_distance_14 < p90) AND session_tokyo >= p90) OR NOT(atr_vs_trailing_100 == p25)) OR (NOT(session_tokyo > p90) AND hour_of_day > p90))` |
| 192 | 292 | -0.1052 | 478404 | 0.000e+00 | -43.191 | `swing_high_distance_14 <= p25` |
| 193 | 294 | -0.0973 | 1732969 | 0.000e+00 | -75.570 | `(((session_ldn_ny_overlap < p90 AND atr_14 < p75) OR NOT(session_tokyo != p10)) OR session_ldn_ny_overlap != p10)` |
| 194 | 295 | -0.1025 | 893201 | 0.000e+00 | -57.586 | `(atr_vs_trailing_100 >= p10 AND ((day_of_week != p25 OR (kijun_26_distance == p25 AND NOT(hour_of_day == p90))) AND session_tokyo != p75))` |
| 195 | 296 | -0.0931 | 1354609 | 0.000e+00 | -63.793 | `((((swing_high_distance_14 == p75 OR session_dead == p50) AND session_ny != p50) OR NOT(session_london == p50)) OR session_tokyo <= p10)` |
| 196 | 297 | -0.1196 | 606008 | 0.000e+00 | -55.278 | `(NOT(spread_vs_trailing_100 < p50) AND (kijun_26_distance >= p10 AND ((NOT(day_of_week > p75) AND distance_to_round_number >= p50) OR NOT(prior_session_high_distance >= p50))))` |
| 197 | 298 | -0.0915 | 1881667 | 0.000e+00 | -73.675 | `((spread_percentile_100 != p90 OR NOT(day_of_week == p50)) OR NOT(atr_vs_trailing_100 > p50))` |
| 198 | 300 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((NOT(distance_to_round_number < p50) OR swing_low_distance_14 != p10) OR (session_london == p75 OR distance_to_round_number < p75))` |
| 199 | 305 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((session_dead <= p90 OR (kijun_26_distance == p75 OR atr_14 > p25)) OR NOT(spread_percentile_100 <= p25))` |
| 200 | 307 | -0.0914 | 1912190 | 0.000e+00 | -74.188 | `((NOT(session_ny < p90) AND (NOT(session_dead <= p90) AND distance_to_round_number == p90)) OR range_close_ratio != p90)` |
| 201 | 311 | -0.0902 | 1416859 | 0.000e+00 | -63.060 | `((((NOT(session_london <= p75) AND NOT(kijun_26_distance < p10)) AND prior_session_low_distance <= p25) OR kijun_26_distance > p25) AND spread_percentile_100 != p50)` |
| 202 | 312 | -0.0956 | 1160657 | 0.000e+00 | -60.766 | `((NOT(session_london != p50) AND swing_low_distance_14 == p90) OR (session_tokyo <= p10 AND NOT(prior_session_high_distance == p90)))` |
| 203 | 313 | -0.0914 | 1908219 | 0.000e+00 | -74.097 | `(NOT(kijun_26_distance >= p75) OR ((NOT(session_tokyo < p75) OR atr_percentile_100 != p90) OR (NOT(atr_vs_trailing_100 != p90) OR NOT(swing_low_distance_14 != p90))))` |
| 204 | 314 | -0.0899 | 1752622 | 0.000e+00 | -69.781 | `hour_of_day <= p90` |
| 205 | 315 | -0.0910 | 1000080 | 0.000e+00 | -54.031 | `(atr_vs_trailing_100 >= p50 OR NOT(hour_of_day != p10))` |
| 206 | 318 | -0.1050 | 1410530 | 0.000e+00 | -73.677 | `(NOT(prior_session_low_distance > p75) OR spread_percentile_100 == p10)` |
| 207 | 321 | -0.0905 | 955400 | 0.000e+00 | -52.882 | `range_close_ratio >= p50` |
| 208 | 322 | -0.0890 | 717217 | 0.000e+00 | -44.325 | `NOT(session_ny == p10)` |
| 209 | 323 | -0.0964 | 1689645 | 0.000e+00 | -73.686 | `(NOT(prior_session_low_distance >= p90) AND kijun_26_distance != p25)` |
| 210 | 325 | -0.0874 | 971197 | 0.000e+00 | -50.650 | `(((session_dead >= p90 OR kijun_26_distance > p25) AND (range_close_ratio >= p90 AND NOT(atr_14 > p50))) OR kijun_26_distance >= p50)` |
| 211 | 328 | -0.0914 | 1912188 | 0.000e+00 | -74.189 | `((NOT(prior_session_low_distance <= p10) OR prior_session_high_distance >= p75) OR NOT(kijun_26_distance == p50))` |
| 212 | 330 | -0.0820 | 1014887 | 0.000e+00 | -48.255 | `((kijun_26_distance >= p75 OR (session_dead < p10 AND NOT(spread_vs_trailing_100 != p25))) OR NOT(session_london != p90))` |
| 213 | 334 | -0.0914 | 1912184 | 0.000e+00 | -74.187 | `(((distance_to_round_number != p50 OR hour_of_day > p50) OR NOT(swing_high_distance_14 <= p10)) OR (NOT(day_of_week <= p10) OR session_london < p90))` |
| 214 | 336 | -0.0916 | 1736673 | 0.000e+00 | -70.835 | `(((hour_of_day != p50 OR NOT(session_dead <= p75)) AND NOT(day_of_week >= p75)) OR (NOT(session_ldn_ny_overlap == p90) AND NOT(day_of_week <= p10)))` |
| 215 | 337 | -0.0922 | 1647213 | 0.000e+00 | -69.532 | `(((prior_session_high_distance == p25 OR NOT(spread_percentile_100 <= p10)) AND (swing_low_distance_14 >= p90 OR hour_of_day != p10)) AND NOT(swing_low_distance_14 == p75))` |
| 216 | 340 | -0.0868 | 1434117 | 0.000e+00 | -60.772 | `((range_close_ratio > p50 OR (atr_14 != p90 OR (NOT(day_of_week >= p10) OR swing_low_distance_14 <= p50))) AND NOT(swing_high_distance_14 < p25))` |
| 217 | 341 | -0.0914 | 1909610 | 0.000e+00 | -74.136 | `(((NOT(distance_to_round_number >= p25) OR NOT(atr_vs_trailing_100 != p10)) OR (distance_to_round_number > p25 AND session_ny <= p90)) OR spread_vs_trailing_100 <= p90)` |
| 218 | 343 | -0.0911 | 1812226 | 0.000e+00 | -72.009 | `(NOT(swing_low_distance_14 <= p10) OR (session_tokyo == p10 AND atr_vs_trailing_100 < p75))` |
| 219 | 344 | -0.0999 | 781532 | 0.000e+00 | -51.942 | `(NOT(session_dead == p10) AND (((NOT(session_ny >= p50) OR atr_percentile_100 != p90) AND atr_percentile_100 != p75) OR NOT(range_close_ratio != p90)))` |
| 220 | 345 | -0.0886 | 1408240 | 0.000e+00 | -61.744 | `(prior_session_low_distance >= p25 AND prior_session_low_distance >= p10)` |
| 221 | 347 | -0.1193 | 398412 | 0.000e+00 | -45.023 | `NOT(hour_of_day <= p75)` |
| 222 | 348 | -0.0914 | 1911886 | 0.000e+00 | -74.170 | `((NOT(session_london != p75) AND (NOT(spread_vs_trailing_100 <= p10) AND session_tokyo <= p50)) OR swing_high_distance_14 != p50)` |
| 223 | 349 | -0.0936 | 1591886 | 0.000e+00 | -69.139 | `(((prior_session_low_distance > p50 AND session_london <= p25) AND NOT(spread_vs_trailing_100 >= p25)) OR (NOT(atr_vs_trailing_100 > p75) OR NOT(day_of_week != p75)))` |
| 224 | 351 | -0.0886 | 1722725 | 0.000e+00 | -68.148 | `distance_to_round_number <= p90` |
| 225 | 352 | -0.0914 | 1912107 | 0.000e+00 | -74.183 | `(NOT(hour_of_day >= p50) OR atr_vs_trailing_100 != p75)` |
| 226 | 353 | -0.0746 | 1010908 | 0.000e+00 | -43.656 | `(((hour_of_day > p50 OR session_london <= p50) AND NOT(prior_session_low_distance < p75)) OR NOT(session_london < p75))` |
| 227 | 354 | -0.0938 | 1242550 | 0.000e+00 | -61.716 | `(NOT(session_tokyo != p50) OR atr_vs_trailing_100 > p90)` |
| 228 | 360 | -0.1028 | 1612891 | 0.000e+00 | -77.422 | `(NOT(atr_14 <= p90) OR distance_to_round_number > p25)` |
| 229 | 361 | -0.0955 | 1195268 | 0.000e+00 | -61.621 | `session_tokyo == p25` |
| 230 | 362 | -0.1268 | 383211 | 0.000e+00 | -47.495 | `day_of_week > p75` |
| 231 | 365 | -0.0878 | 1433605 | 0.000e+00 | -61.570 | `atr_14 > p25` |
| 232 | 367 | -0.0867 | 987722 | 0.000e+00 | -50.544 | `((NOT(prior_session_low_distance > p25) OR (NOT(spread_percentile_100 <= p25) OR NOT(atr_percentile_100 > p50))) AND NOT(session_dead == p75))` |
| 233 | 368 | -0.0913 | 1880822 | 0.000e+00 | -73.521 | `((NOT(atr_14 <= p90) OR prior_session_high_distance != p75) OR ((NOT(session_dead < p10) OR day_of_week < p75) AND prior_session_high_distance > p75))` |
| 234 | 369 | -0.0845 | 716923 | 0.000e+00 | -41.649 | `(NOT(session_tokyo < p75) OR session_tokyo != p25)` |
| 235 | 370 | -0.0839 | 1435215 | 0.000e+00 | -58.713 | `(NOT(distance_to_round_number > p90) AND NOT(distance_to_round_number > p75))` |
| 236 | 371 | -0.0914 | 1912090 | 0.000e+00 | -74.195 | `(range_close_ratio == p25 OR (NOT(swing_low_distance_14 == p25) OR spread_vs_trailing_100 <= p25))` |
| 237 | 372 | -0.0928 | 1194974 | 0.000e+00 | -59.505 | `NOT(session_ny != p25)` |
| 238 | 374 | -0.0835 | 1167608 | 0.000e+00 | -52.592 | `(NOT(distance_to_round_number >= p50) OR (NOT(atr_14 <= p50) AND (NOT(swing_low_distance_14 != p90) OR session_tokyo <= p25)))` |
| 239 | 375 | -0.0921 | 955893 | 0.000e+00 | -52.181 | `((atr_vs_trailing_100 <= p50 AND (session_ny >= p90 OR kijun_26_distance != p10)) AND spread_vs_trailing_100 != p50)` |
| 240 | 377 | -0.0861 | 1592455 | 0.000e+00 | -63.863 | `((NOT(spread_vs_trailing_100 != p75) AND NOT(atr_vs_trailing_100 >= p50)) OR spread_percentile_100 != p90)` |
| 241 | 379 | -0.0915 | 1813160 | 0.000e+00 | -72.478 | `((NOT(session_tokyo != p10) OR atr_vs_trailing_100 > p25) OR (NOT(swing_low_distance_14 >= p50) AND NOT(spread_vs_trailing_100 == p75)))` |
| 242 | 381 | -0.0909 | 1884860 | 0.000e+00 | -73.240 | `((NOT(hour_of_day < p25) OR day_of_week > p50) OR ((hour_of_day < p90 AND NOT(session_dead >= p25)) OR swing_high_distance_14 >= p10))` |
| 243 | 384 | -0.0920 | 1886353 | 0.000e+00 | -74.181 | `((kijun_26_distance > p10 OR session_tokyo != p10) OR (spread_percentile_100 >= p90 OR (session_ldn_ny_overlap <= p25 OR NOT(session_ny >= p75))))` |
| 244 | 385 | -0.0928 | 1194940 | 0.000e+00 | -59.499 | `((kijun_26_distance != p75 OR NOT(session_ny <= p50)) AND session_ny < p90)` |
| 245 | 386 | -0.0915 | 1909800 | 0.000e+00 | -74.194 | `(NOT(spread_percentile_100 == p50) OR NOT(swing_low_distance_14 <= p10))` |
| 246 | 387 | -0.0946 | 1721034 | 0.000e+00 | -73.014 | `(swing_high_distance_14 <= p90 OR (NOT(kijun_26_distance < p75) AND (NOT(atr_percentile_100 >= p90) AND NOT(swing_high_distance_14 != p75))))` |
| 247 | 388 | -0.0911 | 1881102 | 0.000e+00 | -73.386 | `((NOT(range_close_ratio <= p25) OR prior_session_high_distance > p10) OR atr_vs_trailing_100 == p50)` |
| 248 | 389 | -0.0981 | 909325 | 0.000e+00 | -55.292 | `((NOT(session_ldn_ny_overlap == p90) OR NOT(prior_session_low_distance < p90)) AND NOT(session_tokyo != p25))` |
| 249 | 391 | -0.0910 | 923062 | 0.000e+00 | -51.335 | `((NOT(day_of_week == p90) OR distance_to_round_number > p25) AND spread_percentile_100 <= p50)` |
| 250 | 393 | -0.0866 | 1614036 | 0.000e+00 | -64.372 | `(NOT(session_ny <= p10) OR NOT(distance_to_round_number > p75))` |
| 251 | 395 | -0.0863 | 1368971 | 0.000e+00 | -58.914 | `(distance_to_round_number < p25 OR NOT(session_ny != p10))` |
| 252 | 396 | -0.0907 | 872863 | 0.000e+00 | -49.840 | `(kijun_26_distance > p10 AND (NOT(spread_percentile_100 > p50) AND (atr_14 > p10 OR NOT(kijun_26_distance == p25))))` |
| 253 | 398 | -0.0963 | 1721161 | 0.000e+00 | -74.293 | `NOT(swing_low_distance_14 > p90)` |
| 254 | 399 | -0.0831 | 1544964 | 0.000e+00 | -60.220 | `(hour_of_day == p10 OR day_of_week < p90)` |
| 255 | 401 | -0.0914 | 1893864 | 0.000e+00 | -73.847 | `((NOT(swing_high_distance_14 != p10) OR NOT(spread_percentile_100 > p75)) OR (NOT(prior_session_high_distance == p75) OR session_ny < p10))` |
| 256 | 403 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((hour_of_day > p50 OR (NOT(kijun_26_distance == p75) OR NOT(session_ny != p75))) OR session_ldn_ny_overlap <= p90)` |
| 257 | 404 | -0.0955 | 1195126 | 0.000e+00 | -61.602 | `((NOT(session_tokyo > p10) OR (NOT(prior_session_high_distance != p25) AND atr_percentile_100 > p75)) AND (distance_to_round_number >= p90 OR atr_vs_trailing_100 != p90))` |
| 258 | 405 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((NOT(spread_vs_trailing_100 >= p90) AND swing_high_distance_14 == p10) OR NOT(session_ny > p75))` |
| 259 | 406 | -0.0899 | 1798757 | 0.000e+00 | -70.770 | `((atr_percentile_100 < p90 AND NOT(range_close_ratio != p25)) OR (session_london <= p10 OR spread_percentile_100 != p90))` |
| 260 | 407 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((session_london >= p75 OR session_london == p50) OR ((NOT(atr_14 == p10) OR NOT(session_ldn_ny_overlap >= p90)) OR NOT(session_london > p75)))` |
| 261 | 408 | -0.0914 | 1912190 | 0.000e+00 | -74.188 | `(prior_session_high_distance < p25 OR (NOT(atr_vs_trailing_100 >= p10) OR NOT(atr_14 == p50)))` |
| 262 | 410 | -0.1102 | 379822 | 0.000e+00 | -40.127 | `(((kijun_26_distance <= p10 OR NOT(swing_high_distance_14 < p50)) AND swing_low_distance_14 <= p25) OR (NOT(swing_low_distance_14 >= p50) AND NOT(spread_vs_trailing_100 <= p75)))` |
| 263 | 412 | -0.0914 | 1912107 | 0.000e+00 | -74.183 | `(session_ny <= p50 OR atr_vs_trailing_100 != p25)` |
| 264 | 413 | -0.0918 | 1832881 | 0.000e+00 | -72.931 | `(NOT(prior_session_low_distance != p90) OR (NOT(day_of_week < p25) OR range_close_ratio <= p75))` |
| 265 | 414 | -0.0879 | 956183 | 0.000e+00 | -50.501 | `NOT(kijun_26_distance < p50)` |
| 266 | 416 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(swing_low_distance_14 != p90 OR spread_vs_trailing_100 != p90)` |
| 267 | 417 | -0.0968 | 1084543 | 0.000e+00 | -59.440 | `(prior_session_high_distance == p10 OR ((swing_low_distance_14 != p75 AND NOT(prior_session_high_distance <= p25)) AND NOT(kijun_26_distance >= p75)))` |
| 268 | 418 | -0.0817 | 716112 | 0.000e+00 | -40.473 | `((NOT(session_dead == p75) AND (NOT(prior_session_low_distance < p50) AND session_ldn_ny_overlap >= p10)) OR (NOT(prior_session_low_distance == p75) AND session_ldn_ny_overlap != p10))` |
| 269 | 421 | -0.0912 | 1832467 | 0.000e+00 | -72.471 | `hour_of_day != p75` |
| 270 | 424 | -0.0803 | 717077 | 0.000e+00 | -39.701 | `session_london == p75` |
| 271 | 425 | -0.0872 | 1496790 | 0.000e+00 | -62.616 | `(NOT(session_ldn_ny_overlap < p90) OR spread_vs_trailing_100 <= p75)` |
| 272 | 427 | -0.1165 | 661657 | 0.000e+00 | -56.736 | `((session_london < p75 OR NOT(kijun_26_distance >= p25)) AND distance_to_round_number > p50)` |
| 273 | 429 | -0.0913 | 1877282 | 0.000e+00 | -73.428 | `NOT(prior_session_low_distance == p90)` |
| 274 | 431 | -0.0920 | 971499 | 0.000e+00 | -53.587 | `((session_ny >= p50 AND NOT(prior_session_low_distance == p50)) AND (spread_percentile_100 < p90 AND NOT(session_tokyo > p25)))` |
| 275 | 432 | -0.0936 | 1443508 | 0.000e+00 | -66.037 | `spread_percentile_100 >= p25` |
| 276 | 434 | -0.0914 | 1909933 | 0.000e+00 | -74.135 | `((atr_vs_trailing_100 < p75 AND session_dead > p50) OR ((NOT(swing_low_distance_14 > p50) OR NOT(atr_vs_trailing_100 <= p75)) OR atr_14 >= p10))` |
| 277 | 435 | -0.0913 | 1908974 | 0.000e+00 | -74.071 | `((NOT(atr_percentile_100 != p50) OR atr_percentile_100 != p10) OR (NOT(session_london >= p25) OR (NOT(atr_vs_trailing_100 > p25) OR NOT(session_london >= p25))))` |
| 278 | 436 | -0.0816 | 955348 | 0.000e+00 | -46.299 | `((NOT(day_of_week >= p90) OR NOT(session_tokyo <= p75)) AND (NOT(session_ny >= p90) OR (NOT(atr_vs_trailing_100 > p90) AND session_tokyo != p25)))` |
| 279 | 438 | -0.0982 | 1329944 | 0.000e+00 | -66.602 | `(NOT(spread_vs_trailing_100 <= p75) OR (((prior_session_low_distance == p10 AND NOT(atr_14 > p75)) OR NOT(session_london > p10)) OR session_dead == p75))` |
| 280 | 439 | -0.0949 | 1267098 | 0.000e+00 | -62.708 | `(kijun_26_distance < p10 OR (NOT(day_of_week <= p90) OR NOT(session_london > p10)))` |
| 281 | 440 | -0.0918 | 1899676 | 0.000e+00 | -74.273 | `(((session_ldn_ny_overlap != p75 AND NOT(atr_vs_trailing_100 < p25)) AND atr_14 != p25) OR spread_percentile_100 != p25)` |
| 282 | 441 | -0.0904 | 1872522 | 0.000e+00 | -72.589 | `((NOT(range_close_ratio <= p25) OR atr_14 > p25) OR ((NOT(atr_14 <= p90) AND NOT(range_close_ratio == p75)) OR day_of_week != p75))` |
| 283 | 442 | -0.0841 | 1445581 | 0.000e+00 | -59.224 | `spread_percentile_100 <= p75` |
| 284 | 443 | -0.0842 | 1392894 | 0.000e+00 | -58.213 | `((NOT(atr_percentile_100 == p10) AND ((prior_session_low_distance != p75 OR day_of_week <= p90) OR NOT(session_tokyo != p75))) AND NOT(spread_percentile_100 >= p75))` |
| 285 | 445 | -0.0982 | 1567568 | 0.000e+00 | -72.551 | `((NOT(hour_of_day < p90) AND NOT(atr_vs_trailing_100 >= p75)) OR NOT(day_of_week == p25))` |
| 286 | 447 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(hour_of_day != p10 OR ((range_close_ratio != p75 OR NOT(spread_percentile_100 != p10)) OR (NOT(distance_to_round_number > p25) OR session_ny > p25)))` |
| 287 | 448 | -0.0986 | 648501 | 0.000e+00 | -47.010 | `((NOT(prior_session_low_distance >= p75) AND NOT(prior_session_high_distance < p90)) OR (session_ny <= p75 AND (atr_14 <= p25 OR NOT(spread_vs_trailing_100 > p10))))` |
| 288 | 449 | -0.0803 | 717077 | 0.000e+00 | -39.701 | `session_london > p25` |
| 289 | 450 | -0.0871 | 1434094 | 0.000e+00 | -61.193 | `NOT(spread_vs_trailing_100 >= p75)` |
| 290 | 453 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(atr_14 == p90) AND range_close_ratio != p75)` |
| 291 | 454 | -0.0935 | 1593402 | 0.000e+00 | -69.299 | `NOT(session_ldn_ny_overlap >= p90)` |
| 292 | 455 | -0.0748 | 1145847 | 0.000e+00 | -46.320 | `day_of_week <= p50` |
| 293 | 456 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `session_tokyo >= p25` |
| 294 | 457 | -0.0974 | 664425 | 0.000e+00 | -46.494 | `((range_close_ratio >= p90 AND NOT(session_london == p25)) OR ((day_of_week < p75 AND NOT(atr_vs_trailing_100 > p10)) OR NOT(spread_percentile_100 < p75)))` |
| 295 | 458 | -0.0955 | 1195268 | 0.000e+00 | -61.621 | `((session_tokyo != p75 OR session_ny < p25) AND session_tokyo < p90)` |
| 296 | 463 | -0.0914 | 1530010 | 0.000e+00 | -66.626 | `(NOT(day_of_week == p50) AND (NOT(session_ny <= p90) OR (session_dead <= p75 OR distance_to_round_number < p90)))` |
| 297 | 466 | -0.0994 | 796686 | 0.000e+00 | -52.182 | `NOT(session_dead <= p25)` |
| 298 | 467 | -0.0872 | 811274 | 0.000e+00 | -46.204 | `(((atr_vs_trailing_100 > p10 AND NOT(range_close_ratio < p25)) AND distance_to_round_number >= p75) OR (NOT(hour_of_day > p10) OR NOT(day_of_week >= p25)))` |
| 299 | 468 | -0.0894 | 1058327 | 0.000e+00 | -53.707 | `(atr_percentile_100 < p25 OR (NOT(spread_percentile_100 == p90) AND (NOT(prior_session_low_distance > p50) AND NOT(session_ny > p75))))` |
| 300 | 470 | -0.1026 | 1533969 | 0.000e+00 | -75.161 | `((session_london >= p25 AND NOT(spread_vs_trailing_100 >= p10)) OR (atr_vs_trailing_100 <= p10 OR NOT(swing_high_distance_14 >= p75)))` |
| 301 | 472 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(swing_high_distance_14 != p75 OR swing_high_distance_14 > p10)` |
| 302 | 474 | -0.0841 | 1433211 | 0.000e+00 | -58.681 | `(day_of_week <= p50 OR session_tokyo > p25)` |
| 303 | 476 | -0.0913 | 1877282 | 0.000e+00 | -73.428 | `NOT(prior_session_low_distance == p90)` |
| 304 | 477 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `((atr_vs_trailing_100 == p25 OR distance_to_round_number >= p90) OR (atr_14 != p90 OR (atr_14 <= p10 OR NOT(session_dead <= p75))))` |
| 305 | 478 | -0.0935 | 864615 | 0.000e+00 | -51.423 | `(prior_session_high_distance > p50 AND (((NOT(atr_14 <= p25) OR swing_low_distance_14 > p25) OR NOT(atr_vs_trailing_100 < p25)) OR NOT(atr_vs_trailing_100 != p10)))` |
| 306 | 480 | -0.0911 | 1910325 | 0.000e+00 | -73.920 | `((day_of_week != p10 OR NOT(range_close_ratio <= p75)) OR NOT(prior_session_high_distance == p90))` |
| 307 | 485 | -0.0914 | 1912165 | 0.000e+00 | -74.193 | `((kijun_26_distance != p25 OR session_dead < p75) OR (swing_low_distance_14 == p50 AND NOT(session_ny >= p25)))` |
| 308 | 486 | -0.0972 | 1434126 | 0.000e+00 | -68.510 | `(NOT(prior_session_high_distance != p25) OR kijun_26_distance < p75)` |
| 309 | 487 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(NOT(session_london < p10) AND (NOT(session_ldn_ny_overlap > p90) AND (session_tokyo < p90 OR NOT(session_ldn_ny_overlap > p10))))` |
| 310 | 488 | -0.0994 | 796686 | 0.000e+00 | -52.182 | `(NOT(session_dead == p25) OR (NOT(hour_of_day >= p10) AND NOT(distance_to_round_number <= p10)))` |
| 311 | 489 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(session_ldn_ny_overlap >= p25 OR ((day_of_week < p25 OR kijun_26_distance == p75) AND session_ny != p25))` |
| 312 | 490 | -0.0918 | 938765 | 0.000e+00 | -52.393 | `(prior_session_high_distance > p50 AND NOT(spread_vs_trailing_100 == p75))` |
| 313 | 492 | -0.0926 | 1751502 | 0.000e+00 | -71.887 | `(NOT(kijun_26_distance <= p25) OR range_close_ratio <= p75)` |
| 314 | 493 | -0.0911 | 1894761 | 0.000e+00 | -73.608 | `(NOT(distance_to_round_number > p50) OR (NOT(prior_session_high_distance == p10) OR NOT(session_dead <= p75)))` |
| 315 | 495 | -0.0976 | 856909 | 0.000e+00 | -54.055 | `((NOT(kijun_26_distance != p50) OR prior_session_low_distance < p90) AND (range_close_ratio >= p50 OR (NOT(session_london == p10) AND NOT(spread_vs_trailing_100 < p75))))` |
| 316 | 496 | -0.0908 | 1790559 | 0.000e+00 | -71.240 | `((((NOT(atr_vs_trailing_100 >= p25) AND NOT(session_ny < p75)) OR session_ldn_ny_overlap < p90) OR NOT(atr_14 <= p50)) OR session_ny != p75)` |
| 317 | 497 | -0.1043 | 478371 | 0.000e+00 | -42.203 | `(NOT(swing_low_distance_14 > p25) AND (swing_low_distance_14 != p90 OR NOT(swing_low_distance_14 <= p75)))` |
| 318 | 498 | -0.0920 | 711276 | 0.000e+00 | -45.553 | `((spread_percentile_100 > p50 OR ((atr_vs_trailing_100 < p10 AND NOT(hour_of_day > p90)) AND NOT(kijun_26_distance > p75))) AND NOT(spread_vs_trailing_100 > p75))` |
| 319 | 499 | -0.0929 | 1193914 | 0.000e+00 | -59.500 | `(session_ny <= p25 AND NOT(distance_to_round_number == p75))` |
| 320 | 500 | -0.1105 | 956460 | 0.000e+00 | -64.381 | `(NOT(distance_to_round_number < p50) OR ((NOT(swing_low_distance_14 >= p90) AND NOT(session_tokyo >= p25)) AND session_tokyo > p75))` |
| 321 | 501 | -0.0931 | 1642929 | 0.000e+00 | -70.326 | `(((NOT(swing_low_distance_14 > p10) AND NOT(session_tokyo >= p50)) OR (prior_session_high_distance < p90 AND session_tokyo <= p25)) OR NOT(atr_percentile_100 < p25))` |
| 322 | 502 | -0.0972 | 1434175 | 0.000e+00 | -68.515 | `kijun_26_distance <= p75` |
| 323 | 503 | -0.0914 | 1912191 | 0.000e+00 | -74.188 | `(session_ldn_ny_overlap > p10 OR ((NOT(session_tokyo < p50) OR NOT(atr_percentile_100 <= p10)) OR (session_dead != p75 AND NOT(session_london > p10))))` |
| 324 | 504 | -0.0879 | 1529058 | 0.000e+00 | -63.512 | `((atr_vs_trailing_100 == p25 AND session_tokyo < p75) OR (NOT(atr_14 == p10) AND NOT(day_of_week == p75)))` |
| 325 | 509 | -0.0949 | 1338386 | 0.000e+00 | -64.470 | `((day_of_week == p50 OR NOT(session_london != p25)) OR ((session_dead > p90 AND session_tokyo <= p90) AND NOT(spread_vs_trailing_100 == p10)))` |
| 326 | 510 | -0.0913 | 1877963 | 0.000e+00 | -73.418 | `(NOT(prior_session_high_distance == p10) OR NOT(session_ldn_ny_overlap <= p50))` |
| 327 | 400 | -0.1279 | 239104 | 2.782e-307 | -37.526 | `(NOT(session_tokyo != p10) AND (session_london <= p25 AND NOT(session_ny > p50)))` |
| 328 | 316 | -0.1279 | 239074 | 4.497e-307 | -37.513 | `(spread_vs_trailing_100 != p75 AND (NOT(session_london == p75) AND hour_of_day >= p90))` |
| 329 | 124 | -0.0887 | 510272 | 3.473e-302 | -37.181 | `((NOT(prior_session_low_distance < p10) AND NOT(session_ny == p10)) AND (atr_percentile_100 <= p90 AND NOT(atr_percentile_100 < p10)))` |
| 330 | 462 | -0.0934 | 478560 | 4.381e-296 | -36.803 | `NOT(range_close_ratio >= p25)` |
| 331 | 299 | -0.1107 | 311401 | 4.162e-291 | -36.503 | `(prior_session_high_distance != p90 AND ((NOT(range_close_ratio == p10) AND swing_low_distance_14 < p50) AND distance_to_round_number > p75))` |
| 332 | 358 | -0.0711 | 768170 | 2.975e-284 | -36.044 | `((hour_of_day == p10 OR kijun_26_distance >= p75) OR (atr_vs_trailing_100 != p75 AND NOT(distance_to_round_number > p25)))` |
| 333 | 380 | -0.0860 | 500178 | 2.094e-283 | -35.998 | `(prior_session_high_distance <= p50 AND NOT(atr_percentile_100 < p50))` |
| 334 | 293 | -0.0851 | 491464 | 7.815e-281 | -35.833 | `((range_close_ratio >= p75 AND (NOT(day_of_week == p10) OR NOT(distance_to_round_number < p10))) OR day_of_week > p90)` |
| 335 | 331 | -0.0841 | 477944 | 1.444e-263 | -34.704 | `atr_vs_trailing_100 >= p75` |
| 336 | 130 | -0.0847 | 468655 | 1.750e-252 | -33.960 | `NOT(spread_percentile_100 >= p25)` |
| 337 | 461 | -0.0831 | 483589 | 1.262e-250 | -33.833 | `(NOT(session_ldn_ny_overlap >= p50) OR NOT(spread_percentile_100 > p25))` |
| 338 | 234 | -0.0965 | 343834 | 2.763e-248 | -33.681 | `(prior_session_low_distance < p10 OR NOT(range_close_ratio < p90))` |
| 339 | 422 | -0.0847 | 477964 | 4.327e-246 | -33.523 | `(NOT(session_london < p50) AND NOT(atr_vs_trailing_100 > p25))` |
| 340 | 110 | -0.0850 | 412297 | 6.550e-223 | -31.892 | `((spread_percentile_100 < p25 AND NOT(prior_session_low_distance == p50)) AND NOT(prior_session_high_distance > p90))` |
| 341 | 78 | -0.1071 | 244469 | 1.228e-211 | -31.078 | `((NOT(range_close_ratio != p75) OR NOT(atr_vs_trailing_100 >= p90)) AND (spread_vs_trailing_100 > p75 AND (session_ldn_ny_overlap >= p25 AND NOT(session_dead <= p25))))` |
| 342 | 225 | -0.1056 | 238336 | 4.598e-209 | -30.887 | `(NOT(distance_to_round_number <= p75) AND ((atr_percentile_100 > p50 AND atr_14 != p90) AND (NOT(spread_vs_trailing_100 == p90) OR prior_session_high_distance <= p50)))` |
| 343 | 376 | -0.0915 | 334831 | 7.269e-206 | -30.638 | `((NOT(atr_percentile_100 >= p50) AND swing_high_distance_14 != p10) AND (NOT(session_dead < p90) AND spread_percentile_100 != p90))` |
| 344 | 112 | -0.1170 | 189466 | 3.500e-201 | -30.300 | `distance_to_round_number > p90` |
| 345 | 119 | -0.1170 | 189466 | 3.500e-201 | -30.300 | `(NOT(distance_to_round_number <= p90) AND swing_high_distance_14 <= p90)` |
| 346 | 274 | -0.0740 | 478065 | 6.273e-195 | -29.798 | `kijun_26_distance >= p75` |
| 347 | 329 | -0.0740 | 478016 | 8.365e-195 | -29.788 | `kijun_26_distance > p75` |
| 348 | 115 | -0.1028 | 239031 | 1.532e-194 | -29.782 | `(((atr_14 != p50 OR NOT(swing_low_distance_14 != p25)) AND NOT(session_ny == p50)) AND NOT(hour_of_day < p75))` |
| 349 | 254 | -0.0795 | 398148 | 6.037e-188 | -29.256 | `((session_london >= p90 AND NOT(atr_vs_trailing_100 == p50)) AND ((session_ldn_ny_overlap <= p75 AND NOT(atr_14 == p25)) AND NOT(atr_vs_trailing_100 == p50)))` |
| 350 | 147 | -0.1679 | 79535 | 1.791e-184 | -29.042 | `(NOT(hour_of_day != p90) AND NOT(session_london >= p75))` |
| 351 | 98 | -0.1249 | 142803 | 2.513e-167 | -27.607 | `(NOT(spread_percentile_100 != p90) AND ((NOT(prior_session_high_distance != p75) OR NOT(atr_vs_trailing_100 != p10)) OR (session_tokyo < p50 OR session_dead != p50)))` |
| 352 | 475 | -0.1194 | 146816 | 1.755e-161 | -27.113 | `(atr_vs_trailing_100 == p50 OR (NOT(prior_session_high_distance <= p50) AND NOT(spread_percentile_100 != p90)))` |
| 353 | 373 | -0.0951 | 231886 | 1.680e-160 | -27.017 | `((swing_low_distance_14 < p90 AND (atr_percentile_100 > p75 OR spread_vs_trailing_100 > p75)) AND NOT(session_tokyo <= p25))` |
| 354 | 278 | -0.0706 | 430189 | 3.149e-159 | -26.898 | `((NOT(hour_of_day < p10) AND prior_session_high_distance != p50) AND NOT(kijun_26_distance < p75))` |
| 355 | 70 | -0.0810 | 318735 | 8.040e-158 | -26.781 | `(swing_high_distance_14 != p25 AND session_ldn_ny_overlap > p75)` |
| 356 | 26 | -0.0810 | 318789 | 8.435e-158 | -26.779 | `(NOT(session_ny <= p10) AND session_london != p25)` |
| 357 | 27 | -0.0810 | 318789 | 8.435e-158 | -26.779 | `(session_london == p90 AND (NOT(session_ny < p75) AND ((NOT(session_dead < p75) AND NOT(session_london >= p75)) OR NOT(spread_percentile_100 > p90))))` |
| 358 | 192 | -0.0810 | 318789 | 8.435e-158 | -26.779 | `session_ldn_ny_overlap != p75` |
| 359 | 357 | -0.0810 | 318789 | 8.435e-158 | -26.779 | `NOT(session_ldn_ny_overlap == p50)` |
| 360 | 460 | -0.0810 | 318789 | 8.435e-158 | -26.779 | `NOT(session_ldn_ny_overlap == p50)` |
| 361 | 113 | -0.0839 | 267370 | 1.829e-146 | -25.788 | `(spread_percentile_100 <= p25 AND range_close_ratio >= p50)` |
| 362 | 342 | -0.1080 | 159665 | 1.262e-144 | -25.634 | `(NOT(hour_of_day <= p90) OR (NOT(prior_session_high_distance != p90) AND NOT(swing_high_distance_14 == p50)))` |
| 363 | 57 | -0.0799 | 286940 | 4.540e-143 | -25.482 | `(((session_ny != p10 OR (NOT(prior_session_low_distance > p50) OR NOT(session_tokyo < p50))) AND NOT(atr_percentile_100 <= p50)) AND session_tokyo > p25)` |
| 364 | 187 | -0.0826 | 278809 | 8.971e-141 | -25.274 | `(NOT(session_london != p75) AND (NOT(hour_of_day >= p50) AND (atr_percentile_100 == p90 OR range_close_ratio <= p75)))` |
| 365 | 85 | -0.0815 | 274315 | 2.053e-137 | -24.966 | `(NOT(session_ldn_ny_overlap < p90) AND (spread_vs_trailing_100 >= p10 OR (session_london <= p10 AND NOT(session_ldn_ny_overlap >= p10))))` |
| 366 | 390 | -0.0679 | 382471 | 2.361e-129 | -24.207 | `((NOT(session_ldn_ny_overlap > p50) AND atr_vs_trailing_100 == p25) OR day_of_week == p25)` |
| 367 | 320 | -0.1232 | 107270 | 2.011e-121 | -23.462 | `((NOT(atr_percentile_100 < p25) AND (atr_percentile_100 < p10 OR NOT(range_close_ratio >= p10))) AND (range_close_ratio == p90 OR swing_high_distance_14 < p75))` |
| 368 | 355 | -0.0881 | 187050 | 3.812e-109 | -22.210 | `spread_percentile_100 < p10` |
| 369 | 301 | -0.1117 | 107836 | 1.771e-104 | -21.731 | `((prior_session_high_distance > p50 OR NOT(session_dead != p90)) AND ((NOT(session_ldn_ny_overlap < p50) OR atr_14 >= p25) AND distance_to_round_number <= p10))` |
| 370 | 430 | -0.1198 | 91214 | 6.383e-102 | -21.461 | `(((NOT(swing_high_distance_14 > p90) OR hour_of_day == p75) OR distance_to_round_number > p50) AND NOT(distance_to_round_number >= p10))` |
| 371 | 36 | -0.0861 | 178762 | 1.886e-100 | -21.290 | `((NOT(session_dead < p10) AND NOT(spread_percentile_100 > p10)) AND (distance_to_round_number < p90 OR atr_vs_trailing_100 < p10))` |
| 372 | 127 | -0.1024 | 119638 | 5.790e-96 | -20.805 | `(distance_to_round_number <= p10 AND NOT(session_ny == p90))` |
| 373 | 176 | -0.0831 | 163216 | 7.316e-91 | -20.227 | `((NOT(atr_percentile_100 <= p90) AND hour_of_day >= p10) AND (spread_vs_trailing_100 > p10 OR NOT(atr_14 < p10)))` |
| 374 | 464 | -0.0556 | 410732 | 2.230e-90 | -20.164 | `(session_ny <= p25 AND (swing_high_distance_14 > p75 OR atr_vs_trailing_100 < p10))` |
| 375 | 138 | -0.0756 | 191051 | 1.265e-87 | -19.853 | `NOT(range_close_ratio < p90)` |
| 376 | 29 | -0.1090 | 92158 | 6.053e-84 | -19.432 | `((distance_to_round_number <= p10 AND kijun_26_distance <= p50) OR (hour_of_day >= p10 AND (NOT(swing_low_distance_14 != p75) AND kijun_26_distance > p90)))` |
| 377 | 338 | -0.0975 | 120350 | 1.323e-82 | -19.268 | `(range_close_ratio <= p10 AND ((NOT(spread_percentile_100 != p10) AND distance_to_round_number == p10) OR NOT(session_tokyo == p50)))` |
| 378 | 212 | -0.0910 | 123016 | 7.361e-80 | -18.937 | `((NOT(atr_vs_trailing_100 < p50) OR NOT(range_close_ratio <= p50)) AND ((spread_percentile_100 > p90 OR atr_vs_trailing_100 <= p25) AND NOT(spread_vs_trailing_100 >= p75)))` |
| 379 | 255 | -0.0892 | 118921 | 2.218e-72 | -18.005 | `(prior_session_high_distance <= p50 AND (spread_percentile_100 <= p25 AND NOT(session_ny > p25)))` |
| 380 | 51 | -0.0703 | 187700 | 2.956e-72 | -17.985 | `NOT(prior_session_high_distance <= p90)` |
| 381 | 363 | -0.0858 | 126601 | 6.837e-71 | -17.813 | `((kijun_26_distance <= p50 AND session_london >= p10) AND ((NOT(atr_percentile_100 == p25) AND day_of_week >= p25) AND NOT(session_ldn_ny_overlap == p25)))` |
| 382 | 160 | -0.0624 | 241314 | 3.752e-69 | -17.582 | `(kijun_26_distance < p10 OR swing_high_distance_14 >= p90)` |
| 383 | 289 | -0.0443 | 480663 | 8.167e-68 | -17.403 | `(swing_low_distance_14 > p75 OR atr_14 > p90)` |
| 384 | 506 | -0.0684 | 189358 | 9.126e-65 | -17.000 | `(((NOT(atr_percentile_100 == p10) AND NOT(session_london < p25)) AND NOT(kijun_26_distance > p10)) AND NOT(session_dead < p10))` |
| 385 | 219 | -0.0430 | 464386 | 1.354e-61 | -16.563 | `(swing_high_distance_14 >= p75 AND (atr_14 >= p75 AND (distance_to_round_number <= p10 OR day_of_week >= p10)))` |
| 386 | 220 | -0.0954 | 84746 | 1.831e-61 | -16.555 | `(NOT(session_ldn_ny_overlap != p75) AND (swing_high_distance_14 <= p25 AND NOT(spread_percentile_100 >= p25)))` |
| 387 | 335 | -0.0521 | 300029 | 2.171e-59 | -16.255 | `((swing_high_distance_14 > p75 OR NOT(atr_vs_trailing_100 != p75)) AND session_tokyo <= p25)` |
| 388 | 61 | -0.0714 | 146732 | 1.170e-58 | -16.155 | `(NOT(spread_vs_trailing_100 > p10) AND NOT(session_ldn_ny_overlap == p90))` |
| 389 | 226 | -0.0958 | 79724 | 9.263e-57 | -15.889 | `hour_of_day == p75` |
| 390 | 123 | -0.0624 | 191157 | 1.109e-55 | -15.725 | `((NOT(swing_high_distance_14 < p50) AND NOT(swing_high_distance_14 <= p90)) AND (atr_14 >= p75 AND NOT(swing_high_distance_14 <= p90)))` |
| 391 | 235 | -0.0623 | 191216 | 1.452e-55 | -15.708 | `NOT(swing_high_distance_14 < p90)` |
| 392 | 324 | -0.0752 | 126805 | 4.815e-54 | -15.486 | `(spread_percentile_100 != p90 AND (((session_dead >= p90 OR session_ldn_ny_overlap < p90) OR NOT(spread_percentile_100 < p90)) AND NOT(kijun_26_distance > p10)))` |
| 393 | 428 | -0.1130 | 50281 | 1.150e-49 | -14.832 | `((NOT(session_ldn_ny_overlap == p75) AND ((session_dead != p75 OR session_london != p75) OR NOT(atr_vs_trailing_100 > p90))) AND spread_percentile_100 >= p90)` |
| 394 | 327 | -0.0372 | 458168 | 6.143e-46 | -14.230 | `(atr_14 >= p75 AND NOT(hour_of_day == p90))` |
| 395 | 482 | -0.0706 | 116630 | 5.757e-44 | -13.913 | `(NOT(atr_percentile_100 > p25) AND NOT(session_ny == p50))` |
| 396 | 507 | -0.0878 | 60206 | 7.068e-39 | -13.051 | `(NOT(atr_percentile_100 <= p90) AND session_dead >= p90)` |
| 397 | 167 | -0.0470 | 206708 | 9.355e-35 | -12.300 | `(NOT(distance_to_round_number != p25) OR swing_low_distance_14 > p90)` |
| 398 | 302 | -0.0535 | 154392 | 1.730e-31 | -11.677 | `(NOT(atr_vs_trailing_100 > p10) AND (range_close_ratio > p50 OR NOT(day_of_week >= p75)))` |
| 399 | 158 | -0.0453 | 187664 | 8.079e-30 | -11.344 | `(NOT(prior_session_low_distance < p90) AND (swing_low_distance_14 != p75 OR NOT(day_of_week == p10)))` |
| 400 | 55 | -0.0551 | 131808 | 1.779e-29 | -11.276 | `((NOT(day_of_week != p25) AND NOT(swing_high_distance_14 >= p90)) AND session_tokyo != p50)` |
| 401 | 146 | -0.0508 | 119253 | 4.380e-23 | -9.897 | `(range_close_ratio < p50 AND ((NOT(atr_vs_trailing_100 <= p90) OR session_london == p90) AND swing_high_distance_14 > p50))` |
| 402 | 433 | -0.1096 | 19999 | 3.242e-20 | -9.221 | `(NOT(spread_percentile_100 > p75) AND ((NOT(spread_percentile_100 > p25) AND NOT(atr_vs_trailing_100 < p50)) AND spread_vs_trailing_100 >= p90))` |
| 403 | 103 | -0.0883 | 28211 | 3.001e-18 | -8.717 | `(NOT(distance_to_round_number > p75) AND (NOT(swing_high_distance_14 >= p25) AND NOT(spread_vs_trailing_100 <= p90)))` |
| 404 | 77 | -0.0651 | 45066 | 7.892e-17 | -8.336 | `((NOT(session_ldn_ny_overlap >= p50) OR NOT(session_dead != p90)) AND (session_dead > p90 OR range_close_ratio >= p90))` |
| 405 | 250 | -0.1482 | 8535 | 1.068e-16 | -8.314 | `((distance_to_round_number != p50 OR session_tokyo <= p50) AND (NOT(hour_of_day != p90) AND NOT(prior_session_low_distance < p90)))` |
| 406 | 392 | -0.0324 | 191135 | 6.784e-16 | -8.075 | `NOT(kijun_26_distance < p90)` |
| 407 | 218 | -0.0891 | 22556 | 5.420e-15 | -7.822 | `NOT(atr_percentile_100 != p10)` |
| 408 | 151 | -0.0750 | 34584 | 1.248e-14 | -7.714 | `((NOT(range_close_ratio > p10) AND session_dead != p10) AND (NOT(spread_vs_trailing_100 < p90) OR NOT(swing_high_distance_14 <= p75)))` |
| 409 | 201 | -0.0805 | 24031 | 2.429e-13 | -7.327 | `((swing_high_distance_14 > p75 AND session_ny == p75) AND NOT(spread_percentile_100 >= p10))` |
| 410 | 59 | -0.1045 | 12310 | 8.281e-12 | -6.840 | `(NOT(swing_high_distance_14 < p90) AND prior_session_high_distance >= p25)` |
| 411 | 46 | -0.0680 | 18207 | 5.366e-08 | -5.441 | `(NOT(prior_session_high_distance <= p10) AND NOT(spread_percentile_100 != p75))` |
| 412 | 252 | -0.1386 | 3593 | 7.732e-07 | -4.951 | `(((atr_percentile_100 == p50 OR (NOT(distance_to_round_number >= p75) AND NOT(swing_low_distance_14 != p50))) AND session_ldn_ny_overlap == p10) AND NOT(spread_vs_trailing_100 < p75))` |
| 413 | 69 | -0.0429 | 30330 | 2.362e-05 | -4.228 | `(NOT(prior_session_high_distance <= p90) AND (range_close_ratio < p25 OR NOT(range_close_ratio >= p10)))` |
| 414 | 446 | -0.0583 | 16080 | 3.626e-05 | -4.131 | `(swing_high_distance_14 >= p25 AND (atr_percentile_100 <= p75 AND day_of_week > p90))` |
| 415 | 72 | +0.0209 | 119885 | 6.608e-05 | 3.990 | `((NOT(prior_session_high_distance >= p25) OR NOT(day_of_week == p90)) AND (NOT(distance_to_round_number < p10) AND NOT(kijun_26_distance < p90)))` |
| 416 | 60 | -0.2132 | 764 | 1.102e-04 | -3.887 | `(distance_to_round_number == p50 AND (NOT(atr_14 < p10) OR NOT(prior_session_high_distance == p50)))` |
