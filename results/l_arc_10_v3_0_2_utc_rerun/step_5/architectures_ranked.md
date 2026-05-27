# Arc 10 v3.0 — Step 5 Architectures Ranked

- Search window: 2010-01-01 → 2020-12-31 (11-fold anchored)
- Holdout: 2021-01-01 → 2026-04-30 (one-shot per top-3)
- Configs evaluated: **48**  (thin: <50, normal: 50-100, broad: >100)
- Selection-bias flag: **thin**

## All configurations (ranked by worst-fold ROI/DD ratio)

| cluster_id | archetype | architecture | sl_multiplier | exit_policy | exposure | search_worst_ratio | search_worst_roi | search_worst_dd | search_mean_roi | search_sign_consistency | search_n_total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v_shape_recovery | A1 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 5.4185 | 0.2649 | 0.0922 | 0.4991 | 11 | 2162 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | 2.0906 | 0.1355 | 0.1009 | 0.3826 | 11 | 2162 |
| 1 | v_shape_recovery | A1 | 3.5000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 2.0386 | 0.1336 | 0.0857 | 0.2270 | 11 | 1080 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 1.3966 | 0.0293 | 0.0503 | 0.1424 | 10 | 644 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 1.3966 | 0.0293 | 0.0503 | 0.2362 | 10 | 984 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 1.3639 | 0.0271 | 0.0520 | 0.1145 | 10 | 644 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | 1.3639 | 0.0271 | 0.0558 | 0.1872 | 10 | 984 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 0.8909 | 0.0495 | 0.1016 | 0.1753 | 11 | 1080 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | unlimited | -0.9210 | -0.2349 | 0.2563 | -0.0482 | 3 | 984 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_2r | unlimited | -0.9594 | -0.2840 | 0.2960 | -0.0323 | 4 | 2162 |
| 1 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_3r | unlimited | -0.9849 | -0.3475 | 0.3528 | -0.0568 | 3 | 2162 |
| 1 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_2r | unlimited | -0.9880 | -0.3022 | 0.3059 | -0.0194 | 6 | 2162 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | max_per_currency_2 | -0.9947 | -0.0771 | 0.0913 | -0.0083 | 3 | 644 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | max_per_currency_2 | -1.0014 | -0.1228 | 0.1227 | -0.0113 | 4 | 644 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | max_per_currency_2 | -1.0126 | -0.2186 | 0.2158 | -0.0560 | 2 | 644 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | max_per_currency_2 | -1.0128 | -0.1954 | 0.1929 | -0.0377 | 3 | 644 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | unlimited | -1.0209 | -0.2877 | 0.3002 | -0.0787 | 3 | 984 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | unlimited | -1.0326 | -0.1925 | 0.2172 | -0.0199 | 3 | 984 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | unlimited | -1.0349 | -0.1202 | 0.1722 | -0.0126 | 3 | 984 |
| 1 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_3r | max_per_currency_2 | -1.0365 | -0.1954 | 0.1929 | -0.0357 | 4 | 1080 |
| 1 | v_shape_recovery | A1 | 3.5000 | sl_only | unlimited | -1.0491 | -0.5136 | 0.4896 | -0.2387 | 2 | 2162 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_only | unlimited | -1.0492 | -0.5125 | 0.4885 | -0.2687 | 1 | 2162 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_3r | unlimited | -1.0557 | -0.3585 | 0.3396 | -0.0993 | 3 | 2162 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_only | max_per_currency_2 | -1.0714 | -0.3017 | 0.2979 | -0.1475 | 1 | 1080 |
| 1 | v_shape_recovery | A1 | 3.5000 | sl_only | max_per_currency_2 | -1.0717 | -0.2973 | 0.2950 | -0.1304 | 1 | 1080 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_2r | max_per_currency_2 | -1.0801 | -0.1403 | 0.1469 | -0.0205 | 5 | 1080 |
| 1 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_2r | max_per_currency_2 | -1.0818 | -0.1420 | 0.1313 | -0.0079 | 5 | 1080 |
| 1 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_3r | max_per_currency_2 | -1.0883 | -0.2186 | 0.2158 | -0.0595 | 3 | 1080 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_only | max_per_currency_2 | -1.1883 | -0.3017 | 0.2979 | -0.1123 | 2 | 644 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_only | unlimited | -1.1883 | -0.4049 | 0.4146 | -0.1525 | 2 | 984 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_only | max_per_currency_2 | -1.1934 | -0.2973 | 0.2950 | -0.1032 | 3 | 644 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_only | unlimited | -1.1934 | -0.3942 | 0.4087 | -0.1381 | 3 | 984 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | unlimited | -1.8293 | -0.2877 | 0.3002 | -0.0696 | 3 | 1045 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | max_per_currency_2 | -1.8293 | -0.2186 | 0.2158 | -0.0664 | 2 | 689 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_only | unlimited | -1.8293 | -0.3942 | 0.4087 | -0.1502 | 1 | 1045 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | max_per_currency_2 | -1.8293 | -0.1954 | 0.2098 | -0.0519 | 2 | 689 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_only | max_per_currency_2 | -1.8293 | -0.2973 | 0.2950 | -0.1130 | 1 | 689 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | unlimited | -1.8293 | -0.2349 | 0.2563 | -0.0505 | 2 | 1045 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_only | max_per_currency_2 | -1.8293 | -0.3017 | 0.2979 | -0.1237 | 1 | 689 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_only | unlimited | -1.8293 | -0.4049 | 0.4146 | -0.1656 | 0 | 1045 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | -2.0083 | -0.0147 | 0.0428 | 0.1536 | 9 | 689 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | -2.0083 | -0.0147 | 0.0539 | 0.2485 | 9 | 1045 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | unlimited | -2.0971 | -0.1056 | 0.1803 | -0.0195 | 3 | 1045 |
| 1 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | max_per_currency_2 | -2.0971 | -0.0744 | 0.1401 | -0.0257 | 3 | 689 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | max_per_currency_2 | -2.0971 | -0.1228 | 0.1535 | -0.0318 | 4 | 689 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | unlimited | -2.0971 | -0.1925 | 0.2172 | -0.0281 | 3 | 1045 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | -2.2744 | -0.0181 | 0.0521 | 0.1241 | 9 | 689 |
| 1 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | -2.2744 | -0.0181 | 0.0551 | 0.1991 | 9 | 1045 |

## Oracle WFO (per cluster — upper bound)

| cluster_id | archetype | sl_multiplier | exit_policy | search_worst_ratio | search_worst_roi | search_worst_dd | search_mean_roi | search_sign_consistency | search_n_total |
|---|---|---|---|---|---|---|---|---|---|
| 1 | v_shape_recovery | 4.0000 | sl_only | -1.1896 | -0.1174 | 0.1577 | 0.1409 | 8 | 994 |

## Top-3 search candidates with holdout

| architecture | sl_multiplier | exit_policy | exposure | search_worst_ratio | search_worst_roi | search_worst_dd | holdout_worst_ratio | holdout_worst_roi | holdout_worst_dd | verdict_search |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 5.4185 | 0.2649 | 0.0922 | 11.7326 | 0.5907 | 0.0503 | PASS-VIABLE |
| A1 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | 2.0906 | 0.1355 | 0.1009 | 7.8806 | 0.4668 | 0.0592 | FAIL |
| A1 | 3.5000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 2.0386 | 0.1336 | 0.0857 | 6.1411 | 0.2355 | 0.0384 | PASS-VIABLE |
