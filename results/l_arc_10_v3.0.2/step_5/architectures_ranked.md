# Arc 10 v3.0 — Step 5 Architectures Ranked

- Search window: 2010-01-01 → 2020-12-31 (11-fold anchored)
- Holdout: 2021-01-01 → 2026-04-30 (one-shot per top-3)
- Configs evaluated: **48**  (thin: <50, normal: 50-100, broad: >100)
- Selection-bias flag: **thin**

## All configurations (ranked by worst-fold ROI/DD ratio)

| cluster_id | archetype | architecture | sl_multiplier | exit_policy | exposure | search_worst_ratio | search_worst_roi | search_worst_dd | search_mean_roi | search_sign_consistency | search_n_total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | v_shape_recovery | A1 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 6.4273 | 0.2246 | 0.0735 | 0.4988 | 11 | 2059 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 2.7299 | 0.0506 | 0.0494 | 0.2127 | 8 | 778 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | 2.1544 | 0.1071 | 0.0853 | 0.3837 | 11 | 2059 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 1.9959 | 0.0506 | 0.0446 | 0.1310 | 8 | 535 |
| 0 | v_shape_recovery | A1 | 3.5000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 1.9261 | 0.0835 | 0.0434 | 0.2355 | 11 | 1054 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | 0.7835 | 0.0372 | 0.0475 | 0.1528 | 8 | 778 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 0.7032 | 0.0312 | 0.0467 | 0.1823 | 11 | 1054 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 0.6843 | 0.0292 | 0.0427 | 0.0968 | 8 | 535 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | 0.0000 | -0.0199 | 0.0532 | 0.2034 | 9 | 1066 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 0.0000 | -0.0199 | 0.0322 | 0.1031 | 9 | 600 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 0.0000 | -0.0199 | 0.0403 | 0.2609 | 9 | 1066 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_partial_close_1r_runner_trail | max_per_currency_2 | 0.0000 | -0.0199 | 0.0296 | 0.1365 | 9 | 600 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_2r | max_per_currency_2 | -0.9047 | -0.1381 | 0.1579 | -0.0167 | 4 | 1054 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | max_per_currency_2 | -0.9047 | -0.1383 | 0.1589 | -0.0304 | 2 | 600 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | max_per_currency_2 | -0.9605 | -0.1334 | 0.1698 | -0.0426 | 2 | 600 |
| 0 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_3r | unlimited | -0.9677 | -0.2379 | 0.2649 | -0.0552 | 3 | 2059 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | unlimited | -0.9677 | -0.2256 | 0.2331 | -0.0653 | 3 | 1066 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | max_per_currency_2 | -0.9700 | -0.1620 | 0.1847 | -0.0331 | 3 | 600 |
| 0 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_2r | max_per_currency_2 | -0.9842 | -0.1603 | 0.1682 | -0.0143 | 5 | 1054 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | unlimited | -0.9893 | -0.1446 | 0.1670 | -0.0249 | 4 | 1066 |
| 0 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_2r | unlimited | -0.9893 | -0.1794 | 0.2321 | -0.0100 | 5 | 2059 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_2r | unlimited | -0.9968 | -0.1909 | 0.1981 | -0.0151 | 5 | 2059 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | unlimited | -0.9968 | -0.1909 | 0.1943 | -0.0353 | 3 | 1066 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | unlimited | -0.9984 | -0.1664 | 0.1667 | -0.0564 | 2 | 778 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | unlimited | -1.0037 | -0.2473 | 0.2464 | -0.0840 | 3 | 778 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_3r | unlimited | -1.0124 | -0.2524 | 0.2556 | -0.0731 | 3 | 2059 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_only | unlimited | -1.0259 | -0.4055 | 0.4012 | -0.1367 | 2 | 778 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_only | unlimited | -1.0374 | -0.4119 | 0.4063 | -0.1535 | 2 | 778 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_only | max_per_currency_2 | -1.0406 | -0.2310 | 0.2220 | -0.1090 | 1 | 535 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_only | max_per_currency_2 | -1.0416 | -0.2233 | 0.2150 | -0.0968 | 1 | 535 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | max_per_currency_2 | -1.0425 | -0.1809 | 0.1736 | -0.0653 | 2 | 535 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | max_per_currency_2 | -1.0480 | -0.1853 | 0.1769 | -0.0589 | 2 | 535 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_3r | unlimited | -1.0501 | -0.2059 | 0.2058 | -0.0685 | 3 | 778 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | unlimited | -1.0517 | -0.1629 | 0.1549 | -0.0302 | 2 | 778 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_2r | max_per_currency_2 | -1.0753 | -0.1699 | 0.1701 | -0.0540 | 2 | 535 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_only | unlimited | -1.0788 | -0.4545 | 0.4213 | -0.2624 | 1 | 2059 |
| 0 | v_shape_recovery | A1 | 3.5000 | sl_only | unlimited | -1.0788 | -0.4540 | 0.4208 | -0.2367 | 2 | 2059 |
| 0 | v_shape_recovery | A1 | 3.5000 | sl_plus_tp_3r | max_per_currency_2 | -1.0875 | -0.1748 | 0.1812 | -0.0331 | 3 | 1054 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_plus_tp_2r | max_per_currency_2 | -1.0913 | -0.1664 | 0.1583 | -0.0288 | 3 | 535 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_only | max_per_currency_2 | -1.1099 | -0.2334 | 0.2253 | -0.1468 | 2 | 1054 |
| 0 | v_shape_recovery | A1 | 3.5000 | sl_only | max_per_currency_2 | -1.1113 | -0.2329 | 0.2248 | -0.1329 | 2 | 1054 |
| 0 | v_shape_recovery | A1 | 4.0000 | sl_plus_tp_3r | max_per_currency_2 | -1.1254 | -0.1646 | 0.1525 | -0.0458 | 3 | 1054 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_only | unlimited | -3.1790 | -0.4119 | 0.4063 | -0.1901 | 0 | 1066 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_only | max_per_currency_2 | -3.1790 | -0.2647 | 0.2563 | -0.1268 | 0 | 600 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | max_per_currency_2 | -3.1790 | -0.1278 | 0.1487 | -0.0536 | 1 | 600 |
| 0 | v_shape_recovery | A3 | 4.0000 | sl_plus_tp_3r | unlimited | -3.1790 | -0.2216 | 0.2556 | -0.0762 | 2 | 1066 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_only | max_per_currency_2 | -3.2031 | -0.2672 | 0.2588 | -0.1216 | 0 | 600 |
| 0 | v_shape_recovery | A3 | 3.5000 | sl_only | unlimited | -3.2031 | -0.4055 | 0.4012 | -0.1798 | 0 | 1066 |

## Oracle WFO (per cluster — upper bound)

| cluster_id | archetype | sl_multiplier | exit_policy | search_worst_ratio | search_worst_roi | search_worst_dd | search_mean_roi | search_sign_consistency | search_n_total |
|---|---|---|---|---|---|---|---|---|---|
| 0 | v_shape_recovery | 4.0000 | sl_only | -1.0833 | -0.4425 | 0.4084 | -0.3495 | 0 | 1097 |

## Top-3 search candidates with holdout

| architecture | sl_multiplier | exit_policy | exposure | search_worst_ratio | search_worst_roi | search_worst_dd | holdout_worst_ratio | holdout_worst_roi | holdout_worst_dd | verdict_search |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 6.4273 | 0.2246 | 0.0735 | 9.6073 | 0.5283 | 0.0550 | PASS-DEPLOYABLE |
| A3 | 3.5000 | sl_partial_close_1r_runner_trail | unlimited | 2.7299 | 0.0506 | 0.0494 | 7.9505 | 0.2798 | 0.0352 | PASS-DEPLOYABLE |
| A1 | 4.0000 | sl_partial_close_1r_runner_trail | unlimited | 2.1544 | 0.1071 | 0.0853 | 4.4085 | 0.4005 | 0.0908 | PASS-VIABLE |
