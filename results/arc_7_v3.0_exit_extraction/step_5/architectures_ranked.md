# Arc 7 v3.0 exit-extraction - Step 5 ranked architectures

> Reset-floor search engine (run_arc_7 path) + Amendment-3 gate. r_base 0.5%; DD@0.40% = DD_base x 0.8 (reset-floor linear). EET; v3 cost primitives (real HistData bid+ask spreads); SL in {2.5,3.0,3.5}xATR; per-currency exposure in {2,unlimited}. Step 5 re-simulates each config vs live H4 panels (SL-honest live engine, NOT the simulate_path replay).

## Stage A - triage (all 48 configs, worst-of-3 over F1 2010 / F6 2015 / F8 2017)

Ranked by worst-of-3 ROI/DD ratio. screened_out = trailing DD@0.40% > 8% on a triage fold (or no trades). 44/48 screened out.

| config_id | worst_roi_pct | worst_dd_base_pct | worst_dd_at_0p40_pct | worst_ratio | min_trades | screened_out |
|---|---|---|---|---|---|---|
| A3::cl0::sl_partial_close_1r_runner_trail::sl3.5::n5::exp2 | -0.0323 | 0.0772 | 0.0618 | -0.419 | 86 | False |
| A3::cl0::sl_plus_tp_3r::sl3.5::n5::exp2 | -0.0466 | 0.0915 | 0.0732 | -0.509 | 46 | False |
| A3::cl0::sl_partial_close_1r_runner_trail::sl3.0::n5::exp2 | -0.0549 | 0.0823 | 0.0658 | -0.667 | 93 | False |
| A3::cl0::sl_partial_close_1r_runner_trail::sl2.5::n5::exp2 | -0.0683 | 0.0976 | 0.0781 | -0.699 | 98 | False |
| A1::full::sl_plus_tp_3r::sl2.5::expinf | -0.1252 | 0.1992 | 0.1593 | -0.628 | 197 | True |
| A1::full::sl_partial_close_1r_runner_trail::sl2.5::expinf | -0.0917 | 0.1591 | 0.1273 | -0.711 | 365 | True |
| A3::cl0::sl_plus_tp_3r::sl2.5::n5::exp2 | -0.0872 | 0.1212 | 0.0970 | -0.719 | 53 | True |
| A3::cl0::sl_plus_tp_3r::sl3.0::n5::exp2 | -0.0792 | 0.1100 | 0.0880 | -0.720 | 51 | True |
| A3::cl0::sl_plus_tp_2r::sl3.5::n5::exp2 | -0.0738 | 0.1004 | 0.0803 | -0.735 | 51 | True |
| A3::cl0::sl_plus_tp_2r::sl3.0::n5::exp2 | -0.0882 | 0.1163 | 0.0931 | -0.758 | 56 | True |
| A3::cl0::sl_plus_tp_2r::sl2.5::n5::exp2 | -0.0848 | 0.1106 | 0.0885 | -0.767 | 62 | True |
| A3::cl0::sl_partial_close_1r_runner_trail::sl2.5::n5::expinf | -0.0987 | 0.1249 | 0.0999 | -0.790 | 105 | True |
| A1::full::sl_only::sl3.0::exp2 | -0.1933 | 0.2753 | 0.2202 | -0.800 | 18 | True |
| A3::cl0::sl_plus_tp_2r::sl2.5::n5::expinf | -0.1078 | 0.1328 | 0.1062 | -0.812 | 67 | True |
| A1::full::sl_partial_close_1r_runner_trail::sl3.5::expinf | -0.1321 | 0.1894 | 0.1515 | -0.815 | 301 | True |
| A3::cl0::sl_plus_tp_3r::sl3.5::n5::expinf | -0.0973 | 0.1188 | 0.0950 | -0.820 | 57 | True |
| A1::full::sl_partial_close_1r_runner_trail::sl3.0::expinf | -0.1540 | 0.1878 | 0.1502 | -0.820 | 336 | True |
| A1::full::sl_only::sl2.5::exp2 | -0.1913 | 0.3037 | 0.2429 | -0.825 | 25 | True |
| A3::cl0::sl_plus_tp_2r::sl3.0::n5::expinf | -0.1253 | 0.1518 | 0.1214 | -0.826 | 64 | True |
| A1::full::sl_plus_tp_2r::sl3.5::expinf | -0.1441 | 0.1906 | 0.1524 | -0.832 | 181 | True |
| A3::cl0::sl_partial_close_1r_runner_trail::sl3.5::n5::expinf | -0.0844 | 0.1002 | 0.0801 | -0.843 | 99 | True |
| A1::full::sl_plus_tp_3r::sl2.5::exp2 | -0.1069 | 0.1263 | 0.1011 | -0.846 | 117 | True |
| A3::cl0::sl_partial_close_1r_runner_trail::sl3.0::n5::expinf | -0.0992 | 0.1170 | 0.0936 | -0.847 | 101 | True |
| A1::full::sl_plus_tp_2r::sl2.5::expinf | -0.1871 | 0.2208 | 0.1766 | -0.848 | 217 | True |
| A1::full::sl_partial_close_1r_runner_trail::sl2.5::exp2 | -0.1238 | 0.1456 | 0.1165 | -0.850 | 266 | True |
| A3::cl0::sl_plus_tp_3r::sl2.5::n5::expinf | -0.1273 | 0.1475 | 0.1180 | -0.863 | 62 | True |
| A1::full::sl_plus_tp_2r::sl2.5::exp2 | -0.1445 | 0.1672 | 0.1337 | -0.865 | 140 | True |
| A1::full::sl_partial_close_1r_runner_trail::sl3.0::exp2 | -0.1161 | 0.1333 | 0.1067 | -0.871 | 217 | True |
| A1::full::sl_partial_close_1r_runner_trail::sl3.5::exp2 | -0.1184 | 0.1352 | 0.1082 | -0.875 | 159 | True |
| A1::full::sl_plus_tp_3r::sl3.0::expinf | -0.2015 | 0.2329 | 0.1863 | -0.884 | 182 | True |
| A1::full::sl_plus_tp_2r::sl3.0::expinf | -0.1927 | 0.2174 | 0.1740 | -0.886 | 197 | True |
| A3::cl0::sl_plus_tp_3r::sl3.0::n5::expinf | -0.1464 | 0.1590 | 0.1272 | -0.921 | 60 | True |
| A1::full::sl_plus_tp_2r::sl3.5::exp2 | -0.1235 | 0.1352 | 0.1082 | -0.924 | 99 | True |
| A1::full::sl_plus_tp_3r::sl3.0::exp2 | -0.1906 | 0.2058 | 0.1647 | -0.926 | 97 | True |
| A1::full::sl_plus_tp_2r::sl3.0::exp2 | -0.1741 | 0.1879 | 0.1503 | -0.927 | 115 | True |
| A3::cl0::sl_plus_tp_2r::sl3.5::n5::expinf | -0.1367 | 0.1388 | 0.1110 | -0.985 | 59 | True |
| A1::full::sl_plus_tp_3r::sl3.5::expinf | -0.1998 | 0.2134 | 0.1707 | -0.985 | 159 | True |
| A3::cl0::sl_only::sl3.5::n5::exp2 | -0.1789 | 0.1813 | 0.1450 | -0.987 | 21 | True |
| A3::cl0::sl_only::sl3.0::n5::expinf | -0.3215 | 0.3245 | 0.2596 | -0.991 | 42 | True |
| A3::cl0::sl_only::sl3.5::n5::expinf | -0.2902 | 0.2922 | 0.2338 | -0.993 | 39 | True |
| A3::cl0::sl_only::sl3.0::n5::exp2 | -0.1869 | 0.1881 | 0.1505 | -0.994 | 25 | True |
| A1::full::sl_only::sl3.0::expinf | -0.4638 | 0.5174 | 0.4139 | -0.997 | 87 | True |
| A1::full::sl_only::sl3.5::expinf | -0.4535 | 0.4649 | 0.3720 | -0.998 | 75 | True |
| A1::full::sl_only::sl2.5::expinf | -0.4766 | 0.5350 | 0.4280 | -0.998 | 107 | True |
| A1::full::sl_plus_tp_3r::sl3.5::exp2 | -0.1578 | 0.1805 | 0.1444 | -1.000 | 76 | True |
| A3::cl0::sl_only::sl2.5::n5::exp2 | -0.2126 | 0.2126 | 0.1701 | -1.000 | 36 | True |
| A3::cl0::sl_only::sl2.5::n5::expinf | -0.3256 | 0.3256 | 0.2605 | -1.000 | 47 | True |
| A1::full::sl_only::sl3.5::exp2 | -0.2180 | 0.2552 | 0.2041 | -1.000 | 15 | True |

## Stage B - gate (full 11-fold IS + 2021 holdout)

### Triage-selected finalists (top-3 by Stage-A rank - all A3 cl0)

| config_id | verdict | worst_fold_ratio | worst_fold_roi_base_pct | worst_fold_dd_base_pct | worst_fold_dd_at_0p40_pct | chained_max_dd_base_pct | n_negative_folds | min_trades_per_fold | holdout_roi_pct | holdout_dd_base_pct | r_safe_pct | scalable_to_safe | primary_failure_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3::cl0::sl_partial_close_1r_runner_trail::sl3.5::n5::exp2 | fail | -0.6144 | -0.0393 | 0.0789 | 0.0631 | 0.2849 | 5 | 0 | -0.1257 | 0.1823 | 0.0051 | True | step5_chained_dd_above_gate |
| A3::cl0::sl_plus_tp_3r::sl3.5::n5::exp2 | fail | -0.6345 | -0.0725 | 0.1334 | 0.1067 | 0.3850 | 7 | 0 | -0.1032 | 0.2049 | 0.0030 | True | step5_chained_dd_above_gate |
| A3::cl0::sl_partial_close_1r_runner_trail::sl3.0::n5::exp2 | fail | -0.6674 | -0.0549 | 0.0962 | 0.0770 | 0.4406 | 6 | 0 | -0.2763 | 0.3160 | 0.0042 | True | step5_chained_dd_above_gate |

### Primary-hypothesis reference - A1 full-pool + sl_partial_close_1r_runner_trail (the Arc 10 frame)

NOT triage-selected (all 24 A1 configs screened out at Stage A); run through the full gate anyway to document the primary hypothesis with an 11-fold result. The exact Arc 10 extractor is sl3.5.

| config_id | verdict | worst_fold_ratio | worst_fold_roi_base_pct | worst_fold_dd_base_pct | worst_fold_dd_at_0p40_pct | chained_max_dd_base_pct | n_negative_folds | min_trades_per_fold | holdout_roi_pct | holdout_dd_base_pct | r_safe_pct | scalable_to_safe | primary_failure_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1::full::sl_partial_close_1r_runner_trail::sl2.5::expinf | fail | -0.8566 | -0.2430 | 0.3052 | 0.2441 | 0.7703 | 8 | 365 | -0.2468 | 0.3797 | 0.0013 | False | step5_chained_dd_above_gate |
| A1::full::sl_partial_close_1r_runner_trail::sl3.5::exp2 | fail | -0.8754 | -0.1225 | 0.1819 | 0.1455 | 0.5322 | 8 | 159 | -0.0100 | 0.2453 | 0.0022 | True | step5_chained_dd_above_gate |
| A1::full::sl_partial_close_1r_runner_trail::sl2.5::exp2 | fail | -0.8983 | -0.2108 | 0.2577 | 0.2062 | 0.6286 | 7 | 247 | -0.1108 | 0.2914 | 0.0016 | True | step5_chained_dd_above_gate |
| A1::full::sl_partial_close_1r_runner_trail::sl3.5::expinf | fail | -0.9121 | -0.2252 | 0.3104 | 0.2484 | 0.7376 | 9 | 301 | -0.2190 | 0.4156 | 0.0013 | False | step5_chained_dd_above_gate |
| A1::full::sl_partial_close_1r_runner_trail::sl3.0::exp2 | fail | -0.9230 | -0.1489 | 0.1641 | 0.1313 | 0.6482 | 7 | 205 | -0.1793 | 0.3520 | 0.0024 | True | step5_chained_dd_above_gate |
| A1::full::sl_partial_close_1r_runner_trail::sl3.0::expinf | fail | -0.9596 | -0.2638 | 0.3348 | 0.2679 | 0.7864 | 7 | 336 | -0.1992 | 0.4035 | 0.0012 | False | step5_not_scalable |
