# Arc 11 v3.0 — Step 5 Architectures Ranked

Per L_PROTOCOL §2 Step 5 + Amendments 1 + 2.

## Total configs evaluated: 36
Selection bias accounting: **thin** search.

## Top 15 by worst-fold ratio

| rank | config | cluster | arche | arch | SL | exit | cap | worst_ratio | worst_roi% | worst_dd% | sign | n_total | verdict |
|---:|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | cluster0_A2_SL2.5_EXsl_plus_trailing_atr_1r_CAPNone | 0 | Choppy | A2 | 2.5 | sl_plus_trailing_atr_1r | nan | 3.181 | 23.86 | 11.96 | 10 | 2873 | FAIL |
| 2 | cluster0_A2_SL2.0_EXsl_plus_trailing_atr_2r_CAPNone | 0 | Choppy | A2 | 2.0 | sl_plus_trailing_atr_2r | nan | 2.182 | 27.03 | 19.41 | 10 | 2873 | FAIL |
| 3 | cluster0_A2_SL1.5_EXsl_plus_trailing_atr_2r_CAPNone | 0 | Choppy | A2 | 1.5 | sl_plus_trailing_atr_2r | nan | 1.908 | 24.83 | 15.08 | 10 | 2873 | FAIL |
| 4 | cluster0_A2_SL2.5_EXsl_plus_trailing_atr_2r_CAPNone | 0 | Choppy | A2 | 2.5 | sl_plus_trailing_atr_2r | nan | 1.709 | 22.33 | 19.23 | 10 | 2873 | FAIL |
| 5 | cluster0_A2_SL2.5_EXsl_plus_trailing_atr_1r_CAP2 | 0 | Choppy | A2 | 2.5 | sl_plus_trailing_atr_1r | 2.0 | 1.335 | 6.34 | 7.74 | 10 | 1030 | FAIL |
| 6 | cluster0_A2_SL2.0_EXsl_plus_trailing_atr_1r_CAPNone | 0 | Choppy | A2 | 2.0 | sl_plus_trailing_atr_1r | nan | 1.177 | 13.33 | 11.96 | 10 | 2873 | FAIL |
| 7 | cluster0_A2_SL2.0_EXsl_plus_trailing_atr_2r_CAP2 | 0 | Choppy | A2 | 2.0 | sl_plus_trailing_atr_2r | 2.0 | 1.107 | 8.15 | 11.38 | 10 | 1030 | FAIL |
| 8 | cluster0_A2_SL1.5_EXsl_plus_trailing_atr_1r_CAPNone | 0 | Choppy | A2 | 1.5 | sl_plus_trailing_atr_1r | nan | 1.067 | 15.23 | 14.28 | 10 | 2873 | FAIL |
| 9 | cluster0_A2_SL1.5_EXsl_plus_trailing_atr_2r_CAP2 | 0 | Choppy | A2 | 1.5 | sl_plus_trailing_atr_2r | 2.0 | 0.590 | 5.85 | 9.91 | 10 | 1030 | FAIL |
| 10 | cluster0_A2_SL1.5_EXsl_plus_trailing_atr_1r_CAP2 | 0 | Choppy | A2 | 1.5 | sl_plus_trailing_atr_1r | 2.0 | 0.447 | 3.04 | 7.99 | 10 | 1030 | FAIL |
| 11 | cluster0_A2_SL2.5_EXsl_plus_trailing_atr_2r_CAP2 | 0 | Choppy | A2 | 2.5 | sl_plus_trailing_atr_2r | 2.0 | 0.219 | 1.53 | 11.48 | 10 | 1030 | FAIL |
| 12 | cluster0_A2_SL2.0_EXsl_plus_trailing_atr_1r_CAP2 | 0 | Choppy | A2 | 2.0 | sl_plus_trailing_atr_1r | 2.0 | 0.105 | 0.54 | 8.31 | 10 | 1030 | FAIL |
| 13 | cluster0_A2_SL2.0_EXsl_only_CAPNone | 0 | Choppy | A2 | 2.0 | sl_only | nan | -0.384 | -20.84 | 54.20 | 8 | 2873 | FAIL |
| 14 | cluster0_A2_SL1.5_EXsl_only_CAPNone | 0 | Choppy | A2 | 1.5 | sl_only | nan | -0.432 | -26.52 | 61.37 | 9 | 2873 | FAIL |
| 15 | cluster0_A2_SL2.5_EXsl_only_CAPNone | 0 | Choppy | A2 | 2.5 | sl_only | nan | -0.453 | -20.91 | 46.19 | 8 | 2873 | FAIL |

## Oracle WFO

| cluster | worst_ratio | worst_roi% | worst_dd% | mean_ratio | mean_roi% | n_total | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 16.578 | 99.47 | 10.15 | 90.869 | 388.99 | 1411 | FAIL |

## Holdout (top-3 candidates, one-shot)

| config | architecture | cluster | SL | exit | cap | trades | roi% | dd% | ratio | wfo_verdict | holdout_verdict | combined |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|
| cluster0_A2_SL2.5_EXsl_plus_trailing_atr_1r_CAPNone | A2 | 0 | 2.5 | sl_plus_trailing_atr_1r | nan | 1536 | 617.43 | 16.90 | 36.541 | FAIL | FAIL | FAIL |
| cluster0_A2_SL2.0_EXsl_plus_trailing_atr_2r_CAPNone | A2 | 0 | 2.0 | sl_plus_trailing_atr_2r | nan | 1536 | 959.26 | 19.13 | 50.154 | FAIL | FAIL | FAIL |
| cluster0_A2_SL1.5_EXsl_plus_trailing_atr_2r_CAPNone | A2 | 0 | 1.5 | sl_plus_trailing_atr_2r | nan | 1536 | 307.82 | 30.46 | 10.107 | FAIL | FAIL | FAIL |
