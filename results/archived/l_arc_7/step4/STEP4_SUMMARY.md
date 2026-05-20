# Arc 7 — Step 4 extractability summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§8, 10, 17

## Verdict
**FAIL — CLEAN-NULL at Step 4.** Zero (unit, pipeline) pairs clear §8 gate with a valid admission threshold.

## Pipeline summary

| unit | pipeline | n | n_feat | mean AUC | std AUC | gate | pass? | selected t | exclusion | admission | lift | realised_success |
|---|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| c1 | E | 185 | 22 | 0.4841177668 | 0.1269779858 | 0.65 | no | — | — | — | — | — |
| c1 | D1 | 185 | 11 | 0.4200581134 | 0.1304957171 | 0.60 | no | — | — | — | — | — |
| c3 | E | 365 | 22 | 0.5119345553 | 0.09414786501 | 0.65 | no | — | — | — | — | — |
| c3 | D1 | 365 | 11 | 0.5177491081 | 0.09509587406 | 0.60 | no | — | — | — | — | — |
| agg_c1_c3 | E | 550 | 22 | 0.536427014 | 0.02933511701 | 0.65 | no | — | — | — | — | — |
| agg_c1_c3 | D1 | 550 | 11 | 0.495568735 | 0.06543981878 | 0.60 | no | — | — | — | — | — |

## Per-fold AUCs

### c1 / E

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.468 | 35 | 30 | 0.8333333333 |
| 1 | 0.4708994709 | 65 | 30 | 0.7 |
| 2 | 0.6477272727 | 95 | 30 | 0.7333333333 |
| 3 | 0.5358851675 | 125 | 30 | 0.6333333333 |
| 4 | 0.2980769231 | 155 | 30 | 0.8666666667 |

### c1 / D1

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.52 | 35 | 30 | 0.8333333333 |
| 1 | 0.1957671958 | 65 | 30 | 0.7 |
| 2 | 0.4204545455 | 95 | 30 | 0.7333333333 |
| 3 | 0.4736842105 | 125 | 30 | 0.6333333333 |
| 4 | 0.4903846154 | 155 | 30 | 0.8666666667 |

### c3 / E

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.4915254237 | 65 | 60 | 0.01666666667 |
| 1 | 0.4454545455 | 125 | 60 | 0.08333333333 |
| 2 | 0.452 | 185 | 60 | 0.1666666667 |
| 3 | 0.6754807692 | 245 | 60 | 0.1333333333 |
| 4 | 0.4952120383 | 305 | 60 | 0.2833333333 |

### c3 / D1

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.4576271186 | 65 | 60 | 0.01666666667 |
| 1 | 0.5636363636 | 125 | 60 | 0.08333333333 |
| 2 | 0.469 | 185 | 60 | 0.1666666667 |
| 3 | 0.6634615385 | 245 | 60 | 0.1333333333 |
| 4 | 0.4350205198 | 305 | 60 | 0.2833333333 |

### agg_c1_c3 / E

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.5712074303 | 95 | 91 | 0.3736263736 |
| 1 | 0.5561735261 | 186 | 91 | 0.3186813187 |
| 2 | 0.4965483235 | 277 | 91 | 0.4285714286 |
| 3 | 0.5376175549 | 368 | 91 | 0.3626373626 |
| 4 | 0.5205882353 | 459 | 91 | 0.4395604396 |

### agg_c1_c3 / D1

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.4783281734 | 95 | 91 | 0.3736263736 |
| 1 | 0.4404894327 | 186 | 91 | 0.3186813187 |
| 2 | 0.5315581854 | 277 | 91 | 0.4285714286 |
| 3 | 0.4367816092 | 368 | 91 | 0.3626373626 |
| 4 | 0.5906862745 | 459 | 91 | 0.4395604396 |

## Routing per unit

| unit | E pass | D1 pass | route(s) carried |
|---|:---:|:---:|---|
| c1 | no | no | none |
| c3 | no | no | none |
| agg_c1_c3 | no | no | none |

## D1 lag audit

15 spot-checks performed (5 random trades × 3 units). All correct: **YES**.

| trade_id | pair | entry_time | signal_bar | expected ≤ | d1 joined | correct |
|---:|---|---|---|---|---|:---:|
| 85 | GBP_JPY | 2021-02-26 12:00:00 | 2021-02-26 08:00:00 | 2021-02-25 00:00:00 | 2021-02-25 00:00:00 | YES |
| 573 | EUR_NZD | 2023-01-18 12:00:00 | 2023-01-18 08:00:00 | 2023-01-17 00:00:00 | 2023-01-17 00:00:00 | YES |
| 817 | USD_JPY | 2023-12-28 20:00:00 | 2023-12-28 16:00:00 | 2023-12-27 00:00:00 | 2023-12-27 00:00:00 | YES |
| 947 | NZD_USD | 2024-07-29 20:00:00 | 2024-07-29 16:00:00 | 2024-07-28 00:00:00 | 2024-07-26 00:00:00 | YES |
| 1265 | AUD_USD | 2025-11-21 16:00:00 | 2025-11-21 12:00:00 | 2025-11-20 00:00:00 | 2025-11-20 00:00:00 | YES |
| 142 | NZD_USD | 2021-05-13 16:00:00 | 2021-05-13 12:00:00 | 2021-05-12 00:00:00 | 2021-05-12 00:00:00 | YES |
| 690 | AUD_NZD | 2023-07-07 16:00:00 | 2023-07-07 12:00:00 | 2023-07-06 00:00:00 | 2023-07-06 00:00:00 | YES |
| 997 | AUD_JPY | 2024-10-08 04:00:00 | 2024-10-08 00:00:00 | 2024-10-07 00:00:00 | 2024-10-07 00:00:00 | YES |
| 1028 | CAD_JPY | 2024-11-29 12:00:00 | 2024-11-29 08:00:00 | 2024-11-28 00:00:00 | 2024-11-28 00:00:00 | YES |
| 1257 | EUR_AUD | 2025-11-13 12:00:00 | 2025-11-13 08:00:00 | 2025-11-12 00:00:00 | 2025-11-12 00:00:00 | YES |
| 228 | EUR_NZD | 2021-08-24 16:00:00 | 2021-08-24 12:00:00 | 2021-08-23 00:00:00 | 2021-08-23 00:00:00 | YES |
| 492 | EUR_CAD | 2022-09-05 12:00:00 | 2022-09-05 08:00:00 | 2022-09-04 00:00:00 | 2022-09-02 00:00:00 | YES |
| 598 | EUR_JPY | 2023-02-24 04:00:00 | 2023-02-24 00:00:00 | 2023-02-23 00:00:00 | 2023-02-23 00:00:00 | YES |
| 665 | GBP_JPY | 2023-06-06 16:00:00 | 2023-06-06 12:00:00 | 2023-06-05 00:00:00 | 2023-06-05 00:00:00 | YES |
| 1090 | AUD_JPY | 2025-02-28 08:00:00 | 2025-02-28 04:00:00 | 2025-02-27 00:00:00 | 2025-02-27 00:00:00 | YES |

## Top features per surviving (unit, pipeline)

_None surviving._
## Class balance per unit

| unit | n | success_count | success_rate | class_weight_used |
|---|---:|---:|---:|---|
| c1 | 185 | 144 | 0.7783783784 | balanced |
| c3 | 365 | 46 | 0.1260273973 | balanced |
| agg_c1_c3 | 550 | 216 | 0.3927272727 | none |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `d1_lag_audit.csv` | `c879a8953f108579…` | `c879a8953f108579…` | YES |
| `extractability_summary.csv` | `9fe06a6c93528ffb…` | `9fe06a6c93528ffb…` | YES |
| `fold_aucs_agg_c1_c3_D1.csv` | `f9d1603db868ff00…` | `f9d1603db868ff00…` | YES |
| `fold_aucs_agg_c1_c3_E.csv` | `85af2ea894db368a…` | `85af2ea894db368a…` | YES |
| `fold_aucs_c1_D1.csv` | `3445926f90e761df…` | `3445926f90e761df…` | YES |
| `fold_aucs_c1_E.csv` | `1cbcaf6cc0e472c3…` | `1cbcaf6cc0e472c3…` | YES |
| `fold_aucs_c3_D1.csv` | `3079b23d876ce105…` | `3079b23d876ce105…` | YES |
| `fold_aucs_c3_E.csv` | `f2fcc3f489aec691…` | `f2fcc3f489aec691…` | YES |
| `pipeline_routing.csv` | `9658e24bc04f1fd9…` | `9658e24bc04f1fd9…` | YES |

## Kill reasons (per non-passing unit × pipeline)

- `c1/E`: mean AUC 0.4841177668 vs gate 0.65 (margin -0.1658822332).
- `c1/D1`: mean AUC 0.4200581134 vs gate 0.60 (margin -0.1799418866).
- `c3/E`: mean AUC 0.5119345553 vs gate 0.65 (margin -0.1380654447).
- `c3/D1`: mean AUC 0.5177491081 vs gate 0.60 (margin -0.08225089188).
- `agg_c1_c3/E`: mean AUC 0.536427014 vs gate 0.65 (margin -0.113572986).
- `agg_c1_c3/D1`: mean AUC 0.495568735 vs gate 0.60 (margin -0.104431265).

## Files

- `results\l_arc_7\step4/d1_lag_audit.csv`
- `results\l_arc_7\step4/extractability_summary.csv`
- `results\l_arc_7\step4/fold_aucs_agg_c1_c3_D1.csv`
- `results\l_arc_7\step4/fold_aucs_agg_c1_c3_E.csv`
- `results\l_arc_7\step4/fold_aucs_c1_D1.csv`
- `results\l_arc_7\step4/fold_aucs_c1_E.csv`
- `results\l_arc_7\step4/fold_aucs_c3_D1.csv`
- `results\l_arc_7\step4/fold_aucs_c3_E.csv`
- `results\l_arc_7\step4/pipeline_routing.csv`
- `results\l_arc_7\step4/STEP4_SUMMARY.md`
- `configs/arc_7/step4.yaml`
- `scripts/arc_7/step4_extractability.py`

## Step 4 commit
hash: _pending_

