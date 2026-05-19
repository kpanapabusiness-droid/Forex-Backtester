# Arc 10 — Step 4 extractability summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§8, 10, 17

## Verdict
**FAIL — CLEAN-NULL at Step 4.** Zero (unit, pipeline) pairs clear §8 gate with a valid admission threshold.

## Pipeline summary

| unit | pipeline | n | n_feat | mean AUC | std AUC | gate | pass? | selected t | exclusion | admission | lift | realised_success |
|---|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| c1 | E | 228 | 25 | 0.6296093271 | 0.04676161333 | 0.65 | no | — | — | — | — | — |
| c1 | D1 | 228 | 11 | 0.5896962237 | 0.09585755612 | 0.60 | no | — | — | — | — | — |

## Per-fold AUCs

### c1 / E

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.6246537396 | 38 | 38 | 0.5 |
| 1 | 0.6694677871 | 76 | 38 | 0.4473684211 |
| 2 | 0.5512465374 | 114 | 38 | 0.5 |
| 3 | 0.6464285714 | 152 | 38 | 0.2631578947 |
| 4 | 0.65625 | 190 | 38 | 0.5789473684 |

### c1 / D1

| fold | AUC | n_train | n_test | base_success |
|---:|---:|---:|---:|---:|
| 0 | 0.6689750693 | 38 | 38 | 0.5 |
| 1 | 0.6246498599 | 76 | 38 | 0.4473684211 |
| 2 | 0.5180055402 | 114 | 38 | 0.5 |
| 3 | 0.4607142857 | 152 | 38 | 0.2631578947 |
| 4 | 0.6761363636 | 190 | 38 | 0.5789473684 |

## Routing per unit

| unit | E pass | D1 pass | route(s) carried |
|---|:---:|:---:|---|
| c1 | no | no | none |

## D1 lag audit

5 spot-checks performed (5 random trades × 3 units). All correct: **YES**.

| trade_id | pair | entry_time | signal_bar | expected ≤ | d1 joined | correct |
|---:|---|---|---|---|---|:---:|
| 73 | USD_CAD | 2021-03-23 12:00:00 | 2021-03-23 08:00:00 | 2021-03-22 00:00:00 | 2021-03-22 00:00:00 | YES |
| 354 | EUR_AUD | 2023-02-14 04:00:00 | 2023-02-14 00:00:00 | 2023-02-13 00:00:00 | 2023-02-13 00:00:00 | YES |
| 362 | GBP_USD | 2023-03-10 12:00:00 | 2023-03-10 08:00:00 | 2023-03-09 00:00:00 | 2023-03-09 00:00:00 | YES |
| 534 | EUR_CAD | 2024-04-23 12:00:00 | 2024-04-23 08:00:00 | 2024-04-22 00:00:00 | 2024-04-22 00:00:00 | YES |
| 642 | EUR_JPY | 2025-01-20 12:00:00 | 2025-01-20 08:00:00 | 2025-01-19 00:00:00 | 2025-01-17 00:00:00 | YES |

## Top features per surviving (unit, pipeline)

_None surviving._
## Class balance per unit

| unit | n | success_count | success_rate | class_weight_used |
|---|---:|---:|---:|---|
| c1 | 228 | 110 | 0.4824561404 | none |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `d1_lag_audit.csv` | `1cea21c384bf7614…` | `1cea21c384bf7614…` | YES |
| `extractability_summary.csv` | `859c114e23f1b48e…` | `859c114e23f1b48e…` | YES |
| `fold_aucs_c1_D1.csv` | `62126ed270986995…` | `62126ed270986995…` | YES |
| `fold_aucs_c1_E.csv` | `e2e53f83e0982063…` | `e2e53f83e0982063…` | YES |
| `pipeline_routing.csv` | `ea215d6c0fc2c3ab…` | `ea215d6c0fc2c3ab…` | YES |

## Kill reasons (per non-passing unit × pipeline)

- `c1/E`: mean AUC 0.6296093271 vs gate 0.65 (margin -0.02039067289).
- `c1/D1`: mean AUC 0.5896962237 vs gate 0.60 (margin -0.01030377626).

## Files

- `results\l_arc_10\step4/d1_lag_audit.csv`
- `results\l_arc_10\step4/extractability_summary.csv`
- `results\l_arc_10\step4/fold_aucs_c1_D1.csv`
- `results\l_arc_10\step4/fold_aucs_c1_E.csv`
- `results\l_arc_10\step4/pipeline_routing.csv`
- `results\l_arc_10\step4/STEP4_SUMMARY.md`
- `configs/l_arc_10/step4.yaml`
- `scripts/l_arc_10/step4_extractability.py`

## Step 4 commit
hash: _pending_

