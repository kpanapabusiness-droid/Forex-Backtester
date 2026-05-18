# Arc 9 Step 3 - Capturability characterisation

Verdict: **PASS** (1 archetype(s) clear §2 floors at the selected SL)

SL sweep candidates: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
§2 floors (v2.1.2): mono_pre_peak≥0.55, frac_reach_1R≥0.7, frac_wrong_way_pre_peak≤0.3, fwd_mfe_p50≥1.5, size_fraction≥0.1, shape_tag ≠ scattered

## Cluster routing

| cluster | archetype | individual? | sel SL (×ATR) | composite | aggregate? | agg SL | agg composite | disposition |
|---|---|---|---|---|---|---|---|---|
| 0 | Unclassified | PASS | 2.0 | 0.612 | - | - | - | **individual_only** |
| 1 | Early-peak hold | fail | - | - | - | - | - | **dies** |
| 2 | Unclassified | fail | - | - | - | - | - | **dies** |

## Selected SL per surviving archetype

| label | mode | n | size | sel SL | mono_pp | reach_1R | ww_pp | fwd_mfe_p50 | final_r_mean | t | composite | shape | shape_pass | bimodal? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster_0_individual | individual | 365 | 0.170 | 2.0 | 0.562 | 1.000 | 0.000 | 6.18 | +4.423 | +21.81 | 0.612 | tight_unimodal | PASS | - |

## v2.3 §4 (Open-24) - per-archetype pre_t_sl_atr_multiplier

Per v2.3 §4, each Pipeline D1 archetype's pre-t SL multiplier equals its Step 3 selected SL multiplier. This is recorded in `cluster_routing.csv` and `archetype_summaries.csv` (col `pre_t_sl_atr_multiplier`). For Steps 1-4 this is metadata only; engine PR honour pending.

## Files

- per_trade_sl_sweep.csv
- cluster_sl_sweep.csv
- archetype_sl_sweep.csv
- archetype_summaries.csv
- capturability_pass_list.csv
- cluster_routing.csv
- results\l_arc_9\step3_capturability\cluster_0_distribution.csv
