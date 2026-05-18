# Arc 8 — Step 3 capturability characterisation summary

Protocol: `L_ARC_PROTOCOL.md` v2.3 stack (base v2.1.2 + v2.2 + v2.3 amendments); §§2, 7, 11, 17 unchanged. v2.3 §4 adds pre_t_sl_atr_multiplier column to cluster_routing.csv

## Verdict
**PASS** — 3 unit(s) pass §2 floors conjunctively; proceed to Step 4.

## Surviving units (passing §2 at selected SL)

| unit | type | n | size_frac | archetype | sl | R(atr) | composite | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| c1 | cluster | 177 | 0.1334 | V-shape recovery (forward-geometry weak) | 4.0 | 4.0 | 0.6146 | 0.5646 | 1.0000 | 0.0000 | 3.4999 | unclassified |
| c3 | cluster | 405 | 0.3052 | V-shape recovery | 2.0 | 2.0 | 0.4606 | 0.5736 | 0.8420 | 0.0049 | 2.2492 | unclassified |
| agg_c1_c3 | aggregate | 582 | 0.4386 | V-shape recovery | 3.0 | 3.0 | 0.4314 | 0.5669 | 0.8162 | 0.0017 | 2.4739 | unclassified |

## SL sweep summary (per cluster + aggregate)

### c2 (cluster, n=429, peaks_centroid=0.48, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.0153 | 0.2517 | 0.6993 | 0.5078 | unclassified | 0.3233 | 3/7 | -1.3822 | — |
| 1.0 | 0.0153 | 0.0746 | 0.3683 | 0.2955 | unclassified | 0.3233 | 3/7 | -1.2284 | — |
| 1.5 | 0.0153 | 0.0373 | 0.1888 | 0.2200 | unclassified | 0.3233 | 4/7 | -1.0862 | — |
| 2.0 | 0.0153 | 0.0233 | 0.0932 | 0.1708 | unclassified | 0.3233 | 4/7 | -1.0046 | — |
| 3.0 | 0.1774 | 0.1725 | 0.0186 | 0.1750 | heavy_right_tail | 0.3233 | 4/7 | -0.6188 | — |
| 4.0 | 0.2856 | 0.2751 | 0.0047 | 0.2278 | heavy_right_tail | 0.3233 | 4/7 | -0.3940 | — |

### c1 (cluster, n=177, peaks_centroid=33.82, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.3039 | 0.6328 | 0.3785 | 1.7868 | scattered | 0.1334 | 3/7 | -0.3919 | — |
| 1.0 | 0.4445 | 0.7232 | 0.0791 | 7.3939 | scattered | 0.1334 | 5/7 | 0.1385 | — |
| 1.5 | 0.5101 | 0.8531 | 0.0113 | 8.1362 | bimodal_separated | 0.1334 | 6/7 | 0.4020 | — |
| 2.0 | 0.5648 | 1.0000 | 0.0000 | 6.9861 | scattered | 0.1334 | 6/7 | 0.6148 | — |
| 3.0 | 0.5646 | 1.0000 | 0.0000 | 4.6666 | scattered | 0.1334 | 6/7 | 0.6146 | — |
| 4.0 | 0.5646 | 1.0000 | 0.0000 | 3.4999 | unclassified | 0.1334 | 7/7 | 0.6146 | **SEL** |

### c3 (cluster, n=405, peaks_centroid=10.42, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.3020 | 0.6691 | 0.4099 | 1.8056 | heavy_right_tail | 0.3052 | 4/7 | -0.3888 | — |
| 1.0 | 0.4439 | 0.7210 | 0.1086 | 2.4981 | scattered | 0.3052 | 5/7 | 0.1062 | — |
| 1.5 | 0.5144 | 0.7827 | 0.0370 | 2.4153 | scattered | 0.3052 | 5/7 | 0.3101 | — |
| 2.0 | 0.5736 | 0.8420 | 0.0049 | 2.2492 | unclassified | 0.3052 | 7/7 | 0.4606 | **SEL** |
| 3.0 | 0.5680 | 0.7358 | 0.0025 | 1.7102 | unclassified | 0.3052 | 7/7 | 0.3513 | — |
| 4.0 | 0.5684 | 0.6568 | 0.0025 | 1.3531 | unclassified | 0.3052 | 5/7 | 0.2727 | — |

### agg_c1_c3 (aggregate, n=582, peaks_centroid=17.54, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.3025 | 0.6581 | 0.4003 | 1.8028 | heavy_right_tail | 0.4386 | 4/7 | -0.3897 | — |
| 1.0 | 0.4441 | 0.7216 | 0.0997 | 2.8526 | scattered | 0.4386 | 5/7 | 0.1161 | — |
| 1.5 | 0.5131 | 0.8041 | 0.0292 | 3.0948 | scattered | 0.4386 | 5/7 | 0.3380 | — |
| 2.0 | 0.5709 | 0.8900 | 0.0034 | 3.2873 | scattered | 0.4386 | 6/7 | 0.5075 | — |
| 3.0 | 0.5669 | 0.8162 | 0.0017 | 2.4739 | unclassified | 0.4386 | 7/7 | 0.4314 | **SEL** |
| 4.0 | 0.5672 | 0.7612 | 0.0017 | 2.0273 | unclassified | 0.4386 | 7/7 | 0.3767 | — |

## Tentative label disambiguation

- **c2** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold` (pct_peak_and_collapse=0.0233)
- **c1** — tentative `tentative_V-shape recovery` → final `V-shape recovery (forward-geometry weak)` (pct_peak_and_collapse=0.1921)
- **c3** — tentative `tentative_V-shape recovery` → final `V-shape recovery` (pct_peak_and_collapse=0.7284)
- **agg_c1_c3** — tentative `tentative_V-shape recovery` → final `V-shape recovery` (pct_peak_and_collapse=0.4364)

## bimodal_separated test (at selected SL, or SL=2.0 if no SL passed)

| unit | sl_ref | dip stat | p-value | min mode mass | mode separation (R) | result |
|---|---:|---:|---:|---:|---:|:---:|
| c2 | 2.0 | 0.0145 | 0.8298 | 0.0000 | 0.0000 | no |
| c1 | 4.0 | 0.0146 | 0.9953 | 0.0000 | 0.0000 | no |
| c3 | 2.0 | 0.0124 | 0.977 | 0.0000 | 0.0000 | no |
| agg_c1_c3 | 3.0 | 0.0100 | 0.9882 | 0.0000 | 0.0000 | no |

## Per-cluster / per-aggregate routing

| cluster | tentative | individual passes | aggregate passes | disposition | final archetype |
|---:|---|:---:|:---:|---|---|
| c2 | tentative_Early-peak hold OR Peak-and-collapse | no | no | dies | Early-peak hold |
| c1 | tentative_V-shape recovery | yes | yes | proceeds_both | V-shape recovery (forward-geometry weak) |
| c3 | tentative_V-shape recovery | yes | yes | proceeds_both | V-shape recovery |

## Distribution detail

- See `archetype_c1_distribution.csv` for full percentiles, mass-in-band, and bimodal mode info.
- Histograms: `archetype_c1_fwd_mfe_histogram.png`, `archetype_c1_final_r_histogram.png`
- See `archetype_c3_distribution.csv` for full percentiles, mass-in-band, and bimodal mode info.
- Histograms: `archetype_c3_fwd_mfe_histogram.png`, `archetype_c3_final_r_histogram.png`
- See `archetype_agg_c1_c3_distribution.csv` for full percentiles, mass-in-band, and bimodal mode info.
- Histograms: `archetype_agg_c1_c3_fwd_mfe_histogram.png`, `archetype_agg_c1_c3_final_r_histogram.png`

## Kill reasons (per non-surviving unit)

| unit | n | size_frac | best-composite SL | best mono_pp | best reach_1R | best wrong_way_pp | best fwd_mfe_p50 | failing floors at best SL |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| c2 | 429 | 0.3233 | 4.0 | 0.2856 | 0.2751 | 0.0047 | 0.2278 | mono(0.286<0.55), mfe_p50(0.228R<1.5R), reach_1R(0.275<0.70) |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_agg_c1_c3_distribution.csv` | `603d45b762135bb8…` | `603d45b762135bb8…` | YES |
| `archetype_agg_c1_c3_sl_sweep.csv` | `6bb01f9e4bb659a5…` | `6bb01f9e4bb659a5…` | YES |
| `archetype_c1_distribution.csv` | `c4c0920f99201820…` | `c4c0920f99201820…` | YES |
| `archetype_c1_sl_sweep.csv` | `2658e2ec5e9913a4…` | `2658e2ec5e9913a4…` | YES |
| `archetype_c2_distribution.csv` | `9bd13e1c056df576…` | `9bd13e1c056df576…` | YES |
| `archetype_c2_sl_sweep.csv` | `d7c4592bf5cc1ec8…` | `d7c4592bf5cc1ec8…` | YES |
| `archetype_c3_distribution.csv` | `cf4c35cf7c620e6f…` | `cf4c35cf7c620e6f…` | YES |
| `archetype_c3_sl_sweep.csv` | `d7ac2edf1617d857…` | `d7ac2edf1617d857…` | YES |
| `archetype_summaries.csv` | `32dbf9db9cd92b73…` | `32dbf9db9cd92b73…` | YES |
| `capturability_pass_list.csv` | `292f85b4e01ef4a2…` | `292f85b4e01ef4a2…` | YES |
| `cluster_routing.csv` | `cdc34d4673d44ad7…` | `cdc34d4673d44ad7…` | YES |

## Files

- `results\l_arc_8\step3/archetype_agg_c1_c3_distribution.csv`
- `results\l_arc_8\step3/archetype_agg_c1_c3_sl_sweep.csv`
- `results\l_arc_8\step3/archetype_c1_distribution.csv`
- `results\l_arc_8\step3/archetype_c1_sl_sweep.csv`
- `results\l_arc_8\step3/archetype_c2_distribution.csv`
- `results\l_arc_8\step3/archetype_c2_sl_sweep.csv`
- `results\l_arc_8\step3/archetype_c3_distribution.csv`
- `results\l_arc_8\step3/archetype_c3_sl_sweep.csv`
- `results\l_arc_8\step3/archetype_summaries.csv`
- `results\l_arc_8\step3/capturability_pass_list.csv`
- `results\l_arc_8\step3/cluster_routing.csv`
- `results\l_arc_8\step3/STEP3_SUMMARY.md`
- `results\l_arc_8\step3/archetype_*_fwd_mfe_histogram.png`
- `results\l_arc_8\step3/archetype_*_final_r_histogram.png`
- `configs/l_arc_8/step3.yaml`
- `scripts/l_arc_8/step3_capturability.py`

## Step 3 commit
hash: _pending_

