# Arc 7 — Step 3 capturability characterisation summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§2, 7, 11, 17

## Verdict
**PASS** — 3 unit(s) pass §2 floors conjunctively; proceed to Step 4.

## Surviving units (passing §2 at selected SL)

| unit | type | n | size_frac | archetype | sl | R(atr) | composite | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| c1 | cluster | 185 | 0.1436 | V-shape recovery (forward-geometry weak) | 4.0 | 4.0 | 0.6174 | 0.5674 | 1.0000 | 0.0000 | 3.4496 | unclassified |
| c3 | cluster | 365 | 0.2834 | V-shape recovery | 2.0 | 2.0 | 0.4126 | 0.5599 | 0.8082 | 0.0055 | 2.1238 | unclassified |
| agg_c1_c3 | aggregate | 550 | 0.4270 | V-shape recovery | 4.0 | 4.0 | 0.3784 | 0.5575 | 0.7709 | 0.0000 | 1.9761 | unclassified |

## SL sweep summary (per cluster + aggregate)

### c0 (cluster, n=385, peaks_centroid=4.44, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.3506 | 0.7039 | 0.3662 | 1.8216 | scattered | 0.2989 | 4/7 | -0.2617 | — |
| 1.0 | 0.5008 | 0.6545 | 0.1351 | 1.4546 | unclassified | 0.2989 | 4/7 | 0.0703 | — |
| 1.5 | 0.5726 | 0.5662 | 0.0571 | 1.1680 | unclassified | 0.2989 | 5/7 | 0.1317 | — |
| 2.0 | 0.6188 | 0.4805 | 0.0104 | 0.9739 | unclassified | 0.2989 | 5/7 | 0.1389 | — |
| 3.0 | 0.6018 | 0.4052 | 0.0000 | 0.7943 | unclassified | 0.2989 | 5/7 | 0.0570 | — |
| 4.0 | 0.6008 | 0.3870 | 0.0000 | 0.7123 | unclassified | 0.2989 | 5/7 | 0.0378 | — |

### c2 (cluster, n=353, peaks_centroid=0.59, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.0118 | 0.3059 | 0.7025 | 0.5977 | heavy_right_tail | 0.2741 | 3/7 | -1.3348 | — |
| 1.0 | 0.0146 | 0.1076 | 0.3824 | 0.3400 | unclassified | 0.2741 | 3/7 | -1.2102 | — |
| 1.5 | 0.0194 | 0.0652 | 0.1870 | 0.2504 | unclassified | 0.2741 | 4/7 | -1.0525 | — |
| 2.0 | 0.0194 | 0.0340 | 0.1105 | 0.2027 | unclassified | 0.2741 | 4/7 | -1.0071 | — |
| 3.0 | 0.1899 | 0.1728 | 0.0340 | 0.2313 | heavy_right_tail | 0.2741 | 4/7 | -0.6213 | — |
| 4.0 | 0.2885 | 0.2493 | 0.0142 | 0.2439 | heavy_right_tail | 0.2741 | 4/7 | -0.4264 | — |

### agg_c0_c2 (aggregate, n=738, peaks_centroid=2.60, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.1885 | 0.5136 | 0.5271 | 1.0327 | heavy_right_tail | 0.5730 | 3/7 | -0.7750 | — |
| 1.0 | 0.2683 | 0.3930 | 0.2534 | 0.7193 | heavy_right_tail | 0.5730 | 4/7 | -0.5422 | — |
| 1.5 | 0.3080 | 0.3266 | 0.1192 | 0.5710 | unclassified | 0.5730 | 4/7 | -0.4347 | — |
| 2.0 | 0.3321 | 0.2669 | 0.0583 | 0.4554 | unclassified | 0.5730 | 4/7 | -0.4092 | — |
| 3.0 | 0.4048 | 0.2940 | 0.0163 | 0.5364 | heavy_right_tail | 0.5730 | 4/7 | -0.2674 | — |
| 4.0 | 0.4514 | 0.3211 | 0.0068 | 0.5319 | heavy_right_tail | 0.5730 | 4/7 | -0.1843 | — |

### c1 (cluster, n=185, peaks_centroid=33.51, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.3248 | 0.6703 | 0.3405 | 2.3214 | scattered | 0.1436 | 3/7 | -0.2955 | — |
| 1.0 | 0.4412 | 0.7676 | 0.1135 | 8.1475 | scattered | 0.1436 | 5/7 | 0.1452 | — |
| 1.5 | 0.5319 | 0.8703 | 0.0216 | 8.3732 | scattered | 0.1436 | 5/7 | 0.4305 | — |
| 2.0 | 0.5674 | 1.0000 | 0.0000 | 6.8991 | scattered | 0.1436 | 6/7 | 0.6174 | — |
| 3.0 | 0.5674 | 1.0000 | 0.0000 | 4.5994 | scattered | 0.1436 | 6/7 | 0.6174 | — |
| 4.0 | 0.5674 | 1.0000 | 0.0000 | 3.4496 | unclassified | 0.1436 | 7/7 | 0.6174 | **SEL** |

### c3 (cluster, n=365, peaks_centroid=9.84, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.2636 | 0.6356 | 0.4301 | 1.4921 | heavy_right_tail | 0.2834 | 3/7 | -0.4810 | — |
| 1.0 | 0.4011 | 0.6575 | 0.0986 | 1.9673 | scattered | 0.2834 | 4/7 | 0.0101 | — |
| 1.5 | 0.4884 | 0.7534 | 0.0219 | 2.1690 | unclassified | 0.2834 | 6/7 | 0.2699 | — |
| 2.0 | 0.5599 | 0.8082 | 0.0055 | 2.1238 | unclassified | 0.2834 | 7/7 | 0.4126 | **SEL** |
| 3.0 | 0.5553 | 0.7425 | 0.0000 | 1.5847 | unclassified | 0.2834 | 7/7 | 0.3478 | — |
| 4.0 | 0.5524 | 0.6548 | 0.0000 | 1.3399 | unclassified | 0.2834 | 5/7 | 0.2572 | — |

### agg_c1_c3 (aggregate, n=550, peaks_centroid=17.81, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.2842 | 0.6473 | 0.4000 | 1.8050 | heavy_right_tail | 0.4270 | 4/7 | -0.4186 | — |
| 1.0 | 0.4146 | 0.6945 | 0.1036 | 2.3419 | heavy_right_tail | 0.4270 | 5/7 | 0.0555 | — |
| 1.5 | 0.5030 | 0.7927 | 0.0218 | 3.0520 | scattered | 0.4270 | 5/7 | 0.3239 | — |
| 2.0 | 0.5624 | 0.8727 | 0.0036 | 3.2503 | scattered | 0.4270 | 6/7 | 0.4815 | — |
| 3.0 | 0.5594 | 0.8291 | 0.0000 | 2.3684 | scattered | 0.4270 | 6/7 | 0.4385 | — |
| 4.0 | 0.5575 | 0.7709 | 0.0000 | 1.9761 | unclassified | 0.4270 | 7/7 | 0.3784 | **SEL** |

## Tentative label disambiguation

- **c0** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold OR Peak-and-collapse (Step 4 disambiguation)` (pct_peak_and_collapse=0.4675)
- **c2** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold` (pct_peak_and_collapse=0.0340)
- **agg_c0_c2** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold` (pct_peak_and_collapse=0.2602)
- **c1** — tentative `tentative_V-shape recovery` → final `V-shape recovery (forward-geometry weak)` (pct_peak_and_collapse=0.2919)
- **c3** — tentative `tentative_V-shape recovery` → final `V-shape recovery` (pct_peak_and_collapse=0.7151)
- **agg_c1_c3** — tentative `tentative_V-shape recovery` → final `V-shape recovery` (pct_peak_and_collapse=0.3891)

## bimodal_separated test (at selected SL, or SL=2.0 if no SL passed)

| unit | sl_ref | dip stat | p-value | min mode mass | mode separation (R) | result |
|---|---:|---:|---:|---:|---:|:---:|
| c0 | 2.0 | 0.0119 | 0.9904 | 0.0000 | 0.0000 | no |
| c2 | 2.0 | 0.0105 | 0.9952 | 0.0000 | 0.0000 | no |
| agg_c0_c2 | 2.0 | 0.0073 | 0.9961 | 0.0000 | 0.0000 | no |
| c1 | 4.0 | 0.0181 | 0.9763 | 0.0000 | 0.0000 | no |
| c3 | 2.0 | 0.0167 | 0.7442 | 0.0000 | 0.0000 | no |
| agg_c1_c3 | 4.0 | 0.0091 | 0.9935 | 0.0000 | 0.0000 | no |

## Per-cluster / per-aggregate routing

| cluster | tentative | individual passes | aggregate passes | disposition | final archetype |
|---:|---|:---:|:---:|---|---|
| c0 | tentative_Early-peak hold OR Peak-and-collapse | no | no | dies | Early-peak hold OR Peak-and-collapse (Step 4 disambiguation) |
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
| c0 | 385 | 0.2989 | 2.0 | 0.6188 | 0.4805 | 0.0104 | 0.9739 | mfe_p50(0.974R<1.5R), reach_1R(0.481<0.70) |
| c2 | 353 | 0.2741 | 4.0 | 0.2885 | 0.2493 | 0.0142 | 0.2439 | mono(0.288<0.55), mfe_p50(0.244R<1.5R), reach_1R(0.249<0.70) |
| agg_c0_c2 | 738 | 0.5730 | 4.0 | 0.4514 | 0.3211 | 0.0068 | 0.5319 | mono(0.451<0.55), mfe_p50(0.532R<1.5R), reach_1R(0.321<0.70) |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_agg_c0_c2_distribution.csv` | `1c4a8554017fa62d…` | `1c4a8554017fa62d…` | YES |
| `archetype_agg_c0_c2_sl_sweep.csv` | `8f0c55bc8394192b…` | `8f0c55bc8394192b…` | YES |
| `archetype_agg_c1_c3_distribution.csv` | `c25d961526ab578a…` | `c25d961526ab578a…` | YES |
| `archetype_agg_c1_c3_sl_sweep.csv` | `e63097b92df9dfe0…` | `e63097b92df9dfe0…` | YES |
| `archetype_c0_distribution.csv` | `aa255642fba33cca…` | `aa255642fba33cca…` | YES |
| `archetype_c0_sl_sweep.csv` | `1a09e50c3452d2e8…` | `1a09e50c3452d2e8…` | YES |
| `archetype_c1_distribution.csv` | `7a9f284172144173…` | `7a9f284172144173…` | YES |
| `archetype_c1_sl_sweep.csv` | `eb850fe46eb90536…` | `eb850fe46eb90536…` | YES |
| `archetype_c2_distribution.csv` | `0b430e640302d0ee…` | `0b430e640302d0ee…` | YES |
| `archetype_c2_sl_sweep.csv` | `df05cc5187b590e0…` | `df05cc5187b590e0…` | YES |
| `archetype_c3_distribution.csv` | `3569f1a7d2ad7fb2…` | `3569f1a7d2ad7fb2…` | YES |
| `archetype_c3_sl_sweep.csv` | `9b85fdcda0ff60d2…` | `9b85fdcda0ff60d2…` | YES |
| `archetype_summaries.csv` | `c91570f1c5788563…` | `c91570f1c5788563…` | YES |
| `capturability_pass_list.csv` | `2ca6a9910bb5f2e7…` | `2ca6a9910bb5f2e7…` | YES |
| `cluster_routing.csv` | `abb8870e2536425a…` | `abb8870e2536425a…` | YES |

## Files

- `results\l_arc_7\step3/archetype_agg_c0_c2_distribution.csv`
- `results\l_arc_7\step3/archetype_agg_c0_c2_sl_sweep.csv`
- `results\l_arc_7\step3/archetype_agg_c1_c3_distribution.csv`
- `results\l_arc_7\step3/archetype_agg_c1_c3_sl_sweep.csv`
- `results\l_arc_7\step3/archetype_c0_distribution.csv`
- `results\l_arc_7\step3/archetype_c0_sl_sweep.csv`
- `results\l_arc_7\step3/archetype_c1_distribution.csv`
- `results\l_arc_7\step3/archetype_c1_sl_sweep.csv`
- `results\l_arc_7\step3/archetype_c2_distribution.csv`
- `results\l_arc_7\step3/archetype_c2_sl_sweep.csv`
- `results\l_arc_7\step3/archetype_c3_distribution.csv`
- `results\l_arc_7\step3/archetype_c3_sl_sweep.csv`
- `results\l_arc_7\step3/archetype_summaries.csv`
- `results\l_arc_7\step3/capturability_pass_list.csv`
- `results\l_arc_7\step3/cluster_routing.csv`
- `results\l_arc_7\step3/STEP3_SUMMARY.md`
- `results\l_arc_7\step3/archetype_*_fwd_mfe_histogram.png`
- `results\l_arc_7\step3/archetype_*_final_r_histogram.png`
- `configs/arc_7/step3.yaml`
- `scripts/arc_7/step3_capturability.py`

## Step 3 commit
hash: _pending_

