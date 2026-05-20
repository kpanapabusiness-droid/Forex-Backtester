# Arc 11 — Step 3 capturability characterisation summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§2, 7, 11, 17

## Verdict
**PASS** — 3 unit(s) pass §2 floors conjunctively; proceed to Step 4.

## Surviving units (passing §2 at selected SL)

| unit | type | n | size_frac | archetype | sl | R(atr) | composite | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| c1 | cluster | 324 | 0.1409 | V-shape recovery (forward-geometry weak) | 3.0 | 3.0 | 0.6119 | 0.5619 | 1.0000 | 0.0000 | 4.4808 | unclassified |
| c3 | cluster | 565 | 0.2458 | V-shape recovery | 2.0 | 2.0 | 0.4206 | 0.5724 | 0.8088 | 0.0106 | 2.0560 | unclassified |
| agg_c1_c3 | aggregate | 889 | 0.3867 | V-shape recovery | 3.0 | 3.0 | 0.4155 | 0.5668 | 0.8020 | 0.0034 | 2.5021 | unclassified |

## SL sweep summary (per cluster + aggregate)

### c0 (cluster, n=744, peaks_centroid=0.45, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.0115 | 0.2487 | 0.6707 | 0.4934 | unclassified | 0.3236 | 3/7 | -1.3605 | — |
| 1.0 | 0.0142 | 0.0780 | 0.3065 | 0.2826 | unclassified | 0.3236 | 3/7 | -1.1643 | — |
| 1.5 | 0.0142 | 0.0336 | 0.1425 | 0.2062 | unclassified | 0.3236 | 4/7 | -1.0446 | — |
| 2.0 | 0.0142 | 0.0188 | 0.0685 | 0.1667 | unclassified | 0.3236 | 4/7 | -0.9855 | — |
| 3.0 | 0.1848 | 0.1855 | 0.0121 | 0.1959 | heavy_right_tail | 0.3236 | 4/7 | -0.5918 | — |
| 4.0 | 0.2757 | 0.2661 | 0.0000 | 0.2310 | heavy_right_tail | 0.3236 | 4/7 | -0.4081 | — |

### c2 (cluster, n=666, peaks_centroid=4.67, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.3252 | 0.6502 | 0.4009 | 1.5615 | heavy_right_tail | 0.2897 | 4/7 | -0.3756 | — |
| 1.0 | 0.4997 | 0.6201 | 0.1036 | 1.3815 | unclassified | 0.2897 | 4/7 | 0.0662 | — |
| 1.5 | 0.5886 | 0.5811 | 0.0315 | 1.1755 | unclassified | 0.2897 | 5/7 | 0.1882 | — |
| 2.0 | 0.6490 | 0.4820 | 0.0075 | 0.9719 | unclassified | 0.2897 | 5/7 | 0.1735 | — |
| 3.0 | 0.6269 | 0.4129 | 0.0045 | 0.8180 | unclassified | 0.2897 | 5/7 | 0.0853 | — |
| 4.0 | 0.6189 | 0.3904 | 0.0030 | 0.7522 | unclassified | 0.2897 | 5/7 | 0.0563 | — |

### agg_c0_c2 (aggregate, n=1410, peaks_centroid=2.45, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.1597 | 0.4383 | 0.5433 | 0.8362 | heavy_right_tail | 0.6133 | 3/7 | -0.8953 | — |
| 1.0 | 0.2435 | 0.3340 | 0.2106 | 0.5769 | heavy_right_tail | 0.6133 | 4/7 | -0.5831 | — |
| 1.5 | 0.2855 | 0.2922 | 0.0901 | 0.4699 | heavy_right_tail | 0.6133 | 4/7 | -0.4623 | — |
| 2.0 | 0.3141 | 0.2376 | 0.0397 | 0.3989 | heavy_right_tail | 0.6133 | 4/7 | -0.4381 | — |
| 3.0 | 0.3936 | 0.2929 | 0.0085 | 0.4907 | heavy_right_tail | 0.6133 | 4/7 | -0.2720 | — |
| 4.0 | 0.4378 | 0.3248 | 0.0014 | 0.5270 | unclassified | 0.6133 | 4/7 | -0.1888 | — |

### c1 (cluster, n=324, peaks_centroid=32.48, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.2756 | 0.6265 | 0.4074 | 1.7669 | heavy_right_tail | 0.1409 | 4/7 | -0.4552 | — |
| 1.0 | 0.4298 | 0.7377 | 0.1296 | 7.4147 | bimodal_separated | 0.1409 | 6/7 | 0.0878 | — |
| 1.5 | 0.5238 | 0.8704 | 0.0278 | 7.5796 | bimodal_separated | 0.1409 | 6/7 | 0.4164 | — |
| 2.0 | 0.5620 | 1.0000 | 0.0000 | 6.7212 | scattered | 0.1409 | 6/7 | 0.6120 | — |
| 3.0 | 0.5619 | 1.0000 | 0.0000 | 4.4808 | unclassified | 0.1409 | 7/7 | 0.6119 | **SEL** |
| 4.0 | 0.5619 | 0.9907 | 0.0000 | 3.3606 | unclassified | 0.1409 | 7/7 | 0.6026 | — |

### c3 (cluster, n=565, peaks_centroid=9.49, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.2412 | 0.6195 | 0.4690 | 1.4990 | heavy_right_tail | 0.2458 | 3/7 | -0.5584 | — |
| 1.0 | 0.3774 | 0.6088 | 0.1540 | 1.5860 | scattered | 0.2458 | 4/7 | -0.1177 | — |
| 1.5 | 0.4829 | 0.7150 | 0.0425 | 1.9765 | unclassified | 0.2458 | 6/7 | 0.2055 | — |
| 2.0 | 0.5724 | 0.8088 | 0.0106 | 2.0560 | unclassified | 0.2458 | 7/7 | 0.4206 | **SEL** |
| 3.0 | 0.5697 | 0.6885 | 0.0053 | 1.5822 | unclassified | 0.2458 | 6/7 | 0.3029 | — |
| 4.0 | 0.5655 | 0.6425 | 0.0018 | 1.3264 | unclassified | 0.2458 | 5/7 | 0.2562 | — |

### agg_c1_c3 (aggregate, n=889, peaks_centroid=17.87, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.2537 | 0.6220 | 0.4466 | 1.5731 | heavy_right_tail | 0.3867 | 4/7 | -0.5208 | — |
| 1.0 | 0.3965 | 0.6558 | 0.1451 | 2.1138 | scattered | 0.3867 | 4/7 | -0.0428 | — |
| 1.5 | 0.4978 | 0.7717 | 0.0371 | 3.1535 | scattered | 0.3867 | 5/7 | 0.2823 | — |
| 2.0 | 0.5686 | 0.8785 | 0.0067 | 3.3741 | scattered | 0.3867 | 6/7 | 0.4904 | — |
| 3.0 | 0.5668 | 0.8020 | 0.0034 | 2.5021 | unclassified | 0.3867 | 7/7 | 0.4155 | **SEL** |
| 4.0 | 0.5642 | 0.7694 | 0.0011 | 2.0321 | unclassified | 0.3867 | 7/7 | 0.3824 | — |

## Tentative label disambiguation

- **c0** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold` (pct_peak_and_collapse=0.0188)
- **c2** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold OR Peak-and-collapse (Step 4 disambiguation)` (pct_peak_and_collapse=0.4640)
- **agg_c0_c2** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold` (pct_peak_and_collapse=0.2291)
- **c1** — tentative `tentative_V-shape recovery` → final `V-shape recovery (forward-geometry weak)` (pct_peak_and_collapse=0.2438)
- **c3** — tentative `tentative_V-shape recovery` → final `V-shape recovery` (pct_peak_and_collapse=0.7186)
- **agg_c1_c3** — tentative `tentative_V-shape recovery` → final `V-shape recovery` (pct_peak_and_collapse=0.4297)

## bimodal_separated test (at selected SL, or SL=2.0 if no SL passed)

| unit | sl_ref | dip stat | p-value | min mode mass | mode separation (R) | result |
|---|---:|---:|---:|---:|---:|:---:|
| c0 | 2.0 | 0.0096 | 0.961 | 0.0000 | 0.0000 | no |
| c2 | 2.0 | 0.0137 | 0.5797 | 0.0000 | 0.0000 | no |
| agg_c0_c2 | 2.0 | 0.0062 | 0.9912 | 0.0000 | 0.0000 | no |
| c1 | 3.0 | 0.0112 | 0.9946 | 0.0000 | 0.0000 | no |
| c3 | 2.0 | 0.0093 | 0.9925 | 0.0000 | 0.0000 | no |
| agg_c1_c3 | 3.0 | 0.0059 | 0.9994 | 0.0000 | 0.0000 | no |

## Per-cluster / per-aggregate routing

| cluster | tentative | individual passes | aggregate passes | disposition | final archetype |
|---:|---|:---:|:---:|---|---|
| c0 | tentative_Early-peak hold OR Peak-and-collapse | no | no | dies | Early-peak hold |
| c2 | tentative_Early-peak hold OR Peak-and-collapse | no | no | dies | Early-peak hold OR Peak-and-collapse (Step 4 disambiguation) |
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
| c0 | 744 | 0.3236 | 4.0 | 0.2757 | 0.2661 | 0.0000 | 0.2310 | mono(0.276<0.55), mfe_p50(0.231R<1.5R), reach_1R(0.266<0.70) |
| c2 | 666 | 0.2897 | 1.5 | 0.5886 | 0.5811 | 0.0315 | 1.1755 | mfe_p50(1.176R<1.5R), reach_1R(0.581<0.70) |
| agg_c0_c2 | 1410 | 0.6133 | 4.0 | 0.4378 | 0.3248 | 0.0014 | 0.5270 | mono(0.438<0.55), mfe_p50(0.527R<1.5R), reach_1R(0.325<0.70) |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_agg_c0_c2_distribution.csv` | `0dc258f36af8673e…` | `0dc258f36af8673e…` | YES |
| `archetype_agg_c0_c2_sl_sweep.csv` | `b4fc199292ecb1d3…` | `b4fc199292ecb1d3…` | YES |
| `archetype_agg_c1_c3_distribution.csv` | `213df7541b8b1b78…` | `213df7541b8b1b78…` | YES |
| `archetype_agg_c1_c3_sl_sweep.csv` | `7d108f5cc9d1abf7…` | `7d108f5cc9d1abf7…` | YES |
| `archetype_c0_distribution.csv` | `7a7acaac5e89748b…` | `7a7acaac5e89748b…` | YES |
| `archetype_c0_sl_sweep.csv` | `9375291fdd57999e…` | `9375291fdd57999e…` | YES |
| `archetype_c1_distribution.csv` | `752cdd3f638f9dee…` | `752cdd3f638f9dee…` | YES |
| `archetype_c1_sl_sweep.csv` | `6e52e6753b35649c…` | `6e52e6753b35649c…` | YES |
| `archetype_c2_distribution.csv` | `46790b97beef7d67…` | `46790b97beef7d67…` | YES |
| `archetype_c2_sl_sweep.csv` | `78a86a0f0da71f09…` | `78a86a0f0da71f09…` | YES |
| `archetype_c3_distribution.csv` | `ea0f6f660f17827e…` | `ea0f6f660f17827e…` | YES |
| `archetype_c3_sl_sweep.csv` | `ccc8d3505b52582e…` | `ccc8d3505b52582e…` | YES |
| `archetype_summaries.csv` | `a6c806ed921de294…` | `a6c806ed921de294…` | YES |
| `capturability_pass_list.csv` | `64ffb06ec274add5…` | `64ffb06ec274add5…` | YES |
| `cluster_routing.csv` | `bc6f9206d702316f…` | `bc6f9206d702316f…` | YES |

## Files

- `results\l_arc_11\step3/archetype_agg_c0_c2_distribution.csv`
- `results\l_arc_11\step3/archetype_agg_c0_c2_sl_sweep.csv`
- `results\l_arc_11\step3/archetype_agg_c1_c3_distribution.csv`
- `results\l_arc_11\step3/archetype_agg_c1_c3_sl_sweep.csv`
- `results\l_arc_11\step3/archetype_c0_distribution.csv`
- `results\l_arc_11\step3/archetype_c0_sl_sweep.csv`
- `results\l_arc_11\step3/archetype_c1_distribution.csv`
- `results\l_arc_11\step3/archetype_c1_sl_sweep.csv`
- `results\l_arc_11\step3/archetype_c2_distribution.csv`
- `results\l_arc_11\step3/archetype_c2_sl_sweep.csv`
- `results\l_arc_11\step3/archetype_c3_distribution.csv`
- `results\l_arc_11\step3/archetype_c3_sl_sweep.csv`
- `results\l_arc_11\step3/archetype_summaries.csv`
- `results\l_arc_11\step3/capturability_pass_list.csv`
- `results\l_arc_11\step3/cluster_routing.csv`
- `results\l_arc_11\step3/STEP3_SUMMARY.md`
- `results\l_arc_11\step3/archetype_*_fwd_mfe_histogram.png`
- `results\l_arc_11\step3/archetype_*_final_r_histogram.png`
- `configs/l_arc_11/step3.yaml`
- `scripts/l_arc_11/step3_capturability.py`

## Step 3 commit
hash: _pending_

