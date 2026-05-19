# Arc 10 — Step 3 capturability characterisation summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§2, 7, 11, 17

## Verdict
**PASS** — 1 unit(s) pass §2 floors conjunctively; proceed to Step 4.

## Surviving units (passing §2 at selected SL)

| unit | type | n | size_frac | archetype | sl | R(atr) | composite | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| c1 | cluster | 228 | 0.2843 | V-shape recovery | 3.0 | 3.0 | 0.4934 | 0.5574 | 0.8860 | 0.0000 | 3.0828 | unclassified |

## SL sweep summary (per cluster + aggregate)

### c0 (cluster, n=266, peaks_centroid=0.53, tentative: tentative_Early-peak hold OR Peak-and-collapse)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.0182 | 0.2632 | 0.6579 | 0.5653 | unclassified | 0.3317 | 3/7 | -1.3266 | — |
| 1.0 | 0.0219 | 0.1015 | 0.2707 | 0.3461 | unclassified | 0.3317 | 4/7 | -1.0972 | — |
| 1.5 | 0.0282 | 0.0376 | 0.0752 | 0.2524 | unclassified | 0.3317 | 4/7 | -0.9594 | — |
| 2.0 | 0.0282 | 0.0188 | 0.0376 | 0.1932 | unclassified | 0.3317 | 4/7 | -0.9406 | — |
| 3.0 | 0.1875 | 0.2030 | 0.0150 | 0.2073 | heavy_right_tail | 0.3317 | 4/7 | -0.5745 | — |
| 4.0 | 0.2862 | 0.3045 | 0.0075 | 0.3152 | heavy_right_tail | 0.3317 | 4/7 | -0.3668 | — |

### c1 (cluster, n=228, peaks_centroid=21.92, tentative: tentative_V-shape recovery)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.2633 | 0.6009 | 0.4035 | 1.6036 | heavy_right_tail | 0.2843 | 4/7 | -0.4894 | — |
| 1.0 | 0.3911 | 0.6228 | 0.1360 | 1.7039 | scattered | 0.2843 | 4/7 | -0.0721 | — |
| 1.5 | 0.5113 | 0.7982 | 0.0263 | 4.3799 | scattered | 0.2843 | 5/7 | 0.3333 | — |
| 2.0 | 0.5593 | 0.9298 | 0.0044 | 4.4847 | scattered | 0.2843 | 6/7 | 0.5347 | — |
| 3.0 | 0.5574 | 0.8860 | 0.0000 | 3.0828 | unclassified | 0.2843 | 7/7 | 0.4934 | **SEL** |
| 4.0 | 0.5575 | 0.8553 | 0.0000 | 2.4224 | unclassified | 0.2843 | 7/7 | 0.4628 | — |

### c2 (cluster, n=308, peaks_centroid=5.56, tentative: unassigned)

| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|
| 0.5 | 0.3125 | 0.6526 | 0.4156 | 1.6173 | heavy_right_tail | 0.3840 | 4/7 | -0.4005 | — |
| 1.0 | 0.4535 | 0.6136 | 0.1104 | 1.3554 | unclassified | 0.3840 | 4/7 | 0.0067 | — |
| 1.5 | 0.5512 | 0.5714 | 0.0357 | 1.1523 | unclassified | 0.3840 | 5/7 | 0.1369 | — |
| 2.0 | 0.6056 | 0.5130 | 0.0065 | 1.0319 | unclassified | 0.3840 | 5/7 | 0.1621 | — |
| 3.0 | 0.5950 | 0.4253 | 0.0000 | 0.8586 | unclassified | 0.3840 | 5/7 | 0.0703 | — |
| 4.0 | 0.5913 | 0.4026 | 0.0000 | 0.7758 | unclassified | 0.3840 | 5/7 | 0.0439 | — |

## Tentative label disambiguation

- **c0** — tentative `tentative_Early-peak hold OR Peak-and-collapse` → final `Early-peak hold` (pct_peak_and_collapse=0.0188)
- **c1** — tentative `tentative_V-shape recovery` → final `V-shape recovery` (pct_peak_and_collapse=0.4254)
- **c2** — tentative `unassigned` → final `unassigned` (pct_peak_and_collapse=0.5065)

## bimodal_separated test (at selected SL, or SL=2.0 if no SL passed)

| unit | sl_ref | dip stat | p-value | min mode mass | mode separation (R) | result |
|---|---:|---:|---:|---:|---:|:---:|
| c0 | 2.0 | 0.0225 | 0.4809 | 0.0000 | 0.0000 | no |
| c1 | 3.0 | 0.0185 | 0.9017 | 0.0000 | 0.0000 | no |
| c2 | 2.0 | 0.0123 | 0.9927 | 0.0000 | 0.0000 | no |

## Per-cluster / per-aggregate routing

| cluster | tentative | individual passes | aggregate passes | disposition | final archetype |
|---:|---|:---:|:---:|---|---|
| c0 | tentative_Early-peak hold OR Peak-and-collapse | no | no | dies | Early-peak hold |
| c1 | tentative_V-shape recovery | yes | no | proceeds_as_individual | V-shape recovery |
| c2 | unassigned | no | no | dies | unassigned |

## Distribution detail

- See `archetype_c1_distribution.csv` for full percentiles, mass-in-band, and bimodal mode info.
- Histograms: `archetype_c1_fwd_mfe_histogram.png`, `archetype_c1_final_r_histogram.png`

## Kill reasons (per non-surviving unit)

| unit | n | size_frac | best-composite SL | best mono_pp | best reach_1R | best wrong_way_pp | best fwd_mfe_p50 | failing floors at best SL |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| c0 | 266 | 0.3317 | 4.0 | 0.2862 | 0.3045 | 0.0075 | 0.3152 | mono(0.286<0.55), mfe_p50(0.315R<1.5R), reach_1R(0.305<0.70) |
| c2 | 308 | 0.3840 | 2.0 | 0.6056 | 0.5130 | 0.0065 | 1.0319 | mfe_p50(1.032R<1.5R), reach_1R(0.513<0.70) |

## Determinism

**Gate: PASS**

| File | run 1 sha256 | run 2 sha256 | match |
|---|---|---|:---:|
| `archetype_c0_distribution.csv` | `4ebf16a87fc3c4cd…` | `4ebf16a87fc3c4cd…` | YES |
| `archetype_c0_sl_sweep.csv` | `4c7d3213aa782c2e…` | `4c7d3213aa782c2e…` | YES |
| `archetype_c1_distribution.csv` | `5922a1b883d2c6e5…` | `5922a1b883d2c6e5…` | YES |
| `archetype_c1_sl_sweep.csv` | `adf271cad3400c90…` | `adf271cad3400c90…` | YES |
| `archetype_c2_distribution.csv` | `53cb8b6e3710988a…` | `53cb8b6e3710988a…` | YES |
| `archetype_c2_sl_sweep.csv` | `7f52bb3de3383b99…` | `7f52bb3de3383b99…` | YES |
| `archetype_summaries.csv` | `447504a47e1257bf…` | `447504a47e1257bf…` | YES |
| `capturability_pass_list.csv` | `80e616251488d7bc…` | `80e616251488d7bc…` | YES |
| `cluster_routing.csv` | `46a085257b55552b…` | `46a085257b55552b…` | YES |

## Files

- `results\l_arc_10\step3/archetype_c0_distribution.csv`
- `results\l_arc_10\step3/archetype_c0_sl_sweep.csv`
- `results\l_arc_10\step3/archetype_c1_distribution.csv`
- `results\l_arc_10\step3/archetype_c1_sl_sweep.csv`
- `results\l_arc_10\step3/archetype_c2_distribution.csv`
- `results\l_arc_10\step3/archetype_c2_sl_sweep.csv`
- `results\l_arc_10\step3/archetype_summaries.csv`
- `results\l_arc_10\step3/capturability_pass_list.csv`
- `results\l_arc_10\step3/cluster_routing.csv`
- `results\l_arc_10\step3/STEP3_SUMMARY.md`
- `results\l_arc_10\step3/archetype_*_fwd_mfe_histogram.png`
- `results\l_arc_10\step3/archetype_*_final_r_histogram.png`
- `configs/l_arc_10/step3.yaml`
- `scripts/l_arc_10/step3_capturability.py`

## Step 3 commit
hash: _pending_

