# Arc 2 redo2 — Step 3 capturability characterisation summary

Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §§2, 7, 11

**Step 3 disposition: PASS**
Clusters surviving §2 capturability gate: **2 of 4**.

## Per-cluster verdict at selected SL

| cid | label_final | sel_SL | mono_pp | reach_1R | fw_pp | fwd_mfe_p50_R | shape_tag | bimodal_sep | size_frac | composite | verdict |
|---:|---|---|---|---|---|---|---|:---:|---|---|:---:|
| 0 | unclassified | (none) | 0.7169 | 0.4133 | 0.0000 | 0.8296 | heavy_right_tail | NO | 0.3482 | 0.1802 | **FAIL** |
| 1 | Stepwise climber | 3×ATR | 0.5649 | 0.9777 | 0.0000 | 3.8094 | tight_unimodal | NO | 0.1863 | 0.5926 | **PASS** |
| 2 | Early-peak hold | (none) | 0.2637 | 0.2297 | 0.0145 | 0.1025 | heavy_right_tail | NO | 0.3263 | -0.4711 | **FAIL** |
| 3 | unclassified | 2×ATR | 0.5910 | 0.7286 | 0.0000 | 1.7220 | heavy_right_tail | NO | 0.1391 | 0.3696 | **PASS** |

## Per-cluster §2 floor breakdown

### cid 0 — unclassified — best partial-pass SL = 2×ATR (5/7 floors)

| Floor | Threshold | Value | Result |
|---|---|---|:---:|
| mono_pre_peak | >= 0.55 | 0.7169 | PASS |
| frac_reach_1R | >= 0.7 | 0.4133 | FAIL |
| frac_wrong_way_pp | <= 0.3 | 0.0000 | PASS |
| fwd_mfe_p50 | >= 1.5 | 0.8296 | FAIL |
| shape_tag | in ['bimodal_separated', 'heavy_right_tail', 'tight_unimodal'] | heavy_right_tail | PASS |
| local_peaks_ceiling | per §11 (unclassified) | no ceiling for archetype 'unclassified' | PASS |
| size_fraction | >= 0.1 | 0.3482 | PASS |

### cid 1 — Stepwise climber — selected SL = 3×ATR

| Floor | Threshold | Value | Result |
|---|---|---|:---:|
| mono_pre_peak | >= 0.55 | 0.5649 | PASS |
| frac_reach_1R | >= 0.7 | 0.9777 | PASS |
| frac_wrong_way_pp | <= 0.3 | 0.0000 | PASS |
| fwd_mfe_p50 | >= 1.5 | 3.8094 | PASS |
| shape_tag | in ['bimodal_separated', 'heavy_right_tail', 'tight_unimodal'] | tight_unimodal | PASS |
| local_peaks_ceiling | per §11 (Stepwise climber) | 20.07 <= 30 | PASS |
| size_fraction | >= 0.1 | 0.1863 | PASS |

### cid 2 — Early-peak hold — best partial-pass SL = 4×ATR (4/7 floors)

| Floor | Threshold | Value | Result |
|---|---|---|:---:|
| mono_pre_peak | >= 0.55 | 0.2637 | FAIL |
| frac_reach_1R | >= 0.7 | 0.2297 | FAIL |
| frac_wrong_way_pp | <= 0.3 | 0.0145 | PASS |
| fwd_mfe_p50 | >= 1.5 | 0.1025 | FAIL |
| shape_tag | in ['bimodal_separated', 'heavy_right_tail', 'tight_unimodal'] | heavy_right_tail | PASS |
| local_peaks_ceiling | per §11 (Early-peak hold OR Peak-and-collapse) | no ceiling for archetype 'Early-peak hold' | PASS |
| size_fraction | >= 0.1 | 0.3263 | PASS |

### cid 3 — unclassified — selected SL = 2×ATR

| Floor | Threshold | Value | Result |
|---|---|---|:---:|
| mono_pre_peak | >= 0.55 | 0.5910 | PASS |
| frac_reach_1R | >= 0.7 | 0.7286 | PASS |
| frac_wrong_way_pp | <= 0.3 | 0.0000 | PASS |
| fwd_mfe_p50 | >= 1.5 | 1.7220 | PASS |
| shape_tag | in ['bimodal_separated', 'heavy_right_tail', 'tight_unimodal'] | heavy_right_tail | PASS |
| local_peaks_ceiling | per §11 (unclassified) | no ceiling for archetype 'unclassified' | PASS |
| size_fraction | >= 0.1 | 0.1391 | PASS |

## SL sweep — composite + floors-failed per cluster

### cid 0

| X (×ATR) | mono_pp | reach_1R | fw_pp | fwd_mfe_p50_R | shape_tag | composite | floors_failed | overall |
|---|---|---|---|---|---|---|---|:---:|
| 0.5 | 0.3527 | 0.4763 | 0.4040 | 0.8763 | heavy_right_tail | -0.5249 | mono_pre_peak,frac_reach_1R,frac_wrong_way_pp,fwd_mfe_p50 | FAIL |
| 1 | 0.5577 | 0.5222 | 0.1115 | 1.0796 | heavy_right_tail | 0.0185 | frac_reach_1R,fwd_mfe_p50 | FAIL |
| 1.5 | 0.6631 | 0.4874 | 0.0279 | 0.9630 | heavy_right_tail | 0.1726 | frac_reach_1R,fwd_mfe_p50 | FAIL |
| 2 | 0.7169 | 0.4133 | 0.0000 | 0.8296 | heavy_right_tail | 0.1802 | frac_reach_1R,fwd_mfe_p50 | FAIL |
| 3 | 0.6875 | 0.3698 | 0.0009 | 0.6891 | heavy_right_tail | 0.1063 | frac_reach_1R,fwd_mfe_p50 | FAIL |
| 4 | 0.6673 | 0.3607 | 0.0005 | 0.6570 | heavy_right_tail | 0.0774 | frac_reach_1R,fwd_mfe_p50 | FAIL |

### cid 1

| X (×ATR) | mono_pp | reach_1R | fw_pp | fwd_mfe_p50_R | shape_tag | composite | floors_failed | overall |
|---|---|---|---|---|---|---|---|:---:|
| 0.5 | 0.3004 | 0.5033 | 0.4214 | 1.0282 | heavy_right_tail | -0.5678 | mono_pre_peak,frac_reach_1R,frac_wrong_way_pp,fwd_mfe_p50 | FAIL |
| 1 | 0.4463 | 0.6661 | 0.1278 | 4.9359 | tight_unimodal | 0.0346 | mono_pre_peak,frac_reach_1R | FAIL |
| 1.5 | 0.5208 | 0.8254 | 0.0372 | 6.1690 | tight_unimodal | 0.3590 | mono_pre_peak | FAIL |
| 2 | 0.5661 | 0.9930 | 0.0000 | 5.6335 | tight_unimodal | 0.6091 | (none) | PASS |
| 3 | 0.5649 | 0.9777 | 0.0000 | 3.8094 | tight_unimodal | 0.5926 | (none) | PASS |
| 4 | 0.5644 | 0.9505 | 0.0000 | 2.8698 | tight_unimodal | 0.5649 | (none) | PASS |

### cid 2

| X (×ATR) | mono_pp | reach_1R | fw_pp | fwd_mfe_p50_R | shape_tag | composite | floors_failed | overall |
|---|---|---|---|---|---|---|---|:---:|
| 0.5 | 0.0158 | 0.0935 | 0.7223 | -0.8830 | no_magnitude | -1.5631 | mono_pre_peak,frac_reach_1R,frac_wrong_way_pp,fwd_mfe_p50,shape_tag | FAIL |
| 1 | 0.0213 | 0.0405 | 0.4091 | -0.3538 | no_magnitude | -1.2974 | mono_pre_peak,frac_reach_1R,frac_wrong_way_pp,fwd_mfe_p50,shape_tag | FAIL |
| 1.5 | 0.0215 | 0.0135 | 0.2099 | -0.1659 | no_magnitude | -1.1249 | mono_pre_peak,frac_reach_1R,fwd_mfe_p50,shape_tag | FAIL |
| 2 | 0.0215 | 0.0065 | 0.1037 | -0.1017 | no_magnitude | -1.0257 | mono_pre_peak,frac_reach_1R,fwd_mfe_p50,shape_tag | FAIL |
| 3 | 0.1780 | 0.1562 | 0.0335 | 0.0307 | no_magnitude | -0.6493 | mono_pre_peak,frac_reach_1R,fwd_mfe_p50,shape_tag | FAIL |
| 4 | 0.2637 | 0.2297 | 0.0145 | 0.1025 | heavy_right_tail | -0.4711 | mono_pre_peak,frac_reach_1R,fwd_mfe_p50 | FAIL |

### cid 3

| X (×ATR) | mono_pp | reach_1R | fw_pp | fwd_mfe_p50_R | shape_tag | composite | floors_failed | overall |
|---|---|---|---|---|---|---|---|:---:|
| 0.5 | 0.3032 | 0.4713 | 0.4472 | 0.8430 | heavy_right_tail | -0.6228 | mono_pre_peak,frac_reach_1R,frac_wrong_way_pp,fwd_mfe_p50 | FAIL |
| 1 | 0.4420 | 0.5363 | 0.1501 | 1.2175 | heavy_right_tail | -0.1217 | mono_pre_peak,frac_reach_1R,fwd_mfe_p50 | FAIL |
| 1.5 | 0.5233 | 0.6547 | 0.0399 | 1.7006 | heavy_right_tail | 0.1882 | mono_pre_peak,frac_reach_1R | FAIL |
| 2 | 0.5910 | 0.7286 | 0.0000 | 1.7220 | heavy_right_tail | 0.3696 | (none) | PASS |
| 3 | 0.5812 | 0.6413 | 0.0023 | 1.3881 | heavy_right_tail | 0.2701 | frac_reach_1R,fwd_mfe_p50 | FAIL |
| 4 | 0.5774 | 0.5721 | 0.0000 | 1.2198 | heavy_right_tail | 0.1995 | frac_reach_1R,fwd_mfe_p50 | FAIL |

## cid 2 — §11 disambiguation (Early-peak hold vs Peak-and-collapse)

- `pct_peak_and_collapse` (full-window, prior definition): **0.2959**
- Disambiguation branch: **Early-peak hold**
- Thresholds: < 0.3 → Early-peak hold; ≥ 0.5 → Peak-and-collapse; in between → empirical-defer

## cid 1 — Stepwise climber pct_pc (informational; not gated)

- `pct_peak_and_collapse` (full-window): 0.3278
- Used as §11 row 1 exit-policy information; not a §2 gate input.

## Determinism

**Gate: PASS**

Run-1 CSV hashes:
- `archetype_summaries.csv`: `eb113c492a7db084c2ccd53c83969123cf75f9808811c202342ba794d7f71f63`
- `capturability_pass_list.csv`: `930818dd2228cc1ac2e1e1d9b97f67ab49d4d1228f555c5ae02d4d30d97657a1`
- `cluster_0_distribution.csv`: `c0bd1cb6bb0a696a2bcdc87ce873644ea4986edfdeec7112d09489edc9f43b4a`
- `cluster_0_sl_sweep_detail.csv`: `3f70e6f4aaa8c17dbc050ca34642c8fdcaede6a616cfd90a27779d82ad9dcce5`
- `cluster_1_distribution.csv`: `2354d9bf93f818b83afba23af162a6fba063c262e72fab6e5c00f9d7f5aa372e`
- `cluster_1_sl_sweep_detail.csv`: `b485b614abc6230a8bda3a819fa45dd7380272ee8ff50b879802dcff53b887ed`
- `cluster_2_distribution.csv`: `dd74099063beb773f04ae455934554bf85f4f0bdb4ba62196e6365a23ac85ca2`
- `cluster_2_sl_sweep_detail.csv`: `8441b52008cd0192b66cd6995904418ba6db36187e144e13941ecdbff9ea5ec5`
- `cluster_3_distribution.csv`: `96829aab01460268279d18bdcd8609192dd2108a1a6b2e9f293577eae5ac017a`
- `cluster_3_sl_sweep_detail.csv`: `1f3a4d324768eed77f4c89df65ce58b8a15afde9c1eb1312f404b1a299c6284e`
- `cluster_routing.csv`: `41c0f496d3a83ace86af21098f79a70c14fc078ebd20b8bc0cd3b2061ae80fd5`
- `cluster_sl_sweep.csv`: `08ed2527984b9128ab9d544c45f45648ba3686f7a1cf974ce4e887fb934a2226`
- `per_trade_sl_results.csv`: `d0faadb607da2c5d95d0e9bedae5fd397c17ab89648ad369d0951a910eec8281`
- `shape_tag_diagnostics.csv`: `474973f06af9d460fd24a0697c331758e4bf0d038a4971b36be6eb5e33221438`

Run-2 CSV hashes:
- `archetype_summaries.csv`: `eb113c492a7db084c2ccd53c83969123cf75f9808811c202342ba794d7f71f63` (MATCH)
- `capturability_pass_list.csv`: `930818dd2228cc1ac2e1e1d9b97f67ab49d4d1228f555c5ae02d4d30d97657a1` (MATCH)
- `cluster_0_distribution.csv`: `c0bd1cb6bb0a696a2bcdc87ce873644ea4986edfdeec7112d09489edc9f43b4a` (MATCH)
- `cluster_0_sl_sweep_detail.csv`: `3f70e6f4aaa8c17dbc050ca34642c8fdcaede6a616cfd90a27779d82ad9dcce5` (MATCH)
- `cluster_1_distribution.csv`: `2354d9bf93f818b83afba23af162a6fba063c262e72fab6e5c00f9d7f5aa372e` (MATCH)
- `cluster_1_sl_sweep_detail.csv`: `b485b614abc6230a8bda3a819fa45dd7380272ee8ff50b879802dcff53b887ed` (MATCH)
- `cluster_2_distribution.csv`: `dd74099063beb773f04ae455934554bf85f4f0bdb4ba62196e6365a23ac85ca2` (MATCH)
- `cluster_2_sl_sweep_detail.csv`: `8441b52008cd0192b66cd6995904418ba6db36187e144e13941ecdbff9ea5ec5` (MATCH)
- `cluster_3_distribution.csv`: `96829aab01460268279d18bdcd8609192dd2108a1a6b2e9f293577eae5ac017a` (MATCH)
- `cluster_3_sl_sweep_detail.csv`: `1f3a4d324768eed77f4c89df65ce58b8a15afde9c1eb1312f404b1a299c6284e` (MATCH)
- `cluster_routing.csv`: `41c0f496d3a83ace86af21098f79a70c14fc078ebd20b8bc0cd3b2061ae80fd5` (MATCH)
- `cluster_sl_sweep.csv`: `08ed2527984b9128ab9d544c45f45648ba3686f7a1cf974ce4e887fb934a2226` (MATCH)
- `per_trade_sl_results.csv`: `d0faadb607da2c5d95d0e9bedae5fd397c17ab89648ad369d0951a910eec8281` (MATCH)
- `shape_tag_diagnostics.csv`: `474973f06af9d460fd24a0697c331758e4bf0d038a4971b36be6eb5e33221438` (MATCH)

