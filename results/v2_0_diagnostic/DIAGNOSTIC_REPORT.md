# v2.0 archetype clustering diagnostic — path-shape archetypes

> Diagnostic only. No protocol revision, no floor-setting, no verdicts.
> Inputs: KH-24 (553 trades), Arc 1 (45673 trades), Arc 2 (3993 trades).
> All three are weak inputs by acknowledgement; the diagnostic gathers evidence
> against the question: does path-shape clustering identify clean+meaty+predictable
> archetypes that the existing dual-gate clustering missed.

## §1 Read-first verification

- **kh24**: trades = 553, path-shape features computed via `scripts/v2_0_diagnostic/path_features.py`. Loader reused from `scripts/v1_3_calibration/load_paths.py`; R-unit convention (R = 2 x ATR) preserved.
- **arc1**: trades = 45673, path-shape features computed via `scripts/v2_0_diagnostic/path_features.py`. Loader reused from `scripts/v1_3_calibration/load_paths.py`; R-unit convention (R = 2 x ATR) preserved.
- **arc2**: trades = 3993, path-shape features computed via `scripts/v2_0_diagnostic/path_features.py`. Loader reused from `scripts/v1_3_calibration/load_paths.py`; R-unit convention (R = 2 x ATR) preserved.

Loader normalisation decisions reused verbatim. Arc 1 close_r is derived from fwd_logret_cum (no per-bar OHLC); Arc 2 + KH-24 use per-bar OHLC + running mfe/mae. None of the path-shape features depend on intrabar high/low — all computed from close_r and mfe_so_far_r. Arc 1 path-shape features are thus exact under the same close-based approximation as v1.3 axis 2g.

**Spec deviation flag — pullback_magnitude_median:** the spec text reads `mfe_so_far_r at earlier peak - min(mfe_so_far_r over bars between)`, but `mfe_so_far_r` is the per-bar running maximum and therefore non-decreasing, so the literal expression is identically 0 across every trade. Implementation uses `min(close_r)` for the inter-peak trough (close_r is the per-bar mark and does retrace), preserving the spec's intent that the feature captures pullback magnitude between local peaks. Flagged here for chat audit.

**Scope flag — Arc 1 held window:** Arc 1 trades use a 1H time exit; `bars_held = 1` for 96% of trades. The spec scopes path-shape features to `[0, bars_held]`, so Arc 1 features are computed over a 1-2 bar window and are systemically degenerate (see §2). The diagnostic surfaces this rather than re-scoping; chat may want to re-run with a forward-window scope.

## §2 Feature distribution summary

### kh24

| feature | p5 | p25 | p50 | p75 | p95 | mean | std | cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monotonicity_ratio_in_profit | 0 | 0 | 0.4706 | 0.6 | 1 | 0.3767 | 0.3187 | 0.8461 |
| local_peaks_count | 0 | 1 | 2 | 6 | 13 | 3.9711 | 5.1199 | 1.2893 |
| pullback_magnitude_median | 0 | 0 | 0 | 0 | 0.3697 | 0.0525 | 0.1466 | 2.7922 |
| time_to_peak_mfe_relative | 0 | 0.0769 | 0.5 | 0.7778 | 1 | 0.4667 | 0.3535 | 0.7574 |

**Degenerate-distribution flag (>80% trades at single value):**

- `pullback_magnitude_median` — top value 0.0000 shared by 81.6% of trades

### arc1

| feature | p5 | p25 | p50 | p75 | p95 | mean | std | cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monotonicity_ratio_in_profit | 0 | 0 | 0 | 1 | 1 | 0.2589 | 0.4377 | 1.6904 |
| local_peaks_count | 0 | 0 | 0 | 1 | 1 | 0.4687 | 0.6007 | 1.2817 |
| pullback_magnitude_median | 0 | 0 | 0 | 0 | 0 | 0.0002 | 0.0097 | 52.5205 |
| time_to_peak_mfe_relative | 0 | 0 | 0 | 1 | 1 | 0.4585 | 0.4981 | 1.0863 |

**Degenerate-distribution flag (>80% trades at single value):**

- `pullback_magnitude_median` — top value 0.0000 shared by 99.9% of trades

### arc2

| feature | p5 | p25 | p50 | p75 | p95 | mean | std | cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monotonicity_ratio_in_profit | 0 | 0 | 0.494 | 0.5385 | 0.6667 | 0.3791 | 0.2629 | 0.6936 |
| local_peaks_count | 0 | 1 | 4 | 12 | 29 | 7.9088 | 9.5701 | 1.21 |
| pullback_magnitude_median | 0 | 0 | 0 | 0 | 0.4294 | 0.0725 | 0.1966 | 2.7117 |
| time_to_peak_mfe_relative | 0 | 0.0769 | 0.3707 | 0.6845 | 0.9821 | 0.4039 | 0.3347 | 0.8287 |

## §3 Clustering separability (silhouette by K x dataset)

| K | kh24 | arc1 | arc2 |
| --- | --- | --- | --- |
| 3 | 0.4792 | 0.7673 | 0.4056 |
| 4 | 0.4529 | 0.9299 | 0.4615 |
| 5 | 0.4375 | 0.9343 | 0.4479 |
| 6 | 0.4532 | 0.9972 | 0.4470 |
| 7 | 0.4634 | 0.9977 | 0.4600 |

No failed-clustering flags raised.

## §4 Archetype profiles (per dataset, per K)

### kh24

#### K = 3

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Flat-line | 224 | 0.4051 | 0.0813 | 0.5848 | 0.001 | 0.1241 | 2.6716 | 0.75 | 0.5848 | 0.6696 | 0.6696 | -0.6443 | -18.6374 | bimodal |
| 1 | Stepwise climber | 278 | 0.5027 | 0.5965 | 6.6906 | 0.0208 | 0.7 | 3.9179 | 0.8921 | 0.759 | 0.3201 | 0.2194 | 0.7356 | 7.9125 | bimodal |
| 2 | Unlabelled | 51 | 0.0922 | 0.4756 | 4.0196 | 0.4519 | 0.6991 | 2.7196 | 0.7647 | 0.6078 | 0.4902 | 0.3529 | -0.1471 | -1.3426 | bimodal |

#### K = 4

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Flat-line | 211 | 0.3816 | 0.0541 | 0.5261 | 0.001 | 0.1194 | 2.6094 | 0.7441 | 0.5735 | 0.6777 | 0.6682 | -0.6583 | -19.1974 | bimodal |
| 1 | Monotone ascent | 219 | 0.396 | 0.5995 | 3.863 | 0.022 | 0.621 | 3.1089 | 0.8539 | 0.6941 | 0.4247 | 0.3151 | 0.11 | 1.9025 | bimodal |
| 2 | Unlabelled | 49 | 0.0886 | 0.4696 | 3.8571 | 0.4596 | 0.6973 | 2.6848 | 0.7551 | 0.5918 | 0.5102 | 0.3469 | -0.154 | -1.3528 | bimodal |
| 3 | Stepwise climber | 74 | 0.1338 | 0.5756 | 14.1892 | 0.0201 | 0.8473 | 5.4034 | 1 | 0.9595 | 0.0405 | 0.027 | 2.3653 | 11.0161 | bimodal |

#### K = 5

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Monotone ascent | 93 | 0.1682 | 0.6434 | 2.3656 | 0.0112 | 0.2877 | 3.0617 | 0.7957 | 0.6774 | 0.5699 | 0.4946 | -0.2952 | -4.0468 | bimodal |
| 1 | Flat-line | 189 | 0.3418 | 0.0184 | 0.4286 | 0.0011 | 0.1061 | 2.3666 | 0.7302 | 0.5608 | 0.6931 | 0.6984 | -0.7038 | -21.5315 | bimodal |
| 2 | Unlabelled | 50 | 0.0904 | 0.4735 | 3.86 | 0.4553 | 0.6964 | 2.7022 | 0.76 | 0.6 | 0.5 | 0.36 | -0.1536 | -1.3763 | bimodal |
| 3 | Stepwise climber | 68 | 0.123 | 0.5809 | 14.6471 | 0.0218 | 0.8503 | 5.2596 | 1 | 0.9706 | 0.0441 | 0.0294 | 2.3728 | 10.3873 | bimodal |
| 4 | Unlabelled | 153 | 0.2767 | 0.5347 | 4.6144 | 0.0231 | 0.7753 | 3.1932 | 0.8954 | 0.7059 | 0.3399 | 0.2026 | 0.3887 | 5.0141 | bimodal |

#### K = 6

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Unlabelled | 47 | 0.085 | 0.4896 | 3.9149 | 0.4685 | 0.6969 | 2.4977 | 0.7447 | 0.5745 | 0.5106 | 0.3191 | -0.1375 | -1.166 | bimodal |
| 1 | Monotone ascent | 94 | 0.17 | 0.6277 | 2.3085 | 0.0171 | 0.2862 | 2.879 | 0.7872 | 0.6596 | 0.5532 | 0.4894 | -0.3078 | -4.2632 | bimodal |
| 2 | Stepwise climber | 67 | 0.1212 | 0.5804 | 14.7164 | 0.0222 | 0.8525 | 5.3845 | 1 | 0.9701 | 0.0448 | 0.0299 | 2.3949 | 10.3775 | bimodal |
| 3 | Flat-line | 40 | 0.0723 | 0.0238 | 1.7 | 0.0126 | 0.6991 | 3.0891 | 0.775 | 0.625 | 0.575 | 0.575 | -0.3434 | -4.0025 | bimodal |
| 4 | Monotone ascent | 138 | 0.2495 | 0.6084 | 5.0362 | 0.0248 | 0.7558 | 3.2644 | 0.913 | 0.7319 | 0.3043 | 0.1884 | 0.4685 | 5.7265 | bimodal |
| 5 | Flat-line | 167 | 0.302 | 0.015 | 0.2754 | 0 | 0.0541 | 2.3179 | 0.7305 | 0.5569 | 0.7186 | 0.7006 | -0.7383 | -22.0482 | bimodal |

#### K = 7

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Flat-line | 42 | 0.0759 | 0.0385 | 1.7619 | 0.0153 | 0.6896 | 2.9322 | 0.7857 | 0.619 | 0.5476 | 0.5714 | -0.3151 | -3.6009 | bimodal |
| 1 | Monotone ascent | 136 | 0.2459 | 0.6102 | 5.1912 | 0.0154 | 0.7587 | 3.3013 | 0.9118 | 0.7279 | 0.2868 | 0.1618 | 0.5156 | 6.3187 | bimodal |
| 2 | Unlabelled | 13 | 0.0235 | 0.436 | 2.8462 | 0.749 | 0.6415 | 1.8474 | 0.6923 | 0.4615 | 0.6154 | 0.3846 | -0.4348 | -2.3366 | bimodal |
| 3 | Stepwise climber | 61 | 0.1103 | 0.5797 | 15.1475 | 0.0162 | 0.8584 | 5.1346 | 1 | 0.9672 | 0.0492 | 0.0328 | 2.5213 | 10.2767 | bimodal |
| 4 | Flat-line | 167 | 0.302 | 0.015 | 0.2754 | 0 | 0.0541 | 2.3179 | 0.7305 | 0.5569 | 0.7186 | 0.7006 | -0.7383 | -22.0482 | bimodal |
| 5 | Unlabelled | 46 | 0.0832 | 0.5139 | 4.5652 | 0.324 | 0.6923 | 2.8759 | 0.8043 | 0.6739 | 0.4565 | 0.3478 | -0.0023 | -0.0179 | bimodal |
| 6 | Monotone ascent | 88 | 0.1591 | 0.6424 | 2.2614 | 0.0076 | 0.2765 | 3.0372 | 0.7841 | 0.6705 | 0.5682 | 0.4886 | -0.3289 | -4.4658 | bimodal |

### arc1

#### K = 3

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Flat-line | 24704 | 0.5409 | 0.0717 | 0.0002 | 0 | 0 | 2.9193 | 0.803 | 0.6398 | 0.6005 | 0.5875 | -0.2895 | -96.1788 | bimodal |
| 1 | Unlabelled | 20955 | 0.4588 | 0.4795 | 1.018 | 0.0001 | 0.9989 | 3.5429 | 0.8988 | 0.7363 | 0.373 | 0.1329 | 0.3517 | 117.0661 | bimodal |
| 2 | Unlabelled | 14 | 0.0003 | 0.5489 | 5 | 0.4981 | 0.6429 | 1.9504 | 0.9286 | 0.4286 | 0.5 | 0.5714 | -0.3157 | -0.9884 | bimodal |

#### K = 4

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Monotone ascent | 10061 | 0.2203 | 0.9983 | 1.035 | 0.0001 | 0.9985 | 3.741 | 0.93 | 0.7717 | 0.2884 | 0.0015 | 0.5923 | 147.8554 | bimodal |
| 1 | Flat-line | 24704 | 0.5409 | 0.0717 | 0.0002 | 0 | 0 | 2.9193 | 0.803 | 0.6398 | 0.6005 | 0.5875 | -0.2895 | -96.1788 | bimodal |
| 2 | Unlabelled | 14 | 0.0003 | 0.5489 | 5 | 0.4981 | 0.6429 | 1.9504 | 0.9286 | 0.4286 | 0.5 | 0.5714 | -0.3157 | -0.9884 | bimodal |
| 3 | Flat-line | 10894 | 0.2385 | 0.0003 | 1.0023 | 0 | 0.9993 | 3.3675 | 0.8699 | 0.7036 | 0.4511 | 0.2542 | 0.1295 | 40.4115 | bimodal |

#### K = 5

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Monotone ascent | 10034 | 0.2197 | 0.9996 | 1.0037 | 0.0001 | 0.9992 | 3.7439 | 0.9298 | 0.7714 | 0.2889 | 0.0007 | 0.5913 | 150.663 | bimodal |
| 1 | Flat-line | 24704 | 0.5409 | 0.0717 | 0.0002 | 0 | 0 | 2.9193 | 0.803 | 0.6398 | 0.6005 | 0.5875 | -0.2895 | -96.1788 | bimodal |
| 2 | Flat-line | 10895 | 0.2385 | 0.0004 | 1.0028 | 0 | 0.9992 | 3.3673 | 0.8699 | 0.7036 | 0.4511 | 0.2542 | 0.1294 | 40.3398 | bimodal |
| 3 | Stepwise climber | 28 | 0.0006 | 0.5398 | 12.3929 | 0.0295 | 0.7434 | 3.2756 | 1 | 0.8214 | 0.1071 | 0.3214 | 0.82 | 2.5171 | bimodal |
| 4 | Unlabelled | 12 | 0.0003 | 0.5154 | 4.9167 | 0.5382 | 0.6973 | 2.0526 | 0.9167 | 0.5 | 0.5 | 0.5 | -0.0679 | -0.2183 | bimodal |

#### K = 6

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Monotone ascent | 10033 | 0.2197 | 0.9996 | 1.0035 | 0 | 0.9993 | 3.7444 | 0.9298 | 0.7715 | 0.2888 | 0.0006 | 0.5916 | 151.1513 | bimodal |
| 1 | Flat-line | 22934 | 0.5021 | 0 | 0.0002 | 0 | 0 | 2.8897 | 0.7961 | 0.6335 | 0.6163 | 0.6329 | -0.3372 | -112.9664 | bimodal |
| 2 | Flat-line | 10895 | 0.2385 | 0.0004 | 1.0028 | 0 | 0.9992 | 3.3673 | 0.8699 | 0.7036 | 0.4511 | 0.2542 | 0.1294 | 40.3398 | bimodal |
| 3 | Monotone ascent | 13 | 0.0003 | 0.5527 | 4.7692 | 0.5173 | 0.6578 | 1.9971 | 0.9231 | 0.4615 | 0.5385 | 0.5385 | -0.245 | -0.7282 | bimodal |
| 4 | Stepwise climber | 28 | 0.0006 | 0.5398 | 12.3929 | 0.0295 | 0.7434 | 3.2756 | 1 | 0.8214 | 0.1071 | 0.3214 | 0.82 | 2.5171 | bimodal |
| 5 | Monotone ascent | 1770 | 0.0388 | 1 | -0 | -0 | 0 | 3.3341 | 0.8932 | 0.7215 | 0.3949 | 0 | 0.3281 | 53.7159 | bimodal |

#### K = 7

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Flat-line | 22934 | 0.5021 | 0 | 0.0002 | 0 | 0 | 2.8897 | 0.7961 | 0.6335 | 0.6163 | 0.6329 | -0.3372 | -112.9664 | bimodal |
| 1 | Flat-line | 10894 | 0.2385 | 0.0003 | 1.0027 | 0 | 0.9993 | 3.3675 | 0.8699 | 0.7037 | 0.4512 | 0.2542 | 0.1294 | 40.3637 | bimodal |
| 2 | Monotone ascent | 10032 | 0.2196 | 0.9996 | 1.0037 | 0 | 0.9994 | 3.7457 | 0.9298 | 0.7715 | 0.2888 | 0.0005 | 0.5918 | 151.1703 | bimodal |
| 3 | Unlabelled | 5 | 0.0001 | 0.5355 | 3.8 | 0.7554 | 0.8286 | 3.0687 | 1 | 0.6 | 0.4 | 0.2 | 0.6551 | 1.7036 | unclassified |
| 4 | Monotone ascent | 14 | 0.0003 | 0.5754 | 5.2143 | 0.2938 | 0.498 | 1.9384 | 0.9286 | 0.4286 | 0.5 | 0.7143 | -0.567 | -2.153 | unclassified |
| 5 | Monotone ascent | 1770 | 0.0388 | 1 | -0 | -0 | 0 | 3.3341 | 0.8932 | 0.7215 | 0.3949 | 0 | 0.3281 | 53.7159 | bimodal |
| 6 | Stepwise climber | 24 | 0.0005 | 0.5429 | 13.25 | 0.0101 | 0.7542 | 3.0381 | 1 | 0.8333 | 0.0833 | 0.2917 | 0.9314 | 2.5751 | bimodal |

### arc2

#### K = 3

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Stepwise climber | 772 | 0.1933 | 0.535 | 24.2107 | 0.0314 | 0.7882 | 6.3178 | 1 | 0.9896 | 0.0142 | 0.0764 | 3.9082 | 37.1842 | bimodal |
| 1 | Flat-line | 1236 | 0.3095 | 0.0274 | 0.4563 | 0.004 | 0.1204 | 2.462 | 0.7104 | 0.5615 | 0.8277 | 0.6966 | -1.0022 | -70.787 | bimodal |
| 2 | Stepwise climber | 1985 | 0.4971 | 0.5374 | 6.2364 | 0.131 | 0.4315 | 2.8605 | 0.8801 | 0.6741 | 0.4615 | 0.6816 | -0.7146 | -22.1828 | bimodal |

#### K = 4

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Unlabelled | 310 | 0.0776 | 0.4857 | 5.4065 | 0.6156 | 0.6558 | 2.7856 | 0.8742 | 0.6613 | 0.5097 | 0.629 | -0.4836 | -5.3285 | bimodal |
| 1 | Stepwise climber | 1739 | 0.4355 | 0.5399 | 6.3243 | 0.0402 | 0.3787 | 2.8604 | 0.8781 | 0.6722 | 0.4583 | 0.6901 | -0.7499 | -22.0598 | bimodal |
| 2 | Stepwise climber | 765 | 0.1916 | 0.5355 | 24.0993 | 0.0328 | 0.7976 | 6.3193 | 1 | 0.9895 | 0.0183 | 0.0758 | 3.9108 | 37.0438 | bimodal |
| 3 | Flat-line | 1179 | 0.2953 | 0.0123 | 0.3986 | 0.0031 | 0.1194 | 2.48 | 0.7074 | 0.564 | 0.8321 | 0.6955 | -0.9989 | -69.0514 | bimodal |

#### K = 5

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Flat-line | 1114 | 0.279 | 0.0052 | 0.3429 | 0.0031 | 0.0915 | 2.4893 | 0.7083 | 0.5691 | 0.833 | 0.7002 | -1.0065 | -70.2698 | bimodal |
| 1 | Stepwise climber | 621 | 0.1555 | 0.5371 | 26.496 | 0.0275 | 0.7888 | 6.8591 | 1 | 0.9984 | 0.0081 | 0.0467 | 4.4535 | 38.6658 | bimodal |
| 2 | Stepwise climber | 822 | 0.2059 | 0.5229 | 7.6129 | 0.1043 | 0.7259 | 3.2765 | 0.8881 | 0.7324 | 0.4197 | 0.562 | -0.1429 | -2.6174 | bimodal |
| 3 | Unlabelled | 195 | 0.0488 | 0.4766 | 4.6359 | 0.7656 | 0.6188 | 2.5829 | 0.8615 | 0.6 | 0.5538 | 0.6718 | -0.6702 | -6.345 | bimodal |
| 4 | Stepwise climber | 1241 | 0.3108 | 0.525 | 6.0805 | 0.0275 | 0.2449 | 2.8031 | 0.8775 | 0.6632 | 0.4545 | 0.7019 | -0.835 | -19.5676 | bimodal |

#### K = 6

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Flat-line | 1115 | 0.2792 | 0.0052 | 0.3417 | 0.0021 | 0.0933 | 2.4818 | 0.7076 | 0.5677 | 0.8341 | 0.6996 | -1.0057 | -70.267 | bimodal |
| 1 | Unlabelled | 1151 | 0.2883 | 0.5195 | 5.9679 | 0.0227 | 0.2315 | 2.7431 | 0.8723 | 0.6577 | 0.4622 | 0.7003 | -0.8596 | -19.3901 | bimodal |
| 2 | Stepwise climber | 604 | 0.1513 | 0.5371 | 26.7815 | 0.0293 | 0.7956 | 6.9044 | 1 | 0.9983 | 0.0083 | 0.0397 | 4.5318 | 39.1018 | bimodal |
| 3 | Unlabelled | 54 | 0.0135 | 0.4549 | 3.5926 | 1.1939 | 0.7004 | 2.3677 | 0.9074 | 0.6111 | 0.6111 | 0.6481 | -0.6047 | -3.8161 | bimodal |
| 4 | Stepwise climber | 349 | 0.0874 | 0.4962 | 6.0774 | 0.4398 | 0.6046 | 3.0194 | 0.8768 | 0.6905 | 0.4699 | 0.6361 | -0.4839 | -5.4696 | bimodal |
| 5 | Stepwise climber | 720 | 0.1803 | 0.5385 | 8.1097 | 0.0352 | 0.7125 | 3.332 | 0.8958 | 0.7347 | 0.3972 | 0.5639 | -0.0902 | -1.5499 | bimodal |

#### K = 7

| archetype_id | label | size_count | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | fwd_mfe_h240_p50 | frac_reach_1R | frac_reach_2R | frac_wrong_way | pct_peak_and_collapse | final_r_mean | final_r_t_stat | shape_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Stepwise climber | 762 | 0.1908 | 0.5659 | 9.2 | 0.0311 | 0.6665 | 3.5271 | 0.9213 | 0.7756 | 0.3346 | 0.5643 | -0.0385 | -0.629 | bimodal |
| 1 | Flat-line | 205 | 0.0513 | 0.0101 | 1.5122 | 0.0248 | 0.607 | 2.6748 | 0.7317 | 0.5707 | 0.8244 | 0.7073 | -0.9304 | -29.4391 | scattered |
| 2 | Stepwise climber | 585 | 0.1465 | 0.5378 | 27.072 | 0.0292 | 0.8027 | 6.9714 | 1 | 1 | 0.0085 | 0.0308 | 4.6373 | 40.1283 | bimodal |
| 3 | Unlabelled | 84 | 0.021 | 0.4723 | 4.0595 | 1.0409 | 0.6536 | 2.6172 | 0.9405 | 0.631 | 0.5357 | 0.7024 | -0.6591 | -4.851 | bimodal |
| 4 | Stepwise climber | 345 | 0.0864 | 0.504 | 6.342 | 0.3946 | 0.5989 | 2.9832 | 0.8667 | 0.6928 | 0.4638 | 0.6087 | -0.413 | -4.5104 | bimodal |
| 5 | Unlabelled | 1058 | 0.265 | 0.5176 | 5.4588 | 0.0187 | 0.2158 | 2.5917 | 0.8629 | 0.6361 | 0.4924 | 0.7098 | -0.9225 | -20.95 | bimodal |
| 6 | Flat-line | 954 | 0.2389 | 0.0047 | 0.1572 | 0.0002 | 0.0219 | 2.4713 | 0.7013 | 0.5639 | 0.8333 | 0.6918 | -1.0051 | -64.0386 | bimodal |

## §5 Forward geometry per archetype

Forward geometry metrics are embedded in the §4 tables (`fwd_mfe_h240_p50`, `frac_reach_1R`, `frac_reach_2R`, `frac_wrong_way`, `pct_peak_and_collapse`, `final_r_mean`, `final_r_t_stat`). Per-archetype distribution side files are at `results/v2_0_diagnostic/<dataset>/archetype_<id>_K<k>_distribution.csv`.

**Clean-shape-but-magnitude-dead flags** (monotonicity_centroid >= 0.55 AND fwd_mfe_h240_p50 < 1.0):

- (none)

## §6 Predictability per archetype (5-fold ROC-AUC on entry features)

AUC reference: 0.50 random; 0.55-0.60 marginal; 0.60-0.70 usable; 0.70+ strong.

### kh24

#### K = 3

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5538 | 0.0327 | 0.5281 | 0.581 | 0.5046 | 0.5639 | 0.5916 | 220 | 540 |
| 1 | 0.5625 | 0.021 | 0.5861 | 0.5847 | 0.5412 | 0.5645 | 0.536 | 270 | 540 |
| 2 | 0.5227 | 0.0937 | 0.5908 | 0.5194 | 0.6582 | 0.402 | 0.4429 | 50 | 540 |

#### K = 4

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5635 | 0.0411 | 0.5509 | 0.6407 | 0.5253 | 0.5344 | 0.5664 | 207 | 540 |
| 1 | 0.569 | 0.0401 | 0.5936 | 0.4919 | 0.5792 | 0.6057 | 0.5746 | 215 | 540 |
| 2 | 0.5163 | 0.0787 | 0.5208 | 0.5095 | 0.4092 | 0.4888 | 0.6531 | 48 | 540 |
| 3 | 0.5693 | 0.0806 | 0.4992 | 0.5182 | 0.5289 | 0.5783 | 0.7219 | 70 | 540 |

#### K = 5

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5493 | 0.0705 | 0.5309 | 0.4852 | 0.6186 | 0.644 | 0.4678 | 93 | 540 |
| 1 | 0.5362 | 0.0649 | 0.5729 | 0.5542 | 0.5242 | 0.6106 | 0.4191 | 185 | 540 |
| 2 | 0.515 | 0.1031 | 0.5679 | 0.4918 | 0.4194 | 0.4092 | 0.6867 | 49 | 540 |
| 3 | 0.5638 | 0.0802 | 0.5741 | 0.6413 | 0.4704 | 0.4737 | 0.6596 | 66 | 540 |
| 4 | 0.5356 | 0.0759 | 0.5552 | 0.6368 | 0.4435 | 0.5893 | 0.453 | 147 | 540 |

#### K = 6

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.4796 | 0.0443 | 0.4512 | 0.4141 | 0.5163 | 0.5376 | 0.4786 | 46 | 540 |
| 1 | 0.5344 | 0.0668 | 0.5235 | 0.4562 | 0.5027 | 0.657 | 0.5328 | 93 | 540 |
| 2 | 0.5713 | 0.0532 | 0.6348 | 0.6138 | 0.5433 | 0.485 | 0.5798 | 65 | 540 |
| 3 | 0.574 | 0.181 | 0.5263 | 0.7962 | 0.3062 | 0.755 | 0.4862 | 40 | 540 |
| 4 | 0.5556 | 0.0307 | 0.546 | 0.5239 | 0.6081 | 0.5299 | 0.5702 | 133 | 540 |
| 5 | 0.557 | 0.0474 | 0.5935 | 0.499 | 0.6154 | 0.5037 | 0.5736 | 163 | 540 |

#### K = 7

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.618 | 0.0842 | 0.5475 | 0.7613 | 0.5288 | 0.596 | 0.6566 | 42 | 540 |
| 1 | 0.5382 | 0.0555 | 0.5844 | 0.5727 | 0.4953 | 0.5882 | 0.4504 | 131 | 540 |
| 2 | 0.3613 | 0.1234 | 0.3632 | 0.4717 | 0.1651 | 0.2984 | 0.5079 | 13 | 540 |
| 3 | 0.5505 | 0.0427 | 0.553 | 0.6241 | 0.5503 | 0.533 | 0.4922 | 59 | 540 |
| 4 | 0.557 | 0.0474 | 0.5935 | 0.499 | 0.6154 | 0.5037 | 0.5736 | 163 | 540 |
| 5 | 0.3613 | 0.0489 | 0.3012 | 0.4242 | 0.3996 | 0.3086 | 0.3726 | 44 | 540 |
| 6 | 0.5321 | 0.0646 | 0.563 | 0.448 | 0.4617 | 0.5821 | 0.6056 | 88 | 540 |

### arc1

#### K = 3

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5256 | 0.0031 | 0.5262 | 0.5302 | 0.5228 | 0.5272 | 0.5216 | 24704 | 45673 |
| 1 | 0.5254 | 0.0042 | 0.5256 | 0.5324 | 0.5245 | 0.5256 | 0.5191 | 20955 | 45673 |
| 2 | 0.6962 | 0.077 | 0.6629 | 0.6446 | 0.7863 | 0.5994 | 0.7878 | 14 | 45673 |

#### K = 4

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5099 | 0.0026 | 0.5065 | 0.5131 | 0.5113 | 0.5071 | 0.5114 | 10061 | 45673 |
| 1 | 0.5256 | 0.0031 | 0.5262 | 0.5302 | 0.5228 | 0.5272 | 0.5216 | 24704 | 45673 |
| 2 | 0.6962 | 0.077 | 0.6629 | 0.6446 | 0.7863 | 0.5994 | 0.7878 | 14 | 45673 |
| 3 | 0.5269 | 0.0061 | 0.5294 | 0.5338 | 0.5306 | 0.5164 | 0.5241 | 10894 | 45673 |

#### K = 5

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5107 | 0.0038 | 0.5048 | 0.511 | 0.5138 | 0.5083 | 0.5154 | 10034 | 45673 |
| 1 | 0.5256 | 0.0031 | 0.5262 | 0.5302 | 0.5228 | 0.5272 | 0.5216 | 24704 | 45673 |
| 2 | 0.5278 | 0.004 | 0.5306 | 0.5328 | 0.5296 | 0.5225 | 0.5236 | 10895 | 45673 |
| 3 | 0.7315 | 0.078 | 0.6507 | 0.7294 | 0.6397 | 0.8076 | 0.8299 | 28 | 45673 |
| 4 | 0.741 | 0.0579 | 0.6855 | 0.8531 | 0.7218 | 0.7265 | 0.7182 | 12 | 45673 |

#### K = 6

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5112 | 0.0033 | 0.5073 | 0.5168 | 0.5092 | 0.5124 | 0.5102 | 10033 | 45673 |
| 1 | 0.5214 | 0.0017 | 0.5201 | 0.5221 | 0.5197 | 0.5245 | 0.5208 | 22934 | 45673 |
| 2 | 0.5278 | 0.004 | 0.5306 | 0.5328 | 0.5296 | 0.5225 | 0.5236 | 10895 | 45673 |
| 3 | 0.7163 | 0.0921 | 0.6301 | 0.7969 | 0.7984 | 0.5811 | 0.7751 | 13 | 45673 |
| 4 | 0.7315 | 0.078 | 0.6507 | 0.7294 | 0.6397 | 0.8076 | 0.8299 | 28 | 45673 |
| 5 | 0.5401 | 0.0065 | 0.5418 | 0.5443 | 0.5486 | 0.5301 | 0.536 | 1770 | 45673 |

#### K = 7

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5214 | 0.0017 | 0.5201 | 0.5221 | 0.5197 | 0.5245 | 0.5208 | 22934 | 45673 |
| 1 | 0.5274 | 0.0053 | 0.5223 | 0.5366 | 0.5253 | 0.53 | 0.523 | 10894 | 45673 |
| 2 | 0.5105 | 0.0016 | 0.5134 | 0.5089 | 0.51 | 0.5094 | 0.5107 | 10032 | 45673 |
| 3 | 0.7275 | 0.1827 | 0.6655 | 0.8103 | 0.4378 | 0.7276 | 0.9963 | 5 | 45673 |
| 4 | 0.7654 | 0.1729 | 0.9316 | 0.4712 | 0.8799 | 0.6669 | 0.8772 | 14 | 45673 |
| 5 | 0.5401 | 0.0065 | 0.5418 | 0.5443 | 0.5486 | 0.5301 | 0.536 | 1770 | 45673 |
| 6 | 0.6967 | 0.0544 | 0.6396 | 0.7849 | 0.7005 | 0.6397 | 0.7188 | 24 | 45673 |

### arc2

#### K = 3

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5114 | 0.0289 | 0.4769 | 0.5475 | 0.5149 | 0.4799 | 0.5376 | 772 | 3993 |
| 1 | 0.548 | 0.0199 | 0.5155 | 0.5558 | 0.5704 | 0.5356 | 0.5626 | 1236 | 3993 |
| 2 | 0.5354 | 0.0061 | 0.5433 | 0.5372 | 0.5305 | 0.5394 | 0.5264 | 1985 | 3993 |

#### K = 4

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5009 | 0.0304 | 0.5257 | 0.4944 | 0.5192 | 0.5209 | 0.4441 | 310 | 3993 |
| 1 | 0.5435 | 0.013 | 0.5468 | 0.5392 | 0.5604 | 0.5214 | 0.5499 | 1739 | 3993 |
| 2 | 0.5022 | 0.0229 | 0.5423 | 0.4764 | 0.5109 | 0.4926 | 0.4887 | 765 | 3993 |
| 3 | 0.5384 | 0.0225 | 0.5345 | 0.5087 | 0.5784 | 0.5364 | 0.5338 | 1179 | 3993 |

#### K = 5

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5483 | 0.0075 | 0.5453 | 0.5419 | 0.5564 | 0.5398 | 0.5579 | 1114 | 3993 |
| 1 | 0.4989 | 0.0196 | 0.4673 | 0.5014 | 0.5274 | 0.4919 | 0.5064 | 621 | 3993 |
| 2 | 0.5354 | 0.0105 | 0.5418 | 0.527 | 0.5276 | 0.5275 | 0.5533 | 822 | 3993 |
| 3 | 0.4966 | 0.0223 | 0.5318 | 0.4978 | 0.4937 | 0.4983 | 0.4615 | 195 | 3993 |
| 4 | 0.5111 | 0.013 | 0.5225 | 0.52 | 0.4955 | 0.495 | 0.5226 | 1241 | 3993 |

#### K = 6

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5446 | 0.0127 | 0.5224 | 0.5616 | 0.5451 | 0.5447 | 0.5494 | 1115 | 3993 |
| 1 | 0.507 | 0.0114 | 0.5036 | 0.5242 | 0.513 | 0.5041 | 0.4898 | 1151 | 3993 |
| 2 | 0.4929 | 0.011 | 0.4904 | 0.4878 | 0.487 | 0.4845 | 0.5146 | 604 | 3993 |
| 3 | 0.5572 | 0.0397 | 0.594 | 0.5281 | 0.5158 | 0.6152 | 0.533 | 54 | 3993 |
| 4 | 0.5111 | 0.0153 | 0.5093 | 0.4843 | 0.5164 | 0.5138 | 0.5315 | 349 | 3993 |
| 5 | 0.5555 | 0.0155 | 0.5584 | 0.5711 | 0.5728 | 0.5391 | 0.5359 | 720 | 3993 |

#### K = 7

| archetype_id | auc_mean | auc_std | auc_fold_1 | auc_fold_2 | auc_fold_3 | auc_fold_4 | auc_fold_5 | n_pos | n_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5533 | 0.0107 | 0.5559 | 0.5694 | 0.5567 | 0.5371 | 0.5472 | 762 | 3993 |
| 1 | 0.5638 | 0.0436 | 0.5751 | 0.6257 | 0.5551 | 0.5728 | 0.4902 | 205 | 3993 |
| 2 | 0.5043 | 0.018 | 0.4847 | 0.4839 | 0.5207 | 0.5276 | 0.5046 | 585 | 3993 |
| 3 | 0.5541 | 0.0404 | 0.611 | 0.5691 | 0.5134 | 0.5026 | 0.5741 | 84 | 3993 |
| 4 | 0.5171 | 0.0352 | 0.4884 | 0.5025 | 0.5656 | 0.5517 | 0.4771 | 345 | 3993 |
| 5 | 0.5076 | 0.026 | 0.4893 | 0.5541 | 0.481 | 0.4971 | 0.5163 | 1058 | 3993 |
| 6 | 0.5612 | 0.0134 | 0.5746 | 0.5788 | 0.5574 | 0.5502 | 0.5448 | 954 | 3993 |

## §7 Overlap with existing dual-gate clusters (Arc 1 + Arc 2)

Existing-cluster reference is K3_kmeans from `step3_extractability/cluster_assignments.csv` — Arc 1 dual-gate-passing cluster = C0; Arc 2 dual-gate-passing cluster = C2 (per v1.3 calibration §3).

### arc1

**Forward — where does dual-gate-passing cluster C0's majority land?**

| K | existing_cluster | archetype_id | overlap_pct | n_in_existing |
| --- | --- | --- | --- | --- |
| 3 | 0 | 1 | 0.6058 | 14882 |
| 4 | 0 | 1 | 0.3941 | 14882 |
| 5 | 0 | 1 | 0.3941 | 14882 |
| 6 | 0 | 0 | 0.349 | 14882 |
| 7 | 0 | 2 | 0.3491 | 14882 |

**Clean + meaty + predictable archetypes (any K):** none.

### arc2

**Forward — where does dual-gate-passing cluster C2's majority land?**

| K | existing_cluster | archetype_id | overlap_pct | n_in_existing |
| --- | --- | --- | --- | --- |
| 3 | 2 | 0 | 0.7541 | 854 |
| 4 | 2 | 2 | 0.7377 | 854 |
| 5 | 2 | 1 | 0.6628 | 854 |
| 6 | 2 | 2 | 0.6522 | 854 |
| 7 | 2 | 2 | 0.6347 | 854 |

**Clean + meaty + predictable archetypes (any K):** none.

KH-24 has no L-arc step-3 clusters; overlap section N/A.

## §8 v2.0 evidence flag — central output

> **No tuples qualify as v2.0 evidence on these three weak datasets under the first-pass priors. The reframe is neither validated nor refuted by this diagnostic. Re-run when stronger signals are available, or revisit thresholds.**

## §9 Combined view per dataset

### kh24

Dataset **kh24** at K=3 (max-silhouette of non-failed K values, sil=0.4792) produces 3 archetypes ranging from `Stepwise climber` (50.3% of pool) to `Unlabelled` (9.2% of pool). No archetype across any of K in {3,4,5,6,7} qualifies as v2.0 evidence under the first-pass priors (§8 empty for this dataset).

### arc1

Dataset **arc1** at K=7 (max-silhouette of non-failed K values, sil=0.9977) produces 7 archetypes ranging from `Flat-line` (50.2% of pool) to `Unlabelled` (0.0% of pool). No archetype across any of K in {3,4,5,6,7} qualifies as v2.0 evidence under the first-pass priors (§8 empty for this dataset).

### arc2

Dataset **arc2** at K=4 (max-silhouette of non-failed K values, sil=0.4615) produces 4 archetypes ranging from `Stepwise climber` (43.6% of pool) to `Unlabelled` (7.8% of pool). No archetype across any of K in {3,4,5,6,7} qualifies as v2.0 evidence under the first-pass priors (§8 empty for this dataset).

## §10 Files produced (manifest with sha256)

| path | sha256 | bytes |
| --- | --- | --- |
| arc1/archetype_0_K3_distribution.csv | a9ebdcf4088736df5deaaf0a4ae325866c5de3774afa8be904efb5aebfcb26ce | 572 |
| arc1/archetype_0_K4_distribution.csv | 8d1e306eba7a071a5b89914dafba77d04c8d7e378c0f6261ce46ba042b93a514 | 561 |
| arc1/archetype_0_K5_distribution.csv | ae5b69e62cee25e9b66e50fbb8e59cf9de072c45db91a5b0e12dbc192efa80bd | 565 |
| arc1/archetype_0_K6_distribution.csv | bfe737c4892c1a26ba9809e05595008a264597f0c7d1eb28a4fd14fa90842b12 | 561 |
| arc1/archetype_0_K7_distribution.csv | 6ea18d05bce5b7ef2d25893c50e5a1b90f381f3ce99526a448f5112939170915 | 576 |
| arc1/archetype_1_K3_distribution.csv | 157250e1833f1f169f43571a6d7ae6492d6c2c2acdb8b95e360dc2e861e9ff7b | 568 |
| arc1/archetype_1_K4_distribution.csv | a9ebdcf4088736df5deaaf0a4ae325866c5de3774afa8be904efb5aebfcb26ce | 572 |
| arc1/archetype_1_K5_distribution.csv | a9ebdcf4088736df5deaaf0a4ae325866c5de3774afa8be904efb5aebfcb26ce | 572 |
| arc1/archetype_1_K6_distribution.csv | 6ea18d05bce5b7ef2d25893c50e5a1b90f381f3ce99526a448f5112939170915 | 576 |
| arc1/archetype_1_K7_distribution.csv | fde4c27eb17007b00588797086e8a329581a6cdd413bb00f040854df429f7d7e | 570 |
| arc1/archetype_2_K3_distribution.csv | ec8456b384ad9d83297e744444912706a3914fdfb8c4f95708fb4cb73af4b5d8 | 565 |
| arc1/archetype_2_K4_distribution.csv | ec8456b384ad9d83297e744444912706a3914fdfb8c4f95708fb4cb73af4b5d8 | 565 |
| arc1/archetype_2_K5_distribution.csv | 4609b965e28cb5f86fac4365461e97f9f6ec6bf1fcb33e1094978ade303d6051 | 573 |
| arc1/archetype_2_K6_distribution.csv | 4609b965e28cb5f86fac4365461e97f9f6ec6bf1fcb33e1094978ade303d6051 | 573 |
| arc1/archetype_2_K7_distribution.csv | 3299889c4f4588bd1122f9d246fb525c4525ae69bb325a503eecb9abb2e744db | 562 |
| arc1/archetype_3_K4_distribution.csv | 2162167380d5c3a9e508117755163aae97086f612c3ee5a2a3a25a4a08e8dfae | 570 |
| arc1/archetype_3_K5_distribution.csv | e8e59fac394e254044ff63edb1fa0b8ce44560703228f9b901843c091a5c7fdd | 560 |
| arc1/archetype_3_K6_distribution.csv | e1b0a09bf5658e78bf13a47663d8711dfe8a5c51a080651e66024d42afadd43f | 564 |
| arc1/archetype_3_K7_distribution.csv | 8551bcd9fc7c67f228eb8e95900610161449cd2e0886ebaceb928f2008447a30 | 563 |
| arc1/archetype_4_K5_distribution.csv | 000b44c021ee39f009215ac0c59304e5e958de776bc33d6060b046103c6f4689 | 571 |
| arc1/archetype_4_K6_distribution.csv | e8e59fac394e254044ff63edb1fa0b8ce44560703228f9b901843c091a5c7fdd | 560 |
| arc1/archetype_4_K7_distribution.csv | 069c9b93c9e406aa6914eb46129dd5122dd918fd0b7e755f1524d681619e320e | 571 |
| arc1/archetype_5_K6_distribution.csv | c1f2e80fc12973a1576f07b781623ae6b9a9b7c283f2556c86cbea0c3a2f369b | 568 |
| arc1/archetype_5_K7_distribution.csv | c1f2e80fc12973a1576f07b781623ae6b9a9b7c283f2556c86cbea0c3a2f369b | 568 |
| arc1/archetype_6_K7_distribution.csv | 597f0b91c19b810975a288e6575075c8e0f7650204f3f4710cb3bcc7a003331a | 564 |
| arc1/archetype_summaries_K3.csv | e5614c6409834e285c96b8b2f22d7cf3bfd09d684a2f1757ef3678da9e00071c | 1985 |
| arc1/archetype_summaries_K4.csv | 80eb3c7a39e706fb341936be524e3a174d3bdfa28ec592ecbdaa16753b34dc5d | 2506 |
| arc1/archetype_summaries_K5.csv | 714527572f3e7acefe76e7689365ec816ceceed0c9cad5fd048e0be957018d10 | 2964 |
| arc1/archetype_summaries_K6.csv | 7f5f368be9b1648d2cef84dd15eb2817105bb5c350e5f5ba36f0560bfbbb0ae7 | 3519 |
| arc1/archetype_summaries_K7.csv | b2aa089dfbffad5d20a897384f3086d9dff98b78cb400531e78ea0b6a99f429c | 3849 |
| arc1/centroids_K3.csv | 2aa3ddae3f63f18a2dfc9a075c2ebc66991e2725734e777ffeead57b37ff7f93 | 364 |
| arc1/centroids_K4.csv | 29df5ba85c1331c843396f7fdea2dc3fab8428c8a593ddc78fb8543143da7683 | 445 |
| arc1/centroids_K5.csv | a239cf81e7d7a2256c674abcab901e5826208752c3a54271de092748e5bb8f37 | 527 |
| arc1/centroids_K6.csv | aae1b40e73f2347709425a0fbaf5d8b2ae37fbc83bc37babaa5e4357a25f53db | 619 |
| arc1/centroids_K7.csv | 8d74dfda31ab26c80d8fd07c70157e675cb2e3c3d3cccd842f9d5d10d32ba3e2 | 696 |
| arc1/clusters_K3.csv | 5eaf6543982019a35a86d1cae58129d0f9c9561c18420e082125cd3912e77f69 | 354296 |
| arc1/clusters_K4.csv | 850ed8047f9a4afa86239c0cf6f0368e5e9b617181ac2f3d24959a2d2f4016e2 | 354296 |
| arc1/clusters_K5.csv | d29ba445d905b3fb64a5796c6d2cc5fa20fc433518c0ee03740bd5b28abfc31a | 354296 |
| arc1/clusters_K6.csv | 2841bfdeb50ccea013801260dbb141a282ad8dfdbbe9947409c857ae959a8f5b | 354296 |
| arc1/clusters_K7.csv | 03f0ddd9d70cd522723feb8e626e3d31b8bbab4ea19d385bc62c3cfce1ce539a | 354296 |
| arc1/entry_features_basic.csv | f58a75253f907731c4ec6236d84fafc0756451e284752c30308322b8725214fb | 8508059 |
| arc1/overlap_matrix_K3.csv | 97393976031c4f1ffb478b74ec13b272a5d8b592b21f0662d4a568a0bb5236d3 | 61 |
| arc1/overlap_matrix_K4.csv | 6cdf9bf9480196fd1b692fa6c25c5435318bab0aa3256f53c2250175718fffa1 | 78 |
| arc1/overlap_matrix_K5.csv | 568f4e14f589b9d3de50e9f48ea92540a2c2d6c0dbf7ed7f24dd87851400a965 | 87 |
| arc1/overlap_matrix_K6.csv | d5c1fac561b7a7ab19dd3028010ca8b65ec6581abc226a56aa688601db599ba7 | 101 |
| arc1/overlap_matrix_K7.csv | 51ed7059446923431008f6e56a8da83252226c8556bf3072546c4e6f13e682e8 | 109 |
| arc1/overlap_summary.csv | 0e4c9104f432974871b20d275382cbbb7daca1ac71e8ed88a2598d46d13d0089 | 3368 |
| arc1/path_features.csv | cfb0559f88ae27ac617346a859e0e8af3efa57af7349cec4b4506b92263a04ed | 1268768 |
| arc1/path_features_degeneracy.csv | 515758bed51ac6456569ec7df5286d28a2160a6c15eb07ae0d09ebc3a992dcef | 256 |
| arc1/path_features_distributions.csv | bb44e3460b622a1b8d283074f840f3716f828ef9c63513c5be6d9605429ad3b9 | 454 |
| arc1/predictability_K3.csv | 3b2fafdb6e19ef973f820283b8fc2ad3051a8a0facf9bf8a4f638adff2cf6b88 | 540 |
| arc1/predictability_K4.csv | cf0b7a77c658526535f6698266f53fb778085a6524df0c8e903c996fc134ef1b | 688 |
| arc1/predictability_K5.csv | bc0b881fd0bf2d02bf0af6923dfa0540b8a720e1fa145d5d4f3c333f62688ae5 | 836 |
| arc1/predictability_K6.csv | 459004712cf00d49fefbcdd1c3be74c9bfc69c18bdc0bd3efc6c52ff311a00ca | 978 |
| arc1/predictability_K7.csv | 15efb102a6e6c1eda8902ff1de0100e4ed5d373bb2fe9ee901cf4290a9e8a862 | 1126 |
| arc1/silhouette_K3.txt | 942cac7db6e06b2b709373b745e806ca829d0b05e4a5d7646c22605dae3f3bea | 13 |
| arc1/silhouette_K4.txt | 73b12f8491c3ca9aa9cd7931dd1846eb3b8bab0229d84296bc43843ddcd6e603 | 13 |
| arc1/silhouette_K5.txt | 5ab805a54abc289915933be0507656a99a403a1f0cf8c273530c2fbbd6595e87 | 13 |
| arc1/silhouette_K6.txt | 0af48ad8ea566f38a55612477985c7f768c68021e6dbd39f603cad2dd6877f33 | 13 |
| arc1/silhouette_K7.txt | 83a4394e08b0b03ac5acd5af71114ac0b6e2cee4f04b16101399a24a777e3c7c | 13 |
| arc2/archetype_0_K3_distribution.csv | 2acdc9d2af52d3d493b630b817aa5197f42e4a6dbeff38d850ff5800d53c6d18 | 549 |
| arc2/archetype_0_K4_distribution.csv | 2d635d55c8d87731f0e70d4d7ef0ec672e5df1c0f48793f53208ca58ffe6392c | 566 |
| arc2/archetype_0_K5_distribution.csv | 850c9129abeaa6b0d146bc79b3879ca54e1bf6815edbafaa6387539f9b7ae9cc | 578 |
| arc2/archetype_0_K6_distribution.csv | 86b83901b1b4cd40464ba20386f7863262e256f04f956b38930bfcd3c2bae961 | 573 |
| arc2/archetype_0_K7_distribution.csv | aa5fb52f32a5d7bc98b62d8c14878d8b777dd3a735ae203ab53a9f52423d2e03 | 563 |
| arc2/archetype_1_K3_distribution.csv | fa6b3c94a56bfb9651adae081696c651780380e307d530d100b07f64c8dbfa52 | 574 |
| arc2/archetype_1_K4_distribution.csv | f919fa2e351713ffc4df79e1e6619789bb2a10e8efef814929ef1eed6c146a46 | 563 |
| arc2/archetype_1_K5_distribution.csv | 14c6d671b6a8f0ec7372c8176f15408790134264777b4b7ae714f9b672e9a830 | 544 |
| arc2/archetype_1_K6_distribution.csv | 006c5db59018acec2fa4431b77e05ea7d9cf9b5f88e82d7e979f43a2ea8b9cea | 565 |
| arc2/archetype_1_K7_distribution.csv | ec1883da9c5c4d30229593388bf78be65fb699f776e5f4e75bfc27fd7101adea | 572 |
| arc2/archetype_2_K3_distribution.csv | b29442146ee9eb5a9db655c07f3806fe2ca09443907c4a37825c356e3a457ee7 | 566 |
| arc2/archetype_2_K4_distribution.csv | 93e4fbf8fd51c756219756cc26cff8b144fb7b8a2b6657cc47a152880de74322 | 549 |
| arc2/archetype_2_K5_distribution.csv | 7fa7a2487360066af88f6a16812a5c25c6536300f31bc702fc666e1dbdf2ec1e | 566 |
| arc2/archetype_2_K6_distribution.csv | 182b23767ee7da70c66ef8e24b792c275979dbbc8f8c751bc7c705a0fc0a61bd | 544 |
| arc2/archetype_2_K7_distribution.csv | b208e23b1e5a96e535bc4f92920a66792d31bb73dc5a37fe8f62b3fc149b027b | 552 |
| arc2/archetype_3_K4_distribution.csv | a7e6fa9d600a9efe4feee36459e291947a58bcd612182dbfcf94358880a9bb49 | 574 |
| arc2/archetype_3_K5_distribution.csv | af1ea29a0e58c2882b21143a3cf6b90d5597506768d06182daae4b02201d2d4d | 564 |
| arc2/archetype_3_K6_distribution.csv | f285290cd7374463f0a97a91b017a4a619f4ef685d8f9af7a9ef8bd25169a8b1 | 565 |
| arc2/archetype_3_K7_distribution.csv | 31ede71c15bbc6ddd1c61274ca1b43aacb0b7c13743b6dbbf5be672d255a49c5 | 563 |
| arc2/archetype_4_K5_distribution.csv | bd9a208f697e758becea875fe8f71df88c39ed604fd13d9c6699aa089f0b554e | 565 |
| arc2/archetype_4_K6_distribution.csv | c5e0677adfee5cd86032a3bdcfd14d663a81f9d64262095c58eb03d14c173c9c | 564 |
| arc2/archetype_4_K7_distribution.csv | 387adb45d540fd1101ff7f006899d421f22df89d5dc65256d94db05b7f534375 | 564 |
| arc2/archetype_5_K6_distribution.csv | c76a3ae3078748c264a6be6140ba25825f5014d288260f133f0d2300d55f7108 | 566 |
| arc2/archetype_5_K7_distribution.csv | 57a4a98207302ad96dec71c7768baaf399a19762f344562a46421cb3a6ddfce3 | 571 |
| arc2/archetype_6_K7_distribution.csv | 71013da9bc1312cfb69061cf28232bc4b956ad6462d050c62a380dc3de6b3599 | 578 |
| arc2/archetype_summaries_K3.csv | 82d26ce9be2b8b74a5c6d40ecff34013837ebbf163c9dbec21e3cd7f3832219b | 1970 |
| arc2/archetype_summaries_K4.csv | f8b4764ef2f6522ec4e227632331f59185a4e329a98645732420a064fe3ae956 | 2452 |
| arc2/archetype_summaries_K5.csv | a985339b0d1f95c0f7cab21b7b7f90aa31dab642ac664f5bd432a4e0225bc987 | 2964 |
| arc2/archetype_summaries_K6.csv | 63b7f84798fe023da310b0054770bd287f55b0da0ebd48fbb09600932341d4f4 | 3511 |
| arc2/archetype_summaries_K7.csv | ac0e631cdac006297a4b6f1c18a78dc410d6a9e93a4e35c78703b1d6f472c259 | 4006 |
| arc2/centroids_K3.csv | d62ee298dc97e607f89e56fab60d32736ebbe31ab0f7084083d72173c90f02f9 | 355 |
| arc2/centroids_K4.csv | b0a1f6c6988ac17f4d3eb99fa96d6f1df2f75fec5a627de71b6987ce5c8391ee | 435 |
| arc2/centroids_K5.csv | 0007b7aa0dd92c19f5599188716ecee59b7845eae1720e0929f19abb7be7f33f | 509 |
| arc2/centroids_K6.csv | 9a942604a29392925a1e8309a9f89dc7f8de4f834bc98addf243794c9c1579ad | 592 |
| arc2/centroids_K7.csv | dc7d35dd1cc512942c7c45495ebb9944cee624f889ea28c89a58227a5c39c4a3 | 676 |
| arc2/clusters_K3.csv | 6795f1801d6273df40a2dd9c1a715e1d986f417287b63e9d2a97e0cccd8f6b83 | 26863 |
| arc2/clusters_K4.csv | 0d2dd328d9cdbd65a3ba29c1b17137a1a8adb14a80078aa351972769aa3e4773 | 26863 |
| arc2/clusters_K5.csv | 8e0b8fb7a5cc59f870f86889f1cb6f5b9dff59302a0c666ae08a7569e1495be4 | 26863 |
| arc2/clusters_K6.csv | d246c536be6fc628f6cd19f34de3543df22faad8fd59905f12b832c8b47b4dc7 | 26863 |
| arc2/clusters_K7.csv | e8445de424f27c869d43d8718abdf928a004622f23fa1b009b6cf26109c5a00c | 26863 |
| arc2/entry_features_basic.csv | 3dd517778d1f312031a12ffb720977a40fc9029c8b418955cec0d96bdc76dbb4 | 740911 |
| arc2/overlap_matrix_K3.csv | 4257b929252860f70e68a297edb5e470268d5b6b62db4491d8be5a8aef80b5a2 | 69 |
| arc2/overlap_matrix_K4.csv | 2f546cad59fd0c9168ed4a234f036cba0960d524b476c9cbe8da1dd87d7d8030 | 83 |
| arc2/overlap_matrix_K5.csv | 1290b250be574b2e3cdf28dc45b902d5ce638ec14bc7594d2d583595dce8dbef | 96 |
| arc2/overlap_matrix_K6.csv | af77289e612ad1f0a09132d92b9f17e0a99166faa779b5af85ac3cb4004254db | 108 |
| arc2/overlap_matrix_K7.csv | d79d922a55d944ee00f69b854f5050390aa845f54f1bd680cb704c8380091e02 | 119 |
| arc2/overlap_summary.csv | e14507492dcff5034ffddbbdf09d9d60da553f6b194f2b15dd3e310c48ab3b2e | 3469 |
| arc2/path_features.csv | 78a339d0ac05e6103f9111bbab796ae3ce38bfa1625d88951e3056b3b2100614 | 144986 |
| arc2/path_features_degeneracy.csv | 2ec0e35452a9faf72ac27989d24502deaa60aa8f5dac7c87f14290022e95a4f3 | 257 |
| arc2/path_features_distributions.csv | 41dcefc137eb9105a040366a4b53af0cc15e8a4f1774a4e687f2dfb6aeb533dc | 570 |
| arc2/predictability_K3.csv | 3dfbc71bda6191ca794a7825fced04b6195133741ba9690575fce2cb625d0348 | 535 |
| arc2/predictability_K4.csv | cf5884c140667c0c6455ce396e884044b6614e72b60a154c65a12963b22b5e87 | 682 |
| arc2/predictability_K5.csv | be7967d4538a9b533bebb83629cb9749058f368bfee54720854336161f4e3078 | 833 |
| arc2/predictability_K6.csv | 9c7a034e71e918f47a69df48a339e88fb80e9fa1aed289872fdf1fb2719119dc | 970 |
| arc2/predictability_K7.csv | ff9cfe809c431ddbdbef5b3b24af4e67e703d06128d7f4f9cf7e8eec50df76f4 | 1120 |
| arc2/silhouette_K3.txt | 6e5b42638ce9f9d26c9cddf7f81ff1f22d2ec4a24cb8ca824cd1ffb66a1be044 | 13 |
| arc2/silhouette_K4.txt | 0035cd0129e093b8d2dc0847bde5d57a3a3a02aa89415ebc3eba3b6d3969325d | 13 |
| arc2/silhouette_K5.txt | 33b8d29654ec58bfd846817d8df2ae95b720933403a121b5916ff3f5d2f21883 | 13 |
| arc2/silhouette_K6.txt | 0ee6a89478a54dc1f7a31ce15699aee8fa79925a3480593ac43146abea25fbb4 | 13 |
| arc2/silhouette_K7.txt | 263a05de24b351cb21b382cc2a24cc45c48e56fbe93b46b2fb90fe304d688f30 | 13 |
| kh24/archetype_0_K3_distribution.csv | 1151bd3323d8b54b92524827001a15cf9847d5fda285d9a3e7361f849a0af662 | 575 |
| kh24/archetype_0_K4_distribution.csv | 2e4bb1c41a94f02c7b0eb6c732091355e9a46e985565442cb80cac6cf771718c | 578 |
| kh24/archetype_0_K5_distribution.csv | 9bdec70f052f04b3691e1d64edcf0445744e69542d09069a3febcf65a428cf1b | 568 |
| kh24/archetype_0_K6_distribution.csv | 5b89a55c31050127a00a6f471b66682cf6e2c358b4bfecce0dec0ba053f421bc | 572 |
| kh24/archetype_0_K7_distribution.csv | 37f094dc3d1b42ab2a2e15694f8537ed64d3c952580ccb80a4cd831b5f86ec7b | 570 |
| kh24/archetype_1_K3_distribution.csv | 0848da0095e45bc177857b4b2ad284ddb10d80b7fe154f0641c3228fce05012b | 555 |
| kh24/archetype_1_K4_distribution.csv | b98901537a0c4ec3e4b241078949c0c142fd5a1c702f192bed43d0b27c8e378b | 567 |
| kh24/archetype_1_K5_distribution.csv | c0711e0ac5be4e5e2690b0532b88fb93549d014a8a83c295b2eb42088eb2f939 | 578 |
| kh24/archetype_1_K6_distribution.csv | cd4346907b53e4c2fc0783d963d69041f175faf75ab808027874be8ff49f86d6 | 570 |
| kh24/archetype_1_K7_distribution.csv | 7c00dc316eb995e304b61c826d8add961574ddde317ff8a3ba56ad0884a201f2 | 561 |
| kh24/archetype_2_K3_distribution.csv | 1f67635ad1f4177c6bc38cfe3dbe30198c6df6b9cf0c071918f9ccf06f9c9959 | 564 |
| kh24/archetype_2_K4_distribution.csv | 5d0f621ef7667c496478523b3ea6bc54ca952e13a7d2af3f41fb9ae36cc47b18 | 567 |
| kh24/archetype_2_K5_distribution.csv | 7bea19f309a6f4973aaa79c74d80ffc1cf16a43cf0403a03ddade664b8e4daed | 568 |
| kh24/archetype_2_K6_distribution.csv | edd423ddd3b703cdf63fb33b7e40c99c5a8af96f0dc387bdac471a68185d91f0 | 555 |
| kh24/archetype_2_K7_distribution.csv | eb5f5482da37b64469b05dca6d0ddc65bf5ecac2002e64b4520fc3309866db15 | 570 |
| kh24/archetype_3_K4_distribution.csv | a2657f70c35ec90d6e2e0c7ddfbdb00cff5efc49d0b383b585d4d6dbf72655ea | 553 |
| kh24/archetype_3_K5_distribution.csv | 705a079e41506a3602e058ffcb0ae121f0f52b6d109e8a7099eecf3eb1d33a8e | 556 |
| kh24/archetype_3_K6_distribution.csv | 89d72987b0244cc78492d01371b1b43f61d74d7050c10eb098b32e83e30273d6 | 573 |
| kh24/archetype_3_K7_distribution.csv | 1af23a4fa74f89c27a69434d17f4f967a7da59c51cf72975ff93cba3558f3700 | 551 |
| kh24/archetype_4_K5_distribution.csv | b101f146aa186ee0ab38b54214b1426485eced1425761813dc79b93bb45f9ab7 | 565 |
| kh24/archetype_4_K6_distribution.csv | 5885e061435d0cfc8a7e2f67ae802b8243f532ffcba009b0fc5fcf842b6d2e4b | 561 |
| kh24/archetype_4_K7_distribution.csv | 0a0d5628cb4de9f294745d5583e7ec953370b9fc5e04fe49243d829a44c7baae | 576 |
| kh24/archetype_5_K6_distribution.csv | 0a0d5628cb4de9f294745d5583e7ec953370b9fc5e04fe49243d829a44c7baae | 576 |
| kh24/archetype_5_K7_distribution.csv | c2187c550ae6d90aa91df9707f3cccf72b7ab2fd123b156d35ad4764dd9e6424 | 563 |
| kh24/archetype_6_K7_distribution.csv | 78aa2953fe4a14d3b8cfccdafb8cc40d20b4bff1af67e7f1b8db552f28c0d16b | 572 |
| kh24/archetype_summaries_K3.csv | 469e65480b8e7f568f701f8387adefb2fd101124041f071e9b437a7e04a87599 | 1977 |
| kh24/archetype_summaries_K4.csv | c9fd028b55802261fb617a9af16370a68e54e1e45542803e3e12c31378c9c9e8 | 2471 |
| kh24/archetype_summaries_K5.csv | d8e4b7834bf693ff06fd9bc61122c4f47fc6815a9f0a76ba2cb431eaf04bf472 | 2854 |
| kh24/archetype_summaries_K6.csv | 810a2006082e6df8cbd7217c0306a2999dd218dabcfed75c4defa51210b2cceb | 3389 |
| kh24/archetype_summaries_K7.csv | 71ecf3972f9768178879ee1d8999bd8fcdc45c65b8e0814eba26c3eb532cd5c7 | 4015 |
| kh24/centroids_K3.csv | 5b7ae4e5826ce3c7cdbf34dd30b806331c0cf45903f42c7a3d588f6a6c416e64 | 352 |
| kh24/centroids_K4.csv | 577feec0cd40185d6b47b598e6ee66f392dfaff26b2d6f04b6b8f6b4eddfbdb2 | 434 |
| kh24/centroids_K5.csv | 272a74d6a3762b3fe4c62e5173791acc9a887af7dc24c59a6fecc3737fd9e89a | 499 |
| kh24/centroids_K6.csv | b54bc1704b253d38b878f8aeeddee61cf6ae4a4e6fc9fa32e2956d01e5097360 | 596 |
| kh24/centroids_K7.csv | ee06a221d7a2191b841aa1108ae8e6c77637ee4ba0d49238a3af35ff57d2933a | 678 |
| kh24/clusters_K3.csv | f99462fd1b375bc315f64defefbf43ca7ebdf531257c9a86cab2e70707e35b38 | 16612 |
| kh24/clusters_K4.csv | 7013e0faeef26bf8bae6ee926ef0d668ef1063012b576fcda0e70a2752f72bc9 | 16612 |
| kh24/clusters_K5.csv | 118765f8407222b7b5b4317053fe0dcde0572f8156fcc55e0debecaa95f9e859 | 16612 |
| kh24/clusters_K6.csv | fb8dd0aef85f22ed457daf8dfd095a56c391110a30a2d591a63c07de19250e8f | 16612 |
| kh24/clusters_K7.csv | 219ced6f59f482c412eef2a0cae9a41d3d55f4965919d431fb09d63415a8d726 | 16612 |
| kh24/entry_features_basic.csv | 90843ccd543f86feb39d5d75fea0f717ad954d887524997079852c0945d3905c | 113398 |
| kh24/path_features.csv | e93decbfa07a5ee7c3bcceb0aa604abbac20f1f91de8ab994f4a0fb6f0d38fca | 31603 |
| kh24/path_features_degeneracy.csv | 6d24c8da6263a5e974c11da93c9490ddc49aa4e879aea47286236f4f34e5846a | 259 |
| kh24/path_features_distributions.csv | 5d80ea5a941ab2e9cb64197b596a019c0fe78c2ad205c181ef0cc27391070814 | 524 |
| kh24/predictability_K3.csv | fe1016e67314ab3cf0f6ebc9e7826cd413fb8663e20fec27d09b75ee7d74047d | 528 |
| kh24/predictability_K4.csv | 444bee1aed80e90f5e19414d1ca24de4805518af437e9fba6c0f856c79843a79 | 672 |
| kh24/predictability_K5.csv | 19fb3e1619971326f38e9998a3df2428e555f46280ad4e894d49470878423434 | 819 |
| kh24/predictability_K6.csv | 3dc8b2460f4adf015ca1bdd75c2319dfeae4457ec2617b221a34b3188ff11006 | 927 |
| kh24/predictability_K7.csv | a92d84113e6daea89a615517a7990d885a26febf6bbc2e215faf69c8426c965d | 1077 |
| kh24/silhouette_K3.txt | 71b006e8292709fdf4c13da621b621531b1b71c192ae1320b72a8c61cafa5f32 | 13 |
| kh24/silhouette_K4.txt | 080b7f4746af5a298250283a0f995d96c0f5782066377527c7ec3dba22e010c7 | 13 |
| kh24/silhouette_K5.txt | b6aec97f20f3f95aea58bd69327bcdd0091e798670e33a9575818a5d27811010 | 13 |
| kh24/silhouette_K6.txt | 69570f64a0df6f1d1def9f1a579dec35d71983a4f935c53c63ac281a9ae76e38 | 13 |
| kh24/silhouette_K7.txt | b0062fe49e6c9c50f3d32e50ff9f02589db026e307683e6137694c9a70c852c6 | 13 |
| v2_0_evidence_flags.csv | 1ed444b77f80e79b8eea6419afb42cfc17de05c50b88c14c7676ab737d68c05b | 16879 |

## §11 Open observations

- **kh24** path-shape feature `pullback_magnitude_median` is degenerate (81.6% at value 0.0000). Clustering on degenerate features is uninformative; the rest of this dataset's results should be read with caution.
- **arc1** path-shape feature `pullback_magnitude_median` is degenerate (99.9% at value 0.0000). Clustering on degenerate features is uninformative; the rest of this dataset's results should be read with caution.
- **arc1** silhouette is artificially high at K in [4, 5, 6, 7] (>0.90). Inflation source: 96% of arc1 trades have bars_held=1, producing path-shape features that are zero on most features. K-means trivially separates the zero-vector mass from the small "moved" minority, yielding high silhouette without informative archetypes.
- **No (dataset, K, archetype) qualifies as v2.0 evidence under first-pass priors.** Per spec §8 plainness: this neither validates nor refutes the reframe.
- **4 near-miss tuple(s)** meet 4 of 5 first-pass priors. Inspect `v2_0_evidence_flags.csv` for which prior failed — most commonly `predictable=False` (AUC < 0.60), suggesting entry-feature predictivity is the binding constraint on these inputs, not path-shape distinctness.
