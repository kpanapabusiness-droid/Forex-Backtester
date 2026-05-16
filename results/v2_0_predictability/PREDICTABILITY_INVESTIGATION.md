# v2.0 predictability investigation — is the wall real?

> Tests whether the predictability wall from PR #129 (logistic AUC < 0.60 on
> all v2.0 archetypes) is real or an artifact. Three angles per target:
> A) RF on the same 8 features, B) expanded feature set, C) t>0 path-so-far.
> Scope: target archetypes meeting RELAXED clean+meaty+size on KH-24 + Arc 2
> (Arc 1 excluded — diagnostic's bars_held=1 issue makes its clusters invalid).
> Computation only; §6 categorises the finding mechanically (A/B/C).

## §1 Target archetypes investigated

| dataset | K | archetype_id | size_fraction_of_pool | monotonicity_centroid | local_peaks_centroid | pullback_magnitude_centroid | time_to_peak_relative_centroid | final_r_mean | frac_reach_1R | frac_wrong_way | fwd_mfe_h240_p50 | auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc2 | 3 | 0 | 0.1933 | 0.535 | 24.2107 | 0.0314 | 0.7882 | 3.9082 | 1 | 0.0142 | 6.3178 | 0.5114 |
| arc2 | 4 | 2 | 0.1916 | 0.5355 | 24.0993 | 0.0328 | 0.7976 | 3.9108 | 1 | 0.0183 | 6.3193 | 0.5022 |
| arc2 | 5 | 1 | 0.1555 | 0.5371 | 26.496 | 0.0275 | 0.7888 | 4.4535 | 1 | 0.0081 | 6.8591 | 0.4989 |
| arc2 | 6 | 2 | 0.1513 | 0.5371 | 26.7815 | 0.0293 | 0.7956 | 4.5318 | 1 | 0.0083 | 6.9044 | 0.4929 |
| arc2 | 7 | 2 | 0.1465 | 0.5378 | 27.072 | 0.0292 | 0.8027 | 4.6373 | 1 | 0.0085 | 6.9714 | 0.5043 |
| kh24 | 3 | 1 | 0.5027 | 0.5965 | 6.6906 | 0.0208 | 0.7 | 0.7356 | 0.8921 | 0.3201 | 3.9179 | 0.5625 |
| kh24 | 4 | 1 | 0.396 | 0.5995 | 3.863 | 0.022 | 0.621 | 0.11 | 0.8539 | 0.4247 | 3.1089 | 0.569 |
| kh24 | 4 | 3 | 0.1338 | 0.5756 | 14.1892 | 0.0201 | 0.8473 | 2.3653 | 1 | 0.0405 | 5.4034 | 0.5693 |
| kh24 | 5 | 3 | 0.123 | 0.5809 | 14.6471 | 0.0218 | 0.8503 | 2.3728 | 1 | 0.0441 | 5.2596 | 0.5638 |
| kh24 | 5 | 4 | 0.2767 | 0.5347 | 4.6144 | 0.0231 | 0.7753 | 0.3887 | 0.8954 | 0.3399 | 3.1932 | 0.5356 |
| kh24 | 6 | 2 | 0.1212 | 0.5804 | 14.7164 | 0.0222 | 0.8525 | 2.3949 | 1 | 0.0448 | 5.3845 | 0.5713 |
| kh24 | 6 | 4 | 0.2495 | 0.6084 | 5.0362 | 0.0248 | 0.7558 | 0.4685 | 0.913 | 0.3043 | 3.2644 | 0.5556 |
| kh24 | 7 | 1 | 0.2459 | 0.6102 | 5.1912 | 0.0154 | 0.7587 | 0.5156 | 0.9118 | 0.2868 | 3.3013 | 0.5382 |
| kh24 | 7 | 3 | 0.1103 | 0.5797 | 15.1475 | 0.0162 | 0.8584 | 2.5213 | 1 | 0.0492 | 5.1346 | 0.5505 |

**Exit-family groupings (size >= 2 archetypes sharing a tag):**

| dataset | K | exit_family_tag | archetype_ids_in_group | n_archetypes_in_group | total_size_fraction |
| --- | --- | --- | --- | --- | --- |
| kh24 | 6 | stepwise_lock | 2,4 | 2 | 0.3707 |

## §2 Angle A — Random Forest on the 8 basic entry features

| dataset | K | target_id | exit_family_tag | target_size | n_total | auc_mean | auc_std | auc_logistic_baseline | lift_vs_logistic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc2 | 3 | arch_0 | mixed | 772 | 3993 | 0.5082 | 0.0238 | 0.5114 | -0.0032 |
| arc2 | 4 | arch_2 | mixed | 765 | 3993 | 0.4911 | 0.0161 | 0.5022 | -0.0111 |
| arc2 | 5 | arch_1 | mixed | 621 | 3993 | 0.5021 | 0.0121 | 0.4989 | 0.0032 |
| arc2 | 6 | arch_2 | mixed | 604 | 3993 | 0.4951 | 0.0243 | 0.4929 | 0.0023 |
| arc2 | 7 | arch_2 | mixed | 585 | 3993 | 0.492 | 0.0309 | 0.5043 | -0.0123 |
| kh24 | 3 | arch_1 | stepwise_lock | 270 | 540 | 0.5333 | 0.061 | 0.5625 | -0.0291 |
| kh24 | 4 | arch_1 | untradeable | 215 | 540 | 0.5102 | 0.0275 | 0.569 | -0.0588 |
| kh24 | 4 | arch_3 | stepwise_lock | 70 | 540 | 0.5506 | 0.058 | 0.5693 | -0.0187 |
| kh24 | 5 | arch_3 | stepwise_lock | 66 | 540 | 0.5015 | 0.061 | 0.5638 | -0.0623 |
| kh24 | 5 | arch_4 | mixed | 147 | 540 | 0.4754 | 0.0536 | 0.5356 | -0.0602 |
| kh24 | 6 | arch_2 | stepwise_lock | 65 | 540 | 0.5315 | 0.0537 | 0.5713 | -0.0398 |
| kh24 | 6 | arch_4 | stepwise_lock | 133 | 540 | 0.4434 | 0.0681 | 0.5556 | -0.1123 |
| kh24 | 7 | arch_1 | stepwise_lock | 131 | 540 | 0.4592 | 0.0311 | 0.5382 | -0.079 |
| kh24 | 7 | arch_3 | mixed | 59 | 540 | 0.5682 | 0.0503 | 0.5505 | 0.0177 |
| kh24 | 6 | group_stepwise_lock | stepwise_lock | 198 | 540 | 0.4789 | 0.0344 | 0.5395 | -0.0606 |

No Angle-A target clears AUC >= 0.6.

## §3 Angle B — expanded entry feature set

**kh24** expanded feature list (24 features):

```
body_to_range_ratio, upper_wick_ratio, lower_wick_ratio, range_to_atr_14, ret_5bar_atr, ret_20bar_atr, pos_in_20bar_range, rsi_14, atr_7, atr_28, ret_50bar_atr, ret_100bar_atr, body_to_range_mean_5, upper_wick_mean_5, bars_since_high20, bars_since_low20, rsi_7, rsi_28, cci_14, bb_pos_20, realized_vol_20, realized_vol_60, dist_to_swing_high_20, dist_to_swing_low_20
```

**arc2** expanded feature list (38 features):

```
body_to_range_ratio, upper_wick_ratio, lower_wick_ratio, range_to_atr_14, ret_5bar_atr, ret_20bar_atr, pos_in_20bar_range, rsi_14, atr_at_signal_1h, atr_baseline_1h_200, signal_bar_open, signal_bar_high, signal_bar_close, signal_bar_low, day_of_week, concurrent_signals_within_3h, vol_realized_1h_24h_decile, concurrent_signals_same_bar, vol_realized_1h_24h, atr_ratio_to_baseline, signal_bar_volume, trade_overlap_at_execution_time, cum_logret_1h_72, hour_utc, hour_in_d1_bar, bars_to_next_d1_close, currency_basket_3h_USD, cum_logret_1h_24, cum_logret_1h_3, currency_basket_3h_EUR, cum_logret_1h_168, cum_logret_1h_6, currency_basket_3h_GBP, dist_close_to_high30_atr, hour_in_4h_bar, bars_to_next_4h_close, signal_bar_abs_log_return, currency_basket_3h_JPY
```

| dataset | K | target_id | exit_family_tag | target_size | n_features_used | auc_logistic_expanded_mean | auc_rf_expanded_mean | auc_logistic_baseline | lift_logistic_expanded_vs_baseline | lift_rf_expanded_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc2 | 3 | arch_0 | mixed | 772 | 38 | 0.5382 | 0.5788 | 0.5114 | 0.0269 | 0.0675 |
| arc2 | 4 | arch_2 | mixed | 765 | 38 | 0.5315 | 0.576 | 0.5022 | 0.0293 | 0.0738 |
| arc2 | 5 | arch_1 | mixed | 621 | 38 | 0.5266 | 0.5545 | 0.4989 | 0.0277 | 0.0556 |
| arc2 | 6 | arch_2 | mixed | 604 | 38 | 0.5263 | 0.5548 | 0.4929 | 0.0334 | 0.0619 |
| arc2 | 7 | arch_2 | mixed | 585 | 38 | 0.5336 | 0.5679 | 0.5043 | 0.0292 | 0.0636 |
| kh24 | 3 | arch_1 | stepwise_lock | 270 | 24 | 0.5712 | 0.534 | 0.5625 | 0.0087 | -0.0285 |
| kh24 | 4 | arch_1 | untradeable | 215 | 24 | 0.5501 | 0.5267 | 0.569 | -0.0189 | -0.0424 |
| kh24 | 4 | arch_3 | stepwise_lock | 70 | 24 | 0.6647 | 0.6421 | 0.5693 | 0.0954 | 0.0728 |
| kh24 | 5 | arch_3 | stepwise_lock | 66 | 24 | 0.6614 | 0.6171 | 0.5638 | 0.0976 | 0.0533 |
| kh24 | 5 | arch_4 | mixed | 147 | 24 | 0.5324 | 0.536 | 0.5356 | -0.0031 | 0.0005 |
| kh24 | 6 | arch_2 | stepwise_lock | 65 | 24 | 0.64 | 0.6146 | 0.5713 | 0.0687 | 0.0432 |
| kh24 | 6 | arch_4 | stepwise_lock | 133 | 24 | 0.522 | 0.514 | 0.5556 | -0.0336 | -0.0416 |
| kh24 | 7 | arch_1 | stepwise_lock | 131 | 24 | 0.5136 | 0.5027 | 0.5382 | -0.0246 | -0.0355 |
| kh24 | 7 | arch_3 | mixed | 59 | 24 | 0.6556 | 0.6516 | 0.5505 | 0.1051 | 0.101 |
| kh24 | 6 | group_stepwise_lock | stepwise_lock | 198 | 24 | 0.5509 | 0.5537 | 0.5395 | 0.0114 | 0.0142 |

**Angle B clears AUC >= 0.6** for:
- kh24 K=4 arch_3: best AUC = 0.6647
- kh24 K=5 arch_3: best AUC = 0.6614
- kh24 K=6 arch_2: best AUC = 0.6400
- kh24 K=7 arch_3: best AUC = 0.6556

## §4 Angle C — t > 0 path-so-far predictability

| dataset | K | target_id | exit_family_tag | auc_rf_basic_only | target_size_at_t1 | n_excluded_at_t1 | auc_rf_mean_at_t1 | lift_vs_entry_only_at_t1 | target_size_at_t3 | n_excluded_at_t3 | auc_rf_mean_at_t3 | lift_vs_entry_only_at_t3 | target_size_at_t5 | n_excluded_at_t5 | auc_rf_mean_at_t5 | lift_vs_entry_only_at_t5 | target_size_at_t10 | n_excluded_at_t10 | auc_rf_mean_at_t10 | lift_vs_entry_only_at_t10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc2 | 3 | arch_0 | mixed | 0.5082 | 772 | 112 | 0.6196 | 0.1115 | 772 | 454 | 0.6366 | 0.1284 | 772 | 741 | 0.6516 | 0.1435 | 772 | 1202 | 0.6718 | 0.1637 |
| arc2 | 4 | arch_2 | mixed | 0.4911 | 765 | 112 | 0.6108 | 0.1197 | 765 | 454 | 0.6335 | 0.1424 | 765 | 741 | 0.6481 | 0.157 | 765 | 1202 | 0.6689 | 0.1778 |
| arc2 | 5 | arch_1 | mixed | 0.5021 | 621 | 112 | 0.6063 | 0.1043 | 621 | 454 | 0.6274 | 0.1254 | 621 | 741 | 0.6471 | 0.145 | 621 | 1202 | 0.6733 | 0.1712 |
| arc2 | 6 | arch_2 | mixed | 0.4951 | 604 | 112 | 0.6085 | 0.1133 | 604 | 454 | 0.6274 | 0.1322 | 604 | 741 | 0.6322 | 0.1371 | 604 | 1202 | 0.6653 | 0.1701 |
| arc2 | 7 | arch_2 | mixed | 0.492 | 585 | 112 | 0.612 | 0.1201 | 585 | 454 | 0.6269 | 0.1349 | 585 | 741 | 0.6349 | 0.1429 | 585 | 1202 | 0.6685 | 0.1765 |
| kh24 | 3 | arch_1 | stepwise_lock | 0.5333 | 270 | 0 | 0.7237 | 0.1904 | 237 | 85 | 0.82 | 0.2867 | 219 | 177 | 0.8401 | 0.3067 | 188 | 278 | 0.8299 | 0.2965 |
| kh24 | 4 | arch_1 | untradeable | 0.5102 | 215 | 0 | 0.6839 | 0.1737 | 182 | 85 | 0.7443 | 0.2341 | 162 | 177 | 0.7443 | 0.2342 | 127 | 278 | 0.6176 | 0.1074 |
| kh24 | 4 | arch_3 | stepwise_lock | 0.5506 | 70 | 0 | 0.5761 | 0.0255 | 70 | 85 | 0.6377 | 0.0871 | 70 | 177 | 0.6534 | 0.1028 | 70 | 278 | 0.7037 | 0.153 |
| kh24 | 5 | arch_3 | stepwise_lock | 0.5015 | 66 | 0 | 0.5199 | 0.0184 | 66 | 85 | 0.5839 | 0.0823 | 66 | 177 | 0.6584 | 0.1569 | 66 | 278 | 0.6572 | 0.1557 |
| kh24 | 5 | arch_4 | mixed | 0.4754 | 147 | 0 | 0.6032 | 0.1279 | 120 | 85 | 0.675 | 0.1996 | 103 | 177 | 0.6625 | 0.1872 | 87 | 278 | 0.6548 | 0.1794 |
| kh24 | 6 | arch_2 | stepwise_lock | 0.5315 | 65 | 0 | 0.5752 | 0.0437 | 65 | 85 | 0.6252 | 0.0937 | 65 | 177 | 0.6637 | 0.1322 | 65 | 278 | 0.6732 | 0.1417 |
| kh24 | 6 | arch_4 | stepwise_lock | 0.4434 | 133 | 0 | 0.6072 | 0.1639 | 116 | 85 | 0.7012 | 0.2578 | 104 | 177 | 0.6571 | 0.2137 | 91 | 278 | 0.6294 | 0.186 |
| kh24 | 7 | arch_1 | stepwise_lock | 0.4592 | 131 | 0 | 0.6039 | 0.1447 | 114 | 85 | 0.7097 | 0.2506 | 102 | 177 | 0.6856 | 0.2265 | 90 | 278 | 0.6793 | 0.2201 |
| kh24 | 7 | arch_3 | mixed | 0.5682 | 59 | 0 | 0.5682 | -0 | 59 | 85 | 0.6239 | 0.0556 | 59 | 177 | 0.6536 | 0.0853 | 59 | 278 | 0.6814 | 0.1132 |
| kh24 | 6 | group_stepwise_lock | stepwise_lock | 0.4789 | 198 | 0 | 0.6433 | 0.1643 | 181 | 85 | 0.7309 | 0.252 | 169 | 177 | 0.7369 | 0.258 | 156 | 278 | 0.8034 | 0.3245 |

**Angle C clears AUC >= 0.6** at smallest t for:

| dataset | K | target_id | smallest_t_clearing | auc_at_t | n_excluded_at_t | target_size_at_t |
| --- | --- | --- | --- | --- | --- | --- |
| arc2 | 3 | arch_0 | 1 | 0.6196 | 112 | 772 |
| arc2 | 4 | arch_2 | 1 | 0.6108 | 112 | 765 |
| arc2 | 5 | arch_1 | 1 | 0.6063 | 112 | 621 |
| arc2 | 6 | arch_2 | 1 | 0.6085 | 112 | 604 |
| arc2 | 7 | arch_2 | 1 | 0.612 | 112 | 585 |
| kh24 | 3 | arch_1 | 1 | 0.7237 | 0 | 270 |
| kh24 | 4 | arch_1 | 1 | 0.6839 | 0 | 215 |
| kh24 | 4 | arch_3 | 3 | 0.6377 | 85 | 70 |
| kh24 | 5 | arch_3 | 5 | 0.6584 | 177 | 66 |
| kh24 | 5 | arch_4 | 1 | 0.6032 | 0 | 147 |
| kh24 | 6 | arch_2 | 3 | 0.6252 | 85 | 65 |
| kh24 | 6 | arch_4 | 1 | 0.6072 | 0 | 133 |
| kh24 | 7 | arch_1 | 1 | 0.6039 | 0 | 131 |
| kh24 | 7 | arch_3 | 3 | 0.6239 | 85 | 59 |
| kh24 | 6 | group_stepwise_lock | 1 | 0.6433 | 0 | 198 |

## §5 Cross-angle synthesis

| dataset | K | target_id | logistic_basic | rf_basic | logistic_expanded | rf_expanded | rf_at_t3 | rf_at_t5 | best_>=_0.60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc2 | 3 | arch_0 | 0.5114 | 0.5082 | 0.5382 | 0.5788 | 0.6366 | 0.6516 | True |
| arc2 | 4 | arch_2 | 0.5022 | 0.4911 | 0.5315 | 0.576 | 0.6335 | 0.6481 | True |
| arc2 | 5 | arch_1 | 0.4989 | 0.5021 | 0.5266 | 0.5545 | 0.6274 | 0.6471 | True |
| arc2 | 6 | arch_2 | 0.4929 | 0.4951 | 0.5263 | 0.5548 | 0.6274 | 0.6322 | True |
| arc2 | 7 | arch_2 | 0.5043 | 0.492 | 0.5336 | 0.5679 | 0.6269 | 0.6349 | True |
| kh24 | 3 | arch_1 | 0.5625 | 0.5333 | 0.5712 | 0.534 | 0.82 | 0.8401 | True |
| kh24 | 4 | arch_1 | 0.569 | 0.5102 | 0.5501 | 0.5267 | 0.7443 | 0.7443 | True |
| kh24 | 4 | arch_3 | 0.5693 | 0.5506 | 0.6647 | 0.6421 | 0.6377 | 0.6534 | True |
| kh24 | 5 | arch_3 | 0.5638 | 0.5015 | 0.6614 | 0.6171 | 0.5839 | 0.6584 | True |
| kh24 | 5 | arch_4 | 0.5356 | 0.4754 | 0.5324 | 0.536 | 0.675 | 0.6625 | True |
| kh24 | 6 | arch_2 | 0.5713 | 0.5315 | 0.64 | 0.6146 | 0.6252 | 0.6637 | True |
| kh24 | 6 | arch_4 | 0.5556 | 0.4434 | 0.522 | 0.514 | 0.7012 | 0.6571 | True |
| kh24 | 7 | arch_1 | 0.5382 | 0.4592 | 0.5136 | 0.5027 | 0.7097 | 0.6856 | True |
| kh24 | 7 | arch_3 | 0.5505 | 0.5682 | 0.6556 | 0.6516 | 0.6239 | 0.6536 | True |
| kh24 | 6 | group_stepwise_lock | 0.5395 | 0.4789 | 0.5509 | 0.5537 | 0.7309 | 0.7369 | True |

## §6 Headline finding

### **(A) Wall broken at entry.**

At least one target archetype has a non-t>0 angle AUC >= 0.6.

- Angle B (expanded):
  - kh24 K=4 arch_3: AUC = 0.6647
  - kh24 K=5 arch_3: AUC = 0.6614
  - kh24 K=6 arch_2: AUC = 0.6400
  - kh24 K=7 arch_3: AUC = 0.6556

**Reframe is validated; pipeline retains entry-time filter.**

## §7 Open observations

- 15 target(s) gained >= 0.10 AUC by observing path-so-far at small t. Suggests archetype identifiability is path-driven, not entry-driven.
- kh24: 278/553 (50.3%) trades exited before t=10 — deferred-identification using t=10 cannot act on those, halving the addressable pool.
- **kh24**: 4/10 target(s) clear AUC >= 0.6 at entry (Angle A or B): K=4 arch_3, K=5 arch_3, K=6 arch_2, K=7 arch_3. Entry-time filter feasible for those targets.
- **arc2**: 0/5 targets clear AUC >= 0.6 at entry. Predictability wall holds at entry on this dataset; only t>0 angle clears it.

## Files produced

| path | sha256 | bytes |
| --- | --- | --- |
| angle_A_rf_basic.csv | 79e78baa23fb7c31afa000149202fc03f7698e49a438ba4bd9e7476f6ef9fd31 | 3511 |
| angle_B_expanded.csv | f64d0187e18354301dabe97a18c12127c144922e507706ee89767ba45120e1e9 | 3070 |
| angle_C_t_observation.csv | 12514e0e6d32a9482c7778af0731450d2eab257b46dee508af72680b8239a152 | 5282 |
| arc2/entry_features_expanded.csv | dd877a8fcae28935915d97c5c35a3392a2796d6435a1e660e427557c48353d29 | 1870656 |
| arc2/expanded_feature_list.csv | 9f0826eee0150bf75feea798168c82609dd8eca63e80f3adf180b4b04d392856 | 730 |
| exit_family_grouping.csv | a02abebbe3b8dfbd0dbf891188e78fd9439e9775ab66b2b51a732102fa318a15 | 1226 |
| grouped_targets.csv | 9152da942b70e54c24df323f250defbebd4a483e9de7afbc8f98644591bbc9b9 | 316 |
| kh24/entry_features_expanded.csv | 24988f22ef9832b18786e742630b1381e4b6004dfefe698fe20886ef26b2c6e9 | 264977 |
| kh24/expanded_feature_list.csv | c4b57e3a57054ee05a0c1c4c9f60a881ff8c2ce7060863e8c762a29aca88ae2f | 352 |
| target_archetypes.csv | 98d8d0a60f9b45538bea81319a524ab1092858e4fe89a95ab146498511ccd7f9 | 4572 |
| targets_long.csv | 127e6f1057674b0e97c43b8c1202c7bc4935f85a302a00115484ca4b2a805478 | 617 |
