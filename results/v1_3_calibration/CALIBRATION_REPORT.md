# v1.3 capturability calibration — KH-24 + Arc 1 + Arc 2 metrics

> Date: 2026-05-15
> Branch: feature/v1.3-calibration-data

Pure computation output for the v1.3 capturability calibration. Three datasets — KH-24 (the project's only known real edge, must pass any v1.3 floors), Arc 1 (verbatim, step-3 cluster context), Arc 2 (verbatim, step-3 cluster context). The chat sets v1.3 floors after reading this.

---

## §1 Schema audit

See `schema_audit.md` for the full mapping table. Summary:

- KH-24: compatible (reference); ATR-units → R-units via /SL_MULT=2.
- Arc 1: **partial** — no per-bar OHLC. TP/SL/MFE-lock simulations use running mfe/mae (exact at first-cross bar). Trail simulation falls back to close-based detection (derived `close_r` from `fwd_logret_cum`).
- Arc 2: compatible — OHLC + running mfe/mae available.

R-unit convention: all values in SL-distance R-units, R = 2 × ATR_at_entry. KH-24 uses 4H ATR; Arc 1 / Arc 2 use 1H ATR. The 2× SL multiplier is constant; the ATR-timeframe mismatch is acknowledged for absolute-magnitude metrics, immune for dimensionless capture ratios. See `scripts/v1_3_calibration/loader_decisions.md` for full conversions.

---

## §2 Side-by-side metric table (pool level)

| Metric | KH-24 | Arc 1 | Arc 2 |
|---|---|---|---|
| **Axis 1: Peak Magnitude** | | | |
| n_trades | 553 | 45673 | 3993 |
| pool_median_fwd_mfe_h120 (R) | 2.2926 | 2.3137 | 2.4224 |
| pool_median_fwd_mfe_h240 (R) | 3.2262 | 3.2098 | 3.3590 |
| pool_frac_reach_1R | 0.8228 | 0.8470 | 0.8507 |
| pool_frac_reach_1_5R | 0.7468 | 0.7653 | 0.7806 |
| pool_frac_reach_2R | 0.6745 | 0.6840 | 0.7002 |
| pool_p90_fwd_mfe_h240 (R) | 8.2581 | 8.0861 | 8.0477 |
| pool_p95_fwd_mfe_h240 (R) | 10.4779 | 9.8734 | 9.6264 |
| pool_p99_fwd_mfe_h240 (R) | 15.5861 | 14.4790 | 13.3442 |
| **Axis 2: Peak Capture (best of family)** | | | |
| time_exit_best_h | 24 | 1 | 1 |
| time_exit_best_capture | 0.0988 | 0.0066 | 0.0464 |
| trail_exit_best_W | 1.0000 | 0.3000 | 0.3000 |
| trail_exit_best_capture | 0.0238 | -0.0020 | 0.0053 |
| tp_exit_best_X | 3.0000 | 0.5000 | 3.0000 |
| tp_exit_best_capture | 0.0479 | -0.0006 | 0.0094 |
| mfe_lock_best_X | 0.5000 | 0.5000 | 0.5000 |
| mfe_lock_best_capture | 0.0372 | 0.0332 | 0.0470 |
| local_peaks_pool_median | 2.0000 | 0.0000 | 4.0000 |
| monotonicity_pool_median | 0.5385 | 0.5000 | 0.5217 |
| time_to_peak_cv | 0.7181 | 0.6907 | 0.6883 |
| time_to_peak_p50 (bars) | 114.0 | 122.0 | 126.0 |
| frac_reentry_candidates | 0.5461 | 0.5317 | 0.5520 |
| **Axis 3: Path Hostility** | | | |
| race_condition_median (bars) | -14.0 | 2.0 | -3.0 |
| mae_mfe_ratio_winners_median | 0.1770 | 0.3093 | 0.1376 |
| pct_peak_and_collapse | 0.4141 | 0.3789 | 0.5692 |
| pct_wrong_way | 0.4774 | 0.4961 | 0.4884 |
| **Shape (pool) + Mass-in-band** | | | |
| shape_tag | bimodal | bimodal | bimodal |
| shape_p50 | 3.2262 | 3.2098 | 3.3590 |
| shape_p95 | 10.4779 | 9.8734 | 9.6264 |
| shape_p99 | 15.5861 | 14.4790 | 13.3442 |
| mass_band_0_to_0_5R | 0.0940 | 0.0737 | 0.0741 |
| mass_band_0_5_to_1R | 0.0832 | 0.0793 | 0.0751 |
| mass_band_1_to_2R | 0.1483 | 0.1630 | 0.1505 |
| mass_band_2_to_5R | 0.3617 | 0.3943 | 0.3974 |
| mass_band_above_5R | 0.3128 | 0.2897 | 0.3028 |

---

## §3 Per-cluster metrics (Arc 1 + Arc 2)

### arc1 — K3_kmeans clusters

| cluster_id | n_trades | frac_of_pool | pool_median_fwd_mfe_h240 | pool_frac_reach_1R | pool_frac_reach_2R | axis3_pct_peak_and_collapse | axis3_pct_wrong_way | shape_tag | mass_band_2_to_5R | mass_band_above_5R |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 14882 | 0.3258 | 5.7419 | 0.9996 | 0.9889 | 0.2855 | 0.0618 | bimodal | 0.3819 | 0.6070 |
| 1 | 16230 | 0.3554 | 1.6616 | 0.6954 | 0.4120 | 0.3353 | 0.6658 | bimodal | 0.3677 | 0.0443 |
| 2 | 14561 | 0.3188 | 3.0007 | 0.8600 | 0.6756 | 0.5230 | 0.7507 | bimodal | 0.4367 | 0.2389 |

### arc2 — K3_kmeans clusters

| cluster_id | n_trades | frac_of_pool | pool_median_fwd_mfe_h240 | pool_frac_reach_1R | pool_frac_reach_2R | axis3_pct_peak_and_collapse | axis3_pct_wrong_way | shape_tag | mass_band_2_to_5R | mass_band_above_5R |
|---|---|---|---|---|---|---|---|---|---|---|
| -2 | 112 | 0.0280 | 2.4534 | 0.6875 | 0.5804 | 0.6161 | 0.8304 | bimodal | 0.3750 | 0.2054 |
| 0 | 693 | 0.1736 | 3.1611 | 0.9798 | 0.7937 | 0.6912 | 0.1328 | bimodal | 0.6075 | 0.1861 |
| 1 | 2334 | 0.5845 | 2.4611 | 0.7656 | 0.5733 | 0.7104 | 0.7558 | bimodal | 0.3612 | 0.2121 |
| 2 | 854 | 0.2139 | 6.0700 | 1.0000 | 0.9871 | 0.0785 | 0.0012 | bimodal | 0.3290 | 0.6581 |

---

## §4 Distribution shape commentary

### kh24
Shape: **bimodal**. Pool fwd MFE p50 = 3.2262 R, p99 = 15.5861 R. `frac_reach_1R` = 0.8228; `frac_reach_2R` = 0.6745. Best time-exit capture 0.0988 at h=24; best trail capture 0.0238 at W=1.0000. Race-condition median = -14.0 bars (negative = direction wins; first +1R hit before first −1R hit).

### arc1
Shape: **bimodal**. Pool fwd MFE p50 = 3.2098 R, p99 = 14.4790 R. `frac_reach_1R` = 0.8470; `frac_reach_2R` = 0.6840. Best time-exit capture 0.0066 at h=1; best trail capture -0.0020 at W=0.3000. Race-condition median = 2.0 bars (negative = direction wins; first +1R hit before first −1R hit).

### arc2
Shape: **bimodal**. Pool fwd MFE p50 = 3.3590 R, p99 = 13.3442 R. `frac_reach_1R` = 0.8507; `frac_reach_2R` = 0.7002. Best time-exit capture 0.0464 at h=1; best trail capture 0.0053 at W=0.3000. Race-condition median = -3.0 bars (negative = direction wins; first +1R hit before first −1R hit).

---

## §5 Unavailable / approximated metric flags

- **Arc 1** has no per-bar high/low/close in step 2's `trade_paths.csv`. `high_r` / `low_r` are emitted as NaN. Downstream impact:
  - TP / SL / MFE-lock simulations: **exact** (use running mfe/mae which increments at the first bar that crosses the threshold).
  - Trail-exit simulation: **close-based fallback** (intrabar-low trail detection not possible; uses derived `close_r` from `fwd_logret_cum`). Approximation only; documented in `loader_decisions.md`.
  - Axis 2e conditional predictivity uses derived `close_r` for the same reason.
- **Arc 1 / Arc 2** `entry_px` uses `signals_features.signal_bar_close` (one 1H-bar approximation of the true entry-bar open). Sub-ATR drift across one bar; documented.
- **ATR-timeframe mismatch**: KH-24 uses 4H ATR for SL; Arc 1 / Arc 2 use 1H ATR. Absolute-magnitude comparisons (e.g. `frac_reach_1R`) inherit this; capture-ratio comparisons are dimensionless and immune.

---

## §6 Files produced

| path | sha256 | bytes |
|---|---|---|
| _all_pool_metrics.csv | b5f53cc0a5805dbf483408e504b0f358fa186ce334f31b699a611a5a5e176918 | 3467 |
| arc1/axis1_full_distribution.csv | b2b37719886ecb1c33e88e5e42d07be573f3e9e8d781f9fffc00a45e1fc67c34 | 305 |
| arc1/axis2a_time_exit_curve.csv | 09bf2ab313497af12ef858016c6928f5a9184e3e16b2273930aab15987c2acba | 564 |
| arc1/axis2b_trail_exit_curve.csv | f06616e584126ee46318bf2146fb0db74b4c17bd7e9a2f9cfa008f03d325151d | 383 |
| arc1/axis2c_tp_exit_curve.csv | 5b8d946de6ecbe6c62ca2b0144d01c23fc79db09e9d7fdd8afe6d6883e6bd33e | 449 |
| arc1/axis2d_mfe_lock_curve.csv | ab15109ecb32d8294b3b3f0589fbace079333948dc0de5970e4868f660aa33a3 | 300 |
| arc1/axis2e_conditional_predictivity.csv | 177aaa0855fd81c32c77bdfc4897fc19b01005657b4be6a325c3623b50254e4d | 1233 |
| arc1/axis2f_reentry_descriptive.csv | 58217582bd7cc898e34d5df817610a40115ef19212abb4eef0413ff7d1ec2276 | 75 |
| arc1/axis2g_in_trade_smoothness.csv | 05d01b518e573279d01091b91b1867dfa83ca8101b3ee2f24f21f45f75e92e29 | 487326 |
| arc1/axis2g_in_trade_smoothness_pool_summary.csv | 15f9532fb82b5e977097353aab9ce7d3a196f5e3c16fac8ac8f96ed545683238 | 99 |
| arc1/axis3_distributions.csv | f516c639b0ce963c234799dd783707ff5b0ff8679eed71a36043720b67ae0d23 | 807082 |
| arc1/per_cluster_metrics.csv | 369c189f4d2fe46490bd5503eaf61111327688ed70a3594396d05e34a05ad23c | 2223 |
| arc2/axis1_full_distribution.csv | ef928b0cae3d6c5fbd99dae831d41a0513fa7af5e6ab699830e577912dc09e84 | 306 |
| arc2/axis2a_time_exit_curve.csv | df082ecb92819994421e583ea820fe6a829e5850d2867b9177feff75decb515b | 553 |
| arc2/axis2b_trail_exit_curve.csv | c19444877c29dec9dae752efa4e65fab40efea808af080b0ccee9da0706f2b3a | 377 |
| arc2/axis2c_tp_exit_curve.csv | 55679b9023324be113a648b27fa55dfccbdcc0afab89e568fd9e4cb656a38640 | 434 |
| arc2/axis2d_mfe_lock_curve.csv | 4ab7888b867bd664655452dd09a27697b5dadb747dffd4b3db66715e1892adc3 | 296 |
| arc2/axis2e_conditional_predictivity.csv | 6597cdf3995388e3bf09c531ca6ec2c83fc757226413e4fb7f5a7cf998cafe0f | 1216 |
| arc2/axis2f_reentry_descriptive.csv | 4cdeb64dab77f4109c5aef5706c5cec9b8b76d81b39d5bc055a78eddf283e398 | 74 |
| arc2/axis2g_in_trade_smoothness.csv | 026b817e5b69244cffaf3f1b2591358db3a9b24184eee5a94134d464ca61442c | 72857 |
| arc2/axis2g_in_trade_smoothness_pool_summary.csv | d8075c7d51a4405ad176a1a168686e4c1e49d6742d77ced33a6d9626b4bd2b57 | 112 |
| arc2/axis3_distributions.csv | c599efa0ff55f06e57f5eca2201703f0882456edfef7df21af38390264ba7ea9 | 57782 |
| arc2/per_cluster_metrics.csv | a8a2172e963b22eba3fbdbedf3eb766f12a5fd04888c8776aed2389720e24069 | 2668 |
| axis1_peak_magnitude.csv | 5be710f2da8699253807728a3cf0adfa05bd2ec79deb0391a0c2fbcd7001547e | 749 |
| axis2_capture_ceilings.csv | ba0ad279c90b92ba82b12e047b32ff7bd69db38b83997d4f9620fd9dbdf55e9b | 883 |
| axis2h_time_to_peak_cv.csv | 92ddf7f130f463017ec70b2977c2b0d6104509095e4b848a7872ba1dbf8ca472 | 464 |
| axis3_path_hostility.csv | fe801c3e196c3251146c96eb47ebea575ff202fabdb27af69e99072ffff8c1ec | 453 |
| kh24/axis1_full_distribution.csv | c323fa59ba10b09306addf8dfd67f6b713f40a00af6ffee6b5ab61cb341c33b2 | 305 |
| kh24/axis2a_time_exit_curve.csv | 72dba4c97a4f2fbb72737162f2885883054f5073f0dc702d4a464976272a068a | 537 |
| kh24/axis2b_trail_exit_curve.csv | d6c5b49a97c793f17df6ccbda08b3ea25fcc7d409b4715d7c75a0d2d768eefbe | 364 |
| kh24/axis2c_tp_exit_curve.csv | 250bec4b378d190fb00e54c3229bfbbd40f2586701770f768cb2c67cd46e08c2 | 425 |
| kh24/axis2d_mfe_lock_curve.csv | c5a8336afea50b0636f91d71de95590d87e0afbf0297e1731b6359a8720c9f01 | 298 |
| kh24/axis2e_conditional_predictivity.csv | 26205ab341a6e53e78a61b37c736a77a61d8d45267c2ea96c3a401f032345439 | 1225 |
| kh24/axis2f_reentry_descriptive.csv | aa381cacff5f1e2c51f3656a14482bab6edfa566491d8b30bcba91c1a204d262 | 71 |
| kh24/axis2g_in_trade_smoothness.csv | 64e927f9341c1266ac603bb3eae0824670ee61e1b0812c8e84f38ede617f6595 | 21803 |
| kh24/axis2g_in_trade_smoothness_pool_summary.csv | e3d76829a4af9e12c4b92de783fa28e54a1c9c47d0e6053e92078b47ed22d2ec | 113 |
| kh24/axis3_distributions.csv | aee6e28f774b53d041007a05b867062476f823d580e4f75a28263fa8cd48bc63 | 21692 |
| mass_in_band_pool.csv | 54d8c24b711f9d5c3d9174149e3cb2193af3a533ef9a3b0ccfaf7ed224825181 | 445 |
| shape_tags_pool.csv | 1b4617333bcabd89a562d1859aaf2f68e7d9af3fbeb6fc7496fe6d965be107af | 588 |

---

## §7 Status

**proceed-to-floor-setting** — all three datasets produced full pool-level metrics; both arcs produced per-cluster decompositions; only Arc 1's intrabar-low-trail variant is flagged as approximate (close-based fallback). The chat can now read this report and lock v1.3 floors.
