# ARC_11_CLOSURE — l_arc_11

> **Closed:** 2026-05-22T10:53:04Z
> **Branch:** arc/l_arc_11
> **Closure doc path:** results/l_arc_11/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:
  arc_name: l_arc_11
  signal: swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4)
  tf: H4
  sub_protocol: vanilla
  closed_timestamp: '2026-05-22T10:53:04Z'
  closure_doc_link: results/l_arc_11/ARC_CLOSURE.md
  verdict: FAIL
  one_line: Capturable cohort + RF AUC 0.687; A2 worst-fold ratio 3.18 — FAIL on worst-fold DD 11.96% > 10% gate at risk=0.5%.
  failed_at_step: 5
  primary_failure_mode: step5_dd_above_gate
  pool_metadata:
    total_n: 7149
    window_start: '2010-01-01'
    window_end: '2026-04-30'
    kh24_co_fire_pct: null
    configs_evaluated_step5: 36
    search_scope_flag: thin
  best_architecture:
    name: A2 classifier_filter
    cluster: 0
    archetype: Choppy
    config: cluster0_A2_SL2.5_EXsl_plus_trailing_atr_1r_CAPNone
    sl_atr: 2.5
    exit_policy: sl_plus_trailing_atr_1r
    exposure_cap: unlimited
    worst_fold_ratio: 3.1808
    worst_fold_roi_pct: 23.858
    worst_fold_dd_pct: 11.9581
    mean_fold_ratio: 6.0271
    mean_fold_roi_pct: 45.1052
    sign_pos_folds: 10/11
    n_trades_total: 2873
    holdout_roi_pct: 617.4289
    holdout_dd_pct: 16.8967
    holdout_passed: false
    oracle_worst_ratio: 16.5779
    oracle_real_gap_sharpe: 13.3971
    features_in_winning_config:
    - w1_close_slope_sign
    - kijun_26_distance
    - day_of_week
    - break_magnitude_atr
    - eur_strength_index
    - swing_low_distance_14
    - atr_vs_trailing_100
    - atr_percentile_100
    - distance_to_round_number
    - usd_strength_index
  cost_decomposition:
    admit_pool:
      n_fraction: 0.3229
      mean_r: 2.0041
    reject_pool:
      n_fraction: 0.6771
      mean_r: -0.995
    early_exit_pool:
      n_fraction: 0.0
      mean_r: 0.0
  clusters:
    c0:
      n: 2266
      archetype: Choppy
      sl_atr: 2.0
      step3_composite: 0.9908
      mfe_p50_r: 4.6619
      ww_pp: 0.015
      reach_1r: 0.9846
      step4_e_auc: 0.6873
      step4_d1_auc: null
      outcome: dies_step5
    c1:
      n: 4752
      archetype: Unclassified
      sl_atr: 3.0
      step3_composite: 0.2723
      mfe_p50_r: 0.4715
      ww_pp: 0.1084
      reach_1r: 0.1303
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
  architectures_tested:
  - A1
  - A2
  architecture_results:
    A1:
      tested: true
      won: false
      worst_fold_ratio: -0.6697
    A2:
      tested: true
      won: true
      worst_fold_ratio: 3.1808
  archetypes_observed:
  - Choppy
  - Unclassified
  cross_arc_tags:
  - dd_gated_at_chosen_risk_size
  - strong_holdout_blocked_by_wfo_dd
  - step4_auc_above_065_v3_first
  - choppy_label_misnamed_capturable_cohort
  - shb_swing_detection_causal_clean_arc9_lesson_passed
```

---

## §2 Why failed

Arc 11 produces a verdict-FAIL result that sits squarely on the §3 PASS-VIABLE boundary — by every other gate the arc clears, but it fails the chosen-risk-size DD ceiling.

**Proximate cause.** The best A2 config (cluster0_A2_SL=2.5×ATR, sl_plus_trailing_atr_1r exit, unlimited per-currency exposure) reaches worst-fold ratio 3.18 (≥ 2.0 ✓), mean-fold ratio 6.03 (≥ 2.5 ✓), worst-fold ROI +23.86% (> 0 ✓), 10/11 sign-pos folds (PASS-VIABLE permits one negative ✓), and 0 daily-DD breaches (✓). It FAILS only on worst-fold DD 11.96% > 10% — the 5ers hard-limit gate cited in PASS-VIABLE. Even Oracle WFO (true cluster-0 membership filter, no classifier) records worst_dd 10.15% — the cohort's intrinsic drawdown at risk=0.5% is right at the boundary.

**Structural cause.** Cluster 0 has extreme MFE potential (reach_1R 0.985, mfe_p50 4.66R) but the give-back from peak to time-exit means concentrated negative tails on losing trades. With trade size locked at 0.5% per protocol, the worst fold's loss concentration exceeds 10pp of equity.

**What this tells us.** The §3 DD gate is sized 'at chosen risk size' — for cohorts whose intrinsic ratio is healthy but volatility is high relative to the 10% ceiling, risk-size choice at arc-open is load-bearing. SHB at risk=0.42% would clear PASS-VIABLE (DD scales linearly). The signal class is viable; the arc-open risk choice was not. The v3 protocol's DD-at-fixed-risk rule correctly classifies this as FAIL rather than letting a re-sizing trick across the gate after the fact.

---

## §3 Cross-arc observations

- First v3.0 arc to clear Step 4 entry-feature gate (RF mean OOS AUC 0.687, above the 0.65 disjunctive floor) and survive all four steps to Step 5 — establishes that the v3 27-feature default envelope CAN extract for this signal class with the right cluster.
- Cluster 0 holdout +617% ROI / 16.9% DD over 2021-2026-04 (5+ years, one-shot) is the strongest holdout signal in v3.0 to date. WFO–holdout sign-consistency is preserved: both fail the 10% DD ceiling, neither shows the classic holdout-collapse pattern seen in v2 arcs.
- DD-at-chosen-risk-size as a failure mode: first instance under v3.0 of an arc that clears every §3 ratio + sign + ROI check but fails purely on the absolute DD cap. Suggests adding a 'risk-size sensitivity' diagnostic to Step 5: report min risk_pct at which the arc would clear PASS-VIABLE / PASS-DEPLOYABLE.
- Choppy-archetype label is structurally misleading for high-MFE-with-give-back cohorts (cluster 0: 180-bar median hold, 46 mean peaks per trade, but mfe_p50 4.66R and ww_pp 0.015). Step 5 architecture mapping treats Choppy → no archs; this arc applied an override (Choppy + §3 candidate → Stepwise architecture set) which yielded the only viable Step 5 candidate. The taxonomy could use a 'long-runner' / 'oscillating-trender' tag distinct from the failure-mode Choppy.
- Swing-detection producer-level causal audit (Arc 9 lesson) PASS at 10/10 lookahead spot-check trades with full h_ref re-compute match. The 3-bar swing + RIGHT_EDGE_OFFSET=4 idiom is causally clean by construction — confirms the dispatch's whitelist of confirmation-lag variants as a viable design.
