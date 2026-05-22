# ARC_8_CLOSURE — l_arc_8

> **Closed:** 2026-05-22T11:00:00Z
> **Branch:** arc/l_arc_8
> **Closure doc path:** results/l_arc_8/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:

  # ────── Identity ──────
  arc_name: l_arc_8
  signal: pullback_resume_hhhl_long_v0.1 (HH/HL uptrend, pullback >=0.5xATR, bullish-close break of prior bar)
  tf: 4H
  sub_protocol: vanilla
  closed_timestamp: 2026-05-22T11:00:00Z
  closure_doc_link: results/l_arc_8/ARC_CLOSURE.md

  # ────── Verdict ──────
  verdict: FAIL
  one_line: Search WFO worst-fold ratio 1.749 (gate >=2.0) on top A6 config; oracle 999 confirms V-shape capturable but Step 4 AUC 0.53 entry-feature ceiling caps real WFO.
  failed_at_step: 5
  primary_failure_mode: entry_feature_auc_ceiling

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: 6757
    window_start: 2010-01-01
    window_end: 2026-04-10
    kh24_co_fire_pct: 0.0
    configs_evaluated_step5: 72
    search_scope_flag: normal

  # ────── Best architecture (top of search-WFO ranking; FAIL so candidate not deployable) ──────
  best_architecture:
    name: A6 meta_labeling
    cluster: 2
    archetype: V-shape
    config: A6_SL1.5_TP3R_unlimited_thr_0.3_0.5
    sl_atr: 1.5
    exit_policy: sl_plus_tp_3r
    exposure_cap: unlimited
    worst_fold_ratio: 1.749
    worst_fold_roi_pct: 1.7486
    worst_fold_dd_pct: 1.9826
    mean_fold_ratio: 7.468
    mean_fold_roi_pct: 5.902
    sign_pos_folds: "11/11"
    n_trades_total: 4434
    holdout_roi_pct: 42.7803
    holdout_dd_pct: 1.7369
    holdout_passed: true
    oracle_worst_ratio: 999.0
    oracle_real_gap_sharpe: null
    features_in_winning_config: [swing_low_distance_14, prior_session_low_distance, atr_vs_trailing_100, kijun_26_distance, dollar_bloc_state, atr_percentile_100, spread_vs_trailing_100, spread_percentile_100, usd_strength_index, distance_to_round_number]

  # ────── Cost decomposition ──────
  # A6 is sizing-based, not binary admit/reject — the admit/reject framing
  # doesn't capture (0x / 0.5x / 1.0x) tiers cleanly. Set null per template
  # ambiguity; the (lo, hi) threshold pair + per-tier R lives in Step 5's
  # per_fold_metrics.csv.
  cost_decomposition: null

  # ────── Per-cluster results ──────
  clusters:
    c0:
      n: 1657
      archetype: Monotonic_down
      sl_atr: 4.0
      step3_composite: 0.393
      mfe_p50_r: 0.069
      ww_pp: 0.0284
      reach_1r: 0.0
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c1:
      n: 3560
      archetype: Choppy
      sl_atr: 1.5
      step3_composite: 0.562
      mfe_p50_r: 1.246
      ww_pp: 0.4003
      reach_1r: 0.5978
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c2:
      n: 1540
      archetype: V-shape
      sl_atr: 1.5
      step3_composite: 1.000
      mfe_p50_r: 7.527
      ww_pp: 0.0
      reach_1r: 1.0
      step4_e_auc: 0.5300
      step4_d1_auc: null
      outcome: dies_step5

  # ────── Architecture results ──────
  architectures_tested: [A1, A6]
  architecture_results:
    A1: {tested: true, won: false, worst_fold_ratio: 0.171}
    A6: {tested: true, won: true,  worst_fold_ratio: 1.749}
    A3: {tested: false, won: false, worst_fold_ratio: null}

  # ────── Archetypes observed ──────
  archetypes_observed: [Monotonic_down, Choppy, V-shape]

  # ────── Cross-arc observation tags ──────
  cross_arc_tags:
    - v_shape_auc_ceiling
    - step4_classifier_at_chance
    - oracle_real_gap_massive
    - multi_tf_features_absent_at_step1
    - search_wfo_fail_holdout_pass_pattern
```

---

## §2 Why failed

The arc died at the L_PROTOCOL §3 PASS-DEPLOYABLE gate on Step 5 — best A6 worst-fold ratio 1.749 against a 2.0 threshold (margin −0.251). The proximate cause is the gate miss; the structural cause is **Step 4's entry-feature AUC ceiling at 0.53** across all three classifiers (RF 0.530, LGBM 0.521, Logistic 0.523), which means the meta-labeling sizing in A6 has almost no separation power. With AUC ≈ chance, A6 effectively sizes all trades alike and the architecture collapses toward a baseline-with-noise — which produces 11/11 positive folds with bounded magnitude (worst ROI +1.75% over a 1.98% DD) but cannot lift the worst-fold ratio above the gate.

The Step 5 **oracle** WFO at TRUE cluster-2 labels hits worst-fold ratio 999 (effectively infinite — every fold clean +60-135% ROI on the 638-980 cluster-2 trades, DD ≤ 1%) — the cluster is genuinely capturable, the failure is at the entry-time identification step. This is the same V-shape AUC-ceiling pattern that closed Arc 10 (DLR, 2026-05-18): Step 4 near-miss on the disjunctive gate, oracle WFO comfortable, real WFO blocked at the classifier.

The A1 (no-filter) family is much worse: best A1 worst-fold ratio 0.171 (config 4: SL=1.5, sl_plus_tp_3r, unlimited). A1 admits the full 6,757-trade pool where 77% are losers (C0 + C1) — the small cluster-2 edge gets washed out. So A6's edge over A1 (1.749 vs 0.171, ~10×) is real even at AUC 0.53 — sizing differentiation matters — but the absolute ratio is still below gate.

Note on what the holdout shows: the top-3 holdout (2021-04 → 2026-04, one-shot) is **PASS-DEPLOYABLE on all three** (ROI +19% to +43%, DD ≤ 1.7%, ratios 19-35). L_PROTOCOL §2 Step 5 holdout-decision-rule requires BOTH the 11-fold search WFO and the holdout to pass the §3 gate; search WFO doesn't, so the arc verdict is FAIL despite the holdout strength. Per §2 Step 5: "Re-evaluating the same candidate after parameter tweaks is forbidden (selection bias laundering)." The asymmetric pattern (search-WFO worst-ratio just below gate, holdout comfortably above) is itself a documented cross-arc tag — see §3.

Two methodology caveats noted in the closure for cross-arc context: (1) the pool-level WFO approximation does NOT use the bar-by-bar multipair backtester (`MultiPairBacktester`); per-trade R outcomes from Step 1 carry through with exit-policy approximation. This understates intraday DD timing and ignores cross-pair concurrency interactions; (2) Step 1 multi_tf features (`d1_close_slope_*`, `w1_close_slope_sign`, `d1_atr_percentile_100`) came back all-NaN because `compute_feature_matrix` requires `panel.aux={"d1":..., "w1":...}` but the Step 1 driver only passed an H4 panel; the fix is staged but not re-run. Effective Step 4 feature catalogue was 23 features (the 27-feature v3 catalogue minus those four multi_tf entries). This is the load-bearing engine-integration gap of this arc.

---

## §3 Cross-arc observations

- **Third documented V-shape near-miss in 6 months** — Arc 6 (failed_breakout_reversal, Step 4 E AUC 0.600), Arc 10 (DLR c1, E AUC 0.6296 / D1 0.5897), Arc 8 (this arc, Step 4 RF AUC 0.5300). All three have clean capturability (Step 3 composite ≥ 0.49) and fail at the entry-time AUC barrier. Arc 8 is the lowest AUC of the three — confirms the pattern is a function of the v3 27-feature envelope, not signal-specific. EXP-05 cross-arc pool framing (Arc 10's c1 + Arc 7's c3 pooled, AUC 0.6348 with shared 17-feature subset) is now joinable with Arc 8 c2 — three-arc V-shape clusterifier pool would add +1,540 trades to that envelope.
- **Oracle/real WFO ratio gap is the largest documented yet** — Oracle worst ratio 999.0 vs real worst ratio 1.749 (gap factor >500×). Arc 10 showed Sharpe 4.61 vs −1.29 (gap +5.90). Arc 8's gap is wider because cluster-2 is more homogeneous (100% reach_1R, 0% ww_pp) — the upside is whoppingly capturable; the entry classifier just can't find it. Confirms that "envelope expansion" (richer entry-time features) is the deployable lever for V-shape archetypes, not architecture variation.
- **Multi-TF feature absence amplifies AUC ceiling** — Step 4 ran on 23 features (D1 + W1 slope/percentile features absent due to engine-integration gap). The top permutation-importance features came from price geometry, distance, and spread regime classes; cross-asset features (`dollar_bloc_state`, `usd_strength_index`) ranked top-5/9. Re-running Step 1 with the aux-panel fix would restore 4 features, two of which (D1 slope sign, D1 ATR percentile) are HTF context that historically lifts V-shape classification (Arc 10 EXP-05 found `L1_minus_L0_atr`, an HTF magnitude feature, carried 116% of the cross-arc LOO drop). Engine-PR scope.
- **Search WFO FAIL + holdout PASS-DEPLOYABLE is a new pattern** — Top-3 holdout all clear the deployable gate (worst holdout ratio 19.4) while search WFO worst ratio is 1.749. Per §2 Step 5 the arc verdict is FAIL (both gates required). But the asymmetry is worth tagging: 2021-2025 holdout had a more favourable market regime for V-shape pullback-resume than 2010-2020 search. Cross-arc question: do other v3 arcs show this asymmetry? If yes, the 11-fold 2010-2020 search WFO may be biased toward a different regime than the deployment-window holdout.
- **A6 with chance-AUC classifier still adds value** — A6 best (ratio 1.749) is 10× A1 best (ratio 0.171). Even at AUC 0.53, sizing differentiation moves the needle. This argues for A6-style sizing being a useful default architecture regardless of classifier strength, provided sizes are bounded (here: 0x/0.5x/1.0x mapping). Worth a v2.4 calibration thought.
- **KH-24 co-fire on PR-HHHL: 0.000% across 6,757 signals** — Confirms the spec's "independence expected" prior. PR-HHHL bullish-resume and KH-24 bearish-exhaustion are structurally orthogonal. This frees PR-HHHL from any portfolio-overlap accounting if it's ever resurrected.
