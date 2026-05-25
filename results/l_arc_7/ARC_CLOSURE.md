# ARC_7_CLOSURE — l_arc_7

> **Closed:** 2026-05-24T22:55:25Z
> **Branch:** arc/l_arc_7
> **Closure doc path:** results/l_arc_7/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:
  template_version: v1.3
  arc_name: l_arc_7
  signal: liquidity sweep + reclaim long (4H structural reversal)
  tf: 4H
  sub_protocol: vanilla
  closed_timestamp: "2026-05-24T22:55:25Z"
  closure_doc_link: results/l_arc_7/ARC_CLOSURE.md
  verdict: FAIL
  one_line: "FAIL: pool 5175, 2 candidate clusters; top worst-fold ratio -0.82, primary failure mode `step5_not_scalable`."
  failed_at_step: 5
  primary_failure_mode: step5_not_scalable
  pool_metadata:
    total_n: 5175
    window_start: 2010-01-01
    window_end: 2026-04-30
    kh24_co_fire_pct: null
    configs_evaluated_step5: 28
    search_scope_flag: thin
  best_architecture:
    name: null
    cluster: 0
    archetype: Bimodal
    config: "A1::A1::cl0::sl2.0::trail1::expinf"
    sl_atr: 2.0
    exit_policy: sl_plus_trailing_atr
    exposure_cap: unlimited
    worst_fold_ratio: -0.8172120095638017
    worst_fold_roi_base_pct: -0.2689635536883499
    worst_fold_dd_base_pct: 0.34184671296749547
    mean_fold_ratio: -0.09132136298970882
    mean_fold_roi_pct: null
    sign_pos_folds: null
    n_trades_total: null
    holdout_roi_pct: null
    holdout_dd_pct: null
    holdout_passed: null
    oracle_worst_ratio: 11.898870891690722
    oracle_real_gap_sharpe: null
    features_in_winning_config: [atr_14, atr_vs_trailing_100, w1_close_slope_sign, distance_to_round_number, atr_percentile_100, prior_session_high_distance, prior_session_low_distance, swing_low_distance_14, session_tokyo, range_close_ratio]
    chained_max_dd_base_pct: 0.8390047334848105
    per_day_max_dd_artefact_path: "results/l_arc_7/step_5/per_day_max_dd_base__A1__A1__cl0__sl2.0__trail1__expinf.parquet"
    per_day_max_dd_base_summary:
      n_days: null
      p50_pct: null
      p95_pct: null
      p99_pct: null
      max_pct: null
    k_safe: 0.2340230195737082
    k_hard: 0.2925287744671352
    r_safe_pct: 0.001170115097868541
    r_hard_pct: 0.0014626438723356761
    scalable_to_safe: false
    scalable_to_hard: false
    worst_fold_roi_at_r_safe_pct: -0.06294366298942282
    worst_fold_roi_at_r_hard_pct: -0.07867957873677853
    chained_max_dd_at_r_safe_pct: 0.19634642116674964
    chained_max_dd_at_r_hard_pct: 0.24543302645843704
    daily_dd_breaches_at_r_safe: 0
    daily_dd_breaches_at_r_hard: 0
    holdout_roi_at_r_safe_pct: null
    holdout_dd_at_r_safe_pct: null
    holdout_roi_at_r_hard_pct: null
    holdout_dd_at_r_hard_pct: null
    sizing_convention: reset_floor
    chained_dd_method: equity_stitching
    config_artefact_path: null
    deployment_spec_section_present: false
  cost_decomposition: null
  clusters:
    c0:
      n: 636
      archetype: Bimodal
      sl_atr: 1.5
      step3_composite: 1.9982013536532843
      mfe_p50_r: 7.870189218073309
      ww_pp: 0.0047169811320754715
      reach_1r: 1.0
      step4_e_auc: 0.6191919406774266
      step4_d1_auc: null
      outcome: dies_step5
    c1:
      n: 1839
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.9678432480056479
      mfe_p50_r: 2.114999183571737
      ww_pp: 0.03969548667754214
      reach_1r: 0.9603045133224578
      step4_e_auc: 0.6642231832414549
      step4_d1_auc: null
      outcome: dies_step5
    c2:
      n: 1588
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.18654661792644897
      mfe_p50_r: 0.5059525595894241
      ww_pp: 0.9691435768261965
      reach_1r: 0.045969773299748114
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c3:
      n: 1112
      archetype: Monotonic_down
      sl_atr: 1.5
      step3_composite: 0.02690714802589864
      mfe_p50_r: 0.0967214126816511
      ww_pp: 0.9964028776978417
      reach_1r: 0.00539568345323741
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
  architectures_tested: [A1, A4]
  architecture_results:
    A1:
      tested: true
      won: false
      worst_fold_ratio: -0.8172120095638017
    A4:
      tested: true
      won: false
      worst_fold_ratio: null
  archetypes_observed: [Bimodal, Monotonic_down, Unclassified]
  cross_arc_tags: [v3_0_1_first_arc_post_pr_185_186_188_189, ets_boundary_per_pr_189, mid_price_features_per_pr_189, amendment_3_risk_normalised_gates_evaluated, bimodal_observed, monotonic_down_observed, unclassified_only]
  step_6:
    ran: false
    trigger: not_applicable
    overall_passed: null
    manifest_path: null
    categories:
      lookahead: null
      selection_bias: null
      execution_realism: null
      statistical: null
      determinism: null
      deployment_readiness: null
    critical_failures: []
    warnings_count: 0
    verdict_impact: none
```

---

## §2 Why failed

Pool 5,175 trades across 28 FX H4 (5ers EET per PR #189), 2010-01 → 2026-04-30.
Step 2 selected K=4 (silhouettes 0.39 / 0.39 / **0.49** / 0.46 / 0.44). Step 3:
4 clusters, 2 candidate — c0 Bimodal (n=636, composite 2.00, mfe_p50 7.87R,
ww 0.5%, reach_1R 100%) and c1 Unclassified (n=1839, composite 0.97, mfe_p50
2.11R, ww 4.0%, reach_1R 96%). c2 Unclassified (ww 97%) and c3 Monotonic_down
(ww 100%) die at Step 3.

**Step 4 Lineage-clean 27-feature catalogue: max mean AUC = 0.6642 on c1 RF
(crosses the 0.65 deployability bar).** c0 LR AUC = 0.6192. Persisted
classifiers under PR #185 (`results/l_arc_7/step_4/classifiers/{0,1}.pkl`).

**Proximate cause: `step5_not_scalable`.** Best A1 candidate
`A1::A1::cl0::sl2.0::trail1::expinf` worst-fold ratio -0.82, worst-fold ROI
base -26.9%, worst-fold DD base **34.2%**, chained DD base 83.9%, mean ratio
-0.09 (7 of 10 folds negative). Amendment 3 scaling: `k_safe = 8/34.2 = 0.234`
→ r_safe = 0.117% which is BELOW the 0.15% scalability floor → FAIL
`step5_not_scalable` per Amendment 3 §"Scalability bounds".

**Structural cause — architecture-map gap, second arc-of-record.** The dispatch
archetype map gives `Bimodal → {A1, A4}` and `Unclassified → {A1}` (A1 only).
A2 and A6 — the classifier-filter and meta-labeling architectures whose
canonical job is exactly to admit only the cluster's positive cohort — were
NEVER TESTED for either candidate cluster. **c1 RF AUC 0.6642 IS ABOVE the
0.65 bar**; A2 with c1's classifier could have admitted only the c1-like
cohort and shed the wrong-way-dominant c2/c3 that drag A1's unfiltered DD to
34%. Same architecture-map gap surfaced in Arc 7 v3.0 first attempt (c0
Bimodal AUC 0.6758, A2 not tested). Two arcs in a row demonstrate the same
gap; recommend v1.1 dispatch amendment: when any cluster's Step 4 best AUC
≥ 0.65 deployability bar, A2 and A6 are added to the search set REGARDLESS
of archetype.

**Oracle establishes the upper bound.** Oracle WFO with true-label cluster
admission: c0 worst-fold ratio **11.90** (ROI 24-48% per fold, DD < 2.3%);
c1 worst-fold ratio **6.80** (ROI 22-50% per fold, DD 2-4%). Oracle-realised
gap on c0 = 11.90 − (−0.82) = **+12.72** in worst-fold ratio. Same pattern
as Arc 10 v2.3 (oracle Sharpe 4.61 vs base −1.29) and Arc 7 v3.0 first
attempt (oracle ratio 6.85 vs −0.74). Third arc-of-record with phenomenal
latent edge unrecovered by the deployed architecture set.

**Step 6 not auto-dispatched.** No PASS-tier candidate cleared §3 #1-9 per
Amendment 4 evaluation order; framework skipped correctly.

---

## §3 Cross-arc observations

- **First v3.0.1 arc — full canonical infra exercised under PRs #185/#186/#188/#189.**
  Step 4 classifier persistence + holdout-window training exclusion (PR #185),
  Amendment 3 risk-normalised gates with chained DD + per-day max DD parquet +
  holdout re-runs at scaled risk (PR #186), Step 6 framework auto-dispatch
  (PR #188), mid-price features + EET bar boundary (PR #189). Composition
  driver per chat note A; orchestrator's `auto_arch_specs` API requires
  pre-knowledge of candidate cluster IDs, blocking single-pass orchestrator
  use for archetype-based architecture selection.

- **Architecture-map gap: second arc-of-record. RECOMMEND v1.1 DISPATCH AMENDMENT.**
  Dispatch §"Step 5" archetype map has Bimodal → {A1, A4}, Unclassified → {A1}.
  Arc 7 v3.0 first attempt: c0 Bimodal Step-4 AUC 0.6758 above 0.65 bar, A2 not
  tested. Arc 7 v3.0.1: c1 Unclassified Step-4 AUC 0.6642 above bar, A2 not
  tested. Same structural gap, two arcs in a row. **Recommended amendment:**
  when any cluster's Step 4 best AUC ≥ 0.65, A2 and A6 are added to the search
  set regardless of archetype. This unlocks the architecture whose canonical
  job (per L_PROTOCOL §2 Step 5 A2 spec) is to admit only the positive
  cohort — the very mechanism missing here.

- **Oracle PASS-DEPLOYABLE both candidate clusters; realised step5_not_scalable.**
  Oracle c0 worst-fold ratio **11.90** (ROI 24-48%/fold, DD < 2.3%);
  c1 worst-fold ratio **6.80** (ROI 22-50%/fold, DD 2-4%). Realised top A1
  worst-fold DD 34.2% → k_safe = 0.234 → r_safe = 0.117% < 0.15% floor →
  step5_not_scalable. Oracle-realised gap on c0 = +12.72 in worst-fold ratio
  terms — third arc with this pattern (Arc 10 v2.3: oracle Sharpe 4.61 vs
  base -1.29; Arc 7 v3.0: oracle ratio 6.85 vs -0.74). Pattern is reliable
  enough to constitute a research-class tag: `oracle_realised_gap_dominant`.

- **EET + mid-price refactor (PR #189) shifted which cluster has higher AUC.**
  v3.0 first attempt: c0 Bimodal AUC 0.6758, c1 Unclassified AUC 0.6184. v3.0.1:
  c0 Bimodal AUC 0.6192 (LR best), c1 Unclassified AUC 0.6642 (RF best). The
  higher-AUC cluster swapped between attempts. Bar boundary + mid-price feature
  computation moves enough probability mass to flip which archetype carries
  signal — cross-arc consideration for any feature/clustering replication
  attempt under different boundary conventions.

- **`step5_not_scalable` failure mode first invocation on a real arc.**
  Worst-fold DD 34.2% (driven by wrong-way-dominant c2 + monotonic_down c3
  composing 52% of the pool) → k_safe = 0.234 takes r_safe BELOW the 0.15%
  scalability floor. Amendment 3 §"Scalability bounds" correctly fires the
  step5_not_scalable mode. v1 verdict logic would have classified this as
  step5_dd_above_gate at base risk (DD 34% > 8%); Amendment 3 reframes it
  as a scalability failure because at any risk that gets DD under 8%, the
  ROI is below the 0.15% floor's "actionable trade" threshold. Same outcome,
  more precise diagnosis.

- **Cluster registry rows seeded for v3.0.1.** This arc adds 4 cluster rows
  under EET + mid-price conventions (c0 through c3). Comparison vs Arc 7 v3.0
  first attempt (UTC + mid-price pre-#189): cluster sizes shifted modestly
  (c0 636 vs 613, c1 1839 vs 1906, c2 1588 vs 1519, c3 1112 vs 1166) but
  archetype assignments unchanged.

- **Step 6 not dispatched correctly.** No PASS-tier candidate cleared §3 #1-9
  per Amendment 4 evaluation order item 2; framework skipped per spec. Step 6
  manual CLI invocation available per Amendment 4 §"Manual CLI" if chat wants
  retrospective audit; not invoked here per dispatch §E (auto only).

- **Step 6 §6.3 UTC vs EET ambiguity (per chat note G).** Step 6 framework
  was authored against UTC bar boundary literal; this arc runs under EET (PR
  #189). Framework-gap not arc-failure if §6.3 audit fires on this mismatch;
  master chat owns framework fix. N/A for this closure since Step 6 didn't
  dispatch (FAIL verdict).

- **Selection-bias scope:** thin (28 configs evaluated). Skipped exit-policy
  variants explicit in `results/l_arc_7/step_5/skipped_configs.md`. Recommended
  follow-up if the dispatch amendment lands: re-run with A2/A6 across c0+c1
  with full SL × trail × exposure grid — config count likely ~60-90 (normal scope).

