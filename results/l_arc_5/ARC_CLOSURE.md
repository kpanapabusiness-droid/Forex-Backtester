# ARC_5_CLOSURE — l_arc_5

> **Closed:** 2026-05-22T21:41:14+00:00
> **Branch:** arc/l_arc_5
> **Closure doc path:** results/l_arc_5/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:
  arc_name: l_arc_5
  signal: mtf_alignment.2_down_mixed.kijun.h_120 (LCHAR Entry 5, h=120 override)
  tf: H1
  sub_protocol: vanilla
  closed_timestamp: '2026-05-22T21:41:14+00:00'
  closure_doc_link: results/l_arc_5/ARC_CLOSURE.md
  verdict: FAIL
  one_line: 'top-1 A1_sl3.0_trailON_expunl failed: search=fail, holdout=fail (Arc 5 v3.0 vanilla on h=120 trigger).'
  failed_at_step: 5
  primary_failure_mode: step5_dd_above_gate
  pool_metadata:
    total_n: 126801
    window_start: '2010-01-01'
    window_end: '2026-04-30'
    kh24_co_fire_pct: 0.0
    configs_evaluated_step5: 24
    search_scope_flag: thin
  best_architecture:
    name: A1 system_level_filter
    cluster: aggregate
    archetype: Unclassified
    config: A1_sl3.0_trailON_expunl
    sl_atr: 3.0
    exit_policy: trail_atr_2.0_1.5
    exposure_cap: unlimited
    worst_fold_ratio: -0.904154221501165
    worst_fold_roi_pct: -41.90427508702806
    worst_fold_dd_pct: 51.71586505851277
    mean_fold_ratio: -0.5167988921567164
    mean_fold_roi_pct: null
    sign_pos_folds: 2/10
    n_trades_total: 9551
    holdout_roi_pct: -72.91446142213387
    holdout_dd_pct: 75.22039983903525
    holdout_passed: false
    oracle_worst_ratio: null
    oracle_real_gap_sharpe: null
    features_in_winning_config: []
  cost_decomposition: null
  clusters:
    c0:
      n: 19091
      archetype: bimodal
      sl_atr: 1.5
      step3_composite: 1.5795188309429498
      mfe_p50_r: 5.5169237472430295
      ww_pp: 0.006338065056833
      reach_1r: 1.0
      step4_e_auc: 0.7404224388022451
      step4_d1_auc: null
      outcome: passed_step3
    c1:
      n: 38037
      archetype: unclassified
      sl_atr: 1.5
      step3_composite: 0.9358595925460222
      mfe_p50_r: 1.9340777055080811
      ww_pp: 0.0388306122985514
      reach_1r: 0.960801325025633
      step4_e_auc: 0.5402625464907423
      step4_d1_auc: null
      outcome: dies_step4
    c2:
      n: 37749
      archetype: unclassified
      sl_atr: 1.5
      step3_composite: 0.1844462178421674
      mfe_p50_r: 0.5044693650148979
      ww_pp: 0.9703833214124876
      reach_1r: 0.0457230655116691
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c3:
      n: 31924
      archetype: monotonic_down
      sl_atr: 1.5
      step3_composite: 0.015708075033374
      mfe_p50_r: 0.0575229703022982
      ww_pp: 0.9991542413231423
      reach_1r: 0.0024433028442551
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
  architectures_tested:
  - A1
  architecture_results:
    A1:
      tested: true
      won: true
      worst_fold_ratio: -0.904154221501165
  archetypes_observed:
  - bimodal
  - monotonic_down
  - unclassified
  cross_arc_tags:
  - v3_first_complete_arc
  - thin_search_scope_24_configs
  - h120_mtf_alignment_v3_replay
```

---

## §2 Why failed

Top-1 candidate `A1_sl3.0_trailON_expunl` failed §3 gates: worst-fold ratio -0.90 (need ≥ 2.0), holdout ratio -0.97, search negative folds 8.

Step 1 pool: 126,801 trades over 2010-01-01..2026-04-30. Step 2 clustering surfaced archetypes: ['bimodal', 'monotonic_down', 'unclassified']. Step 3 flagged 2 candidate cluster(s) under (reach_1R ≥ 0.50 ∧ ww_pp ≤ 0.30 ∧ mfe_p50 ≥ 1.5R). Step 5 ran 24 A1 configs (thin search per dispatch "Selection-bias accounting") on the 11-fold 2010-2020 WFO.

Methodology note: v3.0 first arc to land end-to-end on the mtf_alignment_2_down_mixed_kijun trigger. v2.x history on this trigger (Arc 2 redo KILL at Step 3, Arc 5 v2 SHELVED at Step 6 under Pipeline D1 admit-vs-deployment failure) is historical context, NOT input to this verdict per dispatch line 11.

Architecture scope: Step 5 exercised A1 only (system_level_filter baseline). A2/A3/A4/A6 require Step 4 classifier wiring + path-so-far classifier training and are deferred to follow-up arcs once the A1 baseline establishes the pool's deployability ceiling.

---

## §3 Cross-arc observations

- **v3.0 first complete arc end-to-end.** Pool size 126,801 trades across 28 FX pairs over 2010-01-01..2026-04-30; signal port + canonical architectures + WFO orchestrator + holdout one-shot all exercised on real data.
- **Cluster archetypes observed on h=120 trigger under v3.0:** ['bimodal', 'monotonic_down', 'unclassified']. First v3.0 data point on this signal class — feeds the per-archetype recurrence registry.
- **A1 baseline thin-search scope** (24 configs: 6 SL × 2 trail × 2 exposure). Per dispatch "Selection-bias accounting" thin (<50) is flagged. A2/A6 (classifier-based) and A3/A4 (path-so-far classifier) deferred — establishes minimum-viable Step 5 scope; future arcs on this trigger can layer classifier architectures.
