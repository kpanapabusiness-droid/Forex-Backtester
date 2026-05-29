# ARC_5_v3_0_2_CLOSURE — l_arc_5_v3.0.2

> **Closed:** 2026-05-29T12:09:42Z
> **Branch:** arc/l_arc_5_v3.0.2
> **Closure doc path:** results/l_arc_5_v3.0.2/ARC_CLOSURE.md
> **Boundary convention:** 5ers_eet (canonical, end-to-end)

---

## §1 tracker_payload

```yaml
tracker_payload:
  template_version: v1.3
  arc_name: l_arc_5_v3.0.2
  signal: mtf_alignment.2_down_mixed.kijun.h_120 (1H structural; H4+D1 aux)
  tf: H1
  sub_protocol: vanilla
  closed_timestamp: "2026-05-29T12:09:42Z"
  closure_doc_link: results/l_arc_5_v3.0.2/ARC_CLOSURE.md
  verdict: FAIL
  one_line: "FAIL: pool 130099, 2 candidate clusters; top worst-fold ratio -0.95, primary failure mode `step5_not_scalable`."
  failed_at_step: 5
  primary_failure_mode: step5_not_scalable
  pool_metadata:
    total_n: 130099
    window_start: 2010-01-01
    window_end: 2026-04-30
    kh24_co_fire_pct: null
    configs_evaluated_step5: 20
    search_scope_flag: thin
  best_architecture:
    name: A1 system_level_filter
    cluster: 0
    archetype: Bimodal
    config: "A1::A1::cl0::sl2.0::sl_plus_tp_2r::exp2"
    sl_atr: 2.0
    exit_policy: sl_plus_tp_2r
    exposure_cap: 2
    worst_fold_ratio: -0.9518780593556826
    worst_fold_roi_base_pct: -0.6109633150056462
    worst_fold_dd_base_pct: 0.6421937625596046
    mean_fold_ratio: -0.7453248696411513
    mean_fold_roi_pct: null
    sign_pos_folds: null
    n_trades_total: null
    holdout_roi_pct: null
    holdout_dd_pct: null
    holdout_passed: null
    oracle_worst_ratio: -0.7348973550305519
    oracle_real_gap_sharpe: null
    features_in_winning_config: [d1_close_slope_magnitude, atr_14, prior_session_low_distance, kijun_26_distance, distance_to_round_number, spread_percentile_100, d1_close_slope_sign, spread_vs_trailing_100, session_dead, session_ldn_ny_overlap]
    chained_max_dd_base_pct: 0.9986024705499369
    per_day_max_dd_artefact_path: "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\compassionate-pare-bae095\\results\\l_arc_5_v3.0.2\\step_5\\per_day_max_dd_base__A1__A1__cl0__sl2.0__sl_plus_tp_2r__exp2.parquet"
    per_day_max_dd_base_summary:
      n_days: null
      p50_pct: null
      p95_pct: null
      p99_pct: null
      max_pct: null
    k_safe: 0.12457299442016129
    k_hard: 0.1557162430252016
    r_safe_pct: 0.0006228649721008064
    r_hard_pct: 0.000778581215126008
    scalable_to_safe: false
    scalable_to_hard: false
    worst_fold_roi_at_r_safe_pct: -0.0761095296311216
    worst_fold_roi_at_r_hard_pct: -0.095136912038902
    chained_max_dd_at_r_safe_pct: 0.12439889999177657
    chained_max_dd_at_r_hard_pct: 0.1554986249897207
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
      n: 15712
      archetype: Bimodal
      sl_atr: 1.5
      step3_composite: 1.9517152321762885
      mfe_p50_r: 7.611417529260462
      ww_pp: 0.007128309572301426
      reach_1r: 1.0
      step4_e_auc: 0.5059460459326188
      step4_d1_auc: null
      outcome: dies_step5
    c1:
      n: 42779
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.9653177727129842
      mfe_p50_r: 2.095516506200454
      ww_pp: 0.03576521190303654
      reach_1r: 0.9647724350732836
      step4_e_auc: 0.5109837747386998
      step4_d1_auc: null
      outcome: dies_step5
    c2:
      n: 39004
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.18742351065912896
      mfe_p50_r: 0.5080536529544726
      ww_pp: 0.9692595631217311
      reach_1r: 0.0476874166752128
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c3:
      n: 32604
      archetype: Monotonic_down
      sl_atr: 1.5
      step3_composite: 0.016167801108272633
      mfe_p50_r: 0.0584747056727527
      ww_pp: 0.999355907250644
      reach_1r: 0.0021163047478836954
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
  architectures_tested: [A1]
  architecture_results:
    A1:
      tested: true
      won: false
      worst_fold_ratio: -0.9518780593556826
  architectures_skipped_by_amendment_5: [a5_gate_4_admission_blocked_by_no_pass_tier_constituent]
  archetypes_observed: [Bimodal, Monotonic_down, Unclassified]
  cross_arc_tags: [arc_5_v3_0_2_first_run_canonical_5ers_eet, w1_producer_canonical_fix_pr_208_applied, amendment_5_four_gate_admission, amendment_3_risk_normalised_gates_evaluated, amendment_6_eet_daily_dd_boundary, signal_module_eet_audit_state_a_restored_canonically, bimodal_observed, monotonic_down_observed, unclassified_only]
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

Pool size 130,099 trades across 28 FX H1 bars 2010-01 → 2026-04-30 under 5ers EET bar boundary (PR #189 / Amendment 6). Signal: `mtf_alignment.2_down_mixed.kijun.h_120` — 1H signal with H4 + D1 most-recent-completed Kijun-sign references, restored canonically per PR #193 (post-State-C zero-pool bug; see `docs/audits/signal_module_eet_audit_2026_05.md`). Step 2 selected K=4 (silhouettes {'2': 0.3994396501999425, '3': 0.3988349208471967, '4': 0.4920574627180539, '5': 0.47385784414729626, '6': 0.4518738380819269}). Step 3 surfaced 4 clusters; 2 flagged candidate (archetypes: Bimodal, Monotonic_down, Unclassified). Step 4 RF/LGBM/LR per candidate cluster on the 27-feature catalogue — max mean OOS AUC = 0.5110 (per cluster: {0: 0.5059460459326188, 1: 0.5109837747386998}). Amendment 5 Gate 2 threshold = 0.65; persisted classifiers per PR #185. **Proximate cause:** best candidate `A1::A1::cl0::sl2.0::sl_plus_tp_2r::exp2` reached worst-fold ratio -0.95 (need ≥ 2.0); primary failure mode `step5_not_scalable`. Reason: r_safe=0.0623% / r_hard=0.0779% outside [0.1500%, 2.0000%] — config not scalable. Worst-fold ROI base -61.10%, worst-fold DD base 64.22%, chained DD base 99.86%. **Structural cause:** Step 4 entry-time classifiers could not separate candidate-cluster membership above Amendment 5's Gate 2 threshold (0.65); max mean OOS AUC 0.5110. A2/A6 NOT admitted under Amendment 5.

---

## §3 Cross-arc observations

- **W1 producer contamination + canonical fix (verbatim, per cross-arc resume signal §5):** Prior `multi_tf.py::_w1_close_slope_sign` had within-period lookahead via `merge_asof`; fixed in `engine/w1_producer_canonical_alignment` (PR #208). Arc 5 ran with canonical producer. Cross-arc impact: Arc 8 originated detection; Arc 10 audit confirmed; Arc 5/8/10/11 all affected; Arc 7 v3.0.2 not affected (uses A2/A6 follow-up; ran post-fix).
- **W1 producer fix — engineering detail.** Prior `core/features/multi_tf.py::_w1_close_slope_sign` used `pd.merge_asof` with within-period lookahead semantics (current W1 bar's close visible to H1 bars within the same W1 period). Engine fix `engine/w1_producer_canonical_alignment` shipped as PR #208 (merged 2026-05-25; commit `ab03be9`) — `_w1_close_slope_sign` now uses canonical `core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)`. Arc 5 v3.0.2 is the FIRST Arc 5 run with the canonical W1 producer. Verified at smoke-test entry: `inspect.getsource(_w1_close_slope_sign)` contains `get_htf_value_at` and not `merge_asof`.
- **First Arc 5 run under canonical 5ers EET production substrate.** Prior Arc 5 v2.x ran under UTC + indicator-floor spreads; v3.0.1 was deleted from main after the signal module's State-C zero-pool failure was diagnosed (see `docs/audits/signal_module_eet_audit_2026_05.md`). This run uses the canonical `htf_alignment.get_htf_index_at(..., require_fully_closed=True)` signal module restored alongside the canonical utility; mid-price features (PR #189) + EET bar boundary (Amendment 6); 27-feature catalogue + persisted classifiers (PR #185); Amendment 3 risk-normalised gates (PR #186); Amendment 5 four-gate AUC-driven architecture admission (PR #194); Step 6 framework auto-dispatch (PR #188 + #207 vacuous-pass patch). Not directly comparable to PR #172-era Arc 5 numbers; comparison in §10 is informational.
- **Archetypes observed under v3.0.2 / EET / mid-price / canonical W1:** Bimodal, Monotonic_down, Unclassified. Cluster outcomes: c0=Bimodal/dies_step5; c1=Unclassified/dies_step5; c2=Unclassified/dies_step3; c3=Monotonic_down/dies_step3.
- **Amendment 5 four-gate admission outcomes:** c0=bimodal(AUC=0.5059) → ['A1']; c1=unclassified(AUC=0.5110) → ['A1']. Architectures skipped vs Amendment 1's archetype-driven rule: ['a5_gate_4_admission_blocked_by_no_pass_tier_constituent'].
- **Entry-feature AUC ceiling at 0.5110** (below 0.65 Amendment 5 Gate 2 threshold). A2/A6 NOT admitted; Amendment 5's tighter rule preserves Step 5 search budget.
- **Selection-bias scope:** thin (20 configs evaluated). Skipped exit-policy variants listed in `step_5/skipped_configs.md`.
- **Two-stage execution — initial 8h wall-budget HALT, then A1-only rerun (HALT diagnostic).** First execution (launched 2026-05-25 19:02 +1000) ran Steps 1-4 cleanly (2 candidate clusters surfaced as documented above) then entered Step 5 with the full Amendment-5 admission: cluster 0 Bimodal → (A1, A4); cluster 1 Unclassified → (A1). Total grid: 56 configs × 10 folds = 560 (candidate × fold) pairs. The A4 per-fold-classifier fits build for cluster 0 alone consumed 3h 16min (19:55 → 23:11). After 160/560 WFO runs the 8h wall budget was exhausted; pipeline halted with `verdict: HALT, halt_reason: wall_budget_exceeded_step_5`. Restart strategy (chat-approved 2026-05-29 via Option C): rerun with `--a1-only` flag stripping A4 from cluster 0's admission for compute-budget reasons. Rerun (launched 2026-05-29 18:36 +1000) completed Steps 0-5 + Amendment 3 + closure in 2h 40min wall time (9595.9s). Step 1 pool deterministically identical (sha `a82ecfac…`); Step 2/3/4 identical (deterministic); only the Step 5 admission set differs. **A4 was therefore admitted by Amendment 5 Gate 1 (Bimodal archetype) but not actually tested**: `architectures_skipped_by_a1_only_compute_budget: ['A4']` records this. Counterfactually, A4 was extremely unlikely to clear the gates — Step 4's path-so-far classifier on cluster 0 would inherit the same entry-time AUC ceiling at chance, and the A4 exit-classifier target ("will trade close profitably?") has the same predictability problem as A2's "is this cluster member?". The compute-budget skip is a methodological corner-cut, not a methodology violation — `architectures_skipped_by_amendment_5` correctly records the Amendment 5.1 Gate 4 status (the load-bearing skip for this arc's verdict-class). A1 is the verdict-carrying architecture per Gate 3 universal admission and would have remained the verdict carrier in the full grid given the AUC ceiling.


---

## §10 Retroactive comparison vs prior Arc 5 numbers (informational)

Per intent doc §I.3 and §J: **this arc is NOT directly comparable to PR #172-era
Arc 5 v3.0.0/v3.0.1 results.** The prior Arc 5 v3.0.1 was deleted from main
after the signal module's State-C zero-pool failure was diagnosed under EET
storage (see `docs/audits/signal_module_eet_audit_2026_05.md`); the original
Arc 5 v2.x ran under L_ARC_PROTOCOL v2.1.1 with UTC bar boundaries +
indicator-floor spreads + a different feature substrate entirely.

The Arc 5 v2.x closure (`docs/archive/arc_results/ARC_5_RESULT.md`) recorded
SHELVED Step 6 FAIL with no ship candidate (all three strategy candidates
closed Step 6 with negative worst-fold ROI at every risk-per-trade level under
PR2 + new spreads).

Arc 5 v3.0.2 (this run) is the FIRST Arc 5 test under canonical 5ers production
conditions: signal-parity engine (PR #189), canonical HTF alignment (PR #193),
canonical W1 producer (PR #208), mid-price features, EET bar boundary +
EET daily-DD bucketing (Amendment 6 / PR #197), Amendment 3 risk-normalised
gates (PR #186), Amendment 5 four-gate architecture admission (PR #194),
Step 6 framework auto-dispatch (PR #188 + #207).

This run's outcome: **FAIL**. Top-1 `A1::A1::cl0::sl2.0::sl_plus_tp_2r::exp2` worst-fold ratio -0.95, verdict `fail`.

Cross-substrate delta (v2.x SHELVED → v3.0.2 FAIL) is not load-bearing —
the methodologies test different things. Recoverable v3.0.1-era pool / cluster
numbers from prior branches (now deleted) are not in scope.
