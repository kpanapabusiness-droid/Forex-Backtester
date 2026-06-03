# ARC_11_V3.0_CLOSURE — arc_11_v3.0_exit_extraction

> **Closed:** 2026-06-03T00:00:00Z
> **Branch:** arc/arc_11_v3.0_exit_extraction
> **Closure doc path:** results/arc_11_v3.0_exit_extraction/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:

  template_version: v1.3

  # ────── Identity ──────
  arc_name: arc_11_v3.0_exit_extraction
  signal: swing-high breakout in trend (SHB), long, 4H, causal 3-bar swing (right-edge t-4)
  tf: 4H
  sub_protocol: vanilla
  closed_timestamp: 2026-06-03T00:00:00Z
  closure_doc_link: results/arc_11_v3.0_exit_extraction/ARC_CLOSURE.md

  # ────── Verdict ──────
  verdict: FAIL
  one_line: Exit-extraction route does NOT close the SHB oracle gap; best (A1 runner-trail) worst-fold ratio -0.85, 6/10 negative folds, chained DD 50.8%.
  failed_at_step: 5
  primary_failure_mode: step5_chained_dd_above_gate

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: 17281
    window_start: 2010-01-01
    window_end: 2026-04-30
    kh24_co_fire_pct: null   # not computed (FAIL — no deployment; decorrelation gate not binding)
    configs_evaluated_step5: 54
    search_scope_flag: normal

  # ────── Best architecture (Top-1 least-bad; all FAIL) ──────
  best_architecture:
    name: A1 system_level_filter
    cluster: aggregate
    archetype: Bimodal   # primary capturable cohort c0; A1 runs full-pool (no cluster filter)
    config: A1::sl3.5::sl_partial_close_1r_runner_trail::exp2
    sl_atr: 3.5
    exit_policy: sl_partial_close_1r_runner_trail
    exposure_cap: 2
    worst_fold_ratio: -0.847
    worst_fold_roi_base_pct: -15.38
    worst_fold_dd_base_pct: 18.35
    mean_fold_ratio: -0.053
    mean_fold_roi_pct: -3.24
    sign_pos_folds: "4/10"
    n_trades_total: 1674
    holdout_roi_pct: -18.10
    holdout_dd_pct: 29.46
    holdout_passed: false
    oracle_worst_ratio: null   # OracleFoldRunner here marks FULL-book DD, not admit-only — see §3
    oracle_real_gap_sharpe: null
    features_in_winning_config: []   # A1 full-pool, no filter rules / no classifier

    # ── Amendment 3 risk-normalised fields ──
    chained_max_dd_base_pct: 50.81
    per_day_max_dd_artefact_path: results/arc_11_v3.0_exit_extraction/step_5/per_day_max_dd_base__A1__A1__sl3.5__sl_partial_close_1r_runner_trail__exp2.parquet
    per_day_max_dd_base_summary:
      n_days: null
      p50_pct: null
      p95_pct: null
      p99_pct: null
      max_pct: null
    k_safe: 0.436
    k_hard: 0.545
    r_safe_pct: 0.2180
    r_hard_pct: 0.2724
    r_safe_intrinsic_pct: 0.2180
    r_hard_intrinsic_pct: 0.2724
    r_safe_capped_at_rmax: false
    r_hard_capped_at_rmax: false
    scalable_to_safe: true
    scalable_to_hard: true
    worst_fold_roi_at_r_safe_pct: -6.70
    worst_fold_roi_at_r_hard_pct: -8.38
    chained_max_dd_at_r_safe_pct: 22.15
    chained_max_dd_at_r_hard_pct: 27.69
    daily_dd_breaches_at_r_safe: 0
    daily_dd_breaches_at_r_hard: 0
    holdout_roi_at_r_safe_pct: -7.89
    holdout_dd_at_r_safe_pct: 12.84
    holdout_roi_at_r_hard_pct: -9.86
    holdout_dd_at_r_hard_pct: 16.05
    sizing_convention: reset_floor
    chained_dd_method: equity_stitching

    # ── v1.2 deployment-spec fields ──
    config_artefact_path: null
    deployment_spec_section_present: false

  # ────── Cost decomposition (winning arch is rule-based A1, not classifier) ──────
  cost_decomposition: null

  # ────── Per-cluster results ──────
  clusters:
    c0:
      n: 2167
      archetype: Bimodal
      sl_atr: 1.5
      step3_composite: 1.984
      mfe_p50_r: 7.785
      ww_pp: 0.003
      reach_1r: 1.000
      step4_e_auc: 0.5056
      step4_d1_auc: null
      outcome: dies_step5
    c1:
      n: 6123
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.967
      mfe_p50_r: 2.108
      ww_pp: 0.035
      reach_1r: 0.965
      step4_e_auc: 0.5056
      step4_d1_auc: null
      outcome: dies_step5
    c2:
      n: 5012
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.190
      mfe_p50_r: 0.519
      ww_pp: 0.966
      reach_1r: 0.043
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c3:
      n: 3977
      archetype: Monotonic_down
      sl_atr: 1.5
      step3_composite: 0.023
      mfe_p50_r: 0.095
      ww_pp: 0.999
      reach_1r: 0.001
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3

  # ────── Architecture results ──────
  architectures_tested: [A1, A4]
  architecture_results:
    A1: {tested: true, won: true,  worst_fold_ratio: -0.847}   # least-bad of all (still FAIL)
    A4: {tested: true, won: false, worst_fold_ratio: -1.000}   # gated variants all expU → 60%+ DD, not scalable

  # ────── Architectures skipped under Amendment 5 ──────
  # A2/A6 not run (exit-extraction dispatch; and Step-4 AUC 0.51 < 0.65 would block them
  # under Amendment 5 Gate 2 regardless). A3 skipped: entry-route DE confirmed FAIL pre-run
  # (ratio 1.04). Listed for cross-arc analytics.
  architectures_skipped_by_amendment_5: [A2, A6]

  # ────── Archetypes observed ──────
  archetypes_observed: [Bimodal, Unclassified, Monotonic_down]

  # ────── Cross-arc observation tags ──────
  cross_arc_tags:
    - "shb_capturable_not_extractable_exit_route"
    - "exit_extraction_fails_after_entry_route_de_fail"
    - "full_pool_long_only_concurrency_carryover_dd"
    - "sl_only_long_hold_oos_carryover_degenerate"

  # ────── Step 6 causal-audit registry ──────
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

**Proximate cause.** Under L_PROTOCOL v3.0 world-#2 (ArcFoldRunner + Amendment 3, reset-floor r_base 0.5%, EET, cost cell 5), every one of the 14 Stage-B candidates FAILED. The Top-1 (least-bad) is `A1::sl3.5::sl_partial_close_1r_runner_trail::exp2`: worst-fold ratio **−0.847**, worst-fold ROI **−15.4%**, **6 of 10 IS folds negative**, holdout **−18.1% / 29.5% DD**, and an IS+holdout **chained max DD of 50.8%** that scales to 22.2% even at the gate-derived r_safe (0.218%) → `step5_chained_dd_above_gate`. It is *scalable* (r_safe above the 0.15% floor) but its edge is negative, so it would also fail on negative worst-fold ROI and 6 negative folds. The unlimited-exposure (`expU`) and `sl_only` variants are far worse — worst-fold DD 32–75%, r_safe < 0.15% → `step5_not_scalable`.

**Structural cause.** The SHB cohort is genuinely *capturable at the path level* — c0 Bimodal reaches +1R with probability 1.00 and has fwd_mfe_p50 **7.79R**; c1 Unclassified reach_1R 0.96, mfe_p50 2.11R — which is why Arc 11 carried "the highest oracle ceiling on record." But the MFE is **early and fleeting**: the asymmetric/differentiated exits tested here (`sl_partial_close_1r_runner_trail`, `sl_plus_tp_2r`, and A4 classifier-timed exits on the Bimodal cluster) bank some of it (~50% win rate, runners reaching +3–4R) yet give back enough on the losers — false breakouts that mean-revert — that the **net per-fold edge is negative in the majority of years**. Entry selection cannot rescue it either: Step-4 entry-time AUC is **0.506 ≈ chance** for both candidate clusters, and the off-protocol Pipeline-DE entry route already FAILED (ratio 1.04). With both the entry route and the exit route exhausted, SHB is **capturable-but-not-extractable** under v3.0.

**Methodology note (this tells us about the signal class).** The full-pool long-only SHB breakout carries a large **concurrency + IS→OOS carryover** drawdown tail: ~1,000 trades/yr held up to 240 bars (40 days) stack into a deep portfolio DD (chained 50.8% at r_base; the two-stage triage flagged 54/54 on the DD-drop). The `sl_only` baseline is a degenerate *measurement* here (40-day holds + exposure cap leave its OOS equity dominated by IS-opened floating positions, yielding 0% completed-trade win rate) and is non-load-bearing; the runner-trail / tp_2r / A4 configs are the valid exit-extraction test, and they FAIL on their own merits.

---

## §3 Cross-arc observations

- **SHB joins the "capturable-not-extractable" tally — now on the EXIT route.** Arc 11's Bimodal cohort (reach_1R 1.0, mfe_p50 7.79R) is path-capturable but neither entry selection (Step-4 AUC 0.51; DE route ratio 1.04 FAIL) nor differentiated/asymmetric exits (A1 runner-trail/tp_2r, A4) extract a positive worst-fold edge. Pairs with Arc 7 (V-shape) and the historical Arc 11 (Bimodal) entry-route findings — the third arc where high path-MFE does not survive realistic exit simulation.
- **Exit-extraction route formally closed for SHB.** The dispatch's hypothesis ("exit extraction, not entry selection, closes the gap") is REJECTED: the best exit config realizes 6/10 negative folds and a −0.85 worst-fold ratio. Both routes (entry DE + exit A4/runner-trail) are now exhausted under v3.0.
- **Full-pool long-only breakout has a concurrency/carryover DD signature.** ~1k trades/yr with 240-bar holds produce chained DD 50.8% at r_base and 17–75% worst-fold DD depending on exposure; unlimited exposure is catastrophic (60–75% DD, not scalable). Exposure cap = 2 is necessary but insufficient. This is the Amendment-7 concurrency tail observed at the architecture level.
- **`sl_only` + long time-exit is a degenerate WFO measurement.** When effective hold (240 bars) is long relative to the OOS fold and exposure-capped entries are sparse, OOS FoldStats are dominated by IS-opened positions floating into OOS (observed: positive fold ROI with 0% completed-trade win rate). Future long-hold arcs should prefer exit policies that close in-window (trail/partial) for clean per-fold attribution, or measure on completed-trade ledgers.
- **Oracle methodology caveat.** `OracleFoldRunner(candidate_cluster_id=…)` as used here marks the FULL signal-book equity (its DD is full-book concurrency DD), so it does NOT reproduce the historical admit-only "+101%/yr / 2.48% DD" cluster figure — that requires isolated cluster-0 sizing (admit-only economics), out of scope for this world-#2 run. The path-level capturability (Step 3) stands; the deployable-oracle number is not re-confirmed here.
- **Deferred next step (only if reconsidered):** this arc is FAIL, so no governed+fixed-initial confirmation run is warranted. Had any config cleared PASS-tier, the Arc-10-canonical governed + fixed-initial confirmation would have been the deferred next step (per chat steering); it is not built here.

---

## §4 deployment_spec

Not applicable — verdict is FAIL. No deployable configuration. (Per template v1.3.1, §4 is OPTIONAL for FAIL verdicts.)
