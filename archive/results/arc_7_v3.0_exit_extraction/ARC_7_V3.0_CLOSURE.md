# ARC_7_V3.0_CLOSURE — arc_7_v3.0_exit_extraction

> **Closed:** 2026-06-02T08:40:00Z
> **Branch:** arc/arc_7_v3.0_exit_extraction
> **Closure doc path:** results/arc_7_v3.0_exit_extraction/ARC_7_V3.0_CLOSURE.md

---

## Result table first

**Direct replication test of the Arc 10 recovery (full-pool A1 + asymmetric `sl_partial_close_1r_runner_trail` exit) on Arc 7's liquidity-sweep-reclaim pool. It does NOT replicate — FAIL.**

### Primary hypothesis — A1 full-pool + runner-trail (the Arc 10 frame), full 11-fold + 2021 holdout

| config (A1 full pool, runner-trail) | worst-fold ratio | worst-fold ROI% (r_base) | worst-fold DD% | DD@0.40% | chained DD% | neg folds | holdout ROI% | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sl2.5 exp2  | −0.90 | −21.08 | 25.77 | 20.62 | **62.86** | 7/11 | −11.08 | FAIL |
| sl2.5 expinf | −0.86 | −24.30 | 30.52 | 24.41 | **77.03** | 8/11 | −24.68 | FAIL |
| sl3.0 exp2  | −0.92 | −14.89 | 16.41 | 13.13 | **64.82** | 7/11 | −17.93 | FAIL |
| sl3.0 expinf | −0.96 | −26.38 | 33.48 | 26.79 | **78.64** | 7/11 | −19.92 | FAIL |
| **sl3.5 exp2** (exact Arc 10 SL) | −0.88 | −12.25 | 18.19 | 14.55 | **53.22** | 8/11 | −1.00 | FAIL |
| sl3.5 expinf | −0.91 | −22.52 | 31.04 | 24.84 | **73.76** | 9/11 | −21.90 | FAIL |

### Triage-selected Stage-B finalists (top-3 by Stage-A rank — all A3 cl0 deferred-entry)

| config | verdict | worst-fold ratio | worst-fold ROI% | worst-fold DD% | DD@0.40% | chained DD% | neg folds | holdout ROI% / DD% | r_safe% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A3 cl0 runner-trail sl3.5 exp2 | FAIL | −0.61 | −3.93 | 7.89 | 6.31 | **28.49** | 5/11 | −12.57 / 18.23 | 0.507 |
| A3 cl0 sl_plus_tp_3r sl3.5 exp2 | FAIL | −0.63 | −7.25 | 13.34 | 10.67 | **38.50** | 7/11 | −10.32 / 20.49 | 0.300 |
| A3 cl0 runner-trail sl3.0 exp2 | FAIL | −0.67 | −5.49 | 9.62 | 7.70 | **44.06** | 6/11 | −27.63 / 31.60 | 0.416 |

**Overall verdict (one sentence): FAIL — the Arc 10 full-pool exit-extraction recovery does NOT replicate on Arc 7, because Arc 7's unfiltered pool is ~52% wrong-way-dominant losers (clusters c2 ww 96.8% + c3 ww 99.6%) with at-chance entry-time separability (best entry-AUC 0.528), so neither the asymmetric runner-trail exit on the full pool (chained DD 53–79%, 7–9 negative folds) nor deferred-entry selection of the strong cohort (A3 chained DD 28–44%, negative 2021 holdout) clears the §3 gate.**

Full ranked tables: [`step_5/architectures_ranked.md`](step_5/architectures_ranked.md). Best candidate detail: [`step_5/best_candidate.md`](step_5/best_candidate.md).

---

## §1 tracker_payload

```yaml
tracker_payload:

  template_version: v1.3

  # ────── Identity ──────
  arc_name: arc_7_v3.0_exit_extraction
  signal: liquidity sweep + reclaim long (4H structural reversal) — exit-extraction re-run
  tf: 4H
  sub_protocol: vanilla
  closed_timestamp: "2026-06-02T08:40:00Z"
  closure_doc_link: results/arc_7_v3.0_exit_extraction/ARC_7_V3.0_CLOSURE.md

  # ────── Verdict ──────
  verdict: FAIL
  one_line: "FAIL: Arc 10 exit-extraction does NOT replicate — full pool ~52% wrong-way losers, entry-AUC 0.50/0.53; A1 runner-trail chained DD 53-79%, best A3 28.5%."
  failed_at_step: 5
  primary_failure_mode: step5_chained_dd_above_gate

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: 5056
    window_start: 2010-01-01
    window_end: 2025-12-31
    kh24_co_fire_pct: 0.0
    configs_evaluated_step5: 48
    search_scope_flag: thin   # 48 configs (dispatch-specified grid: A1+A3 × 4 exits × 3 SL × 2 exp)

  # ────── Best architecture (least-bad Stage-B candidate; still FAIL) ──────
  best_architecture:
    name: A3 pipeline_de
    cluster: 0
    archetype: Bimodal
    config: "A3::cl0::sl_partial_close_1r_runner_trail::sl3.5::n5::exp2"
    sl_atr: 3.5
    exit_policy: sl_partial_close_1r_runner_trail
    exposure_cap: 2
    worst_fold_ratio: -0.6144233009627034
    worst_fold_roi_base_pct: -0.0392652515547377
    worst_fold_dd_base_pct: 0.07887750566537122
    mean_fold_ratio: 0.07042847828945052
    mean_fold_roi_pct: null
    sign_pos_folds: "5/11"
    n_trades_total: 1189
    holdout_roi_pct: -0.12572482990838463
    holdout_dd_pct: 0.1822557355023928
    holdout_passed: false
    oracle_worst_ratio: null
    oracle_real_gap_sharpe: null
    features_in_winning_config: []

    # ── Amendment 3 risk-normalised fields ──
    chained_max_dd_base_pct: 0.28490069604293744
    per_day_max_dd_artefact_path: "results/arc_7_v3.0_exit_extraction/step_5/per_day_max_dd__A3__cl0__sl_partial_close_1r_runner_trail__sl3.5__n5__exp2.parquet"
    per_day_max_dd_base_summary:
      n_days: null
      p50_pct: null
      p95_pct: null
      p99_pct: null
      max_pct: null
    k_safe: 1.0142308548572876
    k_hard: 1.2677885685716096
    r_safe_pct: 0.005071154274286438
    r_hard_pct: 0.006338942842858048
    r_safe_intrinsic_pct: 0.005071154274286438
    r_hard_intrinsic_pct: 0.006338942842858048
    r_safe_capped_at_rmax: false
    r_hard_capped_at_rmax: false
    scalable_to_safe: true
    scalable_to_hard: true
    worst_fold_roi_at_r_safe_pct: -0.03982402965054806
    worst_fold_roi_at_r_hard_pct: -0.04977977298059721
    chained_max_dd_at_r_safe_pct: 0.2889550764970647
    chained_max_dd_at_r_hard_pct: 0.36119384562133087
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

  # ────── Cost decomposition (not computed — failure is full-pool DD, not admit/reject economics) ──────
  cost_decomposition: null

  # ────── Per-cluster results ──────
  clusters:
    c0:
      n: 616
      archetype: Bimodal
      sl_atr: 1.5
      step3_composite: 1.9968099938008559
      mfe_p50_r: 7.86253511123371
      ww_pp: 0.00487012987012987
      reach_1r: 1.0
      step4_e_auc: 0.503706954303192
      step4_d1_auc: null
      outcome: dies_step5
    c1:
      n: 1794
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.9674611705602674
      mfe_p50_r: 2.112746676374748
      ww_pp: 0.038461538461538464
      reach_1r: 0.9604236343366778
      step4_e_auc: 0.5284044859521634
      step4_d1_auc: null
      outcome: dies_step5
    c2:
      n: 1556
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.18654661792644897
      mfe_p50_r: 0.5059525595894241
      ww_pp: 0.968
      reach_1r: 0.044
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c3:
      n: 1089
      archetype: Monotonic_down
      sl_atr: 1.5
      step3_composite: 0.02690714802589864
      mfe_p50_r: 0.0967214126816511
      ww_pp: 0.996
      reach_1r: 0.006
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3

  # ────── Architecture results ──────
  architectures_tested: [A1, A3]
  architecture_results:
    A1: {tested: true, won: false, worst_fold_ratio: -0.8565913}
    A3: {tested: true, won: false, worst_fold_ratio: -0.6144233}

  # ────── Architectures skipped under L_PROTOCOL Amendment 5 ──────
  # A2/A6 (entry-time classifier-filter / meta-labeling) require Step-4 mean OOS AUC ≥ 0.65
  # (Amendment 5 Gate 2). Best entry-AUC = 0.5284 (c1 LGBM) / 0.5037 (c0 RF) — both far
  # below 0.65 → A2/A6 correctly NOT admitted (matches the dispatch's expectation).
  architectures_skipped_by_amendment_5: [A2, A6]

  # ────── Archetypes observed ──────
  archetypes_observed: [Bimodal, Unclassified, Monotonic_down]

  # ────── Cross-arc observation tags ──────
  cross_arc_tags: [capturable_not_extractable_reaffirmed, exit_extraction_does_not_generalise, full_pool_loser_dominated, entry_auc_ceiling_4h, v_shape_relabeled_bimodal_under_eet, co_fire_with_arc_10]

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

**Proximate cause — `step5_chained_dd_above_gate` on every surviving config.** The two-stage Step 5 (48 configs: A1 full-pool + A3 cl0 deferred-entry × {`sl_partial_close_1r_runner_trail`, `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_only`} × SL {2.5, 3.0, 3.5} × exposure {2, unlimited}) screened out 44/48 at Stage-A triage (trailing DD@0.40% > 8% on F1 2010 / F6 2015 / F8 2017). The three triage survivors — all A3 cl0 — FAILed the full 11-fold Stage-B gate: best (A3 runner-trail sl3.5) had worst-fold ratio −0.61, **chained IS+holdout DD 28.5%** (gate ≤ 10%), 5 negative folds, and a **negative 2021 holdout** (−12.6% ROI / 18.2% DD, 438 trades). The dispatch's **primary** architecture — A1 full-pool + the proven Arc 10 `sl_partial_close_1r_runner_trail` exit — was screened out at triage and (run through the full gate anyway, as the primary-hypothesis reference) FAILed every SL: worst-fold ROI −12% to −26%, worst-fold DD 16–33%, **chained DD 53–79%**, 7–9 negative folds, with 159–365 trades/fold (not thin).

**Structural cause — a loser-dominated pool with no entry-time separability.** Step 2 (K=4, silhouette 0.488) reproduces the v3.0.1 EET clustering: c0 Bimodal (n=616, mean +5.61R, mfe_p50 7.86R) and c1 Unclassified (n=1794, mean **−0.64R**) are §2 candidates, but **c2 Unclassified (ww 96.8%) + c3 Monotonic_down (ww 99.6%) compose ~52% of the pool (n=2,645) and are almost all losers**. Step 4 entry-AUC is at chance — c0 RF 0.5037, c1 LGBM 0.5284, both far below the 0.65 Gate-2 bar (top feature `prior_session_high_distance`), so A2/A6 are not admitted (Amendment 5). The Arc 10 recovery worked because a runner-trail exit converts a pool whose winners run far into asymmetric R; it cannot rescue a pool that is majority structural losers — the runner trails the few winners while the loser majority compounds drawdown (hence the 53–79% chained DD on the full pool). Selecting the strong cohort by classifier (A3) is the only lever that could help, but at AUC ≈ 0.51 the deferred-entry admits ~randomly, so A3 also goes net-negative out-of-sample.

**What it tells us about the signal class.** Arc 7 is **capturable-not-extractable** — exactly its v2.3 CLEAN-NULL-at-Step-4 disposition — and that conclusion is now reaffirmed under the v3.0 gates-as-rankings frame *with* the Arc 10 exit-extraction mechanism explicitly tested. Gates-as-rankings removed the old entry-AUC blocker, but the full-pool WFO (the v3.0 deployment gate) re-fails the arc for the deeper economic reason the §8 gate was proxying for: the entry pool has no deployable edge once the wrong-way majority is priced in, and exit engineering does not manufacture one.

---

## §3 Cross-arc observations

- **Exit-extraction does NOT generalise from Arc 10 to a second V-shape/reversal pool.** This is the headline. The Arc 10 recovery (full-pool A1 + asymmetric runner-trail) is **pool-specific**, not a portable mechanism: it requires the unfiltered pool to be winner-skewed enough that a runner-trail's asymmetry dominates. Arc 7's pool is ~52% wrong-way losers, so the identical exit yields chained DD 53–79%. Tag: `exit_extraction_does_not_generalise`.
- **`prior_session_high_distance` is the recurring top entry feature** across both candidate clusters (also surfaced in the v3.0.1 Arc 7 run) yet carries no lift (AUC ≈ 0.50–0.53). Reinforces the 4H entry-feature ceiling pattern on V-shape/reversal cohorts (Arcs 4, 5, 7, 8, 10). Tag: `entry_auc_ceiling_4h`.
- **EET + mid-price clustering relabels Arc 7's "V-shape" cohort as Bimodal/Unclassified** (no V_SHAPE-tagged cluster), consistent with the v3.0.1 note that PR #189 boundary/feature changes shift archetype assignment. The strong cohort (c0) is a high-magnitude long-horizon bimodal (ttp_p50 = 240-bar cap, mfe_p50 7.86R) — the same "real magnitude on unextractable paths" pattern as Arc 2 redo c2. Tag: `v_shape_relabeled_bimodal_under_eet`.
- **`sl_partial_close_1r_runner_trail` beat `sl_plus_tp_2r/3r` and `sl_only` within the A1 full-pool triage** (least-negative worst-of-3 ratio), so the exit *is* the best available exit — it just isn't enough. The exit-policy ranking is informative even on a failing pool: the runner-trail's asymmetry is real but second-order to the entry-pool quality.
- **SL-honest engine note.** This run used the live `MultiPairBacktester` + `ExitPolicyManager` path (SL charged on every bar except the partial-fire bar), NOT the optimistic `simulate_path` replay flagged in [[project_arc10_exit_replay_optimism]] (which ignores pre-+1R-partial SL breaches). Arc 7's runner-trail numbers are therefore SL-honest; part of the gap vs Arc 10's published recovery may be that Arc 10's replay was optimistic, not solely a signal-class difference. Worth verifying when the Arc 10 canonical replay is re-run SL-honest.
- **Co-fire (informational):** KH-24 = **0%** (structural — KH-24 requires close<open, Arc 7 requires close>open; matches the v2.x recorded 0/1288). Arc 10 v3.0.2 = **0.24%** exact (pair, signal-time) / **7.69%** within ±1 H4 bar — legitimate co-occurrence (both fire on D1/structural reversals on overlapping pairs), not redundancy.
- **Selection-bias / noise floor:** N = 48 triaged configs (full grid). Bonferroni-equivalent threshold 0.05/48 ≈ 1.04×10⁻³. Moot — no candidate cleared even the unadjusted gate (best Stage-B verdict FAIL; best worst-fold ratio −0.61 « the 2.0 PASS-DEPLOYABLE bar).
- **Thin-pool note:** A3 cl0 folds 2–11 carry 75–164 trades (not thin); the gate's `min_trades_per_fold=0` is solely the F1 2010 empty-IS artifact (A3's per-fold classifier cannot train with no in-sample data — the project-wide F1 structural-leakage issue). A1 full-pool carries 159–365 trades/fold throughout. No genuine thin-pool flag.

---

## §4 deployment_spec

> N/A — verdict is FAIL. No deployment spec (per template: §4 OPTIONAL for FAIL).

---

## Notes on method / provenance

- **Basis:** reset-floor sizing, r_base 0.5% (engine-emit); Amendment-3 gate evaluated at r_base on trailing worst-fold DD → r_safe (per the operator's 2026-06-02 directive to use the reset-floor Step-5 search engine, NOT the governed fixed-initial harness). DD@0.40% reported = DD_base × (0.40/0.50) (reset-floor linear scaling). EET (`5ers_eet`); v3 cost primitives use real HistData M1 bid+ask spreads. The governed fixed-initial deploy-faithful confirmation run was a deferred contingency for finalists clearing PASS-tier; **not run** (no PASS-tier candidate).
- **Data sourcing:** this worktree has no market data; the H4/D1/W1 `5ers_eet` cache + M1 manifest were sourced read-only from sibling worktree `nice-mirzakhani-baa9e5`. Step 5 re-simulates each config against the live H4 panels (`A1Architecture`/`A3Architecture` → `MultiPairBacktester`), so the {2.5,3.0,3.5}×ATR grid and all four exit policies are produced from real bars — **no forward-price fabrication** (the in-tree `paths.parquet` censored at the 2.0×ATR exit was unused for Step 5).
- **Trigger-set provenance / divergence (HALT-WITHIN-DOC):** pool = **5,056 trades**, sha `2fd4b56dd213f362…`, deterministic across the smoke + full run. This diverges from Arc 7's recorded pools: **v2.3 = 1,288** (UTC boundary, registry-h windowing) and **v3.0.1 EET = 5,175** (window to 2026-04-30). The 5,056-vs-5,175 delta is the window end (this arc ends 2025-12-31 to match the dispatch's 2021–2025 holdout, vs 2026-04-30); the 1,288-vs-5,056 delta is the v2.x→v3.0 EET-boundary + mid-price re-aggregation (PR #189), already documented in the v3.0.1 Arc 7 closure. Per the dispatch's HALT-WITHIN-DOC rule, the divergence is recorded and the regenerated pool used.
- **Step 6:** not auto-dispatched — no candidate cleared §3 #1-9 (all FAIL), so Amendment-4's evaluation order correctly skips the causal audit.
- **Validation checklist:** [✓] trigger-set sha recorded vs Arc 7 v2.x/v3.0.1 (divergence + cause above); [✓] determinism — pool sha reproduced across smoke + full run (seed 42, n_jobs 1); [✓] cost = v3 cell-5-equivalent primitives, spread source = HistData M1 bid+ask; [✓] per-fold n table ([`step_5/per_fold.csv`](step_5/per_fold.csv), [`step_5/a1_primary_per_fold.csv`](step_5/a1_primary_per_fold.csv)) — no thin flag except F1-structural for A3; [✓] selection-bias N=48 + noise floor; [✓] Step 6 auto-dispatch correctly skipped (no PASS-tier). No-lookahead: A1 full-pool carries no classifier (no leakage surface); A3 per-fold classifiers retrained per fold with `train_end` exclusion; D1/W1 one-bar-lag inherited from the canonical pool builder.
- **Artefacts:** [`step_5/architectures_ranked.md`](step_5/architectures_ranked.md), [`step_5/best_candidate.md`](step_5/best_candidate.md), [`step_5/wfo_results.csv`](step_5/wfo_results.csv) (A3 finalists), [`step_5/a1_primary_stageb.csv`](step_5/a1_primary_stageb.csv) (A1 primary), [`step_5/stage_a_triage.csv`](step_5/stage_a_triage.csv), per-fold CSVs, per-day-max-DD parquets, [`run_summary.json`](run_summary.json).
