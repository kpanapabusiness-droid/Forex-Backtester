# ARC_7_V3_0_2_CLOSURE — l_arc_7_v3_0_2

> **Closed:** 2026-05-25T05:26:22Z
> **Branch:** arc/l_arc_7_v3.0.2
> **Closure doc path:** results/l_arc_7_v3.0.2/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:
  template_version: v1.3
  arc_name: l_arc_7_v3_0_2
  signal: liquidity sweep + reclaim long (4H) — A2/A6 follow-up (override)
  tf: 4H
  sub_protocol: vanilla
  closed_timestamp: "2026-05-25T05:26:22Z"
  closure_doc_link: results/l_arc_7_v3.0.2/ARC_CLOSURE.md
  verdict: FAIL
  one_line: "FAIL: A2/A6 follow-up \u2014 pool 5175 (reused v3.0.1); top worst-fold ratio 4.36, primary failure mode `step5_not_scalable`."
  failed_at_step: 5
  primary_failure_mode: step5_not_scalable
  pool_metadata:
    total_n: 5175
    window_start: 2010-01-01
    window_end: 2026-04-30
    kh24_co_fire_pct: null
    configs_evaluated_step5: 45
    search_scope_flag: thin
  best_architecture:
    name: null
    cluster: 1
    archetype: Unclassified
    config: "A6::A6::cl1::sl2.0::thr0.5-0.7::exp2"
    sl_atr: 2.0
    exit_policy: sl_plus_trailing_atr
    exposure_cap: 2
    worst_fold_ratio: 4.363315680255243
    worst_fold_roi_base_pct: 0.044894223720027604
    worst_fold_dd_base_pct: 0.012644805643767906
    mean_fold_ratio: 8.703323738875701
    mean_fold_roi_pct: null
    sign_pos_folds: null
    n_trades_total: null
    holdout_roi_pct: null
    holdout_dd_pct: null
    holdout_passed: null
    oracle_worst_ratio: 6.797246820795108
    oracle_real_gap_sharpe: null
    features_in_winning_config: [w1_close_slope_sign, d1_close_slope_sign, atr_percentile_100, atr_14, d1_close_slope_magnitude, session_tokyo, session_ny, session_dead, kijun_26_distance, atr_vs_trailing_100]
    chained_max_dd_base_pct: 0.020914116125924344
    per_day_max_dd_artefact_path: "C:\\Users\\panap\\Documents\\Forex-Backtester\\.claude\\worktrees\\clever-elbakyan-ad50ee\\results\\l_arc_7_v3.0.2\\step_5\\per_day_max_dd_base__A6__A6__cl1__sl2.0__thr0.5-0.7__exp2.parquet"
    per_day_max_dd_base_summary:
      n_days: null
      p50_pct: null
      p95_pct: null
      p99_pct: null
      max_pct: null
    k_safe: 6.32670855162006
    k_hard: 7.908385689525075
    r_safe_pct: 0.0316335427581003
    r_hard_pct: 0.03954192844762538
    scalable_to_safe: false
    scalable_to_hard: false
    worst_fold_roi_at_r_safe_pct: 0.28403266912784275
    worst_fold_roi_at_r_hard_pct: 0.3550408364098035
    chained_max_dd_at_r_safe_pct: 0.13231751734346053
    chained_max_dd_at_r_hard_pct: 0.1653968966793257
    daily_dd_breaches_at_r_safe: 3
    daily_dd_breaches_at_r_hard: 7
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
      ww_pp: 0.0047169811320754
      reach_1r: 1.0
      step4_e_auc: 0.6191919406774266
      step4_d1_auc: null
      outcome: dies_step4
    c1:
      n: 1839
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.967843248005648
      mfe_p50_r: 2.114999183571737
      ww_pp: 0.0396954866775421
      reach_1r: 0.9603045133224578
      step4_e_auc: 0.6642231832414549
      step4_d1_auc: null
      outcome: dies_step5
  architectures_tested: [A1, A2, A4, A5, A6]
  architecture_results:
    A1:
      tested: true
      won: false
      worst_fold_ratio: null
    A2:
      tested: true
      won: false
      worst_fold_ratio: null
    A4:
      tested: true
      won: false
      worst_fold_ratio: null
    A5:
      tested: true
      won: false
      worst_fold_ratio: null
    A6:
      tested: true
      won: false
      worst_fold_ratio: 4.363315680255243
  archetypes_observed: [Bimodal, Unclassified]
  cross_arc_tags: [v3_0_2_a2_a6_followup_retry, architecture_map_override_applied, v301_artefacts_reused, amendment_3_risk_normalised_gates_evaluated, bimodal_observed, unclassified_only]
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

## §2 Why failed (vs v3.0.1)

v3.0.2 reuses Arc 7 v3.0.1's Step 1-4 artefacts verbatim (pool 5,175 trades, 5ers EET, 27-feature catalogue, persisted classifiers SHA256-verified at load). Only Step 5 onwards re-runs under the **architecture-map override**: cluster c1 (Unclassified, RF AUC 0.6642 ≥ 0.65) augmented to {A1, A2, A6}; c0 (Bimodal, LR AUC 0.6192 < 0.65) stays at {A1, A4}; A5 portfolio composes top-1 per cluster across folds. **Step 5 config grid: 45 configs (vs 28 in v3.0.1, +17 from A2+A6 grids on c1 + A5).** Final verdict: **FAIL** — same letter as v3.0.1, opposite cause.

**v3.0.1 vs v3.0.2 — direct top-1 comparison:**

| Metric | v3.0.1 best (`A1::cl0::sl2.0::trail1::expinf`) | v3.0.2 best (`A6::cl1::sl2.0::thr0.5-0.7::exp2`) | Δ |
|---|---:|---:|---:|
| Worst-fold ratio | -0.817 | **+4.363** | +5.18 |
| Mean fold ratio | ~-0.8 | **+8.703** | ~+9.5 |
| Worst-fold DD base | 34.18% | **1.26%** | -32.92pp |
| n_negative_folds | many | **0** | clean sign |
| Chained max DD base | ~30% | **2.09%** | -28pp |
| k_safe | 0.23 | 6.33 | +6.1× |
| r_safe @ k=8/dd | 0.117% (< 0.15% floor) | **3.163% (> 2.0% ceiling)** | flipped |
| primary_failure_mode | `step5_not_scalable` (DD too high) | `step5_not_scalable` (DD too **low**) | inverted |
| Verdict | FAIL | FAIL | same |

**The architecture-map override worked — and exposed a new failure mode.** A6 (meta-labeling with confidence-band sizing) on c1 filtered the 1,839-trade Unclassified cluster down to 18-70 trades/fold with worst-fold DD 1.26% and uniformly positive fold sign (0 negative folds). That is *better* path quality than v3.0.1 by every primary axis. But Amendment 3 reads the resulting equity as "too clean to scale" — the implied per-trade risk to consume the 8% chained-DD budget would be 3.16%, exceeding the 2.0% prop hard ceiling.

**Top-3 are all A6/c1; #3 is the near-miss.** `A6::cl1::sl2.0::thr0.4-0.6::exp2` reached ratio 3.51, DD base 2.80%, r_safe 1.427% (**inside** [0.15%, 2.0%], `scalable_to_safe=true`) — failure shifts to `step5_daily_dd_breach` (2 days breach 5% daily-DD at r_safe). Tighter threshold band (0.4-0.6 vs 0.5-0.7) admits more trades (70 vs 18 per fold) and brings r_safe back into scalable range, but introduces daily-DD spike clusters that nick the 5%/day prop limit. Two failure modes flanking the candidate that would otherwise PASS-DEPLOYABLE.

**Cluster outcomes:**
- c0 (Bimodal, AUC 0.6192 < 0.65): dies Step 4 — no override augmentation; A1/A4 grids unchanged from v3.0.1, no scalable candidate.
- c1 (Unclassified, AUC 0.6642 ≥ 0.65): dies Step 5 — override surfaced A6 candidates with strong path quality, but flanked by `not_scalable` (too clean, broader threshold) and `daily_dd_breach` (tighter threshold).

Step 6 not dispatched (no PASS-tier candidate to validate). Pool/classifiers byte-identical to v3.0.1; only Step 5 architecture surface differs.

---

## §3 Cross-arc observations

- **Architecture-map override is structurally informative — recommend ratifying.** Rule: when any candidate cluster's Step 4 best-classifier mean OOS AUC ≥ 0.65, augment that cluster's architecture set to include A2 and A6 regardless of archetype. Without the override, c1 (Unclassified) would have run A1/A4 only and the FAIL would have read "thin grid, no edge surfaced" — same letter as v3.0.1 but no information. *With* the override, the FAIL reads "A6/c1 produces strong realised path quality (ratio +4.36, 0 negative folds) bounded by Amendment 3 scalability bounds at both ends" — a richer, more actionable signal. **Recommendation:** ratify the override into L_PROTOCOL as an Amendment 1 update (archetype-map: any cluster with E AUC ≥ 0.65 gains A2+A6 regardless of archetype). The override consistently changes the Step 5 surface even when it does not change the verdict letter; without it, Arc 7's "A6/c1 ratio 4.36 vs A1 -0.82" finding would not exist.
- **Amendment 3 calibration gap: "too clean to scale" is a new failure mode.** v3.0.1's best A1 hit `step5_not_scalable` because DD was *too high* (34% → r_safe 0.117% < 0.15% floor). v3.0.2's best A6 hit `step5_not_scalable` because DD was *too low* (1.26% → r_safe 3.16% > 2.0% ceiling). Both reside on the same Amendment 3 gate, at opposite ends. The "too clean" end is new — this is the first L-arc closure where a candidate fails Amendment 3 by being *too risk-efficient*. Open question for the v3.0.x cycle: when a candidate is `scalable_to_safe=false` *only* because r_safe overshoots 2%, is fixed-risk-cap deployment (r=2% with k_realised < 8 budget headroom) acceptable as PASS-VIABLE? Currently the cap is treated as binary scalability — flag for protocol calibration backlog.
- **Top-3 sandwich: the deployable candidate sits one config-grid step away.** `A6::cl1::sl2.0::thr0.4-0.6::exp2` (rank 3) is `scalable_to_safe=true` (r_safe=1.43% inside bounds) — the only top-3 candidate that clears scalability. Its failure mode is daily_dd_breach (2 days breach 5% at r_safe), not ratio or chained DD. A finer threshold sweep (e.g. (0.35, 0.55), (0.45, 0.65)) or a configurable max_concurrent_total cap to throttle daily volatility could plausibly find a PASS-DEPLOYABLE config without changing the architecture. Flag for follow-up cycle if the signal is reopened.
- **Oracle-realised gap recurs (4th instance).** Oracle WFO (cluster identity known at entry): c0 worst-fold ratio **11.90** (median 24.02), c1 worst-fold ratio **6.80** (median 10.65). Realised top-1 ratio 4.36. The gap is now an empirical regularity across Arc 10 v2.3 (Sharpe 4.61 vs -1.29), Arc 7 v3.0 (ratio 6.85 vs -0.74), Arc 7 v3.0.1 (ratio 11.90 vs -0.82), and Arc 7 v3.0.2 (ratio 11.90 vs 4.36). The v3.0.2 gap is *narrower* than predecessors — A6 closed roughly half the realised-to-oracle distance. Cross-arc clusterifier (v2.4 backlog item from Arc 10) remains the standing follow-up.
- **Branch base correction (per dispatch §0).** v3.0.2 cut from `arc/l_arc_7 @ ba6c5b0` (the v3.0.1 commit) rather than literal `origin/main` because PR #180 has not yet merged; cutting from origin/main would lose access to v3.0.1 Step 1-4 artefacts the dispatch directs CC to reuse. Audit-trail-clean: every v3.0.2 artefact at `results/l_arc_7_v3.0.2/` is provably derived from `results/l_arc_7/` (v3.0.1) plus the augmented Step 5 grid; no Step 1-4 re-execution.
- **v3.0.1 reuse audit clean.** Step 1 pool (5,175 trades, sha256 from v3.0.1 manifest) loaded unchanged; Step 2 K=4 cluster assignments loaded unchanged; Step 4 persisted classifiers (c0 LR, c1 RF) loaded via `core.steps.classifier_persistence.load_classifier` with SHA256 verification, no `ClassifierIntegrityError`; cluster archetype + AUC table identical to v3.0.1's closure §1.clusters block.
- **Search scope:** thin (45 configs vs v3.0.1's 28). Net new under override: A2 grid (6 configs for c1) + A6 grid (18 configs for c1) + A5 portfolio (1 config) − A1 c1 re-uses v3.0.1 surface = +17 configs total.
- **Step 6 not dispatched** — no PASS-tier candidate to validate per Amendment 4 `maybe_dispatch_step_6` gate.
