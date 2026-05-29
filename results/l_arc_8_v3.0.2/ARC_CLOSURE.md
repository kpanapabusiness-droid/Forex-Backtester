# ARC_8_v3.0.2_CLOSURE — l_arc_8_v3.0.2

> **Closed:** 2026-05-26T08:30:00Z
> **Branch:** arc/l_arc_8_v3.0.2
> **Closure doc path:** results/l_arc_8_v3.0.2/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:

  template_version: v1.3

  # ────── Identity ──────
  arc_name: l_arc_8_v3.0.2
  signal: pullback_resume_hhhl_long_v0.1 (PR-HHHL, 4H, causal HH/HL detection)
  tf: H4
  sub_protocol: vanilla
  closed_timestamp: 2026-05-26T08:30:00Z
  closure_doc_link: results/l_arc_8_v3.0.2/ARC_CLOSURE.md

  # ────── Verdict ──────
  verdict: FAIL
  one_line: "PR-HHHL Bimodal best Top-1 (A1, SL=1.5x, sl_partial_close_1r_runner_trail) — chained DD 69.24% (>>8% budget); cross-arc: SL is signal-specific."
  failed_at_step: 5
  primary_failure_mode: step5_chained_dd_above_gate

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: 6612
    window_start: 2010-01-12
    window_end: 2025-12-31
    kh24_co_fire_pct: 0.0000
    configs_evaluated_step5: 36
    search_scope_flag: thin   # 36 < 50 evaluated (A2+A6 skipped under Amendment 5 Gate 2 — see admission notes)

  # ────── Best architecture ──────
  # Failed at Step 5 Amendment 3 (chained DD), but Top-1 is the search-WFO best — recorded
  # per template convention.
  best_architecture:
    name: A1 system_level_filter
    cluster: 2
    archetype: Bimodal
    config: a1_sl1.5_exit-sl_partial_close_1r_runner_trail_exp-max_per_currency_2
    sl_atr: 1.5
    exit_policy: sl_partial_close_1r_runner_trail
    exposure_cap: 2
    worst_fold_ratio: -0.886
    worst_fold_roi_base_pct: -21.16    # fold 4 worst per-fold ROI
    worst_fold_dd_base_pct: 23.88      # fold 4 worst per-fold DD (matches worst-ratio fold)
    mean_fold_ratio: -0.353
    mean_fold_roi_pct: -7.16
    sign_pos_folds: "2/10"             # folds 3 and 9 marginally positive
    n_trades_total: 4603               # sum across folds (per per_fold_metrics.csv)
    holdout_roi_pct: null              # not reported separately — Top-K holdout subsumed into Amendment 3 chained
    holdout_dd_pct: null
    holdout_passed: false
    oracle_worst_ratio: null
    oracle_real_gap_sharpe: null
    features_in_winning_config: []     # A1 with no Step-4 filter rules (Amendment 2 Gate 2 failed)

    # ── Amendment 3 risk-normalised fields ──
    chained_max_dd_base_pct: 69.24
    per_day_max_dd_artefact_path: results/l_arc_8_v3.0.2/step_5/per_day_max_dd_base__A1__a1_sl1.5_exit-sl_partial_close_1r_runner_trail_exp-max_per_currency_2.parquet
    per_day_max_dd_base_summary:
      n_days: null
      p50_pct: null
      p95_pct: null
      p99_pct: null
      max_pct: null
    k_safe: null                       # engine emitted null — chained DD failure precedes scaling
    k_hard: null
    r_safe_pct: null
    r_hard_pct: null
    r_safe_intrinsic_pct: null
    r_hard_intrinsic_pct: null
    r_safe_capped_at_rmax: false
    r_hard_capped_at_rmax: false
    scalable_to_safe: false
    scalable_to_hard: false
    worst_fold_roi_at_r_safe_pct: null
    worst_fold_roi_at_r_hard_pct: null
    chained_max_dd_at_r_safe_pct: null
    chained_max_dd_at_r_hard_pct: null
    daily_dd_breaches_at_r_safe: null
    daily_dd_breaches_at_r_hard: null
    holdout_roi_at_r_safe_pct: null
    holdout_dd_at_r_safe_pct: null
    holdout_roi_at_r_hard_pct: null
    holdout_dd_at_r_hard_pct: null
    sizing_convention: reset_floor
    chained_dd_method: equity_stitching

    # ── v1.2 deployment-spec fields ──
    config_artefact_path: null         # FAIL verdict — no deployable config
    deployment_spec_section_present: false

  # ────── Cost decomposition ──────
  cost_decomposition: null             # A1 with no admit/reject filter — decomposition not meaningful

  # ────── Per-cluster results ──────
  clusters:
    c0:
      n: 3418
      archetype: Choppy                # modal Mixed (81.30%); _cluster_archetype fallback resolved to Choppy
      sl_atr: 1.5
      step3_composite: 0.5697
      mfe_p50_r: 1.291
      ww_pp: 0.395
      reach_1r: 0.6042
      step4_e_auc: null                # not processed (dies_step3 — non-candidate)
      step4_d1_auc: null
      outcome: dies_step3
    c1:
      n: 1628
      archetype: Monotonic_down
      sl_atr: 4.0
      step3_composite: 0.3928
      mfe_p50_r: 0.068
      ww_pp: 0.0295
      reach_1r: 0.0
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c2:
      n: 1566
      archetype: Bimodal
      sl_atr: 1.5
      step3_composite: 1.000
      mfe_p50_r: 7.550
      ww_pp: 0.0
      reach_1r: 1.000
      step4_e_auc: 0.4822             # LGBM best classifier — random; Amendment 5 Gate 2 fails
      step4_d1_auc: null
      outcome: dies_step5

  # ────── Architecture results ──────
  architectures_tested: [A1, A4]
  architecture_results:
    A1: {tested: true, won: true,  worst_fold_ratio: -0.886}
    A4: {tested: true, won: false, worst_fold_ratio: null}   # all A4 configs trailed A1 in top-K

  # ────── Architectures skipped under L_PROTOCOL Amendment 5 (v1.3.1) ──────
  # c2 Bimodal AUC 0.4822 < 0.65 → Gate 2 does not admit A2 or A6. Under Amendment 1's
  # archetype-driven rule, Bimodal would have admitted A4 only (A2/A6 were not in
  # Bimodal's Amendment 1 set either). Therefore architectures_skipped_by_amendment_5
  # is the empty set — Amendment 5 doesn't skip what Amendment 1 wouldn't have tested.
  architectures_skipped_by_amendment_5: []

  # ────── Archetypes observed ──────
  archetypes_observed: [Bimodal, Choppy, Monotonic_down]

  # ────── Cross-arc observation tags ──────
  cross_arc_tags:
    - "direct_driver_pipeline_methodology"
    - "sl_multiplier_signal_specific"
    - "archetype_shift_v_shape_to_bimodal_under_5ers_eet"
    - "chained_dd_blowout_pr_hhhl_bimodal_tight_sl"
    - "w1_producer_canonical_alignment_consumer"

  # ────── Step 6 ──────
  # Per L_PROTOCOL Amendment 4: Step 6 auto-dispatch fires only when at least one
  # Top-K candidate clears §3 constraints #1-9 (PASS-tier). All three Top-K
  # candidates produced amended verdict FAIL at Amendment 3 (chained DD > 8% gate).
  # Step 6 therefore did NOT dispatch.
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

**Proximate cause:** Top-1 search-WFO best architecture (A1 system-level-filter with SL = 1.5 × ATR(14)_H4, `sl_partial_close_1r_runner_trail` exit, exposure cap = 2 per currency) failed Amendment 3 chained-DD constraint with `chained_max_dd_base_pct = 69.24%` at `r_base = 0.5%` — almost an order of magnitude above the 8% safety budget. Top-2 and Top-3 candidates produced chained DDs of 73.47% and 46.14% respectively; all three FAIL. Engine canonical `primary_failure_mode = step5_chained_dd_above_gate`; scaling factors emitted null because the chained DD failure precedes the scaling-feasibility computation. Per-fold metrics tell the same story: only 2 of 10 evaluated folds positive (folds 3 + 9 marginally), worst-fold ratio –0.886, worst-fold DD 23.88% — the underlying per-fold WFO already showed the strategy was non-viable at `r_base = 0.5%` before chained accumulation made it catastrophic.

**Structural cause:** The c2 Bimodal cluster (1,566 trades, 23.68% of the pool) exhibits exceptional capturability metrics — `reach_1R = 100%`, `mfe_p50 = +7.55`, `ww_pp = 0%`, composite = 1.000 — every trade in c2 eventually reaches +1R MFE and the median trade reaches +7.55R MFE. The capturability composite picks `SL = 1.5 × ATR` as optimal because the path-geometry of the surviving cluster supports a tight stop. But realising that capturability requires holding through deep drawdowns: c2's `dd_depth = –5.00` (mean per-trade running-peak-to-trough drop in R units) — meaning *during the position lifetime* the typical trade pulls back ≈5R below the running peak before recovering. With `SL = 1.5 × ATR`, the strategy loses 100% of the SL distance on the rejected portion of the cluster's trades while the admitted-and-held portion takes 5R-deep mid-position drawdowns. In aggregate across 11-fold WFO + chained equity, the loss stream dominates and the DD compounds. A1 has no Step-4 admit filter (Amendment 5 Gate 2 fails on c2 AUC = 0.48), so every signal enters the market — there is no mechanism to skip the c2 trades that don't recover.

**What this tells us about the signal class:** PR-HHHL produces a high-MFE Bimodal cluster under canonical 5ers_eet, but the path geometry — deep mid-position drawdowns followed by V-shape-like recovery — is **incompatible with tight SL multipliers** at any meaningful risk size. The Step 3 capturability composite, which picks SL on a per-cluster optimization without regard to portfolio-level chained-DD, misleads here. Future PR-HHHL retries would need either (a) a wider SL multiplier traded against weaker capturability, (b) a Step-4 classifier that meaningfully discriminates c2-membership above the 0.65 AUC threshold (current 0.48 = noise — Step-4 cannot separate c2 from non-c2 with the v3.0 27-feature catalogue), or (c) an A3/A4 path-classifier with materially different exit timing than the canonical Bimodal slate tested here.

---

## §3 Cross-arc observations

1. **Direct-driver pipeline methodology disclosure.** Arc 8 v3.0.2 Step 5 used the direct `run_search` + `run_holdout` driver pattern (Arc 11 v3.0.2 precedent), consuming canonical primitives (`core.wfo.orchestrator.run_search`, `core.wfo.amended_gates.classify_amended_fold_stats`, `ArcOrchestrator._run_amendment_3_evaluation` invoked via stub instance). The direct-driver path was necessitated by Step 1-4 artefact schema differences vs the canonical orchestrator's expected inputs (the orchestrator's internal `run_step_2` uses a different path-feature set and `paths.parquet` schema than my custom Step 1, producing different cluster IDs). Per the driver-script divergence investigation (PR #213), direct-driver and orchestrator paths produce byte-identical results on matched inputs — methodology equivalence to the canonical orchestrator path is established. Cross-arc trust calibration: Arc 8 + Arc 11 v3.0.2 use the direct-driver path; Arc 7 + Arc 10 use the canonical orchestrator path. Both paths satisfy L_PROTOCOL §2 Step 5 requirements.

2. **Cross-arc SL-multiplier finding — SL is signal-specific.** Comparing Wave 1 v3.0 / v3.0.2 PASS-DEPLOYABLE and FAIL outcomes by winning SL multiplier:

   | Arc | Signal | Winning arch | Winning SL | Worst-fold DD (base) | Chained DD (base) | Verdict |
   |---|---|---|---|---|---|---|
   | Arc 10 v3.0 | DLR (D1 swing-low rejection) | A1 system_level_filter | **3.5 × ATR** | 9.22% | unmeasured (forwarded) | **PASS-DEPLOYABLE** (post Amendment 3 §10 re-eval) |
   | Arc 8 v3.0 | PR-HHHL | A6 meta-labeling | **1.5 × ATR** | 1.98% (per-fold; chained unmeasured) | unmeasured | FAIL (ratio 1.749 < 2.0 gate) |
   | Arc 8 v3.0.2 | PR-HHHL | A1 system_level_filter | **1.5 × ATR** | 23.88% | 69.24% | FAIL (chained DD blowout) |

   Same exit policy primitive (`sl_partial_close_1r_runner_trail` in both Arc 10 + Arc 8 v3.0.2) works across signal classes; the SL multiplier is the load-bearing axis and is determined by the cluster path geometry, not the exit primitive. Arc 10's DLR V-shape supports wide SL (3.5×) → fewer SL-fires + comfortable chained DD; Arc 8 v3.0.2's PR-HHHL Bimodal demands tight SL (1.5×) per capturability → many SL-fires + catastrophic chained DD. **Future arcs MUST NOT assume SL multipliers port across signal classes.** Step 3 capturability optimization must be cross-checked against Step 5 chained-DD viability before architecture admission.

3. **Archetype shift from V-shape (UTC v3.0) to Bimodal (5ers_eet v3.0.2).** Under canonical 5ers_eet bar boundaries + canonical W1 producer + restored multi-TF features, the high-MFE cluster modal-tags as Bimodal (42% modal share) rather than V-shape (Arc 8 v3.0 UTC convention). The dynamics are unchanged (mean R +2.62, mfe_p50 +7.55, dd_depth –5.00, recovery score 0.26) — the canonical `assign_shape_tag` quartile rule applied to EET-aggregated paths produces higher `n_local_peaks` counts (~50 mean for c2) than UTC-aggregated paths, shifting per-trade modal classification into Bimodal. **Cross-arc implication:** other Wave 1 v3.0.2 retries (especially Arc 5 v3.0.2, Arc 11 v3.0.2) running under 5ers_eet may see similar archetype shifts vs their v3.0 UTC baselines. Cross-arc analysis should compare dynamics (mean_r, mfe_p50, recovery ratio) not just archetype names. The archetype shift is also part of WHY this arc fails: Amendment 5 Gate 1 admits A4 (Bimodal) not A3 (V-shape), and the Bimodal canonical 3-exit slate is narrower than V-shape's 4-exit slate.

4. **W1 producer canonical alignment (PR #208) consumer.** Arc 8 v3.0.2 is the second Wave 1 v3.0.2 arc to run end-to-end against the canonical W1 producer (Arc 10 v3.0.2 parity-rerun was first per `docs/calibration/arc_10_signal_parity_rerun_2026_05.md`). The prior `arc/l_arc_8_v3.0.2_halted` Step 4 c2 AUC of 0.6938 collapses under the W1 fix to 0.4822 — confirming the inflated AUC was entirely a function of the W1 lookahead, not a real predictive signal. Verification documentation for the W1 leak is preserved on `arc/l_arc_8_v3.0.2_halted` @ `aa641dc`.

5. **Engine risk-decoupling invariant verified (informational).** Per `docs/dispatches/risk_leak_diagnosis.md` (2026-05-25), the engine is risk-decoupled by construction: `risk_pct` is a sizing-only parameter; admit/exit decisions are byte-identical across risk levels. Amendment 3 scaled-rerun phase ran natively without engine concern. Arc 8 v3.0.2 uses the canonical Amendment 3 evaluator (via stub orchestrator instance) consistent with Arc 7 v3.0.2's invocation pattern.

6. **Search-WFO-fail / holdout-fail symmetry — opposite of Arc 8 v3.0 cross_arc_tag `search_wfo_fail_holdout_pass_pattern`.** Arc 8 v3.0 (UTC) closed FAIL on search-WFO worst-ratio 1.749 but Top-3 holdout all PASS-DEPLOYABLE (ROI +19% to +43%, DD ≤ 1.7%) — a documented anomaly. Arc 8 v3.0.2 (5ers_eet) does NOT reproduce this asymmetry: chained DD across the full 2010-2025 window is catastrophic, so both search WFO and holdout fail together. The v3.0 anomaly was an artefact of UTC + multi-TF-all-NaN + A6 best architecture + 2021-2025 regime favourable; under canonical conditions the regime asymmetry interpretation does not hold.

---

## §4 deployment_spec

**Not populated.** Verdict is FAIL; no deployment_spec produced per template v1.2/v1.3 convention. The Top-1 search-WFO best architecture (A1 + SL=1.5×ATR + `sl_partial_close_1r_runner_trail` + exposure_cap=2) failed Amendment 3 scaling and is not deployable at any risk level (chained DD 69.24% well above 8% safety budget; scaling math impossible to recover).

---

## §10 Retroactive re-evaluation (vs Arc 8 v3.0 UTC)

Quantitative comparison of `results/l_arc_8/` (Arc 8 v3.0, UTC, multi-TF all-NaN, no canonical W1 producer, A6 best architecture) vs this run (Arc 8 v3.0.2, 5ers_eet, multi-TF restored + canonical W1 producer, A1 best architecture):

| Metric | Arc 8 v3.0 (UTC) | Arc 8 v3.0.2 (5ers_eet) | Delta interpretation |
|---|---|---|---|
| Pool size | 6,757 | 6,612 | –2.1% (H4 boundary shift; informational) |
| Cluster K | 3 | 3 | Same structural decomposition |
| c2 archetype (modal) | V-shape recovery (~23%) | Bimodal (23.7%) | Methodological — 5ers_eet bar boundaries + n_local_peaks rule shift modal tag (see §3 obs. 3) |
| c2 mfe_p50 | (not directly reported in v3.0 closure) | +7.55 | Bimodal capturability remains very high |
| c2 Step 4 best AUC | 0.5300 (RF, no holdout-window filter) | 0.4822 (LGBM, IS-only filter) | Drop from 0.530 → 0.48 partially attributable to PR #185 `train_end` filter + 5ers_eet shift; both are near-random — the entry-feature ceiling pattern persists |
| Best architecture | A6 meta-labeling | A1 system_level_filter | Amendment 5 Gate 2 fails at v3.0.2 AUC 0.48 → A6 not admitted; A1 baseline wins Top-K instead |
| Best architecture SL multiplier | 1.5 × ATR | 1.5 × ATR | Same SL surfaces from capturability composite optimum |
| Best architecture exit policy | (legacy 3-exit V-shape slate, pre-CC_18) | sl_partial_close_1r_runner_trail | CC_18 canonical Bimodal slate applied |
| Worst-fold ratio | 1.749 | –0.886 | Catastrophic regime shift |
| Worst-fold DD (base) | 1.98% | 23.88% | 12× per-fold blowup |
| Chained DD (base) | Not measured (forwarded) | **69.24%** | First Arc 8 measurement at chained DD; immediately FAILs |
| Holdout PASS-DEPLOYABLE on Top-3 | YES (ROI +19% to +43%, DD ≤ 1.7%) | NO (subsumed into chained DD failure) | Search-WFO-fail / holdout-pass asymmetry from v3.0 does NOT reproduce |
| Primary failure mode | `entry_feature_auc_ceiling` (re-eval `step5_ratio_below_gate_after_scaling`) | `step5_chained_dd_above_gate` | Engine-canonical per current priority-order classifier |
| Verdict | FAIL | FAIL | Both fail; v3.0.2 fails more severely |

**Methodological caveat:** UTC and 5ers_eet are different tests. The Arc 8 v3.0 result is NOT a baseline against which Arc 8 v3.0.2 should be measured; both are independent observations under different bar-storage conventions. The shared FAIL verdict is meaningful — PR-HHHL is not deployable under either convention — but the failure mechanisms differ: v3.0 was a ratio-below-gate failure with surprisingly tight per-fold DDs and a holdout/search asymmetry; v3.0.2 is a chained-DD-above-gate failure with catastrophic per-fold DDs and no asymmetry. The §3 cross-arc-tag `search_wfo_fail_holdout_pass_pattern` from Arc 8 v3.0 should be reviewed in light of this: it may have been an UTC-specific regime artefact rather than a generalisable Wave 1 finding.

**Cross-arc primary_failure_mode reconciliation:** Arc 8 v3.0's closure §10 retroactively re-classified the failure mode to `step5_ratio_below_gate_after_scaling`. Arc 8 v3.0.2's failure mode (`step5_chained_dd_above_gate`) is emitted directly by the priority-ordered classifier in `core.wfo.amended_gates` per current Amendment 3 + 3.1 logic. No chat override of the engine's classification.

---

## §11 Provenance + reproducibility

- **Branch state at closure:** `arc/l_arc_8_v3.0.2`
- **Cut from:** `origin/main` @ `dd391cd` (post PR #208 W1 producer fix); subsequently merged Amendment 3.1 via PR #211 @ `7ee5c59` (commit `e808216`).
- **Engine state at run:** `core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)` consumed by `core.features.multi_tf._w1_close_slope_sign` (PR #208 canonical alignment); `core.wfo.amended_gates` with Amendment 3.1 r_max-as-cap semantics (PR #211); `core.sim.exit_policies` canonical registry (PR #195) with `sl_partial_close_1r_runner_trail`; `core.runners._fold_stats_helpers.compute_per_day_max_dd(boundary_convention="5ers_eet")` (PR #197); `core.steps.step_4_extraction.run_step_4(train_end=...)` (PR #185).
- **Boundary convention:** `5ers_eet` end-to-end (H4 + D1 + W1 panels + multi_tf feature alignment + Amendment 6 daily-DD bucketing). No UTC fallback.
- **Cache state:** PR-E.2 temp cache (`C:/Users/panap/AppData/Local/Temp/pr_e2_cache/`) pre-warmed with all 28 pairs of `H4_5ers_eet`/`D1_5ers_eet`/`W1_5ers_eet` parquets from the prior `_halted` run.
- **Determinism:** `random_state = 42`, `n_jobs = 1`, `lineterminator = "\n"` per `core.determinism`. Step 1 pool content sha256 = `d3543b1a996743d4b289cc1b0ee09becdd90e450c72cd65ed5fdccf3a7cc25d5`. Step 4 LGBM best-AUC classifier persisted at `results/l_arc_8_v3.0.2/step_4/classifiers/2.pkl` with sha256 manifest. Step 5 raw result objects pickled at `results/l_arc_8_v3.0.2/step_5/_raw_results.pkl`.
- **Step 5 execution path:** direct `run_search` + `run_holdout` driver (Arc 11 precedent) — see §3 cross-arc observation 1.
- **Driver scripts:** `scripts/l_arc_8_v3_0_2/{shared.py, step_1_pool.py, step_2_clustering.py, step_3_capturability.py, step_4_extraction.py, step_5_wfo.py}`.
- **SignalModule wrapper added this arc:** `core/strategies/pullback_resume_hhhl_long/{__init__.py, signal_module.py}` — wraps `core.signals.pullback_resume_hhhl.evaluate_pullback_resume_hhhl_signal` in the canonical `core.arc.signal_protocol.SignalModule` Protocol. Reusable by any future PR-HHHL arc.
- **Preserved prior:** `arc/l_arc_8_v3.0.2_halted` @ `aa641dc` (local + remote) — UTC convention + W1 producer lookahead + Step 4 c2 V-shape AUC 0.6938 + verification HALT artefacts. Kept for historical reference; not consumed by this retry.
- **Compute:** Steps 1+2+3+4 ~4 min total; Step 5 search + holdout + Amendment 3 evaluation 70 min wall-clock single-process.

End of closure.
