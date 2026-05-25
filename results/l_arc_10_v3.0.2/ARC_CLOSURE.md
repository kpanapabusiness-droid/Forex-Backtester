# ARC_10_v3.0.2_CLOSURE — l_arc_10_v3.0.2

> **Closed:** PENDING_STEP_5
> **Branch:** arc/l_arc_10_v3.0.2
> **Closure doc path:** results/l_arc_10_v3.0.2/ARC_CLOSURE.md
>
> **SCAFFOLD STATE:** This closure was pre-written during the Step 1 wall-clock wait. All sections are present per template v1.3.1; non-deferred fields will be backfilled post-Step-5. Two phase-deferral blocks (Amendment 3 + Step 6) are populated per dispatch §2/§4; do not auto-finalise into a non-PROVISIONAL verdict until both addenda land.

---

## §1 tracker_payload

```yaml
tracker_payload:

  template_version: v1.3   # Schema dispatch — accepts v1.3.1 architectures_skipped_by_amendment_5 field.

  # ────── Identity ──────
  arc_name: l_arc_10_v3.0.2
  signal: D1 swing-low rejection long (DLR, v0.1) — bullish rejection of confirmed ascending D1 swing-low, 4H entry
  tf: H4
  sub_protocol: vanilla
  closed_timestamp: PENDING_STEP_5
  closure_doc_link: results/l_arc_10_v3.0.2/ARC_CLOSURE.md
  boundary_convention: 5ers_eet

  # ────── Verdict ──────
  # Provisional pending Amendment 3 scaled-rerun addendum. Engine risk-decoupling
  # bug (docs/bugs/risk_leaking_dependency.md, to land alongside engine fix on
  # engine/risk_decoupling_admit_exit) prevents scaled holdout reruns at r_safe/
  # r_hard — Step 5 verdict is canonical at r_base=0.5% only.
  verdict: PENDING_STEP_5_PROVISIONAL
  one_line: PENDING_STEP_5
  failed_at_step: PENDING_STEP_5
  primary_failure_mode: PENDING_AMENDMENT_3_ADDENDUM

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: PENDING_STEP_1
    window_start: 2010-01-01
    window_end: 2026-04-30
    kh24_co_fire_pct: null
    configs_evaluated_step5: PENDING_STEP_5
    search_scope_flag: PENDING_STEP_5   # thin | normal | broad
    boundary_convention: 5ers_eet

  # ────── Best architecture ──────
  best_architecture:
    name: PENDING_STEP_5
    cluster: PENDING_STEP_5
    archetype: PENDING_STEP_5
    config: PENDING_STEP_5
    sl_atr: PENDING_STEP_5
    exit_policy: PENDING_STEP_5
    exposure_cap: PENDING_STEP_5
    worst_fold_ratio: PENDING_STEP_5
    worst_fold_roi_base_pct: PENDING_STEP_5
    worst_fold_dd_base_pct: PENDING_STEP_5
    mean_fold_ratio: PENDING_STEP_5
    mean_fold_roi_pct: PENDING_STEP_5
    sign_pos_folds: PENDING_STEP_5
    n_trades_total: PENDING_STEP_5
    holdout_roi_pct: PENDING_STEP_5
    holdout_dd_pct: PENDING_STEP_5
    holdout_passed: PENDING_STEP_5
    oracle_worst_ratio: PENDING_STEP_5
    oracle_real_gap_sharpe: null
    features_in_winning_config: []   # populated post-Step-5; A1 expected → [] (rule-based)

    # ── Amendment 3 risk-normalised fields — DEFERRED ──
    # See amendment_3_evaluation block below. All fields populated by post-fix addendum.
    chained_max_dd_base_pct: PENDING_AMENDMENT_3_ADDENDUM
    per_day_max_dd_artefact_path: PENDING_AMENDMENT_3_ADDENDUM
    per_day_max_dd_base_summary:
      n_days: PENDING_AMENDMENT_3_ADDENDUM
      p50_pct: PENDING_AMENDMENT_3_ADDENDUM
      p95_pct: PENDING_AMENDMENT_3_ADDENDUM
      p99_pct: PENDING_AMENDMENT_3_ADDENDUM
      max_pct: PENDING_AMENDMENT_3_ADDENDUM
    k_safe: PENDING_AMENDMENT_3_ADDENDUM
    k_hard: PENDING_AMENDMENT_3_ADDENDUM
    r_safe_pct: PENDING_AMENDMENT_3_ADDENDUM
    r_hard_pct: PENDING_AMENDMENT_3_ADDENDUM
    scalable_to_safe: PENDING_AMENDMENT_3_ADDENDUM
    scalable_to_hard: PENDING_AMENDMENT_3_ADDENDUM
    worst_fold_roi_at_r_safe_pct: PENDING_AMENDMENT_3_ADDENDUM
    worst_fold_roi_at_r_hard_pct: PENDING_AMENDMENT_3_ADDENDUM
    chained_max_dd_at_r_safe_pct: PENDING_AMENDMENT_3_ADDENDUM
    chained_max_dd_at_r_hard_pct: PENDING_AMENDMENT_3_ADDENDUM
    daily_dd_breaches_at_r_safe: PENDING_AMENDMENT_3_ADDENDUM
    daily_dd_breaches_at_r_hard: PENDING_AMENDMENT_3_ADDENDUM
    holdout_roi_at_r_safe_pct: PENDING_AMENDMENT_3_ADDENDUM
    holdout_dd_at_r_safe_pct: PENDING_AMENDMENT_3_ADDENDUM
    holdout_roi_at_r_hard_pct: PENDING_AMENDMENT_3_ADDENDUM
    holdout_dd_at_r_hard_pct: PENDING_AMENDMENT_3_ADDENDUM
    sizing_convention: reset_floor
    chained_dd_method: PENDING_AMENDMENT_3_ADDENDUM   # equity_stitching | full_window_sim

    # ── v1.2 deployment-spec fields ──
    config_artefact_path: PENDING_STEP_5   # configs/l_arc_10_v3.0.2/winning_config.yaml when PASS
    deployment_spec_section_present: false   # set true when §4 populated on PASS verdict

  # ────── Cost decomposition ──────
  cost_decomposition: null   # A1 expected → no admit/reject pools

  # ────── Per-cluster results ──────
  clusters:
    # PENDING_STEP_2 — populate one block per cluster from Step 2 output.
    # Reference shape from Arc 10 v3.0 closure (UTC) for c1 V-shape:
    # c<id>:
    #   n: <int>
    #   archetype: v_shape_recovery | monotonic_down | bimodal | stepwise | unclassified | choppy | monotonic_up
    #   sl_atr: <float>             # Step 3 best per cluster
    #   step3_composite: <float>
    #   mfe_p50_r: <float>
    #   ww_pp: <float>
    #   reach_1r: <float>
    #   step4_e_auc: <float or null>
    #   step4_d1_auc: null          # v3.0 protocol has no D1 pipeline
    #   outcome: passed_step3 | dies_step3 | dies_step4 | wins_step5 | viable_step5 | dies_step5

  # ────── Architecture results ──────
  architectures_tested: PENDING_STEP_5    # subset of [A1, A2, A3, A4, A5, A6]
  architecture_results:
    # PENDING_STEP_5 — one entry per architecture tested
    # A1: {tested: true, won: <bool>, worst_fold_ratio: <float or null>}

  # ────── Architectures skipped under Amendment 5 / 5.1 ──────
  architectures_skipped_by_amendment_5: []   # populated dispatch-time per Step 4 AUC outcome

  # ────── Archetypes observed ──────
  archetypes_observed: PENDING_STEP_2

  # ────── Cross-arc observation tags ──────
  cross_arc_tags:
    - eet_canonical_vs_utc_baseline_diagnostic
    - w1_producer_canonical_alignment_pr_208_first_consumer_after_fix
    - amendment_3_evaluation_deferred_pending_engine_risk_decoupling_admit_exit_fix
    - step_6_deferred_pending_amendment_3_addendum_dependency_chain
    # Populate per Step 5 outcome:
    # - v_shape_archetype_cross_arc_persistence_under_eet (if c1 V-shape surfaces under 5ers_eet)
    # - exit_policy_dominates_classifier_under_eet (if A1 sl_partial_close winner reproduces)
    # - eet_d1_alignment_changes_arc_10_pool_composition (if pool delta vs UTC v3.0 is non-trivial)

  # ────── Amendment 3 evaluation — DEFERRED ──────
  # Engine risk-decoupling bug: docs/bugs/risk_leaking_dependency.md (to land
  # alongside engine fix on engine/risk_decoupling_admit_exit). Same config
  # at different risk_pct values produces different trade counts → scaled-rerun
  # phase produces incorrect holdout-at-r_safe/r_hard numbers. Canonical Steps
  # 1-5 at r_base=0.5% are unaffected (single-risk-level sims work fine).
  amendment_3_evaluation:
    ran: false
    trigger: deferred_pending_engine_fix_engine_risk_decoupling_admit_exit
    manual_rerun_pending: true
    canonical_r_base_pct: 0.005
    notes: |
      Canonical Steps 1-5 at r_base=0.5% complete. Amendment 3
      scaled-rerun phase (holdout reruns at r_safe, r_hard for gate
      evaluation) deferred pending engine risk-decoupling fix
      (PR engine/risk_decoupling_admit_exit). Closure addendum will
      populate Amendment 3 fields + finalise verdict post-fix-merge.

  # ────── Step 6 causal-audit registry — DEFERRED ──────
  # Step 6 framework patch (PR #207 + #208) landed on main 2026-05-25 — auto-
  # dispatch would fire normally. However Step 6 gates on §3 constraints #1-9
  # which depend on Amendment 3 PASS-tier classification. With Amendment 3
  # deferred, Step 6 also defers. Both addenda land together post-engine-fix.
  step_6:
    ran: false
    trigger: deferred_pending_amendment_3_addendum
    manual_cli_pending: true
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
    notes: |
      Step 6 auto-dispatch gates on Amendment 3 PASS-tier evaluation.
      With Amendment 3 evaluation deferred, Step 6 also deferred.
      Both addenda land together post-engine-fix-merge.
```

---

## §2 Why <succeeded | failed> (PROVISIONAL — pending Amendment 3 + Step 6 addenda)

PENDING_STEP_5 — populate post-Step-5 with 100-300 word prose explaining the proximate + structural cause of the canonical-r_base verdict. Tag as PROVISIONAL pending Amendment 3 addendum since gate evaluation occurs at scaled risk, not at r_base.

Provisional framing template (delete this comment + paste actual narrative):

> Verdict provisional pending Amendment 3 scaled-rerun phase (deferred per engine risk-decoupling bug — `docs/bugs/risk_leaking_dependency.md`, fix on `engine/risk_decoupling_admit_exit`). Canonical Steps 1-5 at `r_base=0.5%` complete; gate evaluation at scaled `r_safe` / `r_hard` requires the engine fix before holdout reruns at scaled risk produce valid numbers. Per L_PROTOCOL Amendment 4 §"Discipline rules" the closure addendum sequence will populate Amendment 3 fields + Step 6 audit + finalise verdict; if Step 6 surfaces critical failure post-merge, verdict subject to amendment via closure addendum.

---

## §3 Cross-arc observations (PROVISIONAL stubs — finalise post-Step-5)

- **EET vs UTC methodology framing.** This arc tests the cross-arc V-shape archetype hypothesis under canonical 5ers_eet for the first time. Prior `arc/l_arc_10_v3.0.2` (UTC, deleted 2026-05-25) produced byte-identical numbers to Arc 10 v3.0 — that proved the engine refactor and signal-module canonicalisation are no-op under UTC. Deltas observed under 5ers_eet (pool composition, cluster geometry, AUC, WFO numbers) are the actual diagnostic for whether Arc 10's edge survives canonical production conditions. Arc 7 c1/c3 + Arc 10 v2.3 c1 + Arc 10 v3.0 c1 form the prior V-shape cohort lineage; this arc's c1 (if it surfaces) is the first under EET-canonical D1 alignment.

- **W1 producer canonical alignment verification.** PR #208 (commit `8ce3b3d`) fixed `core/features/multi_tf.py::_w1_close_slope_sign` — replaced raw `merge_asof(direction="backward", allow_exact_matches=False)` (within-period lookahead under any boundary convention) with canonical `get_htf_value_at(..., require_fully_closed=True)`. Arc 10 v3.0.2 is the first Wave 1 retry running under the fixed producer. Pre-fix the bug leaked Sunday end-of-week close into every H4 bar from Mon 00:00:01 onwards of the same week. Verification: runtime assertion `merge_asof` absent + `get_htf_value_at` present in `_w1_close_slope_sign` source. Cross-arc impact note retained per dispatch §5 cross-arc observations: Arc 8 v3.0.2 originated detection; Arc 10 audit confirmed; Arc 5 + Arc 8 + Arc 10 + Arc 11 all affected; Arc 7 v3.0.2 not affected (different feature set / different signal).

- **Engine risk-decoupling bug — Amendment 3 deferral.** Arc 7 r=2% rerun surfaced that the engine doesn't cleanly decouple `risk_pct` from admit/exit decisions — same config at different `risk_pct` produces different trade counts. Affects Amendment 3 scaled-rerun phase (holdout-at-r_safe and -at-r_hard); does NOT affect canonical Steps 1-5 at `r_base=0.5%`. Arc 10 v3.0.2 runs canonical Steps 1-5 normally, defers Amendment 3 scaled-rerun phase to post-fix addendum. First Wave 1 arc to surface this deferral pattern in a v3.0.2 closure. Tracked at `docs/bugs/risk_leaking_dependency.md` (doc + fix to land on `engine/risk_decoupling_admit_exit`). NOTE: doc not present on main at closure-scaffold time; placeholder pending the PR that lands both.

- **Step 6 framework patch already merged — but deferred via Amendment 3 dependency chain.** PR #207 (Step 6 A1 vacuous-pass, `c244568`) landed on main. Step 6 framework auto-dispatch would fire normally on PASS-tier Top-1. However, PASS-tier classification per L_PROTOCOL §3 "Evaluation order" gates Step 6 on Amendment 3 having cleared first. With Amendment 3 deferred, Step 6 also defers. Both addenda land together post-engine-fix-merge. This is the first arc closure exhibiting the "Step 6 patch merged but Step 6 still defers" pattern — distinct from the prior `arc/l_arc_10_v3.0.2` (deleted) "Step 6 patch missing → Step 6 defers" pattern. Documentation precedent.

- **Additional Step-5-outcome-dependent observations** (populate post-Step-5 — examples below, delete or replace per actual surfaced findings):
  - `v_shape_archetype_cross_arc_persistence_under_eet` — only if a V-shape cluster surfaces under EET cluster geometry
  - `exit_policy_dominates_classifier_under_eet` — only if A1 with sl_partial_close_1r_runner_trail wins (Arc 10 v3.0 UTC pattern reproduction under EET)
  - `eet_d1_alignment_changes_pool_composition` — only if pool size delta vs Arc 10 v3.0 UTC (n=3,301) exceeds ±10% (informational HALT threshold per intent doc §8 soft-surface)

---

## §4 deployment_spec

> REQUIRED if verdict ∈ {PASS-DEPLOYABLE, PASS-VIABLE, PASS-*-PROVISIONAL, PASS-*-PENDING-STEP6}. OPTIONAL otherwise.
>
> This arc's verdict is `PENDING_STEP_5_PROVISIONAL` — §4 will be populated only after the Step 5 outcome is known. If Step 5 produces a non-PASS candidate, §4 stays empty (template-v1.3.1 mandates header presence even when content is empty for FAIL/HALT closures).

PENDING_STEP_5 — populate per template v1.3.1 §4.1-§4.11 sub-sections only on PASS-tier verdict.

---

## §10 Retroactive re-evaluation — EET vs UTC delta

PENDING_STEP_5 — populate post-Step-5 with quantitative delta vs Arc 10 v3.0 original (UTC) per intent doc §I.3.

Framing template (delete this comment + paste actual delta narrative):

> The prior `arc/l_arc_10_v3.0.2` (UTC, deleted 2026-05-25) produced byte-identical numbers to Arc 10 v3.0 — confirming the engine refactor, mid-price feature canonicalisation (PR #189), signal-module timezone alignment (PR #193), Amendment 6 EET daily-DD boundary (PR #197), canonical exit-policy registry (PR #195), Amendment 5 dispatch-time selection (PR #194), and Amendment 5.1 Gate 4 qualifier (PR #201) are no-op under UTC. This arc — Arc 10 v3.0.2 under canonical `boundary_convention="5ers_eet"` — is therefore the **first canonical-engine test** of the DLR signal under production-aligned boundaries.
>
> Deltas vs Arc 10 v3.0 (UTC) at canonical r_base = 0.5%:
> - Pool size delta: `<EET_n> - 3301 = <delta>` (`<delta_pct>%`)
> - Cluster geometry: K under EET = `<K_EET>` vs v3.0 K=3; c1 V-shape n delta = `<delta>`
> - Step 4 c1 AUC: `<EET_AUC>` vs v3.0 UTC 0.5199 (`<delta>`)
> - Step 5 winner: `<EET_winner>` at SL=`<EET_sl>` × ATR, exit `<EET_exit>`, exposure `<EET_cap>`
> - Worst-fold ROI: `<EET_roi>%` vs v3.0 UTC 26.49% (`<delta>`)
> - Worst-fold DD: `<EET_dd>%` vs v3.0 UTC 9.22% (`<delta>`)
> - Worst-fold ratio: `<EET_ratio>` vs v3.0 UTC 5.4185 (`<delta>`)
> - Holdout: `<EET_holdout_roi>% / <EET_holdout_dd>%` vs v3.0 UTC 59.07% / 5.03%
>
> **Calibration doc obsolete** — `docs/calibration/arc_10_signal_parity_rerun_2026_05.md` ±2pp tolerances were authored against UTC-vs-UTC parity, not the methodology that runs in production. Under 5ers_eet, large deltas vs Arc 10 v3.0 are EXPECTED and CORRECT, not HALT-triggering.
>
> §10 conclusion: deferred pending Amendment 3 + Step 6 addenda. Full re-evaluation finalised in `ARC_CLOSURE_ADDENDUM.md` post-engine-fix-merge.

---

## §11 Provenance + reproducibility

- **Branch:** `arc/l_arc_10_v3.0.2` cut fresh from `origin/main@a8c02b4` (2026-05-25) post deletion of UTC-contaminated prior branch (`ddc6420`).
- **Plumbing PRs merged into branch:**
  - PR #207 (Step 6 A1 vacuous-pass, `c244568`)
  - PR #208 (W1 producer canonical alignment, `8ce3b3d`)
- **Arc 10 v3.0.2-specific plumbing commits on branch:**
  - `8e6c2e6` — boundary_convention config support + new arc_open.yaml
  - `d56f85c` — step_2 out_dir from cfg
  - `7816cc2` — step_3 out_dir + step_2 input from cfg
  - `f37a9dd` — step_4 out_dir + step_2/3 inputs from cfg
  - `f3a5e9b` — step_5 out_dir + step_2/3/4 inputs from cfg
- **Determinism:** `random_state=42`, `n_jobs=1`, `lineterminator="\n"`. Step 1 two-run sha256 verification: PENDING_STEP_1.
- **Config artefact:** `configs/l_arc_10_v3.0.2/arc_open.yaml` (boundary_convention: 5ers_eet locked).

---

## §12 Addendum sequence post-engine-fix

Per resume signal §6 and L_PROTOCOL Amendment 4 §"Discipline rules":

1. `git fetch origin && git merge origin/main` onto `arc/l_arc_10_v3.0.2` (engine fix lands on main)
2. Re-run only the Amendment 3 scaled-rerun phase on the existing Step 5 canonical artefacts. No Step 5 re-search needed; only the scaled-rerun layer (`_run_amendment_3_evaluation` or equivalent).
3. Step 6 auto-dispatch fires if scaled-rerun produces PASS-tier Top-1.
4. Write `results/l_arc_10_v3.0.2/ARC_CLOSURE_ADDENDUM.md`:
   - Populate all `PENDING_AMENDMENT_3_ADDENDUM` fields in §1 tracker_payload
   - Populate `step_6` block per audit outcome
   - Finalise §2 prose verdict (strip PROVISIONAL)
   - Finalise §10 conclusion
   - If PASS-tier: populate §4 deployment_spec; set `deployment_spec_section_present: true`; reconstruct or land `configs/l_arc_10_v3.0.2/winning_config.yaml` at `config_artefact_path`
5. Run parser. Atomic commit (addendum + tracker delta). PR to main.

---

End of closure scaffold. Real values backfilled post-Step-5 + post-addendum.
