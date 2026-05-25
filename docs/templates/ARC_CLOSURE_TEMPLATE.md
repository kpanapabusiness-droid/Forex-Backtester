# ARC_CLOSURE.md Template (Locked v1.3.1)

> **Location:** `docs/templates/ARC_CLOSURE_TEMPLATE.md`
> **Status:** locked. Every arc closure MUST follow this template.
> **Referenced by:** `L_PROTOCOL.md` §6.
> **Parser status:** implemented at `scripts/update_tracker_from_closure.py`. Invoke per `scripts/tracker_parser/README.md` (run on the arc branch before opening the closure PR).
>
> Section headings are LITERAL — do not rephrase. Field names inside `§1 tracker_payload` are LITERAL — parser depends on exact spelling.
>
> **Design priority:** machine-parseability over human readability. §1 YAML is the source of truth for the tracker. §2 + §3 exist only to preserve cross-arc synthesis quality that prose enables and YAML doesn't.
>
> **v1.1 (2026-05-22, L_PROTOCOL Amendment 3):** risk-normalised gate fields added to `best_architecture` block. Two fields renamed (see §"Schema versioning" at end of template). `primary_failure_mode` enum extended.
> **v1.2 (2026-05-23, deployment-spec addition):** `config_artefact_path`, `deployment_spec_section_present`, `template_version` fields added to `best_architecture` block. New §4 deployment_spec section: REQUIRED for any PASS verdict, OPTIONAL for FAIL / HALT / DISCOVERY_COMPLETE. Parser enforces config path + §4 presence for PASS verdicts.
> **v1.3 (2026-05-24, L_PROTOCOL Amendment 4):** Step 6 causal-audit framework. New `§1 tracker_payload.step_6` block. Step 6 auto-dispatches on any candidate that clears §3 constraints #1-9; manual CLI available for any closure. Step 6 critical-failure downgrades verdict to FAIL with `primary_failure_mode = step6_causal_audit_fail`. Parser v1.3 detection + Phase 2 tightening: for any PASS verdict with `closed_timestamp > 2026-05-23T06:20:59Z`, Amendment 3 fields required; for `template_version: v1.3` PASS, `step_6` block required.
> **v1.3.1 (2026-05-23, L_PROTOCOL Amendment 5):** AUC-gated A2/A6 architecture selection. New top-level optional field `architectures_skipped_by_amendment_5` (subset of `{A1..A6}`, may be `[]`). Field is informational — captures architectures that would have been tested under Amendment 1's archetype-driven rule but were skipped under Amendment 5's four-gate AUC-driven rule. Phase 1: parser accepts presence or absence. Phase 2: parser REQUIRES the field for any PASS verdict whose `closed_timestamp > AMENDMENT_5_CUTOFF_ISO` (placeholder `2026-05-23T00:00:00Z`, to be backfilled with this PR's merge timestamp post-merge). v1.3 / v1.3.1 share the same `template_version: v1.3` declaration — the field's presence/absence is the v1.3.1 discriminator, not a separate version string. Mirrors the v1.2 / v1.2.1 `chained_dd_method` rollout pattern exactly. Amendment 5.1 (2026-05-25) extends the field's reason-string set to include `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` (Gate 4 PASS-tier-constituent qualifier). No template version bump — field accepts the new string as ordinary content.

---

## Template

```markdown
# ARC_<N>_CLOSURE — <arc_name>

> **Closed:** <ISO timestamp>
> **Branch:** arc/<arc_name>
> **Closure doc path:** results/<arc_name>/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:

  template_version: v1.3   # Schema dispatch. Accepts v1.0, v1.1, v1.2, v1.3 (parser detects via this field OR version-exclusive fields).

  # ────── Identity ──────
  arc_name: <arc_name>
  signal: <signal description, terse, ≤80 char>
  tf: <timeframe>
  sub_protocol: vanilla | heavy_ml_probe | signal_discovery_probe | <other>
  closed_timestamp: <ISO>
  closure_doc_link: results/<arc_name>/ARC_CLOSURE.md

  # ────── Verdict ──────
  verdict: PASS-DEPLOYABLE | PASS-VIABLE | FAIL | HALT | DISCOVERY_COMPLETE
  one_line: <≤140 char summary>
  failed_at_step: 1 | 2 | 3 | 4 | 5 | 6 | N/A
  primary_failure_mode: <enum: pool_too_small | no_clusters_separable | no_capturable_cluster | entry_feature_auc_ceiling | step5_not_scalable | step5_wf_roi_below_gate_after_scaling | step5_ratio_below_gate_after_scaling | step5_chained_dd_above_gate | step5_daily_dd_breach | step5_negative_folds | step5_sign_consistency_fail | step5_trade_count_below_gate | step6_causal_audit_fail | selection_bias | holdout_fail_after_is_pass | admit_only_vs_deployment | step5_dd_above_gate | step5_wf_roi_below_gate | other | N/A>

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: <int>
    window_start: <ISO date>
    window_end: <ISO date>
    kh24_co_fire_pct: <float>
    configs_evaluated_step5: <int>
    search_scope_flag: thin | normal | broad   # thin <50, normal 50-99, broad 100+

  # ────── Best architecture (null block if no winner) ──────
  # v1.1 (Amendment 3): two fields renamed — worst_fold_roi_pct → worst_fold_roi_base_pct,
  # worst_fold_dd_pct → worst_fold_dd_base_pct. Risk-normalised fields added below.
  # Closures landed before v1.1 retain v1.0 field names; parser detects via template version.
  best_architecture:
    name: A1 system_level_filter | A2 classifier_filter | A3 pipeline_de | A4 pipeline_d_exits | A5 portfolio_composition | A6 meta_labeling | null
    cluster: <cluster_id> | aggregate | null
    archetype: V-shape | Stepwise | Bimodal | Monotonic_up | Monotonic_down | Choppy | Unclassified | null
    config: <config name / descriptor> | null
    sl_atr: <float or null>
    exit_policy: <policy name or null>
    exposure_cap: <int or unlimited or null>
    worst_fold_ratio: <float or null>
    worst_fold_roi_base_pct: <float or null>     # v1.1 renamed from worst_fold_roi_pct
    worst_fold_dd_base_pct: <float or null>      # v1.1 renamed from worst_fold_dd_pct
    mean_fold_ratio: <float or null>
    mean_fold_roi_pct: <float or null>
    sign_pos_folds: <"X/N" string or null>
    n_trades_total: <int or null>
    holdout_roi_pct: <float or null>
    holdout_dd_pct: <float or null>
    holdout_passed: <bool or null>
    oracle_worst_ratio: <float or null>
    oracle_real_gap_sharpe: <float or null>
    features_in_winning_config: [<feature_1>, <feature_2>, ...]   # empty list if no winner

    # ── Amendment 3 risk-normalised fields ──
    chained_max_dd_base_pct: <float or null>     # IS + holdout chained at r_base
    per_day_max_dd_artefact_path: <path or null> # relative path to per-day series parquet
    per_day_max_dd_base_summary:                 # summary for human inspection only
      n_days: <int or null>
      p50_pct: <float or null>
      p95_pct: <float or null>
      p99_pct: <float or null>
      max_pct: <float or null>
    k_safe: <float or null>
    k_hard: <float or null>
    r_safe_pct: <float or null>           # POST-CAP (Amendment 3.1: min(r_intrinsic, r_max))
    r_hard_pct: <float or null>           # POST-CAP
    r_safe_intrinsic_pct: <float or null> # NEW (Amendment 3.1): pre-cap value, for audit
    r_hard_intrinsic_pct: <float or null> # NEW (Amendment 3.1): pre-cap value, for audit
    r_safe_capped_at_rmax: <bool>         # NEW (Amendment 3.1): true if intrinsic overshot r_max
    r_hard_capped_at_rmax: <bool>         # NEW (Amendment 3.1): true if intrinsic overshot r_max
    scalable_to_safe: <bool or null>
    scalable_to_hard: <bool or null>
    worst_fold_roi_at_r_safe_pct: <float or null>
    worst_fold_roi_at_r_hard_pct: <float or null>
    chained_max_dd_at_r_safe_pct: <float or null>
    chained_max_dd_at_r_hard_pct: <float or null>
    daily_dd_breaches_at_r_safe: <int or null>   # recounted per-day, not count-scaled
    daily_dd_breaches_at_r_hard: <int or null>
    holdout_roi_at_r_safe_pct: <float or null>   # from holdout re-run; null if not re-run
    holdout_dd_at_r_safe_pct: <float or null>
    holdout_roi_at_r_hard_pct: <float or null>
    holdout_dd_at_r_hard_pct: <float or null>
    sizing_convention: reset_floor | equity_pct  # gate FAILs equity_pct unless chat approves
    # Method used to reconstruct chained equity for `chained_max_dd_base_pct`.
    # "equity_stitching" = v3.0.1 default (multiplicative chaining of per-fold
    # OOS returns with continuity adjustment). "full_window_sim" = v3.0.2
    # follow-up (single full-window sim per top-K candidate; chat directive
    # Q6 gold standard). Optional pre-Wave 2 PASS arc; required post Wave 2.
    chained_dd_method: equity_stitching | full_window_sim | null

    # ── v1.2 deployment-spec fields ──
    config_artefact_path: <relative path to canonical config YAML from repo root, or null for non-PASS verdicts>
    deployment_spec_section_present: <bool>   # parser sanity-check — must be true if verdict starts with PASS-

  # ────── Cost decomposition (null if winning arch is not classifier-based) ──────
  cost_decomposition:
    admit_pool:      {n_fraction: <float>, mean_r: <float>}
    reject_pool:     {n_fraction: <float>, mean_r: <float>}
    early_exit_pool: {n_fraction: <float>, mean_r: <float>}
    # or:
    # cost_decomposition: null

  # ────── Per-cluster results (every cluster from Step 2) ──────
  clusters:
    c0:
      n: <int>
      archetype: <archetype>
      sl_atr: <float>
      step3_composite: <float>
      mfe_p50_r: <float>
      ww_pp: <float>
      reach_1r: <float>
      step4_e_auc: <float or null>
      step4_d1_auc: <float or null>
      outcome: passed_step3 | dies_step3 | dies_step4 | wins_step5 | viable_step5 | dies_step5
    c1:
      # same fields
    # repeat per cluster

  # ────── Architecture results (every arch tested in Step 5) ──────
  architectures_tested: [A1, A2, A3, A4, A5, A6]   # subset that ran
  architecture_results:
    A1: {tested: true, won: false, worst_fold_ratio: <float or null>}
    A2: {tested: true, won: true,  worst_fold_ratio: <float>}
    # repeat per architecture tested

  # ────── Architectures skipped under L_PROTOCOL Amendment 5 (v1.3.1) ──────
  # Amendment 5 (ratified 2026-05-23) gates A2/A6 on Step 4 mean OOS AUC ≥ 0.65
  # (instead of the prior Amendment 1 archetype-driven rule). Architectures that
  # WOULD have been tested under the prior rule but are deliberately skipped
  # under Amendment 5's four-gate procedure are recorded here as an informational
  # signal for cross-arc analytics. Empty list `[]` means no architectures were
  # skipped (the Amendment-5 set equals or supersets the Amendment-1 set).
  # Phase 1: OPTIONAL on all closures. Phase 2 (post-Amendment-5-PR merge):
  # REQUIRED for any PASS verdict whose `closed_timestamp` is strictly after
  # `AMENDMENT_5_CUTOFF_ISO` (placeholder `2026-05-23T00:00:00Z`, backfilled
  # to PR-merge timestamp post-merge). Pre-cutoff closures grandfathered.
  # Optional. Subset of {A1..A6} OR reason-string entries like:
  #   - "a5_gate_4_admission_blocked_by_no_pass_tier_constituent"
  #     (Amendment 5.1, 2026-05-25 — see L_PROTOCOL §2 Step 5 Gate 4)
  #   - other amendment-skip strings as protocol evolves
  architectures_skipped_by_amendment_5: []   # subset of {A1..A6}, may be []

  # ────── Archetypes observed ──────
  archetypes_observed: [<archetype_1>, <archetype_2>, ...]   # union across clusters

  # ────── Cross-arc observation tags (terse, one-line each) ──────
  cross_arc_tags: [<tag_1>, <tag_2>]
  # Examples: "v_shape_auc_ceiling", "pipeline_d1_admit_vs_deployment", "co_fire_with_arc_7"

  # ────── Step 6 causal-audit registry (v1.3 / Amendment 4) ──────
  # REQUIRED for v1.3 PASS verdicts. OPTIONAL otherwise (e.g. FAIL/HALT
  # closures that did not run Step 6, or manual CLI invocations on older
  # closures). When ``ran: false`` for a non-PASS closure the rest of the
  # block may be null. Manual CLI invocations record ``trigger: manual``
  # and NEVER modify the verdict per Amendment 4 §"Discipline rules".
  step_6:
    ran: <bool>                  # true when Step 6 dispatched (auto OR manual)
    trigger: auto_pass | manual | not_applicable
    overall_passed: <bool or null>
    manifest_path: results/<arc>/step_6/manifest.json     # null when ran=false
    categories:
      lookahead: <bool or null>
      selection_bias: <bool or null>
      execution_realism: <bool or null>
      statistical: <bool or null>
      determinism: <bool or null>
      deployment_readiness: <bool or null>
    critical_failures: [<list of "category.check_name" entries>]
    warnings_count: <int>
    verdict_impact: none | downgraded_to_fail
```

---

## §2 Why <failed | succeeded>

(Required prose. 100-300 words. Specific and actionable.)

What was the proximate cause? What was the structural cause behind it? What does this tell us about the signal class or the methodology? Reference specific clusters, architectures, or features by name.

---

## §3 Cross-arc observations

(Required bullet list. What does THIS arc add to the project's cumulative findings?)

- <observation 1>
- <observation 2>

Examples of useful cross-arc observations:
- "Third V-shape cohort with Step 4 AUC < 0.55 — confirms entry-feature ceiling pattern"
- "Pipeline D1 admit/reject pools mirror Arc 5 within ±10% on all metrics — fourth instance"
- "Cross-arc co-fire with Arc 7 = 5.4% on c1 trades — worth EXP-05 style pooling"

---

## §4 deployment_spec

> REQUIRED if verdict ∈ {PASS-DEPLOYABLE, PASS-VIABLE, PASS-DEPLOYABLE-PROVISIONAL, PASS-VIABLE-PROVISIONAL, PASS-*-PENDING-STEP6}. OPTIONAL for FAIL / HALT / DISCOVERY_COMPLETE.
>
> Self-contained porting specification. An EA developer must be able to implement this strategy on MT5 (or any platform) using ONLY this section + the YAML at `best_architecture.config_artefact_path`. No reverse-engineering from other artefacts required.

### 4.1 Pair set

- **Pairs:** <explicit list>
- **Timeframe:** <primary TF>
- **Higher-TF references:** <D1 lagged, W1, etc.>

### 4.2 Signal definition

Pseudocode of the entry trigger logic. Concrete enough to translate to MT5 line-by-line. Cite feature names matching `best_architecture.features_in_winning_config`. All thresholds numeric, no symbolic placeholders.

### 4.3 Feature computation specs

For each feature in `best_architecture.features_in_winning_config`:
- **`<feature_name>`** — source bars, formula, lag rule, output type

### 4.4 Filter chain (A1 / A2 / A6 architectures only)

Ordered filter list applied AFTER signal generation, BEFORE entry. For each: type (hard veto / soft score / classifier admit), threshold, on-reject behaviour. For A2/A6 include classifier type, training-window definition, top features by importance, decision threshold.

### 4.5 Entry mechanics

- Trigger bar / fill bar / fill price / order type / slippage assumption

### 4.6 Exit mechanics

- Initial SL anchor, distance, update rule
- Trail activation condition (incl. close/high reference)
- Trail distance (incl. close/high reference for trail level)
- Trail update frequency
- TP, time exit if any
- Bar-by-bar evaluation order

### 4.7 Exposure cap

- Type (unlimited / per-pair-N / per-currency-N / global-N / custom)
- Counter logic, behaviour at cap

### 4.8 Risk sizing

- `r_safe` value from §1
- Risk basis (reset-floor / equity-pct / other)
- Floor reset condition
- Position size formula
- Lot rounding

### 4.9 Session / time-of-day rules

- Trading hours, day-of-week filter, news-window filter, holiday handling

### 4.10 Discrepancies and caveats

Known differences between backtest and expected live deployment.

### 4.11 Deployment readiness checklist

- [ ] Config YAML at `config_artefact_path` exists and is self-contained
- [ ] All features in §4.3 reproduce against KH-24 live MT5 data within tolerance
- [ ] Risk sizing at `r_safe` confirmed feasible against 5ers broker minimums
- [ ] EA implementation matches §4.2-4.9 line-by-line
- [ ] Backtest-vs-EA byte-identical pre-deployment shadow run on 30 days of data
- [ ] Step 6 causal audit clean
- [ ] **Signal parity verified against deployment venue (PR #189 hard requirement).** Re-aggregate HistData M1 → primary TF under `boundary_convention="5ers_eet"` and confirm aggregated closes match 5ers MT5 closes on the deployment pair set within tolerance (mean abs diff <5 pips on majors). Document the comparison artefact under `docs/calibration/` per the dispatch's C.4 procedure. Skipping this check is a hard FAIL of the deployment gate.

```

---

## Section 4 — Closure → ARC_TRACKER mapping (manual update checklist)

Until parser ships, manual updates follow this exact sequence on each arc close. Every action sources fields from the closure doc's `§1 tracker_payload` block.

### A. Active arcs section
Action: REMOVE the row matching `arc_name`.

### B. Closed arcs summary section
Action: APPEND a row with these fields:

| Tracker column | Source field |
|---|---|
| Arc | `arc_name` |
| Signal | `signal` |
| TF | `tf` |
| Sub-protocol | `sub_protocol` |
| Best architecture | `best_architecture.name` |
| Worst-fold ratio | `best_architecture.worst_fold_ratio` |
| Verdict | `verdict` |
| Failed at step | `failed_at_step` |
| Closure doc | `closure_doc_link` |

### C. Per-feature contribution section
For each feature in `best_architecture.features_in_winning_config`:
1. Find or create row matching feature name.
2. Increment `Arcs used`.
3. Update rolling avg `Avg WFO ratio with` using `best_architecture.worst_fold_ratio`.

For each feature in any tracker row but NOT in `features_in_winning_config`:
1. Update rolling avg `Avg WFO ratio without` using this arc's `worst_fold_ratio`.

Re-derive `Verdict` per schema (LIFTS / HURTS / NEUTRAL / INSUFFICIENT).

### D. Per-architecture win rate section
For each architecture in `architectures_tested`:
1. Increment `Arcs tested`.
2. If `architecture_results.<arch>.won == true`: increment `Won (best in arc)`, append `worst_fold_ratio` to rolling avg `Avg ratio when won`.

### E. Per-archetype recurrence section
For each archetype in `archetypes_observed`:
1. Find or create row, increment `Arcs where appeared`.
2. Update rolling avgs `Avg in-cluster R` and `Avg in-cluster reach_1R` using `clusters.<cid>.mfe_p50_r` and `clusters.<cid>.reach_1r` for clusters whose archetype matches.

### F. Per-failure-mode count section
Find row matching `primary_failure_mode`:
1. Increment `Count`.
2. Update `Recent example arc` to `arc_name`.
3. Update `Recent example date` to `closed_timestamp`.

### G. Cross-arc cluster registry
For each cluster in `clusters`:
1. APPEND row: `<arc_name>.<cluster_id>`, archetype, n, mfe_p50_r, ww_pp, reach_1r, step3_composite, step4_e_auc, step4_d1_auc, sl_atr, outcome.
2. Never deduplicate — every cluster across every arc gets a row. Enables cross-arc pooling queries (EXP-05 style).

### H. Cost-decomposition registry
If `cost_decomposition` is non-null:
1. APPEND row: `arc_name`, admit_pool_fraction, admit_mean_r, reject_pool_fraction, reject_mean_r, early_exit_fraction, early_exit_mean_r.
2. Used to track admit-only-vs-deployment pattern across arcs.

### I. Cross-arc tag registry
For each tag in `cross_arc_tags`:
1. Find or create row, increment `Count`, append `arc_name` to `Arcs`.

### J. Update "Last auto-update" line
- Manual: `Last auto-update: manual: YYYY-MM-DD`
- Parser: `Last auto-update: parser: YYYY-MM-DD HH:MM:SS`

### K. Bad-payload handling
If `§1 tracker_payload` block is missing or malformed:
1. DO NOT update the tracker.
2. Flag closure doc in `TODO.md` as needing repair.
3. After repair, apply update.

If a tracker row is suspected wrong post-update:
1. Prepend `⚠️` to the row (do not delete).
2. Add footnote describing suspected issue.
3. Resolve in next protocol calibration review.

### L. v1.2 PASS-verdict validation (parser-enforced)

When `template_version == 1.2` AND `verdict` starts with `PASS-`, the parser additionally enforces (before applying any A-J mapping):

1. `best_architecture.config_artefact_path` MUST be non-null.
2. The file at `config_artefact_path` (interpreted relative to repo root) MUST exist.
3. The closure doc MUST contain a `## §4 deployment_spec` heading.
4. `best_architecture.deployment_spec_section_present` MUST be `true`.

Any of (1)-(4) failing → parser HALTs with exit code 1, no tracker write. Fix the closure doc and re-run.

Note: `config_artefact_path` is read by the parser for validation only — it is NOT written to any tracker column. The tracker schema is unchanged at v1.2.

For non-PASS verdicts in v1.2, (L) is skipped; §4 is optional. For v1.0 / v1.1 closures, (L) is not evaluated.

---

## Section 5 — Parser implementation

Parser is implemented at `scripts/update_tracker_from_closure.py` (see `scripts/tracker_parser/README.md` for usage, error modes, and the standard arc-close workflow).

Invocation is part of the standard arc-close workflow — run on the arc branch before opening the closure PR so the closure doc and the tracker delta land in one atomic commit. See README §"Workflow" for the canonical sequence.

Parser specification (preserved here for reference):
- Input: any `results/<arc_name>/ARC_CLOSURE.md` file.
- Extract: `§1 tracker_payload` YAML block (everything between the fenced ` ```yaml ` and the closing ` ``` `).
- Validate: schema match against this template's §1 spec; fail loudly on missing required fields.
- Output: append-only updates to `ARC_TRACKER.md` per Section 4 A-J above.
- Idempotency: parsing the same closure doc twice produces identical tracker (no double-append). Tracked via sha256 in `scripts/tracker_parser/parsed.log`.
- Determinism: same closure doc + same starting state → byte-identical output tracker.
- **Schema version detection (v1.1+):** parser inspects the closure doc's referenced template version. v1.0 closures use legacy field names (`worst_fold_roi_pct`, `worst_fold_dd_pct`); v1.1+ closures use the renamed fields (`worst_fold_roi_base_pct`, `worst_fold_dd_base_pct`) and may populate Amendment 3 risk-normalised fields. v1.2 closures additionally carry `config_artefact_path`, `deployment_spec_section_present` in `best_architecture` and trigger PASS-verdict validation per Section 4-L. Detection order: 1.2 → 1.1 → 1.0 → error. Parser accepts all three schemas — never rewrites historical closures.

---

## Schema versioning

| Template version | Date | Change |
|---|---|---|
| v1.0 | 2026-05-13 | Initial locked template. |
| v1.1 | 2026-05-22 | L_PROTOCOL Amendment 3. Risk-normalised fields added to `best_architecture`. Two fields renamed: `worst_fold_roi_pct` → `worst_fold_roi_base_pct`, `worst_fold_dd_pct` → `worst_fold_dd_base_pct`. `primary_failure_mode` enum extended. Pre-v1.1 closures retain v1.0 field names; parser handles both via version detection. |
| v1.2 | 2026-05-23 | Deployment-spec addition. Three new fields in `best_architecture`: `config_artefact_path`, `deployment_spec_section_present`, `template_version` (the last was conventional in v1.1; locked at v1.2). New §4 deployment_spec section: REQUIRED for PASS-* verdicts (DEPLOYABLE, VIABLE, *-PROVISIONAL, *-PENDING-STEP6), OPTIONAL otherwise. Parser HALTs on PASS verdict if config path missing / file absent / §4 heading missing (Section 4-L). Pre-v1.2 closures unaffected. |
| v1.2.1 | 2026-05-23 | `chained_dd_method` field added to `best_architecture` per PR-186 review item 1. Records the method used to reconstruct chained equity for `chained_max_dd_base_pct`: `"equity_stitching"` (v3.0.1 engine default) or `"full_window_sim"` (v3.0.2 follow-up). Phase 1: parser accepts as OPTIONAL. Phase 2 (post-Wave-2 first PASS arc): parser REQUIRES the field for any PASS verdict with `closed_timestamp > PR-186 merge date`. Older closures grandfathered by closed_timestamp check. v1.2 / v1.2.1 share the same `template_version: v1.2` declaration — the field's presence/absence is the v1.2.1 discriminator, not a separate version string. |
| v1.3 | 2026-05-24 | L_PROTOCOL Amendment 4 — Step 6 causal-audit framework. New `§1 tracker_payload.step_6` block (`ran`, `trigger`, `overall_passed`, `manifest_path`, `categories`, `critical_failures`, `warnings_count`, `verdict_impact`). REQUIRED for any v1.3 PASS verdict. Parser v1.3 detection precedence: explicit `template_version: v1.3` → v1.3-exclusive `step_6` field → fall-through to v1.2 detection. Phase 2 tightening bundled (PR-186-merge-date cutoff): for any PASS verdict with `closed_timestamp > 2026-05-23T06:20:59Z` the parser REQUIRES Amendment 3 fields in `best_architecture`; for v1.3 PASS verdicts the parser ADDITIONALLY requires the `step_6` block + `step_6.overall_passed: true`. Closures landed before the cutoff (v1.0/v1.1/v1.2/v1.2.1) are grandfathered. |
| v1.3.1 | 2026-05-23 | L_PROTOCOL Amendment 5 — AUC-gated A2/A6 architecture selection. New top-level optional field `architectures_skipped_by_amendment_5` (subset of `{A1..A6}`, may be `[]`). Captures architectures admissible under Amendment 1's archetype-driven rule but skipped under Amendment 5's four-gate AUC-driven rule. Phase 1: parser accepts presence or absence on all closures. Phase 2: parser REQUIRES the field on any PASS verdict whose `closed_timestamp > AMENDMENT_5_CUTOFF_ISO` (placeholder `2026-05-23T00:00:00Z`; backfilled with this PR's merge timestamp post-merge). v1.3 / v1.3.1 share the same `template_version: v1.3` declaration — the field's presence/absence is the v1.3.1 discriminator. Mirrors the v1.2 / v1.2.1 `chained_dd_method` rollout pattern. |
| v1.3.1+ | 2026-05-25 | L_PROTOCOL Amendment 5.1 — Gate 4 PASS-tier-constituent qualifier. Field `architectures_skipped_by_amendment_5` accepts new reason string `a5_gate_4_admission_blocked_by_no_pass_tier_constituent`. No version bump; documentation-only extension. |
| v1.3.1+ (Amendment 3.1) | 2026-05-25 | `r_max` reframed as deployment cap (not gate). New optional fields `r_safe_intrinsic_pct`, `r_hard_intrinsic_pct`, `r_safe_capped_at_rmax`, `r_hard_capped_at_rmax`. `r_safe_pct` / `r_hard_pct` now record post-cap deploy values. No template version bump; no enum change. |

Closures MUST reference the template version they were written against (e.g., `template_version: v1.2` near the top of `§1 tracker_payload` is the convention going forward — pre-v1.1 closures without this field are assumed v1.0; pre-v1.2 closures without `config_artefact_path` are assumed v1.1). v1.2.1 stays under the `v1.2` declaration; the `chained_dd_method` field is the only discriminator and is OPTIONAL during Phase 1. v1.3.1 stays under the `v1.3` declaration; the `architectures_skipped_by_amendment_5` field is the only discriminator and is OPTIONAL during Phase 1.

---

End of template.
