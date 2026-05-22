# ARC_CLOSURE.md Template (Locked v1.0)

> **Location:** `docs/templates/ARC_CLOSURE_TEMPLATE.md`
> **Status:** locked. Every arc closure MUST follow this template.
> **Referenced by:** `L_PROTOCOL.md` §6.
> **Parser status:** spec-only. Manual updates per Section 4 mapping until parser ships.
>
> Section headings are LITERAL — do not rephrase. Field names inside `§1 tracker_payload` are LITERAL — parser depends on exact spelling.
>
> **Design priority:** machine-parseability over human readability. §1 YAML is the source of truth for the tracker. §2 + §3 exist only to preserve cross-arc synthesis quality that prose enables and YAML doesn't.

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
  primary_failure_mode: <enum: pool_too_small | no_clusters_separable | no_capturable_cluster | entry_feature_auc_ceiling | step5_wf_roi_below_gate | step5_dd_above_gate | step5_sign_consistency_fail | step6_causal_audit_fail | selection_bias | holdout_fail_after_is_pass | admit_only_vs_deployment | other | N/A>

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: <int>
    window_start: <ISO date>
    window_end: <ISO date>
    kh24_co_fire_pct: <float>
    configs_evaluated_step5: <int>
    search_scope_flag: thin | normal | broad   # thin <50, normal 50-99, broad 100+

  # ────── Best architecture (null block if no winner) ──────
  best_architecture:
    name: A1 system_level_filter | A2 classifier_filter | A3 pipeline_de | A4 pipeline_d_exits | A5 portfolio_composition | A6 meta_labeling | null
    cluster: <cluster_id> | aggregate | null
    archetype: V-shape | Stepwise | Bimodal | Monotonic_up | Monotonic_down | Choppy | Unclassified | null
    config: <config name / descriptor> | null
    sl_atr: <float or null>
    exit_policy: <policy name or null>
    exposure_cap: <int or unlimited or null>
    worst_fold_ratio: <float or null>
    worst_fold_roi_pct: <float or null>
    worst_fold_dd_pct: <float or null>
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

  # ────── Archetypes observed ──────
  archetypes_observed: [<archetype_1>, <archetype_2>, ...]   # union across clusters

  # ────── Cross-arc observation tags (terse, one-line each) ──────
  cross_arc_tags: [<tag_1>, <tag_2>]
  # Examples: "v_shape_auc_ceiling", "pipeline_d1_admit_vs_deployment", "co_fire_with_arc_7"
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

---

## Section 5 — Parser implementation notes (deferred)

Parser not yet built. Recommendation: build after 2-3 Wave 1 closures land — gives real golden inputs to validate against.

Parser specification:
- Input: any `results/<arc_name>/ARC_CLOSURE.md` file.
- Extract: `§1 tracker_payload` YAML block (everything between the fenced ` ```yaml ` and the closing ` ``` `).
- Validate: schema match against this template's §1 spec; fail loudly on missing required fields.
- Output: append-only updates to `ARC_TRACKER.md` per Section 4 A-J above.
- Idempotency: parsing the same closure doc twice produces identical tracker (no double-append).
- Determinism: same closure doc → same tracker delta byte-for-byte.

Suggested implementation: single Python script `scripts/update_tracker_from_closure.py`. Invoked manually post-PR-merge, or via post-merge git hook (bundles with WORKFLOW §3 auto-cleanup hook trigger point).

---

End of template.
