# L_ARC_PROTOCOL v1.2 AMENDMENT

**Supersedes:** Specific clauses in `L_ARC_PROTOCOL.md` v1.0 (locked 2026-05-13), `L_ARC_PROTOCOL_v1.1_AMENDMENT.md` (locked mid-arc-1), and `L_ARC_OPERATIONAL_SPEC.md` v1.0.
**Trigger:** Arc 2 step 4 disposition surfaced a structural pathology in v1.1 §7.1 / §7.2 — hard-gate framing of metrics that should be planner-disposition inputs.
**Reference:** Arc 2 step 4 outputs (`results/l_arc_2/step4/candidate_component_table.csv`, `tautology_check.csv`, `t_selection.csv` rolled up via `STEP4_PLANNER_REVIEW.md` packet); planning-chat dialogue surfaces the framing error (Chat B planner correction; arc 1-3 chat sign-off on the corrected framing).
**Author:** planning chat
**Lock date:** 2026-05-14
**Effective from:** sign-off date; applies to arc 2 step 4 re-disposition (no recompute, only disposition logic) and onward to arcs 3–5.

---

## Document History

| Version | Date | Scope |
|---------|------|-------|
| v1.0 | 2026-05-13 | Initial protocol lock. 6-step extractability framework. |
| v1.1 | 2026-05-13 (signed off mid-arc-1) | 12 amendments. Family-level calibration, HDBSCAN, partial AUC, expected R-volume ranking, delayed-entry candidate type, hash-seed determinism, cluster-target by mean R. |
| **v1.2** | **2026-05-14** | **3 amendments. Step 4 = compute + disposition table (no auto-disqualification); target/mirror label switches to forward-geometry; single consolidated planner-review doc mandatory.** |

---

## How this doc works

Three amendments to L_ARC_PROTOCOL v1.0 + v1.1 + L_ARC_OPERATIONAL_SPEC v1.0. Each amendment specifies:
- **What changes** — concrete textual edit or framing shift
- **Where it lives** — file and section
- **Why** — justification grounded in arc 2 step 4 findings
- **Cost** — implementation effort
- **Risk** — what could go wrong

Once signed off, this amendment supersedes the relevant v1.0 / v1.1 clauses. The v1.0 + v1.1 docs remain on disk as historical record; the working protocol version becomes v1.2.

**This is a methodology correction, not a finding revision.** Step 3 verdicts stand. Step 4 measurements stand. Only the disposition logic at step 4 changes.

---

## Amendment 1 (v1.2 A1) — Step 4 = compute and dispose, not auto-disqualify

### What changes

Step 4 still computes every per-candidate metric specified in v1.1 §7.1 (filter component table) and §7.2 (exit held-out fold check). The component values feed into a **candidate disposition table** consumed by the planner. **No metric outside of lookahead test and determinism receipts auto-disqualifies a candidate at step 4.**

Step 4's role is:
1. **Instantiate** specific candidates from step 3's surfaced mechanism classes (filter / exit / delayed-entry / exit-only baseline).
2. **Compute** per-candidate metrics — retention, concentration (ΔP_extractable), MFE/race/ratio preservation under the candidate's mechanism, per-fold sign consistency, held-out F6/F7 mean R + capture ratio for exit/delayed-entry candidates, predictor AUC stability, BH tier, cross-arc carry status.
3. **Present** the consolidated disposition table to the planner.
4. **Planner disposes** — selects which candidates advance to step 5, which to step 6, which to drop, with explicit reasoning per candidate.

Hard gates at step 4 reduce to two:
- **Lookahead invariant test passes** (op spec §10.1 for filters, §10.2 for exits; hard fail on any byte difference under perturbation)
- **Two-run determinism receipt PASS** (op spec §11.6)

Everything else is reportorial. No "viable iff all components clear floors" boolean. No auto-disqualification on retention. No mean R floor. No capture ratio floor. No per-fold sign consistency floor.

### Where it lives

- `L_ARC_PROTOCOL.md` §8 (Step 4 — Filter / Exit Derivation): clause "Filter candidate evaluation — component-ranked, not single-composite" reframed; clause "Zero-candidate disposition" removed (planner judges, not the protocol).
- `L_ARC_OPERATIONAL_SPEC.md` §7.1 (Filter candidate component table): "Floor for viability" column replaced with "Reportorial direction" column (e.g., ΔP_extractable: "above 0 is concentration toward target; magnitude is informational"). Removes the "Candidate viable only if ALL components clear their floors" framing.
- `L_ARC_OPERATIONAL_SPEC.md` §7.2 (Exit candidate evaluation): "Held-out fold floors for advancing to step 6" clause replaced with "Held-out fold metrics for planner disposition." Mean R / capture ratio / sign consistency on F6,F7 are reported with directional interpretation, not as auto-disqualifiers.
- `L_ARC_OPERATIONAL_SPEC.md` §4 (Per-Step Deliverables Checklist — Step 4 deliverables): "If zero candidates clear viability" line removed; "Planner decision" line strengthened with explicit one-sentence-reasoning-per-candidate requirement.

### Why

The arc 2 step 4 evidence is conclusive on this:

- `delayed_entry_t_gb` at t=1 produced F6 mean R +0.910, F7 mean R +0.767 (12× verbatim capture ratio on held-out, both folds positive). v1.1 §7.1's retention floor (50/fold auto-disqualify; 100/fold soft floor) killed the candidate at 17/fold retention. The strongest held-out per-trade signal in the entire step 4 run was auto-disposed by a mechanical floor.
- Four filter candidates passed v1.1 §7.1 component-table viability primarily because `mean_r_under_verbatim_exits > 0` happened to clear; two cross-arc carries (EUR, GBP) failed on the same floor. The mean R floor is a function of the (locked but arbitrary) verbatim h=120 exit policy — not a property of the candidate mechanism itself. Filters that may pair productively with a different exit at step 6 are dropped on a metric that pre-supposes the wrong exit.
- v1.1 §7.1's `Per_fold_AUC_stability_CV < 0.30` floor was applied to filters that aren't cluster discriminators (Phase C established cluster membership is not signal-time separable for arc 2). The CV metric is near-undefined for non-discriminator filters and was patched at run-time with a "near_chance_flag" workaround that the spec didn't anticipate. Reportorial framing avoids this entirely.

The structural error is treating step-4 metric values as if they answer the question "is this candidate a winner." They don't. Step 6 WFO answers that. Step 4 metrics answer "does this candidate do what its spec claims, at what robustness, on what trade volume." Those are inputs to planner judgement, not a gate.

Step 3's eligibility check (cluster structure exists; mechanism class opens) plus step 5's re-characterisation (post-filter step 2+3 redo) plus step 6's WFO truth-test already provide three layers of evaluation. v1.1 §7.1 added a fourth, mechanical, layer that systematically discards real edge.

### Cost

Trivial. The computation already happens; the change is to remove "viable" booleans and replace floor-comparison language with directional interpretation in the disposition table output.

### Risk

Without auto-disqualification, the planner may advance candidates to step 5 that are obviously dead. Mitigation:
- The lookahead and determinism gates remain — any candidate failing those is dropped.
- The consolidated disposition table makes per-candidate evidence side-by-side visible; planner reads it and judges. The discipline lives in transparent reasoning per candidate, not in mechanical floors.
- Step 5 re-characterisation is the next filter — a candidate whose post-filter population fails step 3 verdict drops at step 5, not at step 4.
- Step 6 WFO is the actual deployability gate.

Volume concerns (a candidate with too few trades to populate folds 6 and 7 meaningfully) surface as low retention values in the disposition table. The planner makes the volume judgement contextually, with cross-fold sign consistency and held-out economics in view — not against a one-size-fits-all 50/fold or 100/fold floor.

---

## Amendment 2 (v1.2 A2) — Cluster target/mirror label switches to forward-geometry

### What changes

The rule for labelling clusters as "target" vs "mirror" at step 3 switches from mean-net-R-based to forward-geometry-based.

**Previous (v1.1 Amendment 12, superseded):**
> The non-extractable / target cluster is the cluster with the lowest mean net R at the relevant K. Ties broken by highest median fwd_mae_h24.

**Replacement (v1.2 A2):**
> The **target cluster** (the high-drawdown archetype, candidate "filter out") is the cluster with the **highest median fwd_mae_h24_atr** at the relevant K. Ties broken by **lowest median fwd_mfe_h24_atr** (symmetric forward-path metric, not mean R).
>
> The **mirror cluster** (the high-upside archetype, candidate "filter to" or hold-policy-favourable) is the cluster with the **highest median fwd_mfe_h24_atr** at the relevant K. Ties broken by **lowest median fwd_mae_h24_atr** (symmetric forward-path metric, not mean R).
>
> The rule is fixed across all K values and all clustering algorithms. Each (K, algorithm) combination produces one target and one mirror identification. In K ≥ 3, the remaining clusters carry no archetype label (planner reads their characteristics directly from cluster_summary).

Clustering features themselves are unchanged — already pure path-geometry per op spec §6.1 + v1.1 Amendment 4. This amendment changes only the post-clustering label-selection rule.

### Where it lives

- `L_ARC_PROTOCOL.md` §7 (verdict logic): the cluster-target identification clause inherited from v1.1 Amendment 12 is replaced with the v1.2 A2 specification above. v1.1 Amendment 12 is marked SUPERSEDED in the v1.1 amendment doc.
- `L_ARC_OPERATIONAL_SPEC.md` §6.5 (cluster effect sizes): no change required (already uses fwd_mfe_h24 and fwd_mae_h24 as primary effect-size axes). The `cluster_target_selection.csv` output schema gains two columns: `target_rule = highest_median_fwd_mae_h24_atr` and `mirror_rule = highest_median_fwd_mfe_h24_atr`, with the tie-break columns logged.
- Implementation code (`run_step3.py` Phase F): one-line change in the target-selection function. Add equivalent mirror-selection function. Update tests.

### Why

Three reasons:

1. **Policy-independence.** Mean net R bundles entry, SL, time exit, and spread policy into the cluster label. The label is meant to identify which cluster's *forward path* is most drawdown-heavy or most upside-heavy — these are properties of price geometry, not of exit policy. Using mean R confuses "this cluster realised the worst outcome under verbatim" with "this cluster has the worst forward path properties." For some signals these align (e.g., arc 2: both rules pick cluster 1 as target — cluster 1 has both highest fwd_mae AND lowest mean R). For others they could diverge. The protocol should select on the property we actually want to discriminate.

2. **Symmetry.** Mean R is asymmetric — it has a single sign and magnitude. Forward-geometry labelling uses symmetric metrics: target = max drawdown axis; mirror = max upside axis. Symmetric framing avoids edge cases where "lowest mean R" doesn't clearly identify a target (e.g., two clusters with similar mean R but very different forward paths).

3. **Removes mean R from the entire pre-step-6 protocol.** Aligns with the broader v1.2 direction that mean R is exit-dependent and therefore informational until step 6 WFO. v1.0 op spec §8 already said this ("mean R pre-exit is dominated by the wrong exit policy"); v1.1 Amendment 12 partially contradicted it by using mean R for cluster labelling. v1.2 A2 resolves the contradiction in the direction op spec §8 already established.

### Cost

Trivial. One-line code change in target-selection; small extension to add mirror-selection (currently implicit in step 3 narrative). Output CSV schema gains two columns.

### Risk

Forward-geometry-derived target/mirror could differ from mean-R-derived in some arcs. Verified for arc 2: same target (cluster 1) and same mirror (cluster 0) under either rule, no disruption. Arc 1 never reached step 4 so no relabelling required there; cross-arc registry entries for arc 1 carry the original mean-R-derived labels and are not retroactively edited (per protocol §16 no retroactive re-evaluation).

For arcs 3–5 the rule is v1.2 A2 from step 3 onward; cross-arc registry entries use the v1.2 labels. The §3 cluster-archetype registry entry for arc 1 stands; arcs 3–5 entries note "v1.2 A2 forward-geometry labels" in the metadata.

---

## Amendment 3 (v1.2 A3) — Single consolidated `STEP4_PLANNER_REVIEW.md` mandatory deliverable

### What changes

Step 4's primary deliverable to the planner becomes **one consolidated markdown document**: `results/l_arc_N/step4/STEP4_PLANNER_REVIEW.md`. Per-candidate folders still exist for `config.yaml`, `filter_or_exit_spec.md`, and per-candidate evaluation CSVs, but the planner is not required to drill into them — the consolidated doc contains all information needed for disposition.

### Structure of `STEP4_PLANNER_REVIEW.md`

Five sections, in this order:

**§1. Header**
- Arc identifier, step, protocol version (v1.2)
- Candidate count and breakdown by mechanism class
- Target / mirror cluster identification from step 3 (cluster_id, n, fraction of pool, median fwd_mae_h24, median fwd_mfe_h24)
- Reference paths: step 3 phase doc, candidate spec set, CC dispatch
- Methodology gates summary: lookahead test PASS/FAIL per candidate, determinism receipt PASS/FAIL

**§2. Candidate disposition table** (the central artefact)

One row per candidate. Markdown table. Columns:
- `slug`
- `mechanism_class` ∈ {filter, exit, delayed_entry, exit_only}
- `target_cluster_intent` — for filters: cluster_id being filtered TO or OUT; for exits/delayed-entry: predicted cluster being acted on
- `retention_per_fold_min`, `retention_per_fold_max`, `retention_per_fold_mean` — reportorial volume profile
- `delta_p_extractable` — concentration shift toward mirror cluster
- `delta_p_non_extractable` — concentration shift away from target
- `concentration_lift_ratio` — `P(mirror | retains) / P(mirror | pool)` — multiplicative form
- `winner_retention_rate` — `P(retains | mirror)` — concentration's complement
- `mfe_geometry_preservation` — `median(fwd_mfe_h24 | post, mirror) / median(fwd_mfe_h24 | pre, mirror)`
- `race_preservation`, `ratio_preservation` — symmetric forward-geometry preservation checks
- `concentration_lift_sign_consistency_per_fold` — count of folds where concentration_lift_ratio > 1.0 (N/7)
- `concentration_lift_stability_cv` — std/mean of per-fold concentration lift
- `predictor_auc_stability_cv` — for predictor-based candidates only; reportorial
- `mean_r_f6_f7` — for exit / delayed-entry / exit-only candidates; reportorial direction (positive = informational signal, NOT a viability bar)
- `capture_ratio_f6_f7` and `capture_ratio_f6_f7_verbatim_reference` — reportorial
- `per_fold_sign_consistency_f6_f7` — both_positive / one_negative / both_negative — reportorial
- `bh_tier_arc_N` — predictor BH tier this arc
- `cross_arc_carry_status` — registry lookup: count of prior arcs at Tier 1 or 2; reportorial multiplier suggestion (1.5 / 1.2 / 1.0) is planner judgement
- `lookahead_test_passed` — bool (must be True)
- `determinism_receipt_passed` — bool (must be True)
- `notes` — free text per candidate (tautology flags, near-chance flags, edge cases)

Sort by `mechanism_class`, then by `concentration_lift_ratio × winner_retention_rate` descending. The sort is reportorial; no ranking implies disposition.

**§3. Per-mechanism-family narrative**

Interpretive read on each mechanism family, written by the chat-B planner. Sub-sections:
- Cluster-conditional exit candidates — what they show as a group; specific candidate calls
- Early-exit at t candidates — same
- Delayed-entry candidates — same
- Signal-time filter candidates — same
- Registered cross-arc carry candidates — same
- Exit-only / unfiltered baselines — same

Each sub-section discusses cross-candidate patterns, surprising findings, methodology flags (tautology, near-chance discrimination, etc.), and notes any candidate that warrants special planner attention regardless of its position in the §2 table.

**§4. Planner disposition**

Explicit table:

| Candidate slug | Decision | One-sentence reasoning |
|----------------|----------|------------------------|
| `<slug_1>` | advance_to_step_5 | <reasoning grounded in §2 table values + §3 narrative> |
| `<slug_2>` | advance_to_step_6_direct | <reasoning — typically exit-only or cluster-conditional exit candidates that skip step 5> |
| `<slug_3>` | dropped | <reasoning — typically lookahead/determinism failure OR explicit planner judgement> |
| ... | ... | ... |

The decision values are: `advance_to_step_5`, `advance_to_step_6_direct`, `dropped`. Reasoning must reference specific §2 table values; "candidate underperformed" is insufficient.

**§5. Handover to next chat**

Per protocol §13:
- What is complete (step 4 closed)
- Path to this doc and to per-candidate folders
- Docs the next chat reads first (always: L_ARC_PROTOCOL.md, L_ARC_PROTOCOL_v1.2_AMENDMENT.md, L_ARC_OPERATIONAL_SPEC.md, step 3 phase doc, this step 4 doc)
- Open questions / judgement calls deferred to the next chat
- Per-advancing-candidate: which chat takes it (Chat C step 5 for filters / delayed-entry; Chat D step 6 for exits / exit-only baselines)

### Where it lives

- `L_ARC_OPERATIONAL_SPEC.md` §4 (Per-Step Deliverables Checklist — Step 4): the deliverables list updated to make `STEP4_PLANNER_REVIEW.md` the primary deliverable; `candidate_component_table.csv` becomes a supporting artefact embedded in the planner-review doc rather than a stand-alone primary.
- `L_ARC_OPERATIONAL_SPEC.md` §2 (Folder Structure): unchanged at the folder level. `STEP4_PLANNER_REVIEW.md` replaces `PHASE_L_ARC_N_STEP4.md` as the step 4 phase doc (or sits alongside it as the planner-review companion — implementation choice).
- CC dispatch prompts for arcs 3–5 mandate this doc as output of every step 4 run.

### Why

Arc 2 step 4 surfaced the friction directly: with 11 candidates × ~6 output files per candidate = 60+ artefacts to read, the planner ended up requesting consolidated rolled-up tables mid-step. The protocol should not require the planner to drill into 11 folders to write a disposition. One file. Five sections. Planner reads once.

### Cost

Trivial. The CC dispatch writes the consolidated doc from existing per-candidate evaluation outputs. The per-candidate folders still get populated; the consolidated doc is an aggregation step.

### Risk

The consolidated doc may grow unwieldy with arc N's candidate count rising. Mitigation:
- §2 table is the central artefact; everything else is narrative around it.
- Per-mechanism-family narrative (§3) can be terse where candidates align; longer where they diverge.
- Per-candidate evaluation metrics live in per-candidate folders for deep-dive reads if planner wants them.

If a future arc surfaces 30+ candidates, the planner-review doc may need a hierarchical structure (top-N highlighted in §3; full table in §2; appendix with per-candidate detail). Defer to that arc's planner.

---

## Implementation summary — what code / spec / doc changes

### Protocol doc changes

| Doc | Section | Change |
|---|---|---|
| `L_ARC_PROTOCOL.md` | §8 Step 4 | Reframe candidate-evaluation language: "viable only if all components clear floors" → "disposition table populated; planner advances based on §2 table + §3 narrative." Lookahead + determinism remain hard gates. Zero-candidate disposition removed (planner judgement, not mechanical close). |
| `L_ARC_PROTOCOL.md` | §7 verdict logic | v1.1 Amendment 12 target-selection rule replaced with v1.2 A2 forward-geometry rule. |
| `L_ARC_PROTOCOL_v1.1_AMENDMENT.md` | Amendment 12 | Marked **SUPERSEDED by v1.2 A2**. Original text retained as historical record. |
| `L_ARC_OPERATIONAL_SPEC.md` | §4 Step 4 deliverables | `STEP4_PLANNER_REVIEW.md` becomes the primary deliverable. Component-table-viability-check items removed. Held-out-fold-floor items reframed as "metrics reported, planner dispositions." |
| `L_ARC_OPERATIONAL_SPEC.md` | §7.1 Filter component table | "Floor for viability" column replaced with "Reportorial direction." "Candidate viable only if all components clear floors" removed. |
| `L_ARC_OPERATIONAL_SPEC.md` | §7.2 Exit candidate evaluation | "Held-out fold floors for advancing to step 6" replaced with "Held-out fold metrics for planner disposition." Sign-flip auto-disqualification removed. |
| `L_ARC_OPERATIONAL_SPEC.md` | §6.5 Cluster effect sizes | Output schema gains `target_rule` and `mirror_rule` columns documenting v1.2 A2 label selection. |

### Implementation code changes

| File | Change | Cost |
|---|---|---|
| `scripts/l_arc_N/step3/run_step3.py` (or shared library) | Target-cluster selection function updated to v1.2 A2 forward-geometry rule. Mirror-cluster selection function added. | 1-line edit + new ~5-line function |
| `scripts/l_arc_N/step4/run_step4.py` (or equivalent) | Remove `viable_component_table` and `viable_held_out_check` boolean computation. Compute all metrics; write disposition table. Add `STEP4_PLANNER_REVIEW.md` aggregation step. | ~30 line removal; ~80 line aggregation script |
| Regression tests | Add: v1.2 A2 target/mirror selection on synthetic clusters. Add: planner-review-doc generation against fixture. | ~50 lines test code |

---

## Methodology correction vs finding revision (explicit statement)

This amendment changes the disposition logic at step 4. It does NOT revise:
- Step 1 trade-set (signal validation, plumbing, lookahead test)
- Step 2 angle catalogue outputs (descriptive measurements on the raw trade population)
- Step 3 cluster discovery (path-geometry-based clustering; algorithms unchanged)
- Step 3 verdict (PROCEED / AMBIGUOUS / CLOSE remains a real protocol moment driven by dual-gate logic)
- Step 4 per-candidate metric computations (the metrics are still computed; only their gating role changes)
- Step 5 re-characterisation framework (unchanged — still acts as step 2+3 redo on filtered population)
- Step 6 WFO dual-tier disposition (unchanged — the only place hard ROI/DD gates live)

What changes is the interpretive layer at step 4 — from "metrics are auto-disqualifiers" to "metrics are planner-disposition inputs in a consolidated review doc."

Existing arc 2 step 4 measurements stand. The re-disposition is a documentation pass over the existing `candidate_component_table.csv` + `tautology_check.csv` + per-candidate `evaluation_metrics.csv` + `t_selection.csv` + `cluster_0_time_exit_curve.csv` outputs. No CC recompute needed.

---

## Arc 2 step 4 re-disposition handling

Under v1.2 framing, the existing arc 2 step 4 outputs are re-read with the following expected dispositions (planner judgement at re-disposition; reasoning in `PHASE_L_ARC_2_STEP4.md` under v1.2):

- **`delayed_entry_t_gb` at t=1** — held-out F6 +0.910, F7 +0.767, both folds positive, 12× verbatim capture ratio. Retention 17/fold is low and the planner notes the §10.1 step 6 trade-count concern explicitly, but advances to step 5 (Chat C) for re-characterisation. v1.1 §7.1 auto-disq overturned.
- **`filter_concurrent_signals_above_p75`, `filter_atr_at_signal_above_p50`, `filter_basket_jpy_above_p50`** — pass under v1.2 framing for the same reasons they passed under v1.1; advance to step 5.
- **`filter_basket_usd_above_p50`** — `one_negative` on F6,F7 sign consistency flagged in planner reasoning; advance to step 5 with explicit "drop-risk at step 5" note.
- **`filter_basket_eur_above_p50`, `filter_basket_gbp_above_p50`** — mean R under verbatim is negative; under v1.2 this is reportorial, not a floor. Concentration metrics: EUR ΔP_extr +0.002 (essentially zero shift), GBP ΔP_extr −0.002 (filters AWAY from mirror). Planner judgement: GBP dropped (concentration goes in the wrong direction — not arbitrary); EUR advanced to step 5 (positive concentration shift, registered cross-arc carry obligation, marginal but not architecturally broken).
- **`filter_jpy_pairs`** — retention 67/fold falls below v1.1's 100/fold soft floor (and just above the 50/fold auto-disq). Under v1.2 this is reportorial. Strongest mean R signal among filters (+0.154 verbatim, +0.214 F6,F7); ΔP_extr +0.032 (largest among filter candidates). Planner judgement: advance to step 5 with explicit volume-concern flag. Step 6 §10.1 scaled-floor check applies at WFO disposition.
- **`exit_cluster_cond_gb`, `exit_cluster_cond_gb_h240`** — held-out capture ratio < verbatim. Under v1.1 this would have been "viable_held_out_check = False." Under v1.2 reportorial. Planner judgement: dropped — the mechanism doesn't add value over verbatim, and step 3's tautology at t=10 (40% of cluster-1 already SL'd) gives a structural reason this is unlikely to recover. v1.2 doesn't rescue weak mechanisms; it allows planner to drop them with reasoning instead of by mechanical floor.
- **`exit_only_unfiltered_h240`** — marginal absolute edge, plausible PASS-VIABLE at step 6, almost certainly fails PASS-DEPLOYABLE on op cost haircut. Advance to step 6 (Chat D direct, no step 5).

Net advancing from arc 2 step 4 under v1.2: **5 filters + 1 delayed-entry to step 5; 1 exit-only to step 6 direct; 3 candidates dropped (2 cluster-cond exits + GBP carry).**

Compare to v1.1 disposition: 4 filters + 0 delayed-entry to step 5; 1 exit-only to step 6; 6 candidates dropped.

The v1.2 net change is: GBP→drop (under v1.1 it failed mean R floor; under v1.2 it fails concentration-direction judgement — same outcome via cleaner reasoning); EUR→advance (rescued from mean R floor); delayed_entry_t_gb→advance (rescued from retention floor); JPY-pairs→advance (rescued from retention floor).

Three rescues, all with credible per-candidate evidence in the existing step 4 outputs.

---

## Out-of-scope deferrals

The following items were considered for v1.2 and explicitly deferred:

- **Composite mechanism_quality_score with cross-arc multiplier (1.5/1.2/1.0).** Planner judgement under v1.2 reads the disposition table directly; composite scores risk re-introducing mechanical disposition under a different label. Sort the table by `concentration_lift_ratio × winner_retention_rate` for visual ranking; weight cross-arc carry status as a narrative consideration in §3 / §4.
- **Discarded-population metric as gate.** Reported in §2 disposition table (as `delta_p_non_extractable`), not gated.
- **Trade count floor at step 4.** Removed entirely. Step 6 §10.1 retains scaled trade-count floor for PASS-VIABLE / PASS-DEPLOYABLE disposition. Volume matters there, not here.
- **Decision threshold calibration for predictor-based candidates.** Currently locked at 0.5 in the candidate specs. Allowing planner-calibrated thresholds (e.g., to match class prior) would rescue `delayed_entry_t_gb` at higher t-values too. Deferred — orthogonal to v1.2 framing and worth its own amendment when arc 3 or 4 surfaces a similar pattern.
- **Retroactive arc 1 cluster relabelling under v1.2 A2.** Arc 1 never reached step 4 and its step 3 cluster registry entries stand as historical record under v1.1. No retroactive edit. Arcs 3–5 use v1.2 A2 labels from step 3.

---

## Sign-off

- Amendment count: 3 (A1, A2, A3)
- Authoring rationale: arc 2 step 4 disposition surfaced hard-gate pathology in v1.1 §7.1 + §7.2; arc 1-3 chat sign-off confirmed planner model correct on substance, refined framing to "compute but don't auto-disqualify"
- Once signed off, the working protocol version becomes **v1.2**
- v1.0 + v1.1 docs remain on-disk as historical record
- v1.1 Amendment 12 marked SUPERSEDED by v1.2 A2

**Signed off by:** Keanu (project owner)
**Date:** 2026-05-14
**Simulator git commit at sign-off:** _TBD at commit time — chat-B drafts, planner commits with hash_

---

*End of L_ARC_PROTOCOL v1.2 AMENDMENT. Companion docs: `L_ARC_PROTOCOL.md` v1.0, `L_ARC_PROTOCOL_v1.1_AMENDMENT.md`, `L_ARC_OPERATIONAL_SPEC.md` v1.0.*
