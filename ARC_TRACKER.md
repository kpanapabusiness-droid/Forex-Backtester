# ARC_TRACKER — Live State

> **Auto-updated.** Do not edit manually except per `docs/templates/ARC_CLOSURE_TEMPLATE.md` Section 4 mapping (until parser ships).
> Schema locked at v3.0. Replaces STATUS.md, CHANGELOG.md, ARC_QUEUE.md, and ACTIVE_ARCS.md.
> First populated on first arc open under L_PROTOCOL v3.0.

Last auto-update: manual: 2026-05-22

---

## Active arcs

| Arc | Signal | TF mode | Sub-protocol | Started | Branch | Step in progress |
|---|---|---|---|---|---|---|

(empty — no arcs running)

---

## Closed arcs summary

| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Failed at step | Closure doc |
|---|---|---|---|---|---|---|---|---|
| l_arc_5 | mtf_alignment.2_down_mixed.kijun.h_120 (LCHAR Entry 5, h=120 override) | H1 | vanilla | A1 system_level_filter | -0.9042 | FAIL | 5 | [results/l_arc_5/ARC_CLOSURE.md](results/l_arc_5/ARC_CLOSURE.md) |

---

## Per-feature contribution (rolling)

| Feature | Arcs used | Avg WFO ratio with | Avg WFO ratio without | Verdict |
|---|---|---|---|---|

(empty — no data yet)

Schema:
- `Avg WFO ratio with` — mean worst-fold ratio across arcs where this feature appeared in the winning config
- `Avg WFO ratio without` — mean worst-fold ratio across arcs where this feature did NOT appear in the winning config
- `Verdict` ∈ {LIFTS (avg delta > +0.3), NEUTRAL (|delta| ≤ 0.3), HURTS (avg delta < -0.3), INSUFFICIENT (N < 3 arcs)}

---

## Per-architecture win rate

| Architecture | Arcs tested | Won (best in arc) | Avg ratio when won |
|---|---|---|---|
| A1 system_level_filter | 1 | 1 |  -0.9042 |
| A2 classifier_filter | 0 | 0 | — |
| A3 pipeline_de | 0 | 0 | — |
| A4 pipeline_d_exits | 0 | 0 | — |
| A5 portfolio_composition | 0 | 0 | — |
| A6 meta_labeling | 0 | 0 | — |

---

## Per-archetype recurrence

| Archetype | Arcs where appeared | Avg in-cluster R | Avg in-cluster reach_1R |
|---|---|---|---|
| V-shape recovery | 0 | — | — |
| Stepwise climber | 0 | — | — |
| Bimodal | 0 | — | — |
| Monotonic up | 0 | — | — |
| Monotonic down | 0 | — | — |
| Choppy | 0 | — | — |
| Other / unclassified | 3 |  1.2193  |  0.5033 |

---

## Per-failure-mode count

| Failure mode | Count | Recent example arc | Recent example date |
|---|---|---|---|
| pool_too_small | 0 | — | — |
| no_clusters_separable | 0 | — | — |
| no_capturable_cluster | 0 | — | — |
| entry_feature_auc_ceiling | 0 | — | — |
| step5_wf_roi_below_gate | 0 | — | — |
| step5_dd_above_gate | 1 |  l_arc_5  |  2026-05-22 |
| step5_sign_consistency_fail | 0 | — | — |
| step6_causal_audit_fail | 0 | — | — |
| selection_bias | 0 | — | — |
| holdout_fail_after_is_pass | 0 | — | — |
| admit_only_vs_deployment | 0 | — | — |
| other | 0 | — | — |

---

## Cross-arc cluster registry

> Every cluster from every arc gets a row. Append-only, never deduplicated.
> Enables cross-arc pooling queries (e.g., pooling all V-shape clusters with `step4_e_auc` near gate to test multi-arc clusterifier).

| Cluster ID | Archetype | n | mfe_p50_r | ww_pp | reach_1r | step3_composite | step4_e_auc | step4_d1_auc | sl_atr | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| l_arc_5.c0 | bimodal | 19091 | 5.5169 | 0.0063 | 1.0000 | 1.5795 | 0.7404224388022451 | None | 1.50 | passed_step3 |
| l_arc_5.c1 | unclassified | 38037 | 1.9341 | 0.0388 | 0.9608 | 0.9359 | 0.5402625464907423 | None | 1.50 | dies_step4 |
| l_arc_5.c2 | unclassified | 37749 | 0.5045 | 0.9704 | 0.0457 | 0.1844 | None | None | 1.50 | dies_step3 |
| l_arc_5.c3 | monotonic_down | 31924 | 0.0575 | 0.9992 | 0.0024 | 0.0157 | None | None | 1.50 | dies_step3 |

Schema:
- `Cluster ID` — `<arc_name>.<cluster_id>` (e.g., `arc_07.c1`)
- `outcome` — passed_step3 | dies_step3 | dies_step4 | wins_step5 | viable_step5 | dies_step5

---

## Cost-decomposition registry

> One row per arc whose best architecture is classifier-based (A2 / A3 / A4 / A6).
> Tracks admit-only-vs-deployment pattern across arcs.

| Arc | Admit fraction | Admit mean R | Reject fraction | Reject mean R | Early-exit fraction | Early-exit mean R |
|---|---|---|---|---|---|---|

(empty — no classifier-based winners yet)

---

## Cross-arc tag registry

> Tags from §1 `cross_arc_tags` in each closure doc. Used to surface recurring patterns mechanically.

| Tag | Count | Arcs |
|---|---|---|
| v3_first_complete_arc | 1 | l_arc_5 |
| thin_search_scope_24_configs | 1 | l_arc_5 |
| h120_mtf_alignment_v3_replay | 1 | l_arc_5 |

---

## Update mechanism

Until parser ships:
- Manual updates per `docs/templates/ARC_CLOSURE_TEMPLATE.md` Section 4 mapping (steps A through K).
- Each closure doc's `§1 tracker_payload` YAML block is the source of truth — copy fields verbatim.
- "Last auto-update" line bumped to `manual: YYYY-MM-DD` after each update.

Once parser ships (`scripts/update_tracker_from_closure.py`):
- Parser ingests `§1 tracker_payload` YAML.
- Applies same Section 4 mapping mechanically.
- "Last auto-update" line bumped to `parser: YYYY-MM-DD HH:MM:SS`.

Tracker is APPEND-ONLY at the row level. Bad rows can be flagged with a `⚠️` prefix but cannot be deleted (history matters). Schema changes require explicit chat-side redesign event documented in the closure doc that introduced them.
