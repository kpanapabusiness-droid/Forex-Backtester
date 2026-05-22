# ARC_TRACKER — Live State

> **Auto-updated.** Do not edit manually except per `docs/templates/ARC_CLOSURE_TEMPLATE.md` Section 4 mapping (until parser ships).
> Schema locked at v3.0. Replaces STATUS.md, CHANGELOG.md, ARC_QUEUE.md, and ACTIVE_ARCS.md.
> First populated on first arc open under L_PROTOCOL v3.0.

Last auto-update: manual: 2026-05-22 (Arc 11 close via canonical infra; arc_discovery_01 still active per prior update)

---

## Active arcs

| Arc | Signal | TF mode | Sub-protocol | Started | Branch | Step in progress |
|---|---|---|---|---|---|---|
| arc_discovery_01 | discovered_via_search (random search, n=10000) | locked H1 | signal_discovery_probe | 2026-05-22 | arc/discovery_01 | Step 1 — infrastructure landed; full 10k run pending compute slot |

---

## Closed arcs summary

| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Failed at step | Closure doc |
|---|---|---|---|---|---|---|---|---|
| l_arc_11 | swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4) | H4 | vanilla | A2 classifier_filter | -0.7687 | FAIL | 5 | results/l_arc_11/ARC_CLOSURE.md |

---

## Per-feature contribution (rolling)

| Feature | Arcs used | Avg WFO ratio with | Avg WFO ratio without | Verdict |
|---|---|---|---|---|
| w1_close_slope_sign | 1 | -0.769 | — | INSUFFICIENT |
| d1_atr_percentile_100 | 1 | -0.769 | — | INSUFFICIENT |
| prior_session_low_distance | 1 | -0.769 | — | INSUFFICIENT |
| day_of_week | 1 | -0.769 | — | INSUFFICIENT |
| session_london | 1 | -0.769 | — | INSUFFICIENT |
| d1_close_slope_magnitude | 1 | -0.769 | — | INSUFFICIENT |
| distance_to_round_number | 1 | -0.769 | — | INSUFFICIENT |
| atr_percentile_100 | 1 | -0.769 | — | INSUFFICIENT |
| usd_strength_index | 1 | -0.769 | — | INSUFFICIENT |
| spread_vs_trailing_100 | 1 | -0.769 | — | INSUFFICIENT |

Schema:
- `Avg WFO ratio with` — mean worst-fold ratio across arcs where this feature appeared in the winning config
- `Avg WFO ratio without` — mean worst-fold ratio across arcs where this feature did NOT appear in the winning config
- `Verdict` ∈ {LIFTS (avg delta > +0.3), NEUTRAL (|delta| ≤ 0.3), HURTS (avg delta < -0.3), INSUFFICIENT (N < 3 arcs)}

---

## Per-architecture win rate

| Architecture | Arcs tested | Won (best in arc) | Avg ratio when won |
|---|---|---|---|
| A1 system_level_filter | 1 | 0 | — |
| A2 classifier_filter | 1 | 0 | — |
| A3 pipeline_de | 0 | 0 | — |
| A4 pipeline_d_exits | 0 | 0 | — |
| A5 portfolio_composition | 0 | 0 | — |
| A6 meta_labeling | 1 | 0 | — |

Note: Arc 11 has no architecture row marked "Won" because no config met §3 PASS thresholds. A2 is "best of FAIL" (highest worst_fold_ratio among configs that admitted trades); A6 admitted zero trades.

---

## Per-archetype recurrence

| Archetype | Arcs where appeared | Avg in-cluster R | Avg in-cluster reach_1R |
|---|---|---|---|
| V-shape recovery | 0 | — | — |
| Stepwise climber | 0 | — | — |
| Bimodal | 1 | 7.77 | 1.000 |
| Monotonic up | 0 | — | — |
| Monotonic down | 1 | 0.09 | 0.001 |
| Choppy | 0 | — | — |
| Unclassified | 1 | 1.31 | 0.504 |
| Other / unclassified | 0 | — | — |

Note: "in-cluster R" tracked as `mfe_p50_r` per template §4.E. Unclassified row averages over Arc 11's c1 (mfe_p50 2.11) and c2 (mfe_p50 0.51).

---

## Per-failure-mode count

| Failure mode | Count | Recent example arc | Recent example date |
|---|---|---|---|
| pool_too_small | 0 | — | — |
| no_clusters_separable | 0 | — | — |
| no_capturable_cluster | 0 | — | — |
| entry_feature_auc_ceiling | 0 | — | — |
| step5_not_scalable | 0 | — | — |
| step5_chained_dd_above_gate | 0 | — | — |
| step5_daily_dd_breach | 0 | — | — |
| step5_wf_roi_below_gate_after_scaling | 0 | — | — |
| step5_ratio_below_gate_after_scaling | 0 | — | — |
| step5_negative_folds | 0 | — | — |
| step5_sign_consistency_fail | 0 | — | — |
| step5_trade_count_below_gate | 0 | — | — |
| step6_causal_audit_fail | 0 | — | — |
| selection_bias | 0 | — | — |
| holdout_fail_after_is_pass | 0 | — | — |
| admit_only_vs_deployment | 0 | — | — |
| step5_dd_above_gate | 1 | l_arc_11 | 2026-05-22 |  <!-- deprecated by Amendment 3; retained for historical closures -->
| step5_wf_roi_below_gate | 0 | — | — |  <!-- legacy (pre-Amendment-3); retained for historical closures -->
| other | 0 | — | — |

---

## Cross-arc cluster registry

> Every cluster from every arc gets a row. Append-only, never deduplicated.
> Enables cross-arc pooling queries (e.g., pooling all V-shape clusters with `step4_e_auc` near gate to test multi-arc clusterifier).

| Cluster ID | Archetype | n | mfe_p50_r | ww_pp | reach_1r | step3_composite | step4_e_auc | step4_d1_auc | sl_atr | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| l_arc_11.c0 | Bimodal | 2192 | 7.7664 | 0.0018 | 1.0000 | 1.9803 | 0.6543 | — | 1.5 | dies_step5 |
| l_arc_11.c1 | Unclassified | 6287 | 2.1087 | 0.0350 | 0.9648 | 0.9678 | 0.6316 | — | 1.5 | dies_step4 |
| l_arc_11.c2 | Unclassified | 5032 | 0.5111 | 0.9680 | 0.0425 | 0.1899 | — | — | 1.5 | dies_step3 |
| l_arc_11.c3 | Monotonic_down | 4022 | 0.0901 | 0.9993 | 0.0007 | 0.0221 | — | — | 1.5 | dies_step3 |

Schema:
- `Cluster ID` — `<arc_name>.<cluster_id>` (e.g., `arc_07.c1`)
- `outcome` — passed_step3 | dies_step3 | dies_step4 | wins_step5 | viable_step5 | dies_step5

---

## Cost-decomposition registry

> One row per arc whose best architecture is classifier-based (A2 / A3 / A4 / A6).
> Tracks admit-only-vs-deployment pattern across arcs.

| Arc | Admit fraction | Admit mean R | Reject fraction | Reject mean R | Early-exit fraction | Early-exit mean R |
|---|---|---|---|---|---|---|
| l_arc_11 | 0.125 | 5.4272 | 0.875 | -0.8789 | 0.0 | 0.0 |

---

## Cross-arc tag registry

> Tags from §1 `cross_arc_tags` in each closure doc. Used to surface recurring patterns mechanically.

| Tag | Count | Arcs |
|---|---|---|
| step4_auc_above_065_v3_first | 1 | l_arc_11 |
| shb_swing_detection_causal_clean_arc9_lesson_passed | 1 | l_arc_11 |
| canonical_orchestrator_step5_run_context_gap | 1 | l_arc_11 |
| step1_pool_uncapped_canonical_vs_capped_handrolled_2_5x_delta | 1 | l_arc_11 |

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
