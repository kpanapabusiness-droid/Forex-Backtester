# ARC_TRACKER — Live State

> **Auto-updated.** Do not edit manually except per `docs/templates/ARC_CLOSURE_TEMPLATE.md` Section 4 mapping (until parser ships).
> Schema locked at v3.0. Replaces STATUS.md, CHANGELOG.md, ARC_QUEUE.md, and ACTIVE_ARCS.md.
> First populated on first arc open under L_PROTOCOL v3.0.

Last auto-update: parser: 2026-05-22 00:00:00

---

## Active arcs

| Arc | Signal | TF mode | Sub-protocol | Started | Branch | Step in progress |
|---|---|---|---|---|---|---|
| arc_discovery_01 | discovered_via_search (random search, n=10000) | locked H1 | signal_discovery_probe | 2026-05-22 | arc/discovery_01 | Step 1 — infrastructure landed; full 10k run pending compute slot |

---

## Closed arcs summary

> `Re-evaluated verdict` column added 2026-05-22 per L_PROTOCOL Amendment 3 retrospective re-evaluation dispatch. Holds the Amendment-3-gate result for arcs closed before the amendment landed; `Verdict` retains the original-protocol result for audit trail. `—` indicates not yet re-evaluated.

| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Re-evaluated verdict | Failed at step | Closure doc |
|---|---|---|---|---|---|---|---|---|---|
| l_arc_10 | D1 swing-low rejection long (DLR, v0.1) — bullish rejection of confirmed ascending D1 swing-low, 4H entry | H4 | vanilla | A1 system_level_filter | 5.4185 | PASS-VIABLE | PASS-DEPLOYABLE | N/A | results/l_arc_10/ARC_CLOSURE.md |
| l_arc_8 | pullback_resume_hhhl_long_v0.1 (HH/HL uptrend, pullback >=0.5xATR, bullish-close break of prior bar) | 4H | vanilla | A6 meta_labeling | 1.749 | FAIL | FAIL | 5 | results/l_arc_8/ARC_CLOSURE.md |
| l_arc_11 | swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4) | H4 | vanilla | A2 classifier_filter | -0.7687 | FAIL | FAIL | 5 | results/l_arc_11/ARC_CLOSURE.md |

---

## Per-feature contribution (rolling)

| Feature | Arcs used | Avg WFO ratio with | Avg WFO ratio without | Verdict |
|---|---|---|---|---|
| w1_close_slope_sign | 1 | -0.769 | 3.584 | INSUFFICIENT |
| d1_atr_percentile_100 | 1 | -0.769 | 3.584 | INSUFFICIENT |
| prior_session_low_distance | 2 | 0.490 | 5.418 | INSUFFICIENT |
| day_of_week | 1 | -0.769 | 3.584 | INSUFFICIENT |
| session_london | 1 | -0.769 | 3.584 | INSUFFICIENT |
| d1_close_slope_magnitude | 1 | -0.769 | 3.584 | INSUFFICIENT |
| distance_to_round_number | 2 | 0.490 | 5.418 | INSUFFICIENT |
| atr_percentile_100 | 2 | 0.490 | 5.418 | INSUFFICIENT |
| usd_strength_index | 2 | 0.490 | 5.418 | INSUFFICIENT |
| spread_vs_trailing_100 | 2 | 0.490 | 5.418 | INSUFFICIENT |
| swing_low_distance_14 | 1 | 1.749 | 5.418 | INSUFFICIENT |
| atr_vs_trailing_100 | 1 | 1.749 | 5.418 | INSUFFICIENT |
| kijun_26_distance | 1 | 1.749 | 5.418 | INSUFFICIENT |
| dollar_bloc_state | 1 | 1.749 | 5.418 | INSUFFICIENT |
| spread_percentile_100 | 1 | 1.749 | 5.418 | INSUFFICIENT |

Schema:
- `Avg WFO ratio with` — mean worst-fold ratio across arcs where this feature appeared in the winning config
- `Avg WFO ratio without` — mean worst-fold ratio across arcs where this feature did NOT appear in the winning config
- `Verdict` ∈ {LIFTS (avg delta > +0.3), NEUTRAL (|delta| ≤ 0.3), HURTS (avg delta < -0.3), INSUFFICIENT (N < 3 arcs)}

---

## Per-architecture win rate

| Architecture | Arcs tested | Won (best in arc) | Avg ratio when won |
|---|---|---|---|
| A1 system_level_filter | 3 | 1 | 5.418 |
| A2 classifier_filter | 1 | 0 | — |
| A3 pipeline_de | 1 | 0 | — |
| A4 pipeline_d_exits | 0 | 0 | — |
| A5 portfolio_composition | 0 | 0 | — |
| A6 meta_labeling | 3 | 1 | 1.749 |

Note: Arc 11 has no architecture row marked "Won" because no config met §3 PASS thresholds. A2 is "best of FAIL" (highest worst_fold_ratio among configs that admitted trades); A6 admitted zero trades.

---

## Per-archetype recurrence

| Archetype | Arcs where appeared | Avg in-cluster R | Avg in-cluster reach_1R |
|---|---|---|---|
| V-shape recovery | 2 | 6.40 | 0.986 |
| Stepwise climber | 0 | — | — |
| Bimodal | 1 | 7.77 | 1.000 |
| Monotonic up | 0 | — | — |
| Monotonic down | 3 | 1.31 | 0.303 |
| Choppy | 1 | 1.25 | 0.598 |
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
| entry_feature_auc_ceiling | 1 | l_arc_8 | 2026-05-22 |
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
| l_arc_8.c0 | Monotonic_down | 1657 | 0.0690 | 0.0284 | 0.0000 | 0.3930 | — | — | 4.0 | dies_step3 |
| l_arc_8.c1 | Choppy | 3560 | 1.2460 | 0.4003 | 0.5978 | 0.5620 | — | — | 1.5 | dies_step3 |
| l_arc_8.c2 | V-shape | 1540 | 7.5270 | 0.0000 | 1.0000 | 1.0000 | 0.5300 | — | 1.5 | dies_step5 |
| l_arc_10.c0 | monotonic_down | 1771 | 1.9950 | 0.5280 | 0.7100 | 0.6347 | — | — | 1.5 | dies_step3 |
| l_arc_10.c1 | v_shape_recovery | 1528 | 5.2670 | 0.5060 | 0.9730 | 0.8628 | 0.5199 | — | 4.0 | wins_step5 |
| l_arc_10.c2 | monotonic_down | 2 | 3.0710 | 0.5000 | 0.5000 | 0.6750 | — | — | 1.5 | dies_step3 |

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
| v_shape_auc_ceiling | 1 | l_arc_8 |
| step4_classifier_at_chance | 1 | l_arc_8 |
| oracle_real_gap_massive | 1 | l_arc_8 |
| multi_tf_features_absent_at_step1 | 1 | l_arc_8 |
| search_wfo_fail_holdout_pass_pattern | 1 | l_arc_8 |
| v_shape_archetype_cross_arc_persistence | 1 | l_arc_10 |
| exit_policy_dominates_classifier | 1 | l_arc_10 |
| no_classifier_needed_for_v_shape_pass_viable | 1 | l_arc_10 |
| v3_engine_first_pass_viable | 1 | l_arc_10 |
| step4_auc_chance_full_pool_still_passes_via_exit_policy | 1 | l_arc_10 |
| oracle_locked_to_sl_only_unfair_upper_bound | 1 | l_arc_10 |

---

## Update mechanism

Parser at `scripts/update_tracker_from_closure.py` applies the Section 4 A-K mapping from each closure doc's `§1 tracker_payload` YAML block. Invoke per `scripts/tracker_parser/README.md` — invocation is part of the standard arc-close workflow (run on the arc branch before opening the closure PR; tracker delta lands in the same atomic commit as the closure doc).

- Parser writes the "Last auto-update" line as `parser: YYYY-MM-DD HH:MM:SS` (UTC, derived from the closure's `closed_timestamp`).
- Manual updates (rare; for one-off cleanups or retroactive backfills) write `manual: YYYY-MM-DD` and an optional parenthetical describing the change.

Tracker is APPEND-ONLY at the row level. Bad rows can be flagged with a `⚠️` prefix but cannot be deleted (history matters). Schema changes require explicit chat-side redesign event documented in the closure doc that introduced them.
