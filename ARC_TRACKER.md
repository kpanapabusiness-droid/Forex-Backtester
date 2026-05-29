# ARC_TRACKER — Live State

> **Auto-updated.** Do not edit manually except per `docs/templates/ARC_CLOSURE_TEMPLATE.md` Section 4 mapping (until parser ships).
> Schema locked at v3.0. Replaces STATUS.md, CHANGELOG.md, ARC_QUEUE.md, and ACTIVE_ARCS.md.
> First populated on first arc open under L_PROTOCOL v3.0.

Last auto-update: manual: 2026-05-29 (arc_discovery_01_partial archived; supersedes parser: 2026-05-25 21:12:41)

---

## Active arcs

| Arc | Signal | TF mode | Sub-protocol | Started | Branch | Step in progress |
|---|---|---|---|---|---|---|

(empty — no arcs running)

---

## Available sub-protocols

> Sub-protocols are reusable engine paths invokable by arcs via `sub_protocol: <name>` in `ARC_OPEN.md`. Distinct from arcs themselves — they're shipped infrastructure, not research units. A row appears here when a sub-protocol's build lands on main.

| Sub-protocol | Status | Landed | Entry point | Spec |
|---|---|---|---|---|
| signal_discovery_probe | LANDED | (pre-tracker) | scripts/arc_discovery_01/ | docs/sub_protocols/signal_discovery_probe.md |
| heavy_ml_probe | LANDED | 2026-05-25 | scripts/heavy_ml_probe/run_probe.py | docs/sub_protocols/heavy_ml_probe.md |

---

## Closed arcs summary

> `Re-evaluated verdict` column added 2026-05-22 per L_PROTOCOL Amendment 3 retrospective re-evaluation dispatch. Holds the Amendment-3-gate result for arcs closed before the amendment landed; `Verdict` retains the original-protocol result for audit trail. `—` indicates not yet re-evaluated.

| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Re-evaluated verdict | Failed at step | Closure doc |
|---|---|---|---|---|---|---|---|---|---|
| l_arc_10 | D1 swing-low rejection long (DLR, v0.1) — bullish rejection of confirmed ascending D1 swing-low, 4H entry | H4 | vanilla | A1 system_level_filter | 5.4185 | PASS-VIABLE | PASS-DEPLOYABLE | N/A | results/l_arc_10/ARC_CLOSURE.md |
| l_arc_8 | pullback_resume_hhhl_long_v0.1 (HH/HL uptrend, pullback >=0.5xATR, bullish-close break of prior bar) | 4H | vanilla | A6 meta_labeling | 1.749 | FAIL | FAIL | 5 | results/l_arc_8/ARC_CLOSURE.md |
| l_arc_11 | swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4) | H4 | vanilla | A2 classifier_filter | -0.7687 | FAIL | FAIL | 5 | results/l_arc_11/ARC_CLOSURE.md |
| l_arc_7 | liquidity sweep + reclaim long (4H structural reversal) | 4H | vanilla | — | -0.8172120095638017 | FAIL |  | 5 | results/l_arc_7/ARC_CLOSURE.md |
| l_arc_7_v3_0_2 | liquidity sweep + reclaim long (4H) — A2/A6 follow-up (override) | 4H | vanilla | — | 4.363315680255243 | FAIL |  | 5 | results/l_arc_7_v3.0.2/ARC_CLOSURE.md |
| l_arc_10_v3.0.2 | D1 swing-low rejection long (DLR, v0.1) — bullish rejection of confirmed ascending D1 swing-low, 4H entry | H4 | vanilla | A1 system_level_filter | 6.4273 | PASS-DEPLOYABLE |  | N/A | results/l_arc_10_v3.0.2/ARC_CLOSURE.md |
| l_arc_8_v3.0.2 | pullback_resume_hhhl_long_v0.1 (PR-HHHL, 4H, causal HH/HL detection) | H4 | vanilla | A1 system_level_filter | -0.886 | FAIL |  | 5 | results/l_arc_8_v3.0.2/ARC_CLOSURE.md |
| l_arc_11_v3.0.2 | swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4) | H4 | vanilla | A1 system_level_filter | -0.74 | FAIL |  | 5 | results/l_arc_11_v3.0.2/ARC_CLOSURE.md |
| arc_discovery_01_partial | discovered_via_search (random search, n=10000; 511 rules completed before kill) | H1 | signal_discovery_probe | N/A | N/A | ARCHIVED_PARTIAL | N/A | step_1_compute_blowup | archive/probes/arc_discovery_01_partial/README.md |

---

## Per-feature contribution (rolling)

| Feature | Arcs used | Avg WFO ratio with | Avg WFO ratio without | Verdict |
|---|---|---|---|---|
| w1_close_slope_sign | 3 | 0.926 | 2.394 | HURTS |
| d1_atr_percentile_100 | 1 | -0.769 | 2.216 | INSUFFICIENT |
| prior_session_low_distance | 3 | 0.054 | 2.917 | HURTS |
| day_of_week | 1 | -0.769 | 2.216 | INSUFFICIENT |
| session_london | 1 | -0.769 | 2.216 | INSUFFICIENT |
| d1_close_slope_magnitude | 2 | 1.797 | 1.859 | INSUFFICIENT |
| distance_to_round_number | 3 | 0.054 | 2.917 | HURTS |
| atr_percentile_100 | 4 | 1.132 | 2.555 | HURTS |
| usd_strength_index | 2 | 0.490 | 2.294 | INSUFFICIENT |
| spread_vs_trailing_100 | 2 | 0.490 | 2.294 | INSUFFICIENT |
| swing_low_distance_14 | 2 | 0.466 | 2.917 | INSUFFICIENT |
| atr_vs_trailing_100 | 3 | 1.765 | 2.555 | HURTS |
| kijun_26_distance | 2 | 3.056 | 1.881 | INSUFFICIENT |
| dollar_bloc_state | 1 | 1.749 | 2.294 | INSUFFICIENT |
| spread_percentile_100 | 1 | 1.749 | 2.294 | INSUFFICIENT |
| atr_14 | 2 | 1.773 | 1.600 | INSUFFICIENT |
| prior_session_high_distance | 1 | -0.817 | 2.291 | INSUFFICIENT |
| session_tokyo | 2 | 1.773 | 1.600 | INSUFFICIENT |
| range_close_ratio | 1 | -0.817 | 2.291 | INSUFFICIENT |
| d1_close_slope_sign | 1 | 4.363 | 1.600 | INSUFFICIENT |
| session_ny | 1 | 4.363 | 1.600 | INSUFFICIENT |
| session_dead | 1 | 4.363 | 1.600 | INSUFFICIENT |

Schema:
- `Avg WFO ratio with` — mean worst-fold ratio across arcs where this feature appeared in the winning config
- `Avg WFO ratio without` — mean worst-fold ratio across arcs where this feature did NOT appear in the winning config
- `Verdict` ∈ {LIFTS (avg delta > +0.3), NEUTRAL (|delta| ≤ 0.3), HURTS (avg delta < -0.3), INSUFFICIENT (N < 3 arcs)}

---

## Per-architecture win rate

| Architecture | Arcs tested | Won (best in arc) | Avg ratio when won |
|---|---|---|---|
| A1 system_level_filter | 8 | 3 | 3.653 |
| A2 classifier_filter | 2 | 0 | — |
| A3 pipeline_de | 2 | 0 | — |
| A4 pipeline_d_exits | 3 | 0 | — |
| A5 portfolio_composition | 1 | 0 | — |
| A6 meta_labeling | 4 | 1 | 1.749 |

Note: Arc 11 has no architecture row marked "Won" because no config met §3 PASS thresholds. A2 is "best of FAIL" (highest worst_fold_ratio among configs that admitted trades); A6 admitted zero trades.

---

## Per-archetype recurrence

| Archetype | Arcs where appeared | Avg in-cluster R | Avg in-cluster reach_1R |
|---|---|---|---|
| V-shape recovery | 3 | 6.03 | 0.981 |
| Stepwise climber | 0 | — | — |
| Bimodal | 5 | 7.77 | 1.000 |
| Monotonic up | 0 | — | — |
| Monotonic down | 7 | 1.50 | 0.326 |
| Choppy | 2 | 1.27 | 0.601 |
| Unclassified | 4 | 1.43 | 0.569 |
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
| step5_not_scalable | 3 | l_arc_11_v3.0.2 | 2026-05-25 |
| step5_chained_dd_above_gate | 1 | l_arc_8_v3.0.2 | 2026-05-26 |
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
| l_arc_7.c0 | Bimodal | 636 | 7.8702 | 0.0047 | 1.0000 | 1.9982 | 0.6192 | — | 1.5 | dies_step5 |
| l_arc_7.c1 | Unclassified | 1839 | 2.1150 | 0.0397 | 0.9603 | 0.9678 | 0.6642 | — | 1.5 | dies_step5 |
| l_arc_7.c2 | Unclassified | 1588 | 0.5060 | 0.9691 | 0.0460 | 0.1865 | — | — | 1.5 | dies_step3 |
| l_arc_7.c3 | Monotonic_down | 1112 | 0.0967 | 0.9964 | 0.0054 | 0.0269 | — | — | 1.5 | dies_step3 |
| l_arc_7_v3_0_2.c0 | Bimodal | 636 | 7.8702 | 0.0047 | 1.0000 | 1.9982 | 0.6192 | — | 1.5 | dies_step4 |
| l_arc_7_v3_0_2.c1 | Unclassified | 1839 | 2.1150 | 0.0397 | 0.9603 | 0.9678 | 0.6642 | — | 1.5 | dies_step5 |
| l_arc_10_v3.0.2.c0 | v_shape_recovery | 1493 | 5.3090 | 0.4950 | 0.9700 | 0.8640 | 0.5131 | — | 4.0 | wins_step5 |
| l_arc_10_v3.0.2.c1 | monotonic_down | 1658 | 1.9640 | 0.5300 | 0.7140 | 0.6320 | — | — | 1.5 | dies_step3 |
| l_arc_10_v3.0.2.c2 | monotonic_down | 1 | 6.0320 | 0.0000 | 1.0000 | 1.0000 | — | — | 1.5 | dies_step3 |
| l_arc_8_v3.0.2.c0 | Choppy | 3418 | 1.2910 | 0.3950 | 0.6042 | 0.5697 | — | — | 1.5 | dies_step3 |
| l_arc_8_v3.0.2.c1 | Monotonic_down | 1628 | 0.0680 | 0.0295 | 0.0000 | 0.3928 | — | — | 4.0 | dies_step3 |
| l_arc_8_v3.0.2.c2 | Bimodal | 1566 | 7.5500 | 0.0000 | 1.0000 | 1.0000 | 0.4822 | — | 1.5 | dies_step5 |
| l_arc_11_v3.0.2.c0 | Bimodal | 2167 | 7.7855 | 0.0028 | 1.0000 | 1.9835 | 0.5056 | — | 1.5 | dies_step4 |
| l_arc_11_v3.0.2.c1 | Unclassified | 6123 | 2.1082 | 0.0350 | 0.9647 | 0.9675 | 0.5056 | — | 1.5 | dies_step4 |
| l_arc_11_v3.0.2.c2 | Unclassified | 5012 | 0.5187 | 0.9661 | 0.0431 | 0.1903 | — | — | 1.5 | dies_step3 |
| l_arc_11_v3.0.2.c3 | Monotonic_down | 3977 | 0.0945 | 0.9992 | 0.0005 | 0.0228 | — | — | 1.5 | dies_step3 |

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
| shb_swing_detection_causal_clean_arc9_lesson_passed | 2 | l_arc_11, l_arc_11_v3.0.2 |
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
| v3_0_1_first_arc_post_pr_185_186_188_189 | 1 | l_arc_7 |
| ets_boundary_per_pr_189 | 1 | l_arc_7 |
| mid_price_features_per_pr_189 | 1 | l_arc_7 |
| amendment_3_risk_normalised_gates_evaluated | 2 | l_arc_7, l_arc_7_v3_0_2 |
| bimodal_observed | 2 | l_arc_7, l_arc_7_v3_0_2 |
| monotonic_down_observed | 1 | l_arc_7 |
| unclassified_only | 2 | l_arc_7, l_arc_7_v3_0_2 |
| v3_0_2_a2_a6_followup_retry | 1 | l_arc_7_v3_0_2 |
| architecture_map_override_applied | 1 | l_arc_7_v3_0_2 |
| v301_artefacts_reused | 1 | l_arc_7_v3_0_2 |
| eet_canonical_first_pass_deployable_under_amendment_5 | 1 | l_arc_10_v3.0.2 |
| v_shape_archetype_persists_under_eet | 1 | l_arc_10_v3.0.2 |
| exit_policy_dominates_classifier_under_eet | 1 | l_arc_10_v3.0.2 |
| dd_lower_under_eet_than_utc_at_r_base | 1 | l_arc_10_v3.0.2 |
| w1_producer_canonical_alignment_pr_208_first_consumer_after_fix | 1 | l_arc_10_v3.0.2 |
| amendment_3_evaluation_deferred_pending_engine_risk_decoupling_admit_exit_fix | 1 | l_arc_10_v3.0.2 |
| step_6_deferred_pending_amendment_3_addendum_dependency_chain | 1 | l_arc_10_v3.0.2 |
| a6_skipped_under_amendment_5_gate_2_auc_below_065 | 1 | l_arc_10_v3.0.2 |
| step3_outlier_cluster_n_1_required_min_n_filter_in_step_4_and_step_5 | 1 | l_arc_10_v3.0.2 |
| direct_driver_pipeline_methodology | 1 | l_arc_8_v3.0.2 |
| sl_multiplier_signal_specific | 1 | l_arc_8_v3.0.2 |
| archetype_shift_v_shape_to_bimodal_under_5ers_eet | 1 | l_arc_8_v3.0.2 |
| chained_dd_blowout_pr_hhhl_bimodal_tight_sl | 1 | l_arc_8_v3.0.2 |
| w1_producer_canonical_alignment_consumer | 1 | l_arc_8_v3.0.2 |
| canonical_orchestrator_step5_run_context_gap_resolved_v3_0_2 | 1 | l_arc_11_v3.0.2 |
| amendment_5_1_a5_gate_4_blocked_dispatch_time | 1 | l_arc_11_v3.0.2 |
| eet_aggregation_signal_state_a_no_pool_shift | 1 | l_arc_11_v3.0.2 |
| mid_feature_eet_htf_alignment_drift_attribution_in_section_10 | 1 | l_arc_11_v3.0.2 |
| holdout_window_extended_2026_04_30_to_2026_05_25_4w | 1 | l_arc_11_v3.0.2 |
| worst_fold_ratio_negative_at_step5_canonical_engine | 1 | l_arc_11_v3.0.2 |
| worst_fold_dd_above_10pct_canonical_engine | 1 | l_arc_11_v3.0.2 |

---

## Step 6 audit registry

> Auto-dispatched Step 6 runs (per Amendment 4) append one row per arc. Manual CLI
> invocations DO NOT append (read-only diagnostic per chat Q6).
> Schema: Arc | Verdict | Step 6 ran | Step 6 passed | Critical fails | Warnings | Manifest

| Arc | Verdict | Step 6 ran | Step 6 passed | Critical fails | Warnings | Manifest |
|---|---|---|---|---|---|---|

---

## Update mechanism

Parser at `scripts/update_tracker_from_closure.py` applies the Section 4 A-M mapping from each closure doc's `§1 tracker_payload` YAML block. Invoke per `scripts/tracker_parser/README.md` — invocation is part of the standard arc-close workflow (run on the arc branch before opening the closure PR; tracker delta lands in the same atomic commit as the closure doc).

- Parser writes the "Last auto-update" line as `parser: YYYY-MM-DD HH:MM:SS` (UTC, derived from the closure's `closed_timestamp`).
- Manual updates (rare; for one-off cleanups or retroactive backfills) write `manual: YYYY-MM-DD` and an optional parenthetical describing the change.

Tracker is APPEND-ONLY at the row level. Bad rows can be flagged with a `⚠️` prefix but cannot be deleted (history matters). Schema changes require explicit chat-side redesign event documented in the closure doc that introduced them.
