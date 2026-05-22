# ARC_TRACKER — Live State

> **Auto-updated.** Do not edit manually. Schema locked at v3.0.
> Replaces STATUS.md, CHANGELOG.md, ARC_QUEUE.md, and ACTIVE_ARCS.md.
> First populated on first arc open under L_PROTOCOL v3.0.

Last auto-update: manual: 2026-05-22 (Arc 11 close; first arc under v3.0; established schema rows for sections G/H/I per `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.0 §4 mapping)

---

## Active arcs

| Arc | Signal | TF mode | Sub-protocol | Started | Branch | Step in progress |
|---|---|---|---|---|---|---|

(empty — no arcs running)

---

## Closed arcs summary

| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Failed at step | Closure doc |
|---|---|---|---|---|---|---|---|---|
| l_arc_11 | swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4) | H4 | vanilla | A2 classifier_filter | 3.1808 | FAIL | 5 | results/l_arc_11/ARC_CLOSURE.md |

---

## Per-feature contribution (rolling)

| Feature | Arcs used | Avg WFO ratio with | Avg WFO ratio without | Verdict |
|---|---|---|---|---|
| w1_close_slope_sign | 1 | 3.181 | — | INSUFFICIENT |
| kijun_26_distance | 1 | 3.181 | — | INSUFFICIENT |
| day_of_week | 1 | 3.181 | — | INSUFFICIENT |
| break_magnitude_atr | 1 | 3.181 | — | INSUFFICIENT |
| eur_strength_index | 1 | 3.181 | — | INSUFFICIENT |
| swing_low_distance_14 | 1 | 3.181 | — | INSUFFICIENT |
| atr_vs_trailing_100 | 1 | 3.181 | — | INSUFFICIENT |
| atr_percentile_100 | 1 | 3.181 | — | INSUFFICIENT |
| distance_to_round_number | 1 | 3.181 | — | INSUFFICIENT |
| usd_strength_index | 1 | 3.181 | — | INSUFFICIENT |

Schema:
- `Avg WFO ratio with` — mean worst-fold ratio across arcs where this feature appeared in the winning config
- `Avg WFO ratio without` — mean worst-fold ratio across arcs where this feature did NOT appear in the winning config
- `Verdict` ∈ {LIFTS (avg delta > +0.3), NEUTRAL (|delta| ≤ 0.3), HURTS (avg delta < -0.3), INSUFFICIENT (N < 3 arcs)}

---

## Per-architecture win rate

| Architecture | Arcs tested | Won (best in arc) | Avg ratio when won |
|---|---|---|---|
| A1 system_level_filter | 1 | 0 | — |
| A2 classifier_filter | 1 | 1 | 3.181 |
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
| Choppy | 1 | 4.66 | 0.985 |
| Unclassified | 1 | 0.47 | 0.130 |
| Other / unclassified | 0 | — | — |

Note: "in-cluster R" tracked as `mfe_p50_r` per template §4.E (the closure exports this field rather than mean_R). Choppy row reflects Arc 11 cluster 0 (the §3-candidate cohort the dispatch flagged); see `cross_arc_tags: choppy_label_misnamed_capturable_cohort`.

---

## Per-failure-mode count

| Failure mode | Count | Recent example arc | Recent example date |
|---|---|---|---|
| Pool too small | 0 | — | — |
| No clusters separable | 0 | — | — |
| No cluster passes capturability | 0 | — | — |
| Entry-feature AUC ceiling | 0 | — | — |
| Step 5 worst-fold ROI below gate | 0 | — | — |
| Step 5 DD above gate | 1 | l_arc_11 | 2026-05-22 |
| Step 5 sign-consistency fail | 0 | — | — |
| Step 6 causal audit fail | 0 | — | — |
| Selection bias not defensible | 0 | — | — |
| Holdout fail after IS pass | 0 | — | — |
| Other | 0 | — | — |

---

## Cross-arc cluster registry (Template §4.G — append-only, no dedup)

| Arc.cluster | archetype | n | mfe_p50_r | ww_pp | reach_1r | step3_composite | step4_e_auc | step4_d1_auc | sl_atr | outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| l_arc_11.c0 | Choppy | 2266 | 4.6619 | 0.0150 | 0.9846 | 0.9908 | 0.6873 | — | 2.0 | dies_step5 |
| l_arc_11.c1 | Unclassified | 4752 | 0.4715 | 0.1084 | 0.1303 | 0.2723 | — | — | 3.0 | dies_step3 |

---

## Cost-decomposition registry (Template §4.H)

| Arc | admit_pool_fraction | admit_mean_r | reject_pool_fraction | reject_mean_r | early_exit_fraction | early_exit_mean_r |
|---|---:|---:|---:|---:|---:|---:|
| l_arc_11 | 0.3229 | 2.0041 | 0.6771 | -0.9950 | 0.0 | 0.0 |

---

## Cross-arc tag registry (Template §4.I)

| Tag | Count | Arcs |
|---|---:|---|
| dd_gated_at_chosen_risk_size | 1 | l_arc_11 |
| strong_holdout_blocked_by_wfo_dd | 1 | l_arc_11 |
| step4_auc_above_065_v3_first | 1 | l_arc_11 |
| choppy_label_misnamed_capturable_cohort | 1 | l_arc_11 |
| shb_swing_detection_causal_clean_arc9_lesson_passed | 1 | l_arc_11 |

---

## Update mechanism

Overseer parses §1 `tracker_payload` of every `ARC_CLOSURE.md` per `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.0 and updates this tracker on arc close:
- Active arcs: row removed (§4.A)
- Closed arcs summary: row appended (§4.B)
- Per-feature contribution: rolling averages updated for features in the winning config (§4.C)
- Per-architecture win rate: incremented if architecture won; ratio averaged (§4.D)
- Per-archetype recurrence: cluster archetype labels parsed and counted (§4.E)
- Per-failure-mode count: failure mode parsed from closure doc; counted (§4.F)
- Cross-arc cluster registry: per-cluster row appended (§4.G) — append-only, no dedup
- Cost-decomposition registry: row appended if classifier-based architecture won (§4.H)
- Cross-arc tag registry: per-tag count incremented; arcs list appended (§4.I)

Tracker is APPEND-ONLY at the row level. Bad rows can be flagged with a `⚠️` prefix but cannot be deleted (history matters). Schema changes require explicit chat-side redesign event documented in the closure doc that introduced them.

Parser implementation TBD per template §5. Until parser ships, updates are manual per template §4 mapping.
