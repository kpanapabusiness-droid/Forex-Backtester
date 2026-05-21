# ARC_TRACKER — Live State

> **Auto-updated.** Do not edit manually. Schema locked at v3.0.
> Replaces STATUS.md, CHANGELOG.md, ARC_QUEUE.md, and ACTIVE_ARCS.md.
> First populated on first arc open under L_PROTOCOL v3.0.

Last auto-update: never (empty initial state)

---

## Active arcs

| Arc | Signal | TF mode | Sub-protocol | Started | Branch | Step in progress |
|---|---|---|---|---|---|---|

(empty — no arcs running)

---

## Closed arcs summary

| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Failed at step | Closure doc |
|---|---|---|---|---|---|---|---|---|

(empty — no arcs closed under v3.0)

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
| A1 system_level_filter | 0 | 0 | — |
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
| Other / unclassified | 0 | — | — |

---

## Per-failure-mode count

| Failure mode | Count | Recent example arc | Recent example date |
|---|---|---|---|
| Pool too small | 0 | — | — |
| No clusters separable | 0 | — | — |
| No cluster passes capturability | 0 | — | — |
| Entry-feature AUC ceiling | 0 | — | — |
| Step 5 worst-fold ROI below gate | 0 | — | — |
| Step 5 DD above gate | 0 | — | — |
| Step 5 sign-consistency fail | 0 | — | — |
| Step 6 causal audit fail | 0 | — | — |
| Selection bias not defensible | 0 | — | — |
| Holdout fail after IS pass | 0 | — | — |
| Other | 0 | — | — |

---

## Update mechanism

Overseer parses sections 1-3 of every `ARC_CLOSURE.md` and updates this tracker on arc close:
- Active arcs: row removed
- Closed arcs summary: row appended
- Per-feature contribution: rolling averages updated for features in the winning config
- Per-architecture win rate: incremented if architecture won; ratio averaged
- Per-archetype recurrence: cluster archetype labels parsed and counted
- Per-failure-mode count: failure mode parsed from closure doc; counted

Tracker is APPEND-ONLY at the row level. Bad rows can be flagged with a `⚠️` prefix but cannot be deleted (history matters). Schema changes require explicit chat-side redesign event documented in the closure doc that introduced them.
