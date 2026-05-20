# L_ARC_PROTOCOL v2.x Amendment Proposal (REVISED 2026-05-19)

> Status: DRAFT.
> Source: Arc 9 closure (STEP_4_KILL_REAFFIRMED) + Arc 9 producer-level lookahead incident.
> Author: analyst, drafted 2026-05-18, revised 2026-05-19 after causal patch invalidated original §8 evidence.
> Companion artefacts: `results/l_arc_9/ARC_9_CLOSURE.md`, `INCIDENT_2026_05_19_ARC_9_PRODUCER_LEAK.md`.
> Pre-PR requirements: (1) Arc 7 calibration recovery test (separate dispatch, supports §3), (2) analyst sign-off, (3) anchor-preservation verification (KH-24 K=4 archetype 3 must continue to pass under amended protocol).

This proposal collects three amendments surfaced by Arc 9's held-open research cycle, revised after the causal patch invalidated the prior §8 evidence base.

| Amendment | Status | Evidence base | Priority |
|---|---|---|---|
| **NEW: Producer-level causal audit dimension** | **STRONG** | Arc 9 producer-leak incident (single arc, direct empirical) | Highest |
| §3 threshold-grid replacement | WEAKENED but live | Arc 7 calibration recovery (Arc 9 contribution invalidated) | High |
| Step 5 fold-1 warmup convention | LIVE | Arc 9 data-window artifact (independent of leak) | Medium |
| ~~§8 D1 feature-budget expansion~~ | **WITHDRAWN** | Arc 9 evidence collapsed; no remaining empirical base | — |
| Worst-day DD as standard Step 5 output | LIVE | Arc 9 scaled-risk methodology (invalidated for Arc 9; valid as standard) | Medium |

---

## Amendment 1 (NEW) — Producer-level causal audit dimension

### Problem

The Arc 9 producer-level lookahead incident exposed that the existing audit framework checks join-level causality (`merge_asof` direction, days_lag distribution) but does not check whether **values within each joined row are themselves causally constructed**. The two failure modes are distinct:

- **Join-level leak:** the join uses the wrong row (e.g. same-day D1 instead of yesterday's). The classifier sees data from a future row. Detected by `merge_asof` direction audits and days_lag distribution checks.
- **Producer-level leak:** the join uses the correct row, but the value in that row was computed using future data relative to the row's date. The classifier sees data from the "correct" past row that was constructed using bars that hadn't happened yet at the row's nominal date.

The Arc 9 incident: `d1_bars_since_swing_low` used a ±10-bar centred swing detector at the D1 frame level. For each D1 row dated `d`, the swing flag was computed using bars `d-10` through `d+10`. The merge_asof join then correctly fetched the row dated `signal_date − 1 day`. The fetched row's swing flag was constructed using up to 10 future bars relative to the signal's entry time. AUC was inflated by 0.23, the entire deployment thesis was built on the artefact, and the original audit's 8/8 GREEN verdict passed despite the leak being present.

This failure pattern is general. Standard mathematical definitions of trading concepts frequently use centred or bilateral windows:

- Swing/pivot detection, ZigZag, fractal points
- Centred moving averages and smoothers (Hodrick-Prescott, LOESS, Savitzky-Golay)
- Symmetric-window volatility regime classifiers
- Bilateral change-point detection

Any developer implementing these from textbook produces non-causal features by default. The protocol's existing no-lookahead language does not call this out as a distinct audit category.

### Proposed text

> **§N producer-level causal scope (new v2.x).**
>
> For every feature in a classifier's feature matrix, the value at any given row must depend only on data with timestamp ≤ that row's nominal date (after any specified lag). Standard mathematical definitions that use centred or bilateral windows are non-causal by default and must be replaced with one-sided or confirmation-lag variants.
>
> **Audit requirement:** every classifier audit must include a per-feature producer-level causal verification, distinct from join-level causality and end-to-end probability reproduction. The methodology:
>
> 1. For each feature, identify the feature's logical input data at any given date `d`.
> 2. For at least 100 sampled signal bars (stratified across pairs, dates, dataset midpoints / tail / head, boundary cases — Monday morning, post-holiday, session boundaries):
>    - Record the feature value at join time (full producer pipeline)
>    - Re-compute using only data with timestamp ≤ signal_date − specified_lag
>    - Compare; require byte-equivalent match
> 3. Pass criterion: 100% match. Any mismatch → feature has producer-level leak; halt and patch.
>
> **Dispatch instruction language:** any feature-catalogue specification in a dispatch must include the producer-level causal-scope rule explicitly. The rule applies to feature construction at the producer level, not just to the join logic.

### Implementation note

A small library of vetted causal-only implementations should be created for common trading concepts that have textbook non-causal forms:

- `causal_swing_low(series, window=N, confirmation_lag=N)`
- `causal_swing_high(series, window=N, confirmation_lag=N)`
- `causal_zigzag(series, threshold, confirmation_lag)`
- `causal_centred_smoother(series, window, lag)`

Feature catalogues should reference this library rather than asking CC to implement from definition. Reduces surface for the failure mode to recur.

The standard CC lookahead audit dispatch template must include producer-level causal verification as Audit dimension 9 (in addition to the existing 8 dimensions).

### Anchor preservation

KH-24 K=4 archetype 3 — every feature used in its classifier must be reviewed under producer-level audit prior to v2.x landing. Anchor preservation requires the audit; if any KH-24 feature shows producer-level leak, the entire KH-24 deployment is invalidated, with substantial downstream consequences. Audit dispatch for KH-24 is a pre-PR requirement.

### Priority justification

This is the highest-priority amendment because the failure mode it addresses has demonstrably escaped detection by the existing audit framework, the cost of the miss is severe (false deployment candidates), and the fix is mechanical and cheap. Every future arc benefits.

---

## Amendment 2 — §3 threshold-grid replacement (WEAKENED but live)

### Status update from prior version

The original proposal cited Arc 9 evidence (Candidate B at threshold 0.05 producing +11pp ROI over Candidate A at 0.40) as the load-bearing empirical case for replacing the locked grid {0.40, 0.50, 0.60, 0.70} with a recall-floor + precision-gate procedure. **That evidence is invalidated** — both Candidate A and Candidate B were on the leaked classifier. The threshold-gap data does not survive the patch.

What survives:

- **Arc 7 calibration recovery experiment** (separate dispatch, not Arc 9). The Arc 7 surviving D1 units (cluster 1 D1, cluster 3 D1, agg c1+c3 D1) showed the same failure pattern: AUC clears, threshold sweep fails on probability concentration below the locked grid. This is independent of the Arc 9 leak.
- **Arc 9 calibration recovery experiment** (Experiment 2 in Arc 9 sequence) showed RF classifier AUC 0.626 with probability mass concentrated below 0.40, threshold 0.20 reaching recall 0.60 at precision 0.272 (lift 1.58× base rate). This experiment used the **original** D1 classifier, which contains the same leaked features. The pattern's existence is plausible-but-contaminated; the specific numbers are not trustworthy.
- The pattern (RF/LGBM on imbalanced cohorts produces probability mass concentrated outside the locked grid) is general and well-documented in the ML literature, independent of any specific arc's evidence.

### Pre-PR requirement

Arc 7 calibration recovery test must be dispatched and complete before §3 amendment lands. If Arc 7 shows the same pattern on its (causally-clean, separately-implemented) D1 classifier, the amendment has one-arc empirical evidence plus literature support. If Arc 7 returns OUTCOME_B (rank-bound, calibration doesn't help), the amendment still has the threshold-grid-mismatch case from the threshold-sweep itself, but with weaker empirical support.

### Proposed text

Unchanged from prior proposal:

> **Step 4 admission threshold selection (revised v2.x §3).**
>
> Once the classifier clears the AUC gate (§8: ≥ 0.65 for Pipeline E, ≥ 0.60 for Pipeline D1), the deployment operating point is selected via the following procedure:
>
> 1. Compute the classifier's predicted probability for every trade in the Step 1 pool.
> 2. Find the threshold `t*` such that recall = 0.60 on the classifier's predictions.
> 3. Compute precision at `t*`.
> 4. Admit `t*` as the deployment threshold if precision at `t*` ≥ max(2 × base_rate, 0.30).
> 5. If precision at `t*` < the required floor → archetype fails Step 4 (no fallback grid).

### Anchor preservation

KH-24 K=4 archetype 3 does not flow through Step 4 evaluation. No interaction.

---

## Amendment 3 — Step 5 fold-1 warmup convention (LIVE)

### Unchanged from prior proposal

This amendment addresses Arc 9's strict 7-fold §10 failure due to F1 = 0 admits artifact (signal data starts at KH-24 F1 OOS_start, no training data for F1). The convention's applicability survives independent of the classifier leak.

### Proposed text

Unchanged:

> **§10 fold-1 warmup convention (revised v2.x).**
>
> Standard evaluation remains 7-fold WFO with all folds positive required.
>
> Exception: when the arc's Step 1 trade pool start date ≤ KH-24 F1 OOS_start, F1 may report zero admits as a data-window artifact. In such cases:
> - F1 is reported informationally with explicit "data-window artifact" annotation
> - Gate evaluation is performed on F2-F7 instead of F1-F7
> - All pass-deployable and pass-viable thresholds apply to F2-F7 evaluation
> - The arc's closure doc must explicitly note the convention's application

### Anchor preservation

KH-24 K=4 archetype 3 has full 7-fold data and does not trigger the exception. No interaction.

---

## ~~Amendment 4 — §8 D1 feature-budget expansion~~ — WITHDRAWN

### Withdrawal rationale

The original proposal cited Arc 9's Pipeline E retry as direct empirical evidence: classifier AUC lifted from 0.5158 (16-feature baseline) to 0.7508 (28-feature expanded), with 6 of top 15 importances being D1 features. **The +0.23 AUC lift was almost entirely the two leaked swing features.** With causally-clean features, expanded AUC is 0.5190 — essentially identical to baseline 0.5158.

The original "features over classifiers" methodology lesson (drafted as companion artefact) collapses on the same evidence. The lesson in principle may still be valid; Arc 9 does not support it.

### What was correct in the original proposal

The protocol observation that the existing §8 feature catalogue is structured around signal-bar TF features remains accurate. Arc 9 was unable to lift extractability with causally-clean expanded features, but that does not prove no expansion would help — only that the specific 12-feature expansion tested did not help.

### What would need to land an §8 amendment

Different empirical evidence. Possible sources:

- A future arc successfully passing Step 4 with causally-clean feature expansion of any kind
- Re-running prior arcs killed at Step 4 with a vetted causal D1 catalogue (post-Amendment 1 implementation) to test whether feature poverty was a factor
- Direct cross-asset / regime context features (Open-04 escalation, currently out of in-protocol scope)

This amendment is not dead; it is dormant pending a real empirical case.

---

## Amendment 5 — Worst-day DD as standard Step 5 output (LIVE)

### Status

The methodology surfaced by Arc 9's scaled-risk dispatch (worst-day DD at intraday compounded resolution as a binding deployment constraint) survives the leak invalidation because it is engineering-level, not signal-specific. The Arc 9 numerical results that motivated the dispatch are invalidated, but the methodology itself applies to every future Step 5 evaluation.

### Proposed addition

Unchanged:

> **§10 standard outputs (engineering addition v2.x).**
>
> Per-fold metrics extended to include:
> - Worst-day DD (single calendar-day drawdown at intraday compounded resolution)
> - Date of worst-day event
> - Number of contributing trades on that day
> - Number of contributing pairs on that day
>
> Worst-day DD is gated against the in-system 4% target and the 5ers 5% daily hard limit. Pass-deployable adds:
> - Worst-day DD ≤ 4% (in-system)
>
> Pass-viable adds:
> - Worst-day DD ≤ 5% (5ers hard limit)
>
> Methodology: day-DD calculated as running-peak-to-trough within each calendar day, where trough must occur after peak in time. Final day-DD = max(running peak-to-trough, peak-to-day-end-trough, day-start-to-trough).

### Cross-arc note

KH-24's deployed risk of 1.0% per `KH24_SYSTEM_LOCK.md` was set conservatively without explicit worst-day DD measurement. Applying this methodology to KH-24 would either confirm 1.0% is right or expose that a different risk level is appropriate. **Combined with the Amendment 1 producer-level audit requirement for KH-24, the KH-24 audit dispatch becomes load-bearing for protocol confidence.**

### Anchor preservation

KH-24 K=4 archetype 3 deployment is currently characterised without worst-day DD measurement. Adding worst-day DD as a standard output retroactively does not modify KH-24's deployment, but a follow-on dispatch to characterise KH-24's worst-day DD margin is recommended independent of v2.x landing.

---

## Migration

| Action | Owner | Required | Notes |
|---|---|---|---|
| **KH-24 producer-level audit** | Chat (analyst) | **Yes — pre-PR** | Required before Amendment 1 lands; verifies anchor isn't itself affected by the same failure pattern |
| **KH-24 worst-day DD characterisation** | Chat (analyst) | Recommended pre-PR | Validates KH-24 risk parameter; informs Amendment 5 in production |
| **Arc 7 calibration recovery test** | Chat (analyst) | Yes — pre-PR | Required for Amendment 2 evidence base; separate dispatch from any +12 feature expansion (which is withdrawn) |
| Causal-only feature library implementation | Engineering | Yes — pairs with Amendment 1 | Reduces re-occurrence risk of same failure mode |
| Updated CC lookahead audit dispatch template | Chat / Engineering | Yes — pairs with Amendment 1 | Adds dimension 9 (producer-level causal verification) |
| Land v2.x as PR to `L_ARC_PROTOCOL.md` | Engineering (Cursor) | Yes | Amendments 1, 2, 3, 5 together |
| Backtester update to emit worst-day DD | Engineering | Yes — pairs with Amendment 5 | Intraday resolution required |
| Update STATUS / SESSION_ZERO / CHANGELOG / PROTOCOL_IMPROVEMENT_BACKLOG for v2.x | Chat | Yes | Direct-to-main |
| Re-evaluate previously-killed arcs under v2.x | Backlog | Recommended | Post-Amendment-1 audit verification; do NOT re-expand with the withdrawn §8 catalogue |

---

## Open items not closed by this proposal

- Withdrawn §8 amendment — proposal is dormant pending new empirical evidence; not closed forever, just paused
- The "features over classifiers" methodology lesson — withdrawn; would need different empirical support
- Open-04 cross-asset / regime context features — separate escalation, not closed by this proposal
- Future research effort to resurrect IB-trend signal with different feature class (cross-pair, microstructure, longer-TF, ensembles) — backlog, fresh dispatch when proposed

---

## Document control

| Field | Value |
|---|---|
| Version | v2.x DRAFT (revised) |
| Supersedes | v2.3 (drafted, not yet landed) — additive amendments |
| Original draft | 2026-05-18 |
| Revised | 2026-05-19 — §8 withdrawn, Amendment 1 (producer-level audit) added as highest priority |
| Active for arcs | post-PR-land |
| Methodology change | Yes — producer-level audit dimension, §3 threshold selection, §10 fold-1 convention + worst-day DD output |
| Engine change required | Causal-only feature library (paired with Amendment 1); §10 worst-day DD backtester output (paired with Amendment 5) |
| Anchor preservation | Pre-PR audit of KH-24 under Amendment 1 audit dimension required |
| Pre-PR gates | KH-24 producer audit + Arc 7 calibration test + analyst sign-off + anchor verification |
