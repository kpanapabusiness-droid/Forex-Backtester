# Incident — Arc 9 Producer-Level Lookahead Miss

| Field | Value |
|---|---|
| Date detected | 2026-05-19 |
| Date occurred | 2026-05-18 (Pipeline E retry dispatch + lookahead audit dispatch) |
| Severity | HIGH — research result false-positive; deployment chain initiated on fake economics |
| Detected by | Analyst-initiated parallel-chat audit |
| Resolution | Causal patch dispatch (commit 5b6c547); STEP_4_KILL_REAFFIRMED |
| Forward-looking change | Producer-level causal audit dimension added as standard for all future classifier audits |

## What happened

Arc 9 Pipeline E retry (dispatch 2026-05-18, commit 0193334) added 8 D1-lagged features + 4 session/time features to the existing 16-feature catalogue. Classifier AUC lifted from 0.5158 → 0.7508 — a +0.23 improvement that motivated the held-open lifecycle, deployment candidate construction, and a v2.x amendment proposal.

Two of the 8 D1 features — `d1_bars_since_swing_low` and `d1_bars_since_swing_high` — were computed using a ±10-bar centred swing detector at the D1 frame level. The producer suppressed the last 10 bars of the frame (correct partial protection for the dataset tail) but did not suppress the 10 bars after each individual signal's join point. For any signal in the middle of the dataset, the swing flag at the joined D1 row had been computed using up to 10 future D1 bars relative to the signal's entry time.

Causal patch with confirmed-swing detection (10-day lag) re-ran the classifier on causally-clean features. AUC collapsed: LightGBM 0.7508 → 0.5190, RandomForest 0.7759 → 0.5551. The +0.23 lift was almost entirely the two leaked features (27% combined gain, 20% from the top one alone). Forced WFO at the patched classifier produced full-data ROI ≈ 0% with materially negative folds.

The cohort itself is unaffected (Step 3 capturability and Step 5 oracle ceiling both unchanged). The classifier and all downstream economics (Candidate A, Candidate B, scaled-risk recommendation, EA deployment chain) are invalidated.

## How it happened — root cause

Two failures, both in dispatch-and-audit instructions written by the analyst:

### Failure 1 — Dispatch instructions did not specify producer-level causal scope

The Pipeline E retry dispatch specified D1 features must be one-day-lagged via `merge_asof(direction='backward')` on a pre-shifted date column. It did not specify that **each D1 row's value must itself depend only on data with timestamp ≤ that row's date**. The dispatch treated "lagged join" and "causal feature" as equivalent. They are not.

CC implemented the swing detector using the standard mathematical definition (centred window) because:
- That is the default implementation in any TA library
- The dispatch's "no lookahead" language was framed at the join level
- Suppressing the last 10 bars of the frame appeared to handle the future-data concern

The dispatch instruction did not call out that the protocol's no-lookahead requirement extends to the values within each row, not just to which row gets joined.

### Failure 2 — Audit dispatch did not include producer-level causal verification

The lookahead audit dispatch (2026-05-18, commit 9dc4f8a) specified 8 audit dimensions:

1. 4H feature timestamps (join-level)
2. D1 lag integrity (join-level — `merge_asof` direction, days_lag distribution)
3. Session/hour features (deterministic-from-timestamp)
4. Label leakage (forward-geometry features in entry-time matrix)
5. Train/inference fold disjointness
6. Cluster label flow (label flows training-target → model-weights → predictions only)
7. Spread & execution semantics
8. End-to-end probability reproduction

All eight dimensions passed cleanly. The audit was rigorous on the dimensions it covered — three audit-script bugs were caught and corrected during the run, end-to-end reproduction landed at float64 noise floor.

But the eight dimensions did not include: **for each feature, was the value at the join row computed using only data with timestamp ≤ that row's date?** This is a distinct audit dimension. The audit checked that the join itself was causal. It did not check that the values being joined were causally constructed.

The category was the same one I flagged at the start of the chat as "the project-fatal failure mode" and "the most consequential audit." I specified the audit narrowly to the lag and join dimensions. The dimension that mattered was one level deeper, and I did not write an audit for it.

## Why the failure pattern is general

Standard mathematical definitions of trading concepts frequently use centred windows:

- **Swing/pivot detection** — bar N is a swing if it's the local extremum of N±k bars
- **ZigZag / fractal points** — same family
- **Centred moving averages and smoothers** — Hodrick-Prescott filter, LOESS, Savitzky-Golay
- **Some volatility regime classifiers** — symmetric-window vol calculations
- **Breakpoint / change-point detection** — typically uses bilateral data

Any developer implementing these from a textbook will produce non-causal features. The protocol's no-lookahead language did not call this out as a category. The dispatch wrote "no lookahead" and the audit checked join-level causality. Both interpretations are correct as far as they go. Both are incomplete.

This is a recurring failure mode of the form: a rule is stated, an implementation follows the obvious reading of the rule, both pass the obvious audit of the rule, and the bug lives one abstraction layer below where the rule was stated. The fix is not "be more careful" — it is to make the audit dimension explicit and standard.

## Forward-looking changes

### Change 1 — Producer-level causal audit dimension is now standard

Every future classifier audit must include a per-feature producer-level causal-scope verification. The methodology:

For each feature in the classifier's feature matrix:

1. Identify the feature's logical input data at any given date `d` (e.g. `rsi_14` at date `d` uses bars `d-13` through `d`)
2. For N sampled signal bars (N ≥ 100, stratified across pairs and dates):
   - Record the feature value at join time (computed with the full producer pipeline)
   - Re-compute the feature using ONLY data with timestamp ≤ signal_date − lag (where lag is the feature's nominal lag, typically 1 day for D1 features)
   - Compare; require byte-equivalent match
3. Specifically test: dataset midpoints, dataset tail (last 30 bars), dataset head (first 30 bars), boundary cases (Monday morning, post-holiday, session boundaries)

Pass criterion: 100% match. Any mismatch indicates producer-level leak.

This audit dimension is separate from join-level causality verification (`merge_asof` direction, days_lag distribution) and separate from end-to-end probability reproduction. All three are required.

The audit dimension specification is logged in:

- `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` as a v2.x protocol amendment
- The standard CC lookahead audit dispatch template (to be created as `cc_lookahead_audit_template.md`)

### Change 2 — Dispatch instruction template includes causal-scope language

Future dispatches that specify feature catalogues must include the producer-level causal-scope rule explicitly:

> For each feature, the value at any given row must depend only on data with timestamp ≤ that row's nominal date (after any specified lag). Standard mathematical definitions that use centred or bilateral windows (swing detection, pivots, ZigZag, centred moving averages, fractal points, some change-point detectors) are non-causal by default and must be replaced with one-sided or confirmation-lag variants.

This language is added to the CC arc orchestrator template at the feature-catalogue specification section.

### Change 3 — Standard library of causal-only feature implementations

A small library of vetted causal-only implementations is created for common trading concepts:

- `causal_swing_low(series, window=10, confirmation_lag=10)` — confirmed-swing with explicit lag
- `causal_swing_high(series, window=10, confirmation_lag=10)`
- `causal_zigzag(series, threshold, confirmation_lag)`
- (extend as needed)

Future dispatches reference this library rather than asking CC to implement from definition. Less surface for the same failure mode to recur.

### Change 4 — Cross-arc regression

The lookahead audit on Arc 7's three surviving D1 units (recommended in earlier cross-arc note) is **withdrawn** in its current form. The proposed Arc 7 audit would have included the same +12 feature expansion that contains the leaked swing features. Reproducing the leak on Arc 7 contributes nothing.

The Arc 7 calibration recovery test (Platt + isotonic on the existing classifier) remains valid and recommended as a separate dispatch.

If Arc 7's existing classifier or any prior arc's classifier used centred-window swing detection or any of the other non-causal patterns in the failure-pattern list, those arcs require producer-level audit before any deployment claims they support remain valid. Backlog item.

## What this incident does not invalidate

- **Step 1 / Step 2 / Step 3 of Arc 9** — path-shape features used only forward-geometry computation; not entry-time predictors; not vulnerable to the same leak class
- **Step 5 oracle and raw baseline** — used cluster identity or no filter; not classifier-dependent
- **Original lookahead audit on its eight checked dimensions** — those dimensions are correctly verified; the audit's scope was incomplete, not its execution
- **The cohort itself** — cluster_0 is real, has +60% ann ROI / 0% DD in oracle, has clean forward geometry

## What this incident validates

The protocol's gate structure worked correctly. Once the leak was patched, the §8 AUC gate immediately rejected the classifier (AUC 0.5190 below 0.65 floor). Forced WFO at the patched classifier confirmed economic collapse exactly as the gate predicted. The protocol's machinery did its job — the problem was the audit input it was given, not the gate logic.

## Cost

Compute: eight CC dispatches across two days (Pipeline E retry, Step 5 LGBM E, lookahead audit, scaled-risk, candidate spec drafts, EA handover, causal patch, forced WFO). Several hundred Claude tokens. Minor.

Time: roughly one day of analyst time during which Arc 9 was treated as a deployment candidate that it was not. Several hours of EA-deployment planning that does not transfer.

Trust: the original audit was presented as proof of leak-freedom and was wrong on the dimension that mattered. Future audits will need to be designed against this incident's evidence. The producer-level dimension is now standard for that reason.

## Sign-off

Analyst-acknowledged. Forward-looking changes (1-4) entered into protocol amendment proposal and CC dispatch templates. Arc 9 closed as STEP_4_KILL_REAFFIRMED.

Logged in `PROTOCOL_IMPROVEMENT_BACKLOG.md` as cross-arc incident reference.
