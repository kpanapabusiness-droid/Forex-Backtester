# Arc 9 — IB-trend (compression geometry → directional break)

## Closure

| Field | Value |
|---|---|
| Disposition | **STEP_4_KILL_REAFFIRMED** |
| First closed | 2026-05-18 (STEP_4_KILL on extractability FAIL) |
| Held-open opened | 2026-05-18 (analyst-directed diagnostic experiments) |
| Held-open closed | 2026-05-19 (STEP_4_KILL reaffirmed after causal patch) |
| Active protocol | L_ARC_PROTOCOL v2.1.2 + v2.2 + v2.3 amendments |
| Branch | `claude/bold-brattain-d79817` + `claude/arc-9-causal-patch` |
| Anchor preservation | KH-24 K=4 archetype 3 — preserved, no interaction |

## Verdict

Arc 9 closes as STEP_4_KILL with cohort verified deployable in oracle but unreachable through causally-clean in-protocol features. The held-open cycle produced one valuable forward-looking artefact (producer-level causal audit dimension) and one important framework-level correction (a feature-expansion finding that appeared load-bearing was an artefact of two leaked features).

Two months of cycle time, eight experiments, one consequential audit miss, one corrective patch. Honest disposition.

## Step pass/fail table

| Step | Gate | Result | Commit |
|---|---|---|---|
| 1 | Plumbing | PASS — 2153 trades; all 7 sub-gates green | (Step 1 commit) |
| 2 | Clustering | PASS — K=3, silhouette 0.4247 | (Step 2 commit) |
| 3 | Capturability | PASS — 1 archetype (cluster_0_individual @ SL=2.0×ATR) | (Step 3 commit) |
| 4 | Extractability (original) | **FAIL** — E AUC 0.511; D1 AUC 0.626 with threshold sweep recall 0.003 | (Step 4 commit) |

### Held-open experiments (2026-05-18 → 2026-05-19)

| # | Experiment | Outcome | Commit |
|---|---|---|---|
| 1 | Step 5 oracle validation | PASS-DEPLOYABLE @ +39.45% worst-fold ROI / 0.01% DD (cluster 0 only, oracle filter) | d32b506 |
| 2 | Calibration recovery (Platt / isotonic) | OUTCOME_B — rank-bound, not calibration-bound | 2882270 |
| 3 | Step 5 raw baseline | FAIL all gates (−29.64% worst-fold, 62.99% DD; the floor) | 496d772 |
| 4 | Pipeline E retry (LGBM + expanded features) | **AUC 0.7508 (FAKE — see Phase 7)** | 0193334 |
| 5 | Step 5 LGBM Pipeline E WFO | **PASS-DEPLOYABLE F2-F7 (INVALIDATED — see Phase 7)** | 44de1ca |
| 6 | Lookahead audit (8 dimensions) | GREEN on join-level dimensions; **INCOMPLETE — missed producer-level causal scope** | 9dc4f8a |
| 7 | Scaled-risk WFO | **1.0% recommended (INVALIDATED — admit set was on fake classifier)** | 5ce39d6 |
| 8 | Causal patch + forced WFO | **STEP_4_KILL_AFTER_PATCH** — patched AUC 0.5190; forced WFO confirms economic collapse | 5b6c547 |

## What happened

Steps 1-4 ran cleanly. Step 4 KILL was correct: Pipeline E AUC 0.511 (chance) on the in-protocol 16-feature set, D1 AUC 0.626 with probability mass below the threshold grid. The cohort exists in forward-geometry but doesn't separate from the pool on entry-time features.

Held-open experiments confirmed the cohort's reality:
- Step 5 oracle (cluster 0 only): +60% ann ROI / 0% DD across 7/7 folds. The signal class produces clean economics when correctly identified.
- Step 5 raw baseline (no filter): −29% worst-fold ROI, 63% DD. The complement is portfolio-killing; the cluster work is the entire system.
- Calibration recovery: rank-bound failure. Probability mass concentration below 0.40 wasn't a calibration artifact; the classifier genuinely couldn't separate the populations crisply.

**The Pipeline E retry (Experiment 4) appeared to break this.** Adding 8 D1-lagged features + 4 session/time features lifted classifier AUC from 0.511 to 0.7508. Step 5 LGBM Pipeline E WFO (Experiment 5) produced two pass-deployable candidates. Lookahead audit (Experiment 6) passed 8/8 dimensions. Scaled-risk measurement (Experiment 7) identified worst-day DD as the binding constraint and recommended 1.0% per-trade deployment risk producing +41.45% annualised ROI at 2.52% DD.

**Then an external audit detected lookahead in two of the added features.** `d1_bars_since_swing_low` and `d1_bars_since_swing_high` were computed using a ±10-bar centred swing detector at the D1 frame level. The producer suppressed the last 10 bars of the frame, but did not suppress the 10 bars after each individual signal's join point. The `merge_asof` join itself was clean (verified to 560 samples in the prior audit). The values inside the joined rows were not — they had been computed using up to 10 future D1 bars relative to each signal's entry time. The original audit checked join-level causality and end-to-end probability reproduction; it did not check producer-level causal scope.

Causal patch dispatch (Experiment 8) replaced both swing features with a confirmed-swing detector requiring 10 days of lag before a swing is markable. Re-ran Pipeline E retry on the patched feature set. AUC collapsed from 0.7508 to 0.5190. RF on the same patched features dropped from 0.7759 to 0.5551. The baseline 16-feature cells were byte-identical between original and patched runs, confirming the 12 added features (and specifically the 2 swing features) were the load-bearing change.

Forced WFO at the patched classifier confirmed the economic verdict. Candidate A: full-data ROI −0.13%, DD 13.38%, 3 of 6 folds negative. Candidate B: full-data ROI +0.08%, DD 22.65%, well past 5ers' 10% hard limit. The §8 AUC gate correctly identifies these classifiers as non-deployable; forced WFO confirms the gate's verdict.

## The bracket — three Step 5 reference points (final)

| Run | Filter | Worst-fold ROI | Worst-fold DD | Verdict |
|---|---|---|---|---|
| Floor (raw baseline) | none | −29.64% | 43.61% | FAIL all gates (expected) |
| Ceiling (oracle) | post-hoc cluster identity | +39.45% | 0.01% | PASS-DEPLOYABLE (oracle, not deployable as system) |
| Patched LGBM Pipeline E Cand A | classifier @ threshold 0.40 | −6.61% (F6) | 6.83% | FAIL — close to floor |
| Patched LGBM Pipeline E Cand B | classifier @ threshold 0.05 | −19.37% (F6) | 20.90% | FAIL — at floor |

The gap between oracle ceiling (+39%) and patched real-classifier (−6%) is the gap between "we know the future" and "we use causally-clean entry-time features." It is unbridgeable on the current in-protocol feature catalogue.

## What survives

- **Step 1 plumbing** (2153 trades, deterministic, audit-clean on join-level dimensions)
- **Step 2 clustering** (K=3, cluster_0 n=365) — path-shape features only, no entry-time leak risk
- **Step 3 capturability** — cluster_0 fwd_mfe_p50 6.18R, final_r_mean +4.42R, t-stat +21.81 at SL=2.0×ATR. The cohort is real.
- **Step 5 oracle ceiling** — confirms the cohort produces deployable economics if perfectly identified at entry
- **Step 5 raw baseline** — confirms the complement carries portfolio-killing loss-side asymmetry
- **Original lookahead audit on its checked dimensions** — join-level, label, fold disjointness, execution semantics, e2e reproduction all valid; the dimensions checked were correct, the audit scope was incomplete

## What collapses

- **Pipeline E retry AUC 0.7508** — the appearance of extractability was a leak artefact
- **Step 5 LGBM Pipeline E both candidates** — admit sets were on a fake classifier
- **Scaled-risk 1.0% deployment recommendation** — measured on the fake classifier; meaningless
- **"Features over classifiers" methodology lesson** — the empirical lift attributed to feature expansion was the leaked features; lesson has no Arc 9 empirical case
- **v2.x §8 D1 feature-budget expansion proposal** — Arc 9 contributes no evidence; proposal needs different empirical support or different scope

## What is newly valuable

The **producer-level causal scope audit dimension** is the single most valuable forward-looking artefact from the entire held-open cycle. The original audit checked:

- Join-level causality (`merge_asof` direction, days_lag distribution)
- End-to-end probability reproduction
- Label leakage (forward-geometry features in entry-time feature set)
- Fold disjointness (training/inference separation)
- Execution semantics
- Cluster label flow
- 4H feature timestamps
- Session/hour feature determinism

It did not check, for each D1 feature, whether the value at the join row depended only on data with timestamp ≤ that row's date. This is a distinct audit dimension from the eight listed above. It is now standard for every future classifier audit. The audit-dimension specification is captured in the corresponding incident note.

## v2.x amendment evidence — Arc 9's surviving contribution

| Amendment | Arc 9 contribution |
|---|---|
| §3 threshold-grid replacement | WEAKENED — Arc 9's AUC-clears-grid-fails pattern was on a fake classifier. Arc 7 calibration recovery remains the load-bearing evidence. Cross-test on Arc 7 D1 units still recommended. |
| §8 D1 feature-budget expansion | COLLAPSED — Arc 9 contributes no evidence. Causally-clean D1 expansion delivers ≈ 0 AUC lift. Proposal requires re-scoping or different empirical support. |
| Step 5 fold-1 warmup convention | Still applies for future arcs whose data window starts at F1 OOS_start. Survives independent of Arc 9 economics. |
| **Producer-level causal audit dimension (NEW)** | REINFORCED. Promotes from "lesson" to "standard audit requirement" for every future classifier evaluation. |
| Step 4 artefact-on-FAIL persistence | Still valid (process improvement, independent of leak). |

## Cross-arc work corrected

- **Arc 7 D1 feature-expansion test:** REFUTED as a recommendation. Do not run the +12-feature expansion test on Arc 7's three surviving D1 units. The proposed expansion includes the now-known-leaked swing features. Reproducing the leak in Arc 7 contributes nothing.
- **Arc 7 calibration recovery test (Platt + isotonic on existing classifier):** remains valid and recommended. Separate dispatch from feature expansion. Tests whether Arc 7's threshold-sweep failure is a calibration artefact, parallel to the Arc 9 calibration recovery (which returned OUTCOME_B, rank-bound).

## Methodology lessons (revised after patch)

The original held-open finding "features over classifiers" was the Arc 9 empirical case for prioritising feature engineering over classifier-family experimentation when Step 4 fails. **That case collapses with the leak removal.** The lift attributed to feature expansion was almost entirely the two leaked features. Causally-clean expansion produces near-zero lift on this cohort.

The lesson in principle may still be valid — feature engineering is often higher-leverage than classifier tuning — but Arc 9 does not provide empirical support. Future arcs would need to establish this independently.

The replacement lesson, taken directly from this cycle's mistake: **dispatch instructions must specify causality at the producer level, not just at the join level.** When a feature catalogue says "use D1-lagged features via `merge_asof` backward," the analyst must also specify that the values within D1 rows must be causally constructed — meaning each row's value depends only on data with timestamp ≤ that row's date. Standard mathematical definitions of trading concepts (swing detection, pivots, ZigZag, fractal points) frequently use centred windows and are non-causal by default. The dispatch must call this out; the audit must verify it.

## Disposition

**STEP_4_KILL_REAFFIRMED.** The cohort is real but unreachable on causally-clean in-protocol features. The held-open lifecycle terminates here.

Resurrecting this signal requires a fresh research effort proposing a different feature class — cross-pair regime context, intra-bar microstructure, longer-TF context (W1, M1), ensemble methods, or non-standard data sources. A fresh dispatch, not a continuation of Arc 9.

## Files

All under `results/l_arc_9/` on `claude/bold-brattain-d79817` + `claude/arc-9-causal-patch`:
- `step1_plumbing/`, `step2_clustering/`, `step3_capturability/`, `step4_extractability/` — original arc artefacts
- `experiments/step5_validation/` — oracle ceiling (valid)
- `experiments/calibration_recovery/` — OUTCOME_B (valid)
- `experiments/step5_raw_baseline/` — floor (valid)
- `experiments/pipeline_e_retry/` — leaked classifier (invalidated by Phase 8)
- `experiments/step5_lgbm_pipeline_e/` — leaked-classifier WFO (invalidated)
- `experiments/lookahead_audit/` — original audit (valid on its dimensions; scope incomplete)
- `experiments/scaled_risk/` — leaked-classifier scaled risk (invalidated)
- `experiments/causal_patch/` — patched re-run + forced WFO + new audit dimension (load-bearing)
- `ARC_9_LIVE.md` — held-open record, status STEP_4_KILL_REAFFIRMED 2026-05-19
- `ARC_9_CLOSURE.md` — this doc

## Companion documents

- `INCIDENT_2026_05_19_ARC_9_PRODUCER_LEAK.md` — incident note: what was missed, why, what audit dimension is now standard
- `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` — revised amendment proposal reflecting Arc 9's actual evidence contribution
