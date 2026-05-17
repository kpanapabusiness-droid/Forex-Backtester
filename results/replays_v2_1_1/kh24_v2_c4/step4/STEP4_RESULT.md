# KH-24 v2.0 self-test Step 4 — Extractability (Amendment-Conditional) — RESULT

## Conditional disclaimer

This Step 4 work runs under proposed but not yet ratified amendments to v2.1.1 §2 + §11 (see chat-side `AMENDMENT_PROPOSAL_S2_SHAPE_TAG_S11_EXPANSION.md`). Cohort set (c1 Stepwise climber, c4 Slow Trend Trail) depends on amendment ratification. **Classifier target is cluster membership (not profitability)** — policy-independent. Output is diagnostic-grade until Open-18 cross-replay synthesis confirms or modifies the amendments. STATUS / CHANGELOG / §14 governance updates do not follow from this dispatch.

Inputs:
- `trades_all.csv` (Step 1)
- `trades_features_base8.csv` sidecar (Step 1 patch, commit 1a7c9f8)
- `trades_paths.csv` (Step 1; path-so-far source for Angle D1)
- `clusters_K5.csv` (Step 2; cluster membership labels)

All three joined on `trade_id`; row count 842 preserved across the inner join. Zero NaN across the 8 base features post-join.

## Cohort set under test (amendment-state)

| cluster | n (positives) | non-cluster pool (negatives) | size | positive_rate | selected SL | composite | §11 row (proposed) | exit policy (deferred D1 PR 2) |
|---|---|---|---|---|---|---|---|---|
| c1 | 365 | 477 | 43.4% | 0.434 | 2.0×ATR | 0.387 | Stepwise climber | MFE-lock at 1R, trail 0.75R from new high |
| c4 | 122 | 720 | 14.5% | 0.145 | 2.0×ATR | 0.607 | Slow Trend Trail (new row) | Lock at 1R, trail 0.5R from new high |

c1 is roughly balanced; c4 is imbalanced. PR-AUC is the load-bearing metric for c4 — its class baseline of 0.145 means ROC-AUC alone can mislead.

Feature inventory: 8 §8 base + 1 arc-specific (`signal_bar_atr_14`) = 9 features for Angle E. Other `trades_all.csv` columns are outcomes/identifiers (lookahead) or market-condition variables excluded by design (`spread_pips_used`). Angle D1 adds 7 path-so-far features computed at each candidate t for 16 total D1 features. Both well under §8's 38 cap.

## Angle E results

| Cluster | RF ROC-AUC | RF PR-AUC | Logistic ROC-AUC | Logistic PR-AUC | RF–Log gap | Top-3 features (RF importance) | Verdict |
|---|---|---|---|---|---|---|---|
| c1 | 0.465 ± 0.039 | 0.420 ± 0.027 | 0.458 ± 0.065 | 0.416 | +0.007 | rsi_14 (0.139), signal_bar_atr_14 (0.136), ret_5bar_atr (0.116) | **FAIL** (gate 0.65) |
| c4 | 0.521 ± 0.053 | 0.174 ± 0.042 | 0.477 ± 0.026 | 0.152 | +0.044 | signal_bar_atr_14 (0.144), rsi_14 (0.138), ret_5bar_atr (0.124) | **FAIL** (gate 0.65) |

**Pipeline E gate fails for both cohorts.** Neither cluster's archetype membership is predictable from entry-bar features alone above the 0.65 RF ROC-AUC gate. The RF–logistic gap is small (<0.05 for both), indicating no meaningful non-linear lift over logistic — the feature set is the binding constraint, not model complexity (§1.18 / §8 diagnostic).

PR-AUC commentary for c4: RF PR-AUC 0.174 is just above the class baseline of 0.145 — confirms the headline ROC-AUC 0.521 is near random with imbalance accounting. The threshold sweep returned **no admissible threshold** for c4 because no candidate threshold ∈ {0.40, 0.50, 0.60, 0.70} reaches recall ≥ 0.60 at this discriminative level — typical of a near-baseline classifier on an imbalanced cohort.

## Angle D1 results

Per cluster: ROC-AUC + PR-AUC + exclusion% at t ∈ {1, 2, 3, 4, 5, 10}, smallest-t chosen.

| Cluster | t=1 | t=2 | t=3 | t=4 | t=5 | t=10 | Chosen t | Verdict |
|---|---|---|---|---|---|---|---|---|
| c1 | **0.618 / 0.534 / 0.000** | 0.648 / 0.580 / 0.003 | 0.671 / 0.601 / 0.005 | 0.713 / 0.669 / 0.011 | 0.708 / 0.672 / 0.022 | 0.665 / 0.686 / 0.074 | **t=1** | **PASS** (D1 ROC-AUC ≥ 0.60 AND excl ≤ 0.30) |
| c4 | 0.593 / 0.237 / 0.000 | **0.615 / 0.241 / 0.000** | 0.639 / 0.297 / 0.000 | 0.642 / 0.264 / 0.000 | 0.644 / 0.294 / 0.000 | 0.630 / 0.324 / 0.000 | **t=2** | **PASS** (D1 ROC-AUC ≥ 0.60 AND excl ≤ 0.30; PR-AUC marginal — see below) |

Format: `ROC-AUC / PR-AUC / exclusion_rate`. Bold = chosen-t per smallest-t rule.

**c1 (D1, t=1):** RF ROC-AUC 0.618 and PR-AUC 0.534 (well above class baseline 0.434, meaningful margin +0.10). Exclusion 0.0. Threshold 0.50 selected: precision 0.71, recall 0.68. Strong D1 pass.

**c4 (D1, t=2):** RF ROC-AUC 0.615 (passes 0.60 gate). **RF PR-AUC 0.241 — just under the 0.25 "meaningful" threshold** from the dispatch (class baseline 0.145). Margin over baseline is +0.096, comparable to c1's +0.10, so the classifier does carry signal, but the imbalance flag stands: the chosen-t classifier sits at the edge of usability. Note that c4's D1 AUC peaks at t=4–5 (0.642–0.644) but the smallest-t rule (§8) picks t=2 because it's the smallest t clearing both gates — trading classifier strength for earlier policy switching.

Exclusion is **zero across all t for c4** — its 122 positives all have `bars_held ≥ 10`, consistent with c4 being the slow-trend-trail cohort that censoring (87.7% cap-binding in v2.0) made hard to characterize. For c1, exclusion stays under 0.075 even at t=10.

D1 path-so-far features dominate over entry features in the importance ranking (top 3 importances are typically `close_r_at_t`, `mfe_so_far_r_at_t`, `mae_so_far_r_at_t`). The discriminating signal is the early-bar path behaviour, not the entry-bar OHLC features — structurally consistent with the path-shape clustering that defines the cohorts.

## Pipeline assignment

| Cluster | E pass | D1 pass | Pipeline assigned | Artefacts |
|---|---|---|---|---|
| c1 | N | Y (t=1) | **D1** | `c1_D1_classifier.joblib`, `c1_D1_policy.yaml` |
| c4 | N | Y (t=2) | **D1** | `c4_D1_classifier.joblib`, `c4_D1_policy.yaml` |

Both cohorts route exclusively to Pipeline D1. No Pipeline E artefacts produced. **No archetype dies at Step 4** — both have viable D1 classifiers, subject to chat-side review of c4's marginal PR-AUC.

## RF vs logistic gap (§1.18 / §8)

| Cluster | RF ROC-AUC | Logistic ROC-AUC | Gap | Interpretation |
|---|---|---|---|---|
| c1 | 0.465 | 0.458 | +0.007 | Tiny gap — **feature set is binding**. Richer features may help; non-linearity won't. |
| c4 | 0.521 | 0.477 | +0.044 | Small gap — same conclusion. RF gives ~4pp lift over logistic but neither reaches the gate. |

Both gaps well under the ≥0.10 non-linear-dynamics flag. The Pipeline E failure is not a model-complexity failure; it's a feature-set failure. The 8 base + 1 arc-specific feature set does not contain enough entry-bar information to predict cluster membership at 0.65 ROC-AUC.

## Filter selection trace

Per cluster: Step A baseline (full 9-feature RF) failed → Step B top-5 subset attempted → also failed → Step C not attempted (only 9 features available; stacking budget better deployed against richer catalogues per dispatch §C note).

- **c1:** step_a (ROC-AUC 0.465) → step_b_top5 (0.488) → step_c_not_attempted
- **c4:** step_a (ROC-AUC 0.521) → step_b_top5 (0.581) → step_c_not_attempted

Top-5 subsets gave modest lifts (+0.023 for c1, +0.060 for c4) but did not clear the gate. The Step C stacking budget (≤30 combinations) is preserved for clusters with more feature variety in future arcs.

## Diagnostic findings

**1. Feature catalogue is thin.** Only 1 arc-specific feature (`signal_bar_atr_14`) was available beyond the 8 §8 base. The dispatch caps arc-specific at 30 but the actual KH-24 v2.0 Step 1 schema offers 1. A richer feature catalogue at signal-bar emission time (e.g. multi-bar wick ratios, RSI divergences, prior-bar momentum) would likely lift Pipeline E.

**2. Pipeline E unviable, Pipeline D1 viable.** This split is informative on its own. Cluster membership (= path-shape archetype) is not predictable from entry-bar features in this signal's feature space, but *is* predictable from 1–2 bars of path-so-far behaviour. The classifier is essentially observing the path shape's first derivative and inferring which cluster it will land in. Aligns with the clustering-by-path-shape methodology — clusters are defined by path features, so observing path features early is the natural predictor.

**3. c4 PR-AUC imbalance flag.** RF Angle D1 PR-AUC at chosen t=2 is 0.241, just under the dispatch's 0.25 "meaningful" threshold. Above baseline (0.145) by +0.096, comparable to c1's margin over its own baseline, but lower in absolute terms because of the imbalance. The c4 classifier should be considered diagnostic-grade for D1; Step 5 cross-fold stability is the next gate that will determine whether it survives in deployment-track use.

**4. Smallest-t rule trades classifier strength for earlier policy switching.** c4's D1 ROC-AUC at t=4–5 (0.642–0.644) is meaningfully better than at t=2 (0.615), but the smallest-t rule (§8) picks t=2. c1's t=4 ROC-AUC (0.713) is also notably better than t=1 (0.618). Whether this trade-off is right is a §8 calibration question, not a per-arc decision; flagged for future cross-arc review if pattern repeats.

**5. Path-so-far features dominate entry features in D1 importance.** Consistent with point 2 above.

**6. Threshold sweep returned no admissible threshold for c4 D1.** All four candidate thresholds {0.40, 0.50, 0.60, 0.70} fail the recall ≥ 0.60 floor at the chosen t=2 — the classifier's predicted probabilities cluster too tightly around the base rate (0.145). Recorded as `admit_threshold: null` in `c4_D1_policy.yaml`. D1 PR 2 backtester will need to either use a finer threshold sweep, accept a lower recall floor for c4, or back off to a different t value. Flagged for chat-side decision; this is a real protocol-vs-imbalance interaction.

## Next steps

If synthesis ratifies amendments:
- **Step 5 cross-fold stability** dispatch under the same conditional framing. c4's marginal PR-AUC will be tested for fold-by-fold stability.
- **Step 6 WFO truth** is blocked for both cohorts: D1 WFO requires D1 PR 2 (per-archetype exit policy executor; PR #131 only landed the close-at-market path).
- **Pipeline E WFO at Step 6 is moot** for this arc — both cohorts failed Pipeline E.
- **c4 threshold question** must be resolved before D1 PR 2 can ship c4: either (a) finer-grain threshold sweep, (b) lower recall floor for imbalanced cohorts, or (c) abandon c4 D1 admission and use a different selection mechanism.

If synthesis ratifies different amendments (e.g. modified §11 Slow Trend Trail row, modified §2 shape_tag dead zone fix that changes the cohort set):
- Re-derive Step 4 with updated cohort set
- This output retained as diagnostic baseline

If synthesis rejects amendments:
- This output is obsolete. v2.1.1 Step 3 only rescued c4 at SL=1.0×ATR (composite 0.304); under that R-frame, R-denomination shifts and feature scales change, but cluster-membership target is invariant to SL choice — most of this Step 4 work would still apply, only the policy YAMLs would need rewriting against v2.1.1 §11 exits.

## Files

All under `results/replays_v2_1_1/kh24_v2_c4/step4/`:

- `predictability_angle_E.csv` — per-cluster ROC-AUC, PR-AUC, gap, importances, threshold
- `predictability_angle_D1.csv` — per-cluster per-t ROC-AUC, PR-AUC, exclusion, chosen-t
- `extractability_pass_list.csv` — per-cluster pipeline assignment + AUCs
- `c1_D1_classifier.joblib` + `c1_D1_policy.yaml`
- `c4_D1_classifier.joblib` + `c4_D1_policy.yaml`
- `STEP4_RESULT.md` (this file)

No Pipeline E artefacts (both clusters failed Pipeline E).

Step 1 patch commit: 1a7c9f8 (sidecar `trades_features_base8.csv`)
Step 3 closure commit: 30bc376 (Open-18 Replay #2 Step 2)
Step 4 commit: filled in post-commit
