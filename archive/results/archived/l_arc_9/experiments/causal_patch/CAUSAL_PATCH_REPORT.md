# Arc 9 Causal Patch — Swing Features Re-run

> Corrective dispatch triggered by external review finding producer-level
> lookahead in two D1 swing features used by the Candidate A / B LightGBM
> classifier (AUC 0.7508). The prior 8/8 GREEN lookahead audit checked
> join-level causality (merge_asof + days_lag + end-to-end probability
> reproduction) but did NOT check producer-level causal scope — whether
> the value in each joined row depends only on data with timestamp ≤ that
> row's date. Two features failed this dimension; the classifier was
> riding them for 27% of total gain.

## Headline

**STEP_4_KILL_AFTER_PATCH.** Patched LightGBM mean CV AUC drops from **0.7508 → 0.5190** (delta **−0.2318**), well below the §8 Pipeline E gate of 0.65. Phases 4 (Step 5 WFO) and 5 (scaled-risk) correctly skipped per the dispatch's hard rule. Arc 9 reverts to STEP_4_KILL; all leaked-classifier-derived results (Candidate A, Candidate B, scaled-risk 1.0% recommendation) are invalidated.

**The cohort is still real.** Step 3 capturability (cluster_0 individual, n=365, fwd_mfe_p50 6.18R, final_r_mean +4.423R, t-stat +21.81 at SL=2.0×ATR) used path-shape features only — no D1 features, no leak. Step 5 oracle (cluster 0 only, post-hoc filter, §11 Stepwise exit, 7/7 folds PASS-DEPLOYABLE) is unaffected because it doesn't use a classifier. The cohort exists and carries real economic edge under an oracle filter; it just **cannot be identified on causally-clean in-protocol features** at AUC ≥ 0.65.

## Causal patch summary

### Features patched

| Feature | Original implementation | Patched implementation |
|---|---|---|
| `d1_bars_since_swing_low` | Two-sided ±10-bar swing flag (`is_swing_low[k]` uses bars [k-10, k+10]) emitted per D1 row; only the last 10 D1 bars of the history were suppressed | Confirmed-swing-with-10-day-lag (Option A from dispatch): at date d, only swings at bars k ≤ d−10 are confirmed |
| `d1_bars_since_swing_high` | Same pattern as above | Same Option A patch |

Implementation: new function `_bars_since_confirmed_swing(is_swing, lag=10)` in `scripts/l_arc_9/experiments/causal_patch.py`. At index d, it walks forward to find swings as of d-lag, never using bars > d. The two-sided ±10-bar swing detection itself is unchanged (same flag computation); only the **confirmation rule** is added.

Causally clean by construction: `bars_since_confirmed_swing[d]` depends only on `is_swing[:d-lag+1]`, each of whose entries depends only on bars `[:d-lag+lag] = [:d]`.

The other 6 D1 features (`d1_atr14, d1_rsi_14, d1_kijun, d1_trend_state, d1_pos_in_20d_range, d1_close_above_kijun, d1_ret_5d_atr, d1_atr_ratio_to_4h`) are causally clean by construction (one-sided recursive smoothers / rolling-window maxes on [k-N+1, k] / scalar comparisons). Phase 2 verified this.

### Empirical evidence of the original leak

Random sample of 50 EUR_USD signals in the Arc 9 window 2021-2025:
- 24% of (sample, feature) pairs differ between full-pipeline and truncated-to-causal-data computations
- Median shift +25 bars (matches the analyst's external review note of +26 bars)
- Both swing features affected (`d1_bars_since_swing_low`: 15/50 differ; `d1_bars_since_swing_high`: 9/50 differ)

### Producer-level causal verification (NEW audit dimension)

100 stratified samples (70 midpoint Arc 9 window, 15 head first 60 days, 15 tail last 60 days), across 28 pairs. For each (sample, D1 feature) pair: compute the value via the full-pipeline patched producer, compute it via a TRUNCATED D1 history (bars with date ≤ signal_date − 1 day), compare.

| D1 feature | Samples | Match | Verdict |
|---|---|---|---|
| `d1_bars_since_swing_high` (PATCHED) | 100 | 100 (100.0%) | **PASS** |
| `d1_bars_since_swing_low` (PATCHED) | 100 | 100 (100.0%) | **PASS** |
| `d1_close_above_kijun` | 100 | 100 (100.0%) | PASS |
| `d1_pos_in_20d_range` | 100 | 100 (100.0%) | PASS |
| `d1_ret_5d_atr` | 100 | 100 (100.0%) | PASS |
| `d1_rsi_14` | 100 | 100 (100.0%) | PASS |
| `d1_trend_state` | 100 | 100 (100.0%) | PASS |

(`d1_atr_ratio_to_4h` is computed post-merge from `d1_atr14_lag1 / atr14_at_signal`; both components are causally clean — verified at the original join-level audit. Skipped here since it lives outside the D1 frame.)

**Producer-level causal scope: PASS on all 7 D1 features computed in the D1 frame.** This is the audit dimension that the original GREEN audit missed.

## Pipeline E retry — patched (Phase 3)

| Classifier | Features | Original (LEAKED) | Patched | Delta |
|---|---|---|---|---|
| RandomForest | 16 baseline | 0.5158 | **0.5158** | 0.0000 (unchanged — baseline cell is causal) |
| RandomForest | 28 expanded | 0.7759 | **0.5551** | **−0.2208** |
| LightGBM | 16 baseline | 0.5246 | **0.5246** | 0.0000 (unchanged) |
| **LightGBM** | **28 expanded** | **0.7508** | **0.5190** | **−0.2318** |

Per-fold LGBM expanded:

| Fold | Original | Patched | Delta |
|---|---|---|---|
| F1 | 0.848380 | 0.6155 | −0.2329 |
| F2 | 0.735746 | 0.4938 | −0.2419 |
| F3 | 0.717263 | 0.5585 | −0.1588 |
| F4 | 0.714402 | 0.4349 | −0.2795 |
| F5 | 0.738041 | 0.4924 | −0.2456 |
| Mean | 0.7508 | 0.5190 | **−0.2318** |
| Std | 0.0556 | 0.0721 | — |

The drop is consistent across all 5 folds: every fold loses 0.16-0.28 AUC. This is not a single-fold artifact — the leaked features were carrying real predictive power on the leaked data, distributed across the entire OOS span.

**Baseline cells are byte-identical to the original** (RF baseline 0.5158, LGBM baseline 0.5246) — confirms (a) the patch was scoped to D1 swing features only, (b) the other features (4H base + arc-specific structural + non-swing D1 + session) are unchanged.

**Feature expansion lift collapses.** Original LGBM expanded vs baseline: +0.226 AUC (the headline Pipeline E retry finding "feature expansion is the unlock"). Patched LGBM expanded vs baseline: **−0.0056 AUC** — feature expansion no longer helps. The +0.22 lift was almost entirely from the two leaked swing features.

### Patched feature importance (LGBM expanded full-data fit, top 10 by gain)

Per the new `phase_3_feature_importances_patched.csv`:
- `d1_bars_since_swing_low` (PATCHED): now no longer the dominant feature
- The two PATCHED swing features still appear in the importance list but at materially lower gain — the model finds them less informative once they're causal
- Other D1 features and the structural arc-specific features now share the importance mass roughly equally, with no clear winner

The structural reading: with the leak removed, the classifier has no concentrated signal axis. The remaining features individually carry weak predictive signal that doesn't aggregate to discrimination ≥ 0.65 AUC.

## Phase 4 — Step 5 LGBM E WFO with patched classifier

**SKIPPED** per dispatch hard rule: "If patched classifier AUC drops below §8 gate (0.65): report and halt. Do not skip to 'try a different classifier' or 'try different features.' The arc reverts to STEP_4_KILL until a fresh research effort proposes a new approach."

Patched LGBM AUC 0.5190 < gate 0.65. The reason for the skip: a classifier with AUC ~0.52 doesn't discriminate cluster 0 from the rest of the pool — running WFO with it would produce essentially random admission, no different in expectation from the raw baseline (which failed every §10 gate). No diagnostic value.

## Phase 5 — Scaled-risk

**SKIPPED** per dispatch hard rule: Phase 4 did not run; Candidate A doesn't exist as a deployable artefact under the patched classifier.

## What survives, what collapses

### Survives (unaffected by the leak)

- **Step 1 plumbing**: 2,153-trade pool, deterministic, audit-clean (live-execution semantics, real spreads, KH-24 D1 lag pattern). Unchanged.
- **Step 2 clustering**: K=3 selected, silhouette 0.4247, cluster 0 n=365 (17.0% of pool). Unchanged — path-shape features, no D1.
- **Step 3 capturability**: cluster_0_individual passes §2 floors at SL=2.0×ATR. mono_pp 0.562, frac_reach_1R 1.000, frac_wrong_way_pp 0.000, fwd_mfe_p50 6.18R, final_r_mean +4.423R, t-stat +21.81. Unchanged — path features only.
- **Step 5 oracle (cluster 0 only, §11 Stepwise exit)**: 7/7 folds PASS-DEPLOYABLE, worst-fold ann ROI +39.45%, worst DD 0.01%, full-data ann ROI +60.50%. Unchanged — uses post-hoc oracle filter (cluster membership directly, not a classifier).
- **Step 5 raw baseline (no filter, default exit)**: FAIL every §10 gate, worst-fold ann ROI −29.64%, full DD 62.99%. Unchanged — confirms the cohort identification is the entire system without a real filter.
- **Lookahead audit on its checked dimensions**: D1 lag join-level integrity (560/560 samples days_lag ≥ 1, Monday weekend handling correct), 4H feature timestamps (0/3,920 mismatches), label leakage, fold disjointness, execution semantics, end-to-end probability reproduction (100/100 within 4.82e-11) — all still PASS.

### Collapses (invalidated by the leak)

- **Pipeline E retry AUC 0.7508**: FAKE. Real patched AUC 0.5190. The "features over classifiers" narrative collapses — the feature expansion didn't unlock cluster 0 identifiability on causally-clean features. Classifier family change (RF → LGBM) was already refuted in the original; the patched result confirms classifier family is irrelevant when no signal exists.
- **Step 5 LGBM E Candidate A** (worst-fold ann ROI +9.63%, mean +22.92%, worst DD 1.32%, 236 admits): **INVALIDATED**. Built on the fake classifier. The 236-trade admit set was not selecting cluster-0-prone trades; it was selecting trades whose leaked swing values happened to coincide with future-favorable conditions.
- **Step 5 LGBM E Candidate B** (worst-fold +20.68%, mean +36.12%, worst DD 6.80%, 599 admits): **INVALIDATED**. Same reason.
- **Scaled-risk 1.0% deployment recommendation**: **INVALIDATED**. Built on the fake Candidate A admit set. The recommendation, the per-risk-level table, the worst-day cluster identification (2024-08-06 GBP_CHF + NZD_USD) — all built on leaked-classifier admissions.

### Methodology lesson NOT preserved

The "features over classifiers" lesson from Pipeline E retry **does not survive the patch**. The +0.23 AUC lift from feature expansion was almost entirely fake. With causally-clean features:
- LGBM expanded mean AUC 0.5190 vs LGBM baseline 0.5246: **−0.0056** (feature expansion actively slightly hurts under the causally-clean regime)
- RF expanded mean AUC 0.5551 vs RF baseline 0.5158: **+0.0393** (modest but well below §8 gate)

The cleanest reading: D1 swing-low / swing-high features carried no real predictive signal — they only appeared predictive because they were leaking. The honest extractability picture for Arc 9 is closer to Step 4 original: cluster 0 cannot be identified at entry from in-protocol causal features at AUC ≥ 0.65.

### v2.x amendment evidence

| Amendment | Original Arc 9 evidence | Status after patch |
|---|---|---|
| §3 grid relaxation (recall-floor + precision-gate for low-prev/mod-AUC) | Pipeline E retry: AUC 0.7508 clears gate but threshold sweep fails; Candidate B beats A by +11pp full-data ROI — empirical economic validation of grid relaxation | **WEAKENED.** The "AUC-clears-grid-fails" pattern that drove this amendment was demonstrated on a fake AUC. The patched classifier doesn't clear the AUC gate at all — there's no grid issue to relax around. The pattern still exists in Arc 7 D1 (calibration recovery dispatch confirmed) but Arc 9 evidence is gone. Arc 7 cross-test becomes the load-bearing remaining evidence. |
| §8 feature-budget expansion (D1-lagged context separate from entry-bar cap) | Feature expansion +0.26 AUC lift — strongest direct evidence for D1 context as a structural unlock | **COLLAPSED.** The lift was from leaked features. Causal D1 expansion delivers near-zero lift. Arc 9 provides no evidence for this amendment. |
| Step 5 fold-1 warmup convention | Data-window mismatch with KH-24 anchor (Arc 9 data starts at fold 1 OOS_start) | Still surfaced — but the Arc 9 Step 5 itself is invalidated so this evidence channel is moot for Arc 9. Pattern still applies to any future arc whose data window doesn't pre-date fold 1. |
| Step 4 artefact-on-FAIL persistence | Calibration recovery had to reproduce the classifier deterministically because the joblib wasn't saved | Still valid as a process improvement — independent of the leak. |
| **NEW: Producer-level causal scope as standard audit dimension** | This dispatch | The most important new amendment evidence: any future Pipeline E / D1 classifier audit must include per-feature producer-level causal verification as a distinct check, alongside join-level merge_asof and end-to-end probability reproduction. The lookahead audit template should be amended. |

### Cross-arc implications

The Pipeline E retry dispatch recommended testing the +D1 feature expansion on Arc 7's three surviving D1 units. **That recommendation is now refuted** — the expansion lift was a leak artifact, so applying it to Arc 7 would be reproducing the leak with a different signal. **Do not test it.**

The Arc 7 calibration recovery test recommended in earlier dispatches is still valid (it tests calibration on a known-real classifier with no leak issue). That test should proceed independently.

### Recommendation for Arc 9 disposition

**Arc 9 reverts to STEP_4_KILL**, with the formal closure to be re-written by the analyst incorporating:
- The leak finding and the audit-dimension gap
- The list of survives/collapses above
- The corrected v2.x amendment evidence (Arc 9 contributes nothing to §8 case; weakens §3 case to Arc 7 only)
- The "producer-level causal scope" audit dimension as a NEW standard for all future classifier work
- All five prior experiments + this patch as the full Arc 9 record

The cohort is still real. It is just unreachable on in-protocol causally-clean features. A future research effort proposing a different feature class (e.g. cross-pair regime context, intra-bar microstructure, longer-TF context, ensemble methods) could re-attempt extractability — but that would be a fresh dispatch, not a continuation of this one.

## Determinism

All Phase 3 LGBM training uses `random_state=42, deterministic=True, force_row_wise=True` (same as Pipeline E retry). Re-running the patch script produces byte-identical AUCs.

## Files

- `feature_matrix_patched.csv` — 28-feature matrix with PATCHED swing features (replaces leaked originals)
- `phase_2_producer_audit.csv` — 700 (sample × feature) producer-level causal checks
- `phase_2_per_feature_verdict.csv` — per-feature PASS/FAIL summary
- `phase_3_per_fold_aucs_patched.csv` — 4-cell × 5-fold AUC table with mean/std
- `phase_3_feature_importances_patched.csv` — LGBM expanded full-data fit importances
- `summary.json` — machine-readable summary with phase-by-phase verdicts
