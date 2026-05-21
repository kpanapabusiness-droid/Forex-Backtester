# Arc 9 Pipeline E Retry — LightGBM + expanded features

> Held-open experiment under v2.3. Tests three simultaneous changes vs Step 4
> original Pipeline E: H1 classifier family (RF → LightGBM), H2 feature set
> (+8 D1-lagged + 4 session = +12 features, 28 total), H3 pipeline architecture
> (Pipeline E only — no D1 deferred policy).

## Headline

**FAIL §8 — threshold sweep on locked grid {0.40, 0.50, 0.60, 0.70}, despite mean CV AUC 0.7508 PASSING the §8 Pipeline E gate (0.65) with margin +0.10.**

Pipeline E gate: **PASS** (LGBM expanded AUC 0.7508 / RF expanded AUC 0.7759, both ≥ 0.65). v2.2 §3 threshold sweep: **FAIL** (best recall on locked grid 0.340, well below 0.60 floor). The recall=0.60 contour for the LGBM classifier lives at threshold ≈ 0.05, with precision 0.337 (lift 1.95× over base rate 0.176). This extends the AUC-clears-grid-fails pattern from Arc 7 and Arc 9 D1 to Arc 9 Pipeline E — the third arc-unit demonstrating this same structural failure mode.

This is **not** a structural KILL of cluster 0 extractability. The cohort IS reachable on in-protocol features — D1 swing-low context is the unlock. The arc fails v2.2 §3 on grid alignment, not on classifier discrimination. The v2.x threshold-grid amendment case is materially strengthened.

## Hypothesis verdicts

| H | Hypothesis | Verdict |
|---|---|---|
| H1 | RF was the bottleneck — LightGBM lifts AUC by 0.03-0.06 on imbalanced sub-cohort tasks | **REFUTED.** LGBM vs RF on baseline features: +0.009 mean CV AUC (0.5246 vs 0.5158). On expanded features: −0.025 (LGBM 0.7508 vs RF 0.7759 — RF actually marginally better). Classifier family is not the bottleneck for this task. |
| H2 | Feature poverty was the bottleneck — D1 context + session features unlock discrimination | **STRONGLY CONFIRMED.** Same-classifier feature expansion lifts AUC by +0.260 (RF: 0.5158 → 0.7759) and +0.226 (LGBM: 0.5246 → 0.7508). D1 features dominate top importance (`d1_bars_since_swing_low` is top by 3× margin). |
| H3 | Pipeline E avoids the D1 rejected-pool drag — even modest classifier lift maps to clean WFO economics | **PARTIALLY CONFIRMED.** Pipeline E does avoid the D1 cost structure. But the threshold sweep gate still fails on the locked grid even at AUC 0.75 — the failure mode is grid alignment, not pipeline architecture. The substantive lift in cohort-identifiability matters; the gate-grid does not credit it. |

## Method

- **Trade pool:** 2,153 trades from Step 1 (full Arc 9 pool, post exposure cap, sorted chronologically by `entry_time`).
- **Label:** binary `is_cluster_0` from Step 2 (n_pos = 365 = 17.0% base rate).
- **Baseline 16 features (per Step 4):** 8 cross-dataset base (`body_to_range_ratio, upper_wick_ratio, lower_wick_ratio, range_to_atr_14, ret_5bar_atr, ret_20bar_atr, pos_in_20bar_range, rsi_14`) + 8 arc-specific structural (`n_swing_lows, most_recent_sl_lag, swing_low_dist_atr, mother_bar_range_atr, inside_bar_range_atr, ib_range_ratio, break_bar_body_atr, break_close_above_high_atr`). Path-shape leakage check: PASS (no `monotonicity_*`, `local_peaks_*`, `pullback_*`, `time_to_peak_*` in entry features).
- **Added 8 D1-lagged features:** `d1_trend_state, d1_atr_ratio_to_4h, d1_pos_in_20d_range, d1_ret_5d_atr, d1_rsi_14, d1_close_above_kijun, d1_bars_since_swing_high, d1_bars_since_swing_low`. **Lag pattern:** `lookup_date = normalize(signal_bar_time) - 1 day`; `merge_asof(lookup_date, d1_date, direction='backward')` — same engine pattern as KH-24's `_precompute_d1_exit_arrays` in `scripts/phase_kgl_v2_4h_wfo.py`. Each 4H bar sees the D1 bar from D-1, never same-day D1. D1 swing-high/low detection uses a confirmed-only flag (last 10 D1 bars marked unusable to avoid forward peek).
- **Added 4 session features:** `session_london` (hour 08-15 UTC), `session_ny_overlap` (hour 12-15 UTC), `hour_sin`, `hour_cos` (cyclic encoding, both axes per standard practice).
- **Total: 28 features**, well under the 50 hard cap.
- **CV:** `TimeSeriesSplit(n_splits=5)`, chronological order by `entry_time`. OOF predictions cover the last 5/6 of data (1,790 trades; first 358 trades have no OOF prediction by TimeSeriesSplit construction).
- **Classifiers:**
  - RF: `n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42` (same as Step 4).
  - LightGBM: `n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced', random_state=42, deterministic=True, force_row_wise=True` (dispatch config + determinism flags).
- **Determinism:** 2-run byte-identical on `feature_matrix.csv`, `per_fold_aucs.csv`, `threshold_sweep_locked.csv` — PASS.

## Classifier comparison

| Classifier | Feature set | Mean CV AUC | Std | Lift vs RF baseline 16 |
|---|---|---|---|---|
| RandomForest | 16 baseline | 0.5158 | 0.0731 | baseline |
| RandomForest | 28 expanded | **0.7759** | 0.0561 | **+0.260** |
| LightGBM | 16 baseline | 0.5246 | 0.0497 | +0.009 |
| **LightGBM** | **28 expanded** | **0.7508** | 0.0594 | **+0.235** |

The feature-set axis dominates the classifier-family axis by ~25× (+0.26 vs +0.01 lift). Cross-axis: on expanded features, RF marginally outperforms LGBM (+0.025) — both clear the §8 Pipeline E AUC gate (0.65) with margin.

## Per-fold AUC variance

| Fold | RF base 16 | RF expanded 28 | LGBM base 16 | LGBM expanded 28 |
|---|---|---|---|---|
| F1 (test 358) | 0.622 | 0.869 | 0.594 | 0.848 |
| F2 (test 358) | 0.495 | 0.761 | 0.514 | 0.736 |
| F3 (test 358) | 0.557 | 0.724 | 0.501 | 0.717 |
| F4 (test 358) | 0.447 | 0.745 | 0.464 | 0.714 |
| F5 (test 358) | 0.458 | 0.780 | 0.550 | 0.738 |
| Mean | 0.516 | 0.776 | 0.525 | 0.751 |
| Std | 0.073 | 0.056 | 0.050 | 0.059 |

Lift is robust across folds — every fold in both expanded variants is ≥ 0.71. F1 is somewhat higher (~0.85) than F2-F5 (~0.71-0.78) but the rest are tightly clustered. Not a single-fold artifact.

## Feature importance (LightGBM expanded, top 15 by gain)

| Rank | Feature | Gain | Splits | Origin |
|---|---|---|---|---|
| 1 | **d1_bars_since_swing_low** | 5796 | 502 | D1 |
| 2 | **d1_bars_since_swing_high** | 1961 | 566 | D1 |
| 3 | swing_low_dist_atr | 1589 | 597 | arc_specific |
| 4 | **d1_rsi_14** | 1257 | 538 | D1 |
| 5 | **d1_pos_in_20d_range** | 1243 | 422 | D1 |
| 6 | upper_wick_ratio | 1235 | 598 | base |
| 7 | ret_5bar_atr | 1197 | 621 | base |
| 8 | rsi_14 | 1161 | 445 | base |
| 9 | ret_20bar_atr | 1155 | 572 | base |
| 10 | **d1_ret_5d_atr** | 1140 | 478 | D1 |
| 11 | inside_bar_range_atr | 1120 | 602 | arc_specific |
| 12 | break_close_above_high_atr | 1107 | 535 | arc_specific |
| 13 | body_to_range_ratio | 1102 | 557 | base |
| 14 | **d1_atr_ratio_to_4h** | 1039 | 556 | D1 |
| 15 | range_to_atr_14 | 951 | 444 | base |

**Per-axis attribution:** 6 of top 15 are D1 features (top 2 are D1, top 5 includes 3 D1). `d1_bars_since_swing_low` is the dominant feature by gain — 3× the next-highest feature. Session features do not appear in top 15 (top sin/cos importances are mid-pack — they contribute little to discrimination). The signal-spec-derived structural features (swing-low distance, inside-bar range, break-above-high) appear in top 15 alongside D1 context but at materially lower importance than the dominant D1-swing-low timing feature.

**Reading of the result:** trades entered when D1 has recently established a swing-low (low `d1_bars_since_swing_low`) catch the early phase of a multi-day D1 uptrend — the smooth-path-shape cluster 0 behaviour. Trades entered far from a recent D1 swing-low fire later in the D1 cycle and face higher chop / whipsaw risk — they end up in cluster 1 (Early-peak hold) or cluster 2 (mixed). This is the structural relationship the bare-4H feature set was blind to and the D1 context exposes.

## Threshold sweep — v2.2 §3 locked grid (LGBM expanded OOF)

| Threshold | Precision | Recall | n admitted (of 1,790 OOF) | Pass v2.2 §3? |
|---|---|---|---|---|
| 0.40 | 0.469 | 0.340 | 228 | FAIL (recall < 0.60) |
| 0.50 | 0.503 | 0.295 | 185 | FAIL |
| 0.60 | 0.538 | 0.267 | 156 | FAIL |
| 0.70 | 0.529 | 0.200 | 119 | FAIL |

Best recall on the locked grid = 0.340 (at threshold 0.40). Floor = 0.60. **Margin = 0.26.** Per v2.2 §3 no max-F1 fallback — archetype fails Step 4 threshold sweep despite AUC clearing.

## Threshold sweep — extended (101 thresholds 0.00..1.00 step 0.01, diagnostic only)

| Threshold | Precision | Recall | n admitted | Note |
|---|---|---|---|---|
| 0.00 | 0.176 | 1.000 | 1,790 | trivial admit-all (base rate sanity) |
| 0.01 | 0.275 | 0.784 | 897 | |
| 0.02 | 0.292 | 0.705 | 760 | |
| 0.03 | 0.312 | 0.679 | 685 | |
| 0.04 | 0.329 | 0.654 | 626 | |
| **0.05** | **0.337** | **0.619** | **578** | **highest threshold with recall ≥ 0.60 (best precision while clearing floor)** |
| 0.10 | 0.378 | 0.540 | 450 | |
| 0.20 | 0.418 | 0.454 | 342 | |
| 0.30 | 0.463 | 0.400 | 272 | |
| 0.40 | 0.469 | 0.340 | 228 | locked grid floor |
| 0.50 | 0.503 | 0.295 | 185 | |

The recall=0.60 contour for this classifier sits at threshold ≈ 0.05 — eight grid steps below v2.2 §3's lowest threshold. At that operating point, precision is 0.337 (lift **1.92×** over base rate 0.176). Compare to Arc 9 D1 calibration recovery's extended sweep, where the same recall=0.60 contour sat at threshold ≈ 0.20 with precision 0.272 (lift 1.58×). The Pipeline E expanded classifier is materially better than D1 baseline at recall=0.60 (precision +0.065, lift 1.22× higher), but still ranks below the precision-floor a 17%-base-rate task typically needs to be deployment-meaningful.

## Comparison to prior Arc 9 experiments

| Experiment | Classifier | Features | AUC | Recall-0.60 precision | Pass §8? | Pass v2.2 §3? |
|---|---|---|---|---|---|---|
| Step 4 Pipeline E (original) | RF | 16 | 0.511 | — | NO | NO |
| Step 4 Pipeline D1 t=1 | RF | 23 (16 + 7 path-so-far) | 0.626 | 0.272 | YES (≥0.60 D1 floor) | NO (recall ≤ 0.003 on locked grid) |
| Calibration recovery D1 | RF + Platt/Iso | 23 | 0.626 | 0.272 | YES | NO (calibration cannot lift recall on locked grid) |
| **Pipeline E retry (this)** | **LGBM** | **28** | **0.751** | **0.337** | **YES (≥0.65 E floor)** | **NO (recall 0.340 at threshold 0.40)** |

Each iteration has lifted real classifier quality (AUC 0.51 → 0.63 D1 → 0.75 E expanded; recall-0.60 precision 0.27 D1 → 0.34 E expanded). At every iteration the v2.2 §3 locked-grid recall sweep has failed. The pattern is consistent: the {0.40, 0.50, 0.60, 0.70} grid does not align with the recall=0.60 contour for low-prevalence (17%) classification with AUC in [0.60, 0.80].

## Interpretation

**Feature expansion is the unlock for cluster 0 identifiability.** Adding 8 D1-lagged features + 4 session features to the 16-feature Step 4 baseline lifts mean CV AUC by +0.26 (RF) and +0.23 (LGBM), with `d1_bars_since_swing_low` carrying the largest single contribution (3× gain over the next feature). The structural story is intuitive: cluster 0 trades — the high-monotonicity, high-time-to-peak, smooth-path subset — concentrate at the early phase of D1 uptrends, identified at entry time by recency to a confirmed D1 swing-low. This information was not in the 4H bar features alone; it is in the D1 context. The signal-spec's hypothesis that compression geometry at the 4H bar would carry entry-time edge (refuted at Step 4) is replaced by a different and stronger hypothesis: D1 swing-low timing carries the edge, and the 4H entry trigger is the timing mechanism rather than the discriminator.

**Classifier family did not contribute.** LGBM vs RF on the same features differs by only +0.009 (baseline) / −0.025 (expanded) AUC. The dispatch's H1 (RF was the bottleneck) is refuted. The cross-arc recommendation it implied (re-test Arc 7 D1 units with LGBM) does NOT trigger from this evidence — LightGBM is not a general unlock for the AUC ≈ 0.60 regime arcs; feature-set expansion is. The reframed cross-arc question is: would Arc 7's three surviving D1 units benefit from a similar +8 D1 + 4 session expansion? This is testable and worth flagging for analyst review.

**The v2.2 §3 locked grid is not aligned with the recall=0.60 contour for low-prevalence + moderate-AUC tasks.** Three Arc 9 experiments now (Step 4 D1 original, calibration recovery, this Pipeline E retry) plus Arc 7's three D1 units exhibit the same surface failure: classifier AUC clears the relevant pipeline gate (0.60 D1 or 0.65 E), but recall on {0.40, 0.50, 0.60, 0.70} caps below 0.60 because the probability outputs concentrate well below 0.40. Calibration does not fix this (calibration recovery OUTCOME_B established this for the rank-bound case). For a 17% base-rate task, AUC 0.75 with locked-grid best-recall 0.34 / precision 0.47 corresponds to a usable filter at lift 2.7× over base rate — but the gate as drawn rejects it. v2.x amendment scope: relax the grid for low-prevalence positive classes (e.g. base-rate-anchored grid: thresholds at {base, 2×base, 3×base, 4×base}), or add a parallel recall-floor evaluation at sub-grid thresholds with explicit precision floor. The empirical case for either amendment is now triplicate.

**Arc 9 disposition remains held-open per dispatcher.** The dispatch said "If headline is `PASS §8 + threshold sweep clears` → Step 5 WFO dispatch; if any FAIL → analyst writes closure consolidating four experiments + structural conclusion". This headline is FAIL_THRESHOLD_SWEEP, so analyst writes the closure. The structural conclusion to consolidate is materially different from what would have been written after Step 4 alone: cluster 0 IS identifiable from in-protocol features (D1 context unlocks it); the binding constraint is the v2.2 §3 grid alignment, not feature poverty (refuted), classifier family (refuted), pipeline architecture (Pipeline E doesn't help the grid), or cohort quality (Step 5 oracle showed +60% ann ROI under perfect filter).

## Cross-arc note

The dispatch's cross-arc condition for LightGBM re-test on Arc 7 ("If LightGBM lifts AUC by ≥ 0.05 over RF on same features → recommend re-testing Arc 7's three D1 units") does NOT trigger — LGBM lift over RF on baseline 16 was +0.009, far below 0.05.

**The unlocked cross-arc question that does trigger:** Arc 7's three surviving D1 units (cluster 1 D1, cluster 3 D1, agg c1+c3 D1) were tested at Step 4 with the baseline 8-base + arc-specific feature set. If Arc 9's pattern generalises, expanding their feature set with the same 8 D1-lagged + 4 session features (same merge_asof lag pattern) is likely to lift their classifier AUC substantially. This is a cheap test (single script, swap input paths + feature catalogue) and would either confirm the cross-arc finding or refute it. Recommend the analyst include this as the third leg of the "two-arc D1 mis-calibration pattern" investigation alongside the Arc 7 calibration test already recommended.

## Files

- `feature_matrix.csv` — per-trade 28-feature + label matrix (n=2153 after NaN drop)
- `per_fold_aucs.csv` — 4 cells × 5 folds + mean + std
- `threshold_sweep_locked.csv` — v2.2 §3 grid on LGBM expanded OOF
- `threshold_sweep_extended.csv` — 101 thresholds 0..1 step 0.01 on LGBM expanded OOF
- `feature_importances.csv` — LGBM expanded full-data fit, all 28 features with gain + split + origin
- `determinism_check.json` — 2-run byte-identical PASS
- `summary.json` — verdict + per-cell metrics + chosen threshold (None)
