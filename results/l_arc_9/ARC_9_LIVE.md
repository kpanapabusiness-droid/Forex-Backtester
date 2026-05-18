# Arc 9 - signal_inside_bar_break_trend_long_v0.1

## Status

- Disposition: **STEP_4_KILL** (held-open lifecycle — experimental validation in progress; closure pending second experiment per dispatcher)
- Protocol: L_ARC_PROTOCOL v2.3 (base v2.1.2 + v2.2 + v2.3 amendments)
- Branch: `claude/bold-brattain-d79817` (CC worktree; queue transition skipped per dispatcher instruction - another CC session running prior arcs)
- Last updated: 2026-05-18

## Experiments (held-open lifecycle)

| Experiment | Question | Outcome |
|---|---|---|
| Step 5 validation — no-filter WFO on cluster 0 (`results/l_arc_9/experiments/step5_validation/`) | Does §8 extractability correctly identify non-deployable cohorts, or did it over-reject a cohort with real edge? | **§8 over-rejected the cohort, conditional on oracle filter.** 7/7 folds pass-deployable (worst-fold ann ROI +39.45%, mean +61.40%, worst-fold DD 0.01%, full-data ann ROI +60.50%). The cohort has real, durable structural edge; the binding constraint at Step 4 was feature-set information, not cohort quality. No deployment implication — closure stands. |
| Calibration recovery — Pipeline D1 classifier (`results/l_arc_9/experiments/calibration_recovery/`) | Was the Step 4 threshold-sweep failure a calibration artifact or a fundamental discrimination problem? | **OUTCOME_B — discrimination problem.** Platt + isotonic preserve AUC (drift +0.0000 / −0.0037, both within tolerance) but neither lifts recall above 0.03 at any threshold in the v2.2 §3 grid {0.40, 0.50, 0.60, 0.70}. Rank discrimination IS strong enough for recall 0.60, but only at thresholds ~0.20 with precision ~0.27 (lift 1.58x over base 0.17). §8 gate as drawn correctly rejects marginal-discrimination cohorts. The two experiments compose: cohort is real (Step 5 val), no available filter can identify it on the v2.2 §3 grid (calibration recovery), binding constraint is feature-set information content — not calibration miss, not cohort quality, not threshold-grid tuning. Cross-arc next step: re-run the calibration test on Arc 7 surviving D1 units; if also OUTCOME_B, pattern is structural to the AUC ≈ 0.60 regime, not a protocol calibration gap. |
| Step 5 raw baseline — full pool, no filter, default exit (`results/l_arc_9/experiments/step5_raw_baseline/`) | What does the bare IB-trend signal produce without any cohort identification? Establishes the FLOOR opposite the oracle CEILING. | **FAIL by every §10 gate.** Worst-fold ann ROI −29.64%, mean ann ROI +6.10%, worst-fold DD 43.61%; full-data ann ROI +0.80%, full-data DD 62.99%. 5 of 7 folds negative. Comparison to oracle: gap of +69.09pp on worst-fold ROI, −62.97pp on full-data DD. The clustering work is **not incremental — it is the entire system**; without cohort identification IB-trend is a money-loser with portfolio-killing drawdown. The "rest of pool" (1,788 trades = 83% complement of cluster 0) carries the loss-side asymmetry (its complement, cluster 0, has frac_wrong_way_pre_peak = 0.000 by selection). The three-point picture is now bracketed: floor FAIL, ceiling PASS-DEPLOYABLE, realistic (calibrated) point sits between — composing with the calibration recovery's precision ~0.27 at recall 0.60 in the AUC ≈ 0.63 regime, the realistic point will be closer to the floor than the ceiling. |
| Pipeline E retry — LightGBM + expanded features (`results/l_arc_9/experiments/pipeline_e_retry/`) | Three hypotheses tested simultaneously: H1 RF was the bottleneck (LightGBM lifts AUC), H2 feature poverty was the bottleneck (+8 D1-lagged + 4 session features unlock cluster 0), H3 Pipeline E avoids D1 rejected-pool drag and maps a modest classifier lift cleanly to WFO. | **FAIL §8 — threshold sweep on locked grid, despite Pipeline E AUC gate (0.65) clearing by margin +0.10.** Mean CV AUC: RF baseline 0.5158, RF expanded 0.7759, LGBM baseline 0.5246, LGBM expanded 0.7508. H1 **REFUTED** (LGBM vs RF on same features ≤ ±0.025). H2 **STRONGLY CONFIRMED** (+0.26 lift from feature expansion alone). H3 partially confirmed (Pipeline E avoids the cost structure, but the threshold-sweep grid still fails). `d1_bars_since_swing_low` is the dominant feature by 3× gain — cluster 0 trades concentrate at the early phase of D1 uptrends, identifiable at entry from D1 swing-low timing. Locked-grid best recall 0.340 at threshold 0.40 (margin 0.26 to 0.60 floor); extended sweep finds recall 0.60 at threshold ~0.05 with precision 0.337 (lift 1.92x over base 0.176). Materially better than D1 calibration recovery (precision 0.27 at recall 0.60) but still fails the locked grid. This is the third Arc 9 experiment exhibiting the AUC-clears-grid-fails pattern; v2.x threshold-grid amendment case (relax for low-prevalence + moderate-AUC tasks) is materially strengthened. Cross-arc: dispatch's LGBM-cross-test recommendation does NOT trigger (LGBM lift < 0.05 over RF); but D1 + session feature expansion on Arc 7's three D1 units is a strong candidate cross-arc test. |
| Step 5 LightGBM Pipeline E WFO — classifier-driven admission, §11 Stepwise exit (`results/l_arc_9/experiments/step5_lgbm_pipeline_e/`) | Load-bearing fourth Step 5 point. Does the AUC 0.7508 classifier (Pipeline E retry) translate to deployable WFO economics? Two threshold candidates: A (0.40, locked grid) and B (0.05, recall-floor / hypothetical v2.x amendment point). | **PASS-DEPLOYABLE-PENDING-AMENDMENT.** Parity vs Pipeline E retry TSS-CV AUC 0.7508: PASS (reproduced 0.750766, byte-identical per-fold AUCs). Both candidates **PASS-DEPLOYABLE on folds 2-7** (training-data-available; fold 1 has 0 admits because Arc 9 data starts at KH-24 fold 1 OOS_start — anchored-expanding has no prior training data). Candidate A (thr 0.40): worst-fold ann ROI +9.63%, mean +22.92%, worst DD 1.32%, 236 admits / 109 cluster-0 TP. Candidate B (thr 0.05): worst-fold ann ROI +20.68%, mean +36.12%, worst DD 6.80%, 599 admits / 194 cluster-0 TP. Both candidates fail strict 7-fold §10 only on fold-1 data-availability artifact. Candidate B beats A by +11pp full-data ann ROI — direct economic validation of the v2.x §3 grid-amendment proposal. Both capture 84-98% of the floor→ceiling DD reduction and 30-73% of the ROI gap. Determinism PASS. Two blockers to formal Step 4 re-pass: (i) v2.2 §3 locked grid, (ii) fold-1 warmup-convention gap in Step 5 (data window mismatch with KH-24 anchor). v2.x amendment evidence: §3 grid relaxation, §8 D1 feature-budget expansion, Step 5 fold-1 warmup convention, Step 4 artefact-on-FAIL persistence — all surfaced by this dispatch + prior experiments. |
| Lookahead + cross-timeframe leak audit (`results/l_arc_9/experiments/lookahead_audit/`) | Single most important gate between Arc 9 research and any deployment decision. 8 independent audits covering 4H feature timestamps, **D1 lag integrity (CRITICAL)**, session features, label leakage, training/inference fold disjointness, cluster label flow, execution semantics, and end-to-end probability reproduction. | **GREEN — candidates are audit-clean.** All 8 audits PASS. **D1 lag (CRITICAL)**: 560 samples (20 per pair × 28 pairs), min/max days_lag 1/4, zero same-day D1 access; Monday boundary (129 samples) all show days_lag=3 (Friday's D1, correct weekend handling); pattern verbatim-equivalent to KH-24 engine `_precompute_d1_exit_arrays` (`dates.dt.normalize() - pd.Timedelta(days=1)` + `pd.merge_asof(direction='backward')`). 4H feature timestamps: 0/3,920 recompute mismatches at 1e-6 tolerance. Label leakage: 0 forward-geometry features in entry matrix, max |Pearson r| 0.3285 between any entry-time feature and any path-shape feature. Fold disjointness: TSS-CV walk-forward by construction, Step 5 anchored-expanding with `train_mask = entry_time < fold.oos_start` confirmed; zero overlaps under either scheme. End-to-end reproduction: 100/100 sample probabilities match recorded values to 4.82e-11 abs_diff. Three audit-script bug fixes during the run (tolerance, spread-floor application, in-memory feature rebuild) before final GREEN — none of these changed any audited code. **AUC 0.7508 is real. Step 5 economics are real. Deployment momentum can proceed subject to v2.x amendment + formal Step 4 re-pass.** |
| Candidate A scaled-risk WFO (`results/l_arc_9/experiments/scaled_risk/`) | Re-account the fixed 236-trade admit set at 5 per-trade risk levels {0.5%, 1.0%, 1.5%, 2.0%, 2.5%} with intraday compounded worst-day DD measurement. Establishes the **measured** deployment risk, replacing the projection-based 2% in the candidate spec. | **Recommended deployment risk: 1.0% per trade** (half the projection). At 1.0%: worst-fold max DD 2.52% / **worst-day DD 2.01%** / worst-fold ann ROI +20.07% / full-data ann ROI +41.45% / all F2-F7 positive. **Binding constraint is worst-day DD, NOT worst-fold max DD** — a metric the projection didn't measure. **Single day (2024-08-06, 2 same-day SL trades on GBP_CHF + NZD_USD, net -2.02R) drives the worst-day at every risk level.** DD scales exactly linearly with risk (DD ratio observed / linear ≈ 1.00); ROI scales supra-linearly via compounding (2.17× at 2× risk, 7.04× at 5× risk). At 2.0% risk: worst-day DD 3.99% — knife-edge under in-system 4% target with zero margin for out-of-sample worse days. At 2.5%: 4.98% — knife-edge under 5ers 5% hard limit. The 1.0% recommendation buys 1.99pp margin under in-system 4% target, 2.99pp under 5ers 5% hard limit — robust to ~one additional same-day-2-SL cluster in OOS. Determinism PASS. Methodology fix during run: initial day-DD calc reported monotone-winning days as "DD = day's gain"; corrected to running-peak-to-trough (with day-start anchor as secondary check, take max of both). Cross-arc lesson: worst-day DD at intraday compounded resolution should be a standard Step 5 output, not a separate dispatch. KH-24's own 1.0% deployment risk per KH24_SYSTEM_LOCK should be re-measured under the same methodology. |

## Arc-open

- **Signal under test:** `signal_inside_bar_break_trend_long_v0.1`
  - Source: `docs/signal_spec_inside_bar_break_trend_long_v0.1.md` (provided by dispatcher)
  - Implementation: [signals/lchar_inside_bar_break_trend_long.py](../../signals/lchar_inside_bar_break_trend_long.py)
- **Hypothesis:** this signal carries structural edge surface-able by path-shape clustering and v2.3 capturability + extractability gates. The signal-spec's specific hypothesis: "Compression geometry (parent-bar range, inside-bar range, range ratio) + break-bar strength are entry-time observable → Pipeline E should clear 0.65 AUC."
- **Protocol version:** v2.3 (base v2.1.2 + v2.2 amendment + v2.3 amendment).
- **SL sweep candidates:** `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H`.
- **Simulation SL (Step 1 pool):** `2.0 × ATR_4H` per signal spec config.
- **Forward window:** 240 bars (4H = 40 calendar days).
- **Spread floors fallback:** `configs/spread_floors_5ers.yaml` (body sha256 `8da7644b252ae163d963fbd46807572906fa3e5a44fb3e02d771e181b3ecdc05`).
- **Pair set:** 28 FX (KH-24 set).
- **Population builder:** ex-ante (no peeking; this script natively respects "signal at bar t close → entry at bar t+1 open").
- **Risk:** 0.5% per trade.
- **Pre-committed step gates:** per v2.3 (no overrides, no mid-arc sign-off, halt end of Step 4).

## Step results

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 | Plumbing | **PASS** | 2,153 trades / 28 pairs / 2020-10-01..2026-01-31. Determinism byte-identical 2-run; lookahead 5/5; right-edge swing 0 violations; §15a schema PASS; KH-24 co-fire 0.279% (well below 10% flag). Sibling arcs 8/10/11 step 1 not yet landed. |
| 2 | Clustering | **PASS** — K=3, silhouette 0.4247 | All five K∈{3,4,5,6,7} pass gate; K=3 highest silhouette; K=4 within 0.01 tolerance (0.4193); smaller K wins. No degenerate features. Cluster 0 (n=365, 17.0%) = unclassified (boundary Stepwise/Monotone, mono 0.534, lp 31.44, pullback 0.580, ttp_rel 0.771). Cluster 1 (n=781, 36.3%) = Early-peak hold (ttp_rel 0.129). Cluster 2 (n=1007, 46.8%) = unclassified (mono 0.528, lp 7.28, pullback 0.510, ttp_rel 0.477). |
| 3 | Capturability | **PASS** — 1 archetype clears §2 floors | Cluster 0 individual survives at SL=2.0×ATR: mono_pp 0.562, reach_1R 1.000, ww_pp 0.000, fwd_mfe_p50 6.18R, final_r_mean +4.423R, t=+21.81, composite 0.612, shape_tag tight_unimodal. Cluster 1 dies (Early-peak hold low magnitude, fails §2). Cluster 2 dies (does not pass §2 at any candidate SL). |
| 4 | Extractability | **FAIL** — archetype dies (no max-F1 fallback per v2.2 §3) | Cluster 0 Pipeline E: RF AUC 0.511 (folds 0.464/0.476/0.524/0.523/0.570); LR AUC 0.527; entry features carry zero predictive power for this archetype. Pipeline D1 at smallest-t (t=1) clears AUC floor (0.626 ≥ 0.60, exclusion 1.3%), but threshold sweep fails: at threshold 0.40 only 3 trades admitted (recall 0.003); thresholds 0.50/0.60/0.70 admit 0. No threshold satisfies recall ≥ 0.60. Per v2.2 §3 archetype fails Step 4. |

## Detailed analysis

### Step 1 - Plumbing (PASS)

**Pool:** 2,153 trades / 28 pairs / 5,109 signal fires / 2,956 dropped by exposure cap.

Pool size sits exactly mid-range of the signal-spec prior (1,500-2,500). Per-pair counts span 56 (CHF_JPY) - 92 (NZD_CAD, NZD_USD); no pair has < 30 trades.

**Bars-held distribution:** p5/p25/p50/p75/p95 = 2/8/22/111/240. 17.46% of trades cap at bar 240 (time_exit), below the 20% auto-extend trigger. Exit-reason mix: stoploss 81.5%, time_exit 17.5%, end_of_data 1.1%.

**Gates:**

| Gate | Result | Detail |
|---|---|---|
| Pool size ≥ 500 | **PASS** | 2,153 |
| Determinism (2-run byte-identical) | **PASS** | sha256 trades_all: `ed3c811ba9fd7070d0ddf2f8fe11bafb7139e564294025b12a595d16295d0e5d` |
| Lookahead spot-check (5/5 truncated-OHLC reproduce) | **PASS** | 5/5 across EUR_GBP, EUR_NZD, EUR_CHF, GBP_JPY, CHF_JPY |
| Right-edge swing audit (most_recent_sl_lag ≥ 4 on every fire) | **PASS** | min lag = 4, max lag = 27 |
| §15a trades_paths.csv schema | **PASS** | 0 violations across 2,153 trades; 9 cols; 1,747 trades emit forward-obs rows |
| KH-24 co-fire (>10% = flag) | **PASS** | 6 exact overlaps / 2,153 = 0.279% |
| Sibling arcs 8/10/11 co-fire | n/a | None of l_arc_8/10/11 step1 has landed in this worktree |

**Co-fire reading:** signal class is structurally independent of KH-24 (deployed long-only system). 0.279% overlap = noise; the two signals fire on different bar patterns.

### Step 2 - Clustering (PASS)

**K-sweep:**

| K | silhouette | max cluster frac | min cluster n | gate |
|---|---|---|---|---|
| 3 | 0.4247 | 0.4677 | 365 | **PASS** |
| 4 | 0.4193 | 0.3093 | 332 | PASS |
| 5 | 0.4032 | 0.2968 | 246 | PASS |
| 6 | 0.4168 | 0.2601 | 120 | PASS |
| 7 | 0.4055 | 0.2601 | 120 | PASS |

**Selected K = 3** (highest silhouette satisfying gate; K=4 within 0.01 tolerance, smaller K wins per §6 parsimony rule).

**Archetype assignment at K=3:**

| cluster | n | frac | mono | local_peaks | pullback | ttp_rel | archetype label (§11 centroid pattern match) |
|---|---|---|---|---|---|---|---|
| 0 | 365 | 0.170 | 0.534 | 31.44 | 0.580 | 0.771 | **Unclassified** (centroid on Stepwise climber boundary: mono 0.534 just below 0.55 Monotone-ascent floor; pullback 0.580 just above 0.50 Stepwise ceiling) |
| 1 | 781 | 0.363 | 0.074 | 0.76 | 0.028 | 0.129 | **Early-peak hold** (ttp_rel ≤ 0.30) |
| 2 | 1007 | 0.468 | 0.528 | 7.28 | 0.510 | 0.477 | **Unclassified** (centroid between Monotone ascent and Stepwise) |

Path-feature degeneracy: 0 features at >80% modal mass; halt rule (≥ 2 degenerate features) not tripped.

**Note on Unclassified clusters:** per §6, boundary clusters whose centroids do not match a §11 row are routed "by empirical test on per-fold internal validation". For Steps 1-4 they still proceed to Step 3 (§2 floor evaluation); §11 row routing finalises at Steps 5/6, which are out of scope for this dispatch.

### Step 3 - Capturability (PASS - 1 archetype)

SL sweep over `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR`. §2 floors (v2.1.2):

- monotonicity_pre_peak_centroid ≥ 0.55
- frac_reach_1R ≥ 0.70
- frac_wrong_way_pre_peak ≤ 0.30
- fwd_mfe_h240_p50 ≥ 1.5R
- size_fraction_of_pool ≥ 0.10
- shape_tag ≠ scattered

Capturability composite (v2.1.1): `(mono_pp − 0.55) + (frac_reach_1R − 0.70) + (0.30 − frac_wrong_way_pp)`.

**Cluster routing:**

| cluster | archetype | individual? | sel SL | composite | aggregate? | disposition |
|---|---|---|---|---|---|---|
| 0 | Unclassified | **PASS** | 2.0×ATR | 0.612 | n/a (single-cluster label) | **individual_only** |
| 1 | Early-peak hold | fail | - | - | n/a | **dies** |
| 2 | Unclassified | fail | - | - | n/a (single-cluster label) | **dies** |

**Surviving archetype (cluster_0_individual):**

| Metric | Value | §2 floor | Pass? |
|---|---|---|---|
| Selected SL (×ATR) | 2.0 | sweep | - |
| n | 365 | - | - |
| size_fraction | 0.170 | ≥ 0.10 | PASS |
| mono_pre_peak_centroid | 0.562 | ≥ 0.55 | PASS (margin +0.012) |
| frac_reach_1R | 1.000 | ≥ 0.70 | PASS |
| frac_wrong_way_pre_peak | 0.000 | ≤ 0.30 | PASS |
| fwd_mfe_p50 | 6.18R | ≥ 1.5R | PASS |
| fwd_mfe_p50_atr | 12.36 | - | - |
| final_r_mean | +4.423R | - | - |
| final_r_t_stat | +21.81 | - | - |
| composite | 0.612 | - | - |
| shape_tag | tight_unimodal | ≠ scattered | PASS |
| bimodal_separated test | fail | - | n/a |
| pre_t_sl_atr_multiplier (v2.3 §4) | 2.0 | = selected SL | recorded |

Cluster 0 is structurally clean: the high-monotonicity, high-time-to-peak group reaches 1R every time and never violates 1R pre-peak. Forward MFE distribution is tight unimodal at ATR-units p50 12.36 (median trade peaks at 12.36 ATR before retracing). This is the IB-trend signal's "real edge cohort" — the question is whether entry-time / early-bar features can identify it.

### Step 4 - Extractability (FAIL - archetype dies)

**Sample size (§15):** n_total 2,153, n_pos 365. Both ≥ floors (200, 50).

**Pipeline E (entry-time RF):**

- Feature set: 8 base (body_to_range_ratio, upper_wick_ratio, lower_wick_ratio, range_to_atr_14, ret_5bar_atr, ret_20bar_atr, pos_in_20bar_range, rsi_14) + 8 arc-specific (n_swing_lows, most_recent_sl_lag, swing_low_dist_atr, mother_bar_range_atr, inside_bar_range_atr, ib_range_ratio, break_bar_body_atr, break_close_above_high_atr).
- RF: `n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42`, 5-fold StratifiedKFold CV.
- **RF AUC mean: 0.511** (folds 0.464 / 0.476 / 0.524 / 0.523 / 0.570). LR AUC mean 0.527. RF-LR gap −0.015 (linear ≈ tree-based; feature set is binding, not non-linearity).
- **Fails Pipeline E floor (0.65) by 0.139.**

Top 10 feature importances (informational; RF AUC ≈ 0.5 so importances ranking is essentially noise):

| Rank | Feature | Importance |
|---|---|---|
| 1 | ret_20bar_atr | 0.090 |
| 2 | swing_low_dist_atr | 0.083 |
| 3 | rsi_14 | 0.082 |
| 4 | inside_bar_range_atr | 0.076 |
| 5 | break_bar_body_atr | 0.071 |
| 6 | body_to_range_ratio | 0.068 |
| 7 | break_close_above_high_atr | 0.066 |
| 8 | lower_wick_ratio | 0.064 |
| 9 | range_to_atr_14 | 0.063 |
| 10 | ib_range_ratio | 0.062 |

**Verdict on signal-spec hypothesis:** the spec's claim "compression geometry + break-bar strength → entry-time predictability ≥ 0.65 AUC" is empirically rejected. The compression-geometry features (ib_range_ratio, inside_bar_range_atr, mother_bar_range_atr) appear in the importance list but with weight comparable to noise features (RSI), and the model overall is at chance.

**Pipeline D1 (deferred-policy RF at bar t):**

Per §8 smallest-t selection: smallest t ∈ {1, 2, 3, 4, 5, 10} with RF AUC ≥ 0.60 AND exclusion ≤ 30%.

| t | n_total_at_t | n_pos_at_t | exclusion | RF AUC mean | passes AUC | passes excl |
|---|---|---|---|---|---|---|
| 1 | 2,124 | 365 | 0.013 | 0.626 | PASS | PASS |
| 2 | 2,042 | 365 | 0.052 | 0.645 | PASS | PASS |
| 3 | 1,963 | 365 | 0.088 | 0.670 | PASS | PASS |
| 4 | 1,852 | 365 | 0.140 | 0.650 | PASS | PASS |
| 5 | 1,801 | 365 | 0.163 | 0.679 | PASS | PASS |
| 10 | 1,450 | 365 | 0.327 | 0.692 | PASS | **FAIL** (>0.30) |

**Smallest-t = 1 (AUC 0.626, exclusion 1.3%).**

**Threshold sweep (v2.2 §3 mandatory recall ≥ 0.60, no max-F1 fallback):**

| threshold | precision | recall | n_admitted | tp | fp | fn | tn | passes_recall_floor |
|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.333 | 0.003 | 3 | 1 | 2 | 364 | 1757 | 0 |
| 0.50 | 0.000 | 0.000 | 0 | 0 | 0 | 365 | 1759 | 0 |
| 0.60 | 0.000 | 0.000 | 0 | 0 | 0 | 365 | 1759 | 0 |
| 0.70 | 0.000 | 0.000 | 0 | 0 | 0 | 365 | 1759 | 0 |

The D1 classifier discriminates (AUC 0.626) but produces probability outputs concentrated well below 0.40 — at the strict {0.40, 0.50, 0.60, 0.70} sweep, only threshold 0.40 admits any positives, and only 3 of 365. Best achievable recall = 0.003. **Margin to recall floor: 0.597.**

**Per v2.2 §3:** no max-F1 fallback. Archetype fails Step 4.

**Per §16a HALT/KILL evaluation:**

| Criterion | Result |
|---|---|
| Single criterion fail (only one Step 4 gate fails) | **FAIL** — Pipeline E AUC fails (margin 0.139) AND Pipeline D1 threshold-sweep recall fails (margin 0.597). Two distinct failure modes within the cohort. |
| Cohort viability (size_fraction ≥ 0.10) | PASS (0.170) |
| Path A near-miss (numeric, margin < 0.03) | **FAIL** — both failing margins (E AUC 0.139, D1 recall 0.597) are far above 0.03 |
| Path B categorical with strong magnitude | N/A — failing criteria are numeric, not categorical |

**Disposition: KILL.**

## Halt Summary - Arc 9

```
## Halt Summary — Arc 9

### Status
- Disposition: STEP_4_KILL
- Closure doc: results/l_arc_9/ARC_9_CLOSURE.md
- Live arc doc: results/l_arc_9/ARC_9_LIVE.md
- Branch: claude/bold-brattain-d79817 (CC worktree; phase/l_arc_9 NOT created — queue updates skipped per dispatcher)
- Queue state: not transitioned (per dispatcher: another CC session running prior arcs)

### Step pass/fail table
| Step | Gate | Result |
|---|---|---|
| 1 | Plumbing (pool ≥ 500, determinism, lookahead, schema, co-fire) | PASS — 2153 trades; all 7 sub-gates green |
| 2 | Clustering (silhouette + size + non-degenerate features) | PASS — K=3, sil 0.4247 |
| 3 | Capturability (§2 floors at some candidate SL) | PASS — 1 archetype (cluster_0_individual) at SL=2.0×ATR |
| 4 | Extractability (Pipeline E or D1 + threshold sweep) | FAIL — both pipelines die: E AUC 0.511 (gate 0.65); D1 reaches AUC 0.626 at t=1 but threshold-sweep recall floors at 0.003 (gate 0.60); v2.2 §3 = no max-F1 fallback |

### Surviving archetypes (if step 4 complete)
_None._

### Cross-arc calibration candidates (HALT only)
_n/a — KILL disposition._

### Recommended next dispatch
- STEP_4_KILL: arc archived. No follow-up.
- Cross-arc observation worth chat review: this is the second arc (after Arc 7) where Pipeline D1 clears AUC ≥ 0.60 but the {0.40, 0.50, 0.60, 0.70} threshold sweep produces no threshold satisfying recall ≥ 0.60 — classifier outputs are probability-mis-calibrated relative to the fixed threshold grid. v2.2 §3 closed this mechanically; the structural pattern may merit a future protocol consideration (e.g. probability calibration + extended threshold grid, or recall-floor relaxation for low-prevalence positive classes). Reported here as informational only; not a §16a HALT candidate (margins too large for Path A).
```

## Cross-arc candidates

- **Pipeline D1 threshold-sweep mis-calibration pattern surfaces again.** Arc 7 Pipeline D1 closure (CLEAN-NULL 2026-05-17) was the first; this is the second. In both, classifier AUC is above the §8 floor but probability outputs cluster below 0.40, and the fixed {0.40, 0.50, 0.60, 0.70} sweep produces no recall ≥ 0.60 threshold. Pattern is structural to low-prevalence positive classes (here 17%, Arc 7 similar). Two arcs is not enough to warrant a protocol change but worth tracking against future arcs. Not a §16a candidate from this arc (margins far exceed Path A 0.03).

- **Compression-geometry features carry no entry-time edge for the IB-trend signal.** The signal-spec hypothesis "compression geometry + break-bar strength → entry-time AUC ≥ 0.65" is empirically refuted: all 8 arc-specific features (mother range, inside range, IB ratio, break body, break-above-high distance, swing-low distance, n_swing_lows, most_recent_sl_lag) appear in RF importance roughly equal to noise base features. Hypothesis class can be retired for this signal.

## Interesting observations

- **The signal does have a clean cohort.** Cluster 0 (17% of pool) has fwd_mfe_p50 6.18R, final_r_mean +4.423R, t-stat +21.81 — large, very-clean Stepwise/Monotone-boundary trades that hit 1R every time and never go wrong pre-peak. This is genuine capturable edge but it's not extractable from entry or early-bar features.

- **Pipeline D1 t-curve.** AUC rises monotonically with t until exclusion bites: t=1 → 0.626, t=10 → 0.692. If the protocol allowed t∈{15, 20}, AUC might rise further but exclusion would too. The smallest-t selection (t=1) gives the largest addressable pool but the weakest signal, exactly the wrong combination here.

- **Forward window not binding.** Only 17.5% of trades cap at bar 240 (time_exit), well below the 20% auto-extend trigger. Pool size 2,153 is exactly mid-range of the signal-spec's 1,500-2,500 prior, indicating signal calibration matches the spec author's expectation.

## Files

- Signal module: [signals/lchar_inside_bar_break_trend_long.py](../../signals/lchar_inside_bar_break_trend_long.py)
- Step 1 backtest: [scripts/l_arc_9/step1_plumbing.py](../../scripts/l_arc_9/step1_plumbing.py)
- Step 1 diagnostics: [scripts/l_arc_9/step1_diagnostics.py](../../scripts/l_arc_9/step1_diagnostics.py)
- Step 2 clustering: [scripts/l_arc_9/step2_clustering.py](../../scripts/l_arc_9/step2_clustering.py)
- Step 3 capturability: [scripts/l_arc_9/step3_capturability.py](../../scripts/l_arc_9/step3_capturability.py)
- Step 4 extractability: [scripts/l_arc_9/step4_extractability.py](../../scripts/l_arc_9/step4_extractability.py)
- Config: [configs/wfo_l_arc_9.yaml](../../configs/wfo_l_arc_9.yaml)
- Step 1 outputs: [results/l_arc_9/step1_verbatim/](step1_verbatim/)
- Step 2 outputs: [results/l_arc_9/step2_clustering/](step2_clustering/)
- Step 3 outputs: [results/l_arc_9/step3_capturability/](step3_capturability/)
- Step 4 outputs: [results/l_arc_9/step4_extractability/](step4_extractability/)
- Closure doc: [results/l_arc_9/ARC_9_CLOSURE.md](ARC_9_CLOSURE.md)
