# Arc 9 Lookahead and Cross-Timeframe Leak Audit

> Read-only audit of the Arc 9 LightGBM Pipeline E classifier (AUC 0.7508) and
> the Step 5 WFO economics derived from it (Candidate A worst-fold ann ROI
> +9.63% / Candidate B +20.68% on F2-F7). Project history flag: same-day D1
> close has previously invalidated WFO results — this audit is the gate between
> Arc 9 research and any deployment momentum.

## Headline

**GREEN — Arc 9 candidates are audit-clean.**

All 8 audits PASS. The classifier's AUC 0.7508 and the Step 5 economics for
both Candidate A (thr 0.40) and Candidate B (thr 0.05) reflect real
out-of-sample classifier behaviour with no lookahead, no cross-timeframe
leak, no label contamination, no training-set / test-set overlap, and no
execution-semantics violation. Reproduction of recorded probabilities is
byte-exact (max abs diff 4.82e-11 over 100 sample trades — float64 noise
floor).

D1 lag specifically: implemented via the identical `merge_asof(direction='backward')`
on a pre-shifted (signal_date − 1 day) date column pattern that the KH-24 engine
(`scripts/phase_kgl_v2_4h_wfo.py:_precompute_d1_exit_arrays`) uses. 560 sampled
trades all show `days_lag ≥ 1`; zero same-day D1 access detected; Monday
boundary cases all show days_lag = 3 (Friday's D1, correctly).

Audit verdict means: deployment momentum can continue subject to v2.x
amendment and formal Step 4 re-pass. No leak invalidates the AUC 0.7508
result or the Step 5 economics.

## Audit-by-audit verdicts

| Audit | Scope | Verdict | Key evidence |
|---|---|---|---|
| 1 | 4H feature timestamp boundary | GREEN | 0 / 3,920 (feature, sample) recompute mismatches at 1e-6 tolerance |
| 2 | **D1 lag integrity (CRITICAL)** | **GREEN** | 0 / 560 samples with days_lag < 1; code pattern equivalent to KH-24 engine reference |
| 3 | Session / hour features | GREEN | 0 / 80 mismatches (20 samples × 4 features) |
| 4 | Label leakage (cluster_0 forward-geom features in entry matrix) | GREEN | 0 forward-geom features; 0 column overlap; max |Pearson r| 0.3285 |
| 5 | Training/inference fold disjointness | GREEN | 0 TSS-fold overlaps; 0 KH-24-fold train-into-OOS overlaps; walkforward code confirmed in Step 5 |
| 6 | Cluster label flow | GREEN | `y` target-only; inference X is features-only; no k-means at inference time |
| 7 | Spread & execution semantics | GREEN | 20 / 20 samples pass entry-time-at-t+1, spread-from-t+1 (with floor), SL-from-signal-bar-ATR checks |
| 8 | End-to-end probability reproduction | GREEN | 100 / 100 reproduced probs match recorded (max abs diff 4.82e-11) |

Audit-script bug fixes made during this audit (before final GREEN):
- Audit 1 tolerance loosened from 1e-9 → 1e-6 (Wilder RSI accumulates float ops over hundreds of bars; 1e-9 below noise floor). Initial run flagged 224 mismatches all at ~1e-9 abs_diff = FP noise.
- Audit 1 windowing aligned to production: slice to `arc_cfg.date_start` before computing recursive smoothers (RSI/ATR). 10 residual mismatches all in 2020-10/11 reflected my audit loading full historical CSV vs production slicing to date_start — fixed by matching production slice.
- Audit 7 spread check: load `configs/spread_floors_5ers.yaml` and verify `spread_pips_used == max(raw_pips_t+1, floor_pips_for_pair)` matching `core/spread_floor.py:apply_spread_floor_to_pips`. Initial run failed because I only checked raw spread, missing the floor application.
- Audit 8 feature pipeline: rebuild matrix IN-MEMORY (call `_build_feature_matrix`), do not reload from CSV. CSV round-trip via `%.10g` loses precision and propagates through LGBM tree splits to ~1e-2 prob diffs. The "Step 5 production probabilities" were trained on in-memory floats; the audit must do the same to be a faithful reproduction.

None of these bug fixes changed code being audited — they fixed audit-script assumptions about what the production code does. The classifier, feature pipeline, training, threshold logic, and accounting in Step 5 / Pipeline E retry are unchanged.

## D1 lag specific verification (critical section)

Project history flag: a Python bug that used same-day D1 close invalidated all pre-fix KGL WFO results. The EA's `iClose(PERIOD_D1, 1)` was always correct (index 1 = yesterday's D1 close). The Arc 9 LightGBM classifier uses 8 D1-lagged features including the top-importance feature `d1_bars_since_swing_low` (3× gain over the next feature). If the D1 lag here were wrong, AUC 0.7508 would be fake, Candidate B's +20.68% worst-fold ann ROI would be fake, and the v2.x amendment evidence would collapse.

Specific evidence (audit 2):

| Check | Status | Detail |
|---|---|---|
| `pd.merge_asof(direction='backward')` confirmed in code | **YES** | `scripts/l_arc_9/experiments/pipeline_e_retry.py:_attach_d1_features` |
| Pre-shifted date column (`signal_date − 1 day`) confirmed | **YES** | `trades["lookup_date"] = trades["signal_bar_time"].dt.normalize() - pd.Timedelta(days=1)` |
| Sample of 560 trades (20 per pair × 28 pairs) all show `days_lag ≥ 1` | **TRUE** | min lag 1, max lag 4, distribution: 430 × lag=1, 129 × lag=3, 1 × lag=4 |
| Monday-morning boundary cases tested | 129 samples, **all PASS** with days_lag = 3 (Friday's D1, correct weekend handling) |
| 00:00-01:00 UTC boundary cases tested | 194 samples, **all PASS** (days_lag distribution: 149 × lag=1 mid-week, 44 × lag=3 Monday early-hours, 1 × lag=4 Monday post-holiday) |
| D1 lag pattern matches `scripts/phase_kgl_v2_4h_wfo.py:_precompute_d1_exit_arrays` | **YES — verbatim-equivalent** | both: `dates.dt.normalize() - pd.Timedelta(days=1)` + `pd.merge_asof(direction='backward')` |
| Zero same-day D1 access detected | **TRUE** | smallest days_lag = 1 across all 560 samples |

Files:
- `audit_2_d1_lag_code_review.md` — explicit code excerpts vs reference engine pattern
- `audit_2_d1_lag_samples.csv` — 560 samples with (pair, signal_ts, signal_date, lookup_date, d1_date_used, days_lag, leak_flag)
- `audit_2_boundary_cases.csv` — 323 Monday-morning + early-UTC samples called out separately

## Per-audit details

### Audit 1 — 4H feature timestamp boundary

For each of 280 sampled trades (10 per pair × 28 pairs), the 14 production 4H features (8 base + 6 of the 8 arc-specific that depend on 4H bars; `n_swing_lows` and `most_recent_sl_lag` are signal-time scalars not 4H-bar-derived) were recomputed from a truncated 4H series (bars with `date >= arc_cfg.date_start AND date <= signal_bar_time`). All recomputed values match the recorded values in `entry_features.csv` to within 1e-6 (the noise floor for chained float ops on Wilder RSI).

**Result: 0 / 3,920 (feature, sample) pairs show mismatch above 1e-6.** No future bar contributed to any sampled feature.

Files: `audit_1_4h_feature_timestamps.csv`

### Audit 2 — D1 lag integrity (CRITICAL)

See "D1 lag specific verification" section above. **GREEN.**

### Audit 3 — Session / hour features

For 20 sampled trades, recomputed `session_london`, `session_ny_overlap`, `hour_sin`, `hour_cos` directly from each trade's `signal_bar_time` hour-of-day. All 80 (sample × feature) values match the recorded values exactly (abs_diff < 1e-9). Confirms no smoothing, no future-window aggregation, no leakage path.

Files: `audit_3_session_features.csv`

### Audit 4 — Label leakage (cluster_0)

Cluster_0 labels are derived at Step 2 from path-shape features (monotonicity_ratio_in_profit, local_peaks_count, pullback_magnitude_median, time_to_peak_mfe_relative). These path-shape features REQUIRE forward observation (held bars from entry to actual exit). If any path-shape feature appeared in the entry-time feature matrix, the classifier would have a trivially-predictive leak path.

Check 1 — pattern match on forbidden tokens (`monotonicity`, `local_peaks`, `pullback`, `time_to_peak`, `mfe`, `mae`, `final_r`, `fwd_`, `_so_far`, `post`, `peak_mfe`): **0 of 28 features classified as FORWARD-GEOMETRY**.

Check 2 — column-name overlap between EXPANDED_28 and Step 2 `path_features.csv` columns: **0 overlapping columns**.

Check 3 — Pearson correlation between every entry-time feature and every path-shape feature on the same trade pool: **max |r| = 0.3285** (well below the 0.85 flag threshold). No structural similarity that would indicate hidden leakage.

Files: `audit_4_feature_list_review.md`, `audit_4_correlation_matrix.csv`

### Audit 5 — Training/inference fold disjointness

**Pipeline E retry CV (TimeSeriesSplit(5)):** train indices are always strictly less than test indices by construction of `sklearn.model_selection.TimeSeriesSplit`. Audit verified: each of the 5 folds has `train_idx.max() < test_idx.min()` — 0 overlap, 5/5 folds clean.

**Step 5 LGBM E WFO (KH-24 anchored expanding):** code review confirms `train_mask = entry_time < np.datetime64(fold.oos_start)` in `scripts/l_arc_9/experiments/step5_lgbm_pipeline_e.py:_wfo_for_threshold`. For each of the 7 KH-24 folds, `train_date_max < fold.oos_start` — 0 overlap, 7/7 folds clean.

Cross-check via parity: Step 5 reproduces Pipeline E retry's TSS-CV mean AUC 0.7508 byte-equivalently (reproduced 0.750766). If the Step 5 training data had been contaminated by OOS data (relative to either the TSS or KH-24 fold structure), the parity check would have failed.

**No training data overlaps with its own test fold under either fold scheme.**

Files: `audit_5_fold_overlap_table.csv` (12 folds across 2 schemes)

### Audit 6 — Cluster label flow

`is_cluster_0` is computed once at Step 2 from path-shape features (forward-geometry). It is used in three places:

1. **Pipeline E retry training:** `y = df_clean["y"].to_numpy(dtype=int)` — `y` is the classifier's target. The training calls `mdl.fit(X, y)` where `X = df_clean[EXPANDED_28]` — features only.
2. **Step 5 inference:** `mdl.predict_proba(X[oos_mask])[:, 1]` — X is features only. No label column in X.
3. **Step 5 admit logic:** admit-mask is `prob >= threshold` — classifier output only, no label.

`KMeans` is imported only in `scripts/l_arc_9/step2_clustering.py`, not in `step5_lgbm_pipeline_e.py`. No re-fitting of clusters at inference time. The forbidden columns (`cluster_id`, `cluster_0`, `is_cluster_0`, `label`, `target`) do not appear in `EXPANDED_28`.

Label flow: **Step 2 k-means (path-shape) → `is_cluster_0` target → classifier weights → inference prediction**. No path where cluster labels enter the inference X.

Files: `audit_6_label_flow.md`

### Audit 7 — Spread & execution semantics at inference

For 20 sampled admitted trades from Candidate A (or A+B if A < 20):

| Check | Pass rate |
|---|---|
| Entry time equals bar t+1 timestamp | 20 / 20 |
| `spread_pips_used` matches `max(raw_pips_t+1, floor_pips_for_pair)` per `core/spread_floor.py:apply_spread_floor_to_pips` | 20 / 20 |
| `sl_distance_recorded == 2.0 × atr14_at_signal` (ATR computed at signal bar, not bar t+1) | 20 / 20 |

Note on volume veto: Arc 9 signal (`signals/lchar_inside_bar_break_trend_long.py`) has NO volume-based condition. The signal-spec doesn't include a volume veto. Check N/A. (KH-24's c7 volume gate is unrelated to Arc 9.)

Files: `audit_7_execution_semantics.csv`

### Audit 8 — End-to-end probability reproduction

Sampled 100 admitted trades (50 from Candidate A admits, 50 from Candidate B admits, balanced across folds). For each:

1. Rebuilt the 28-feature matrix in-memory using `scripts.l_arc_9.experiments.step5_lgbm_pipeline_e._build_feature_matrix` (the same function Step 5 uses; reused via import, not re-implemented).
2. For each KH-24 fold present in the sample, trained a LightGBM classifier on all trades with `entry_time < fold.oos_start` (the same anchored-expanding training that Step 5 does).
3. Scored each sample trade with its fold's classifier.
4. Compared the reproduced probability to the value recorded in the candidate's `admitted_trades.csv`.

**Result: 100 / 100 trades reproduce within 4.82e-11 abs_diff (float64 noise floor).**

This is the most decisive check: any divergence between the Step 5 training pathway and the audit reproduction would manifest as probability drift. The drift is at the LightGBM internal float-determinism floor, confirming the Step 5 classifier (per-fold trained, applied to OOS signals, threshold applied to admit) IS the same classifier the audit rebuilds.

Files: `audit_8_e2e_reproduction.csv`

## Overall verdict

**GREEN: candidates A and B are audit-clean. AUC 0.7508 and Step 5 economics are real.**

Specifically:

- The LightGBM Pipeline E classifier (mean CV AUC 0.7508 on TimeSeriesSplit(5), reproduced byte-equivalent at 0.750766) is **not** the result of any of: same-day D1 leak, future 4H bar peek, path-shape label leakage, training/test-set overlap, k-means at inference, or t+1-ATR SL placement.
- The Step 5 WFO economics — Candidate A (worst-fold ann ROI +9.63% F2-F7, mean +22.92%, worst DD 1.32%) and Candidate B (worst-fold +20.68%, mean +36.12%, worst DD 6.80%) — reflect real out-of-sample classifier behaviour, including correct one-day-lagged D1 features that the EA's `iClose(PERIOD_D1, 1)` pattern would have used at live execution.
- The fold-1 zero-admits artifact (Arc 9 data starts at KH-24 fold 1 OOS_start) is a data-window / protocol mismatch, NOT a leak. Per Step 5 LGBM E report it can be addressed by extending Arc 9 Step 1 backward to 2018-2019 (engine-touching) or by a Step 5 warmup convention (protocol amendment).
- The v2.x amendment evidence (Candidate B beats Candidate A by +11pp full-data ann ROI) is real economic measurement, not a leak artifact.

Deployment momentum can proceed subject to v2.x amendment landing and a formal Step 4 re-pass under the amended protocol.

## RNG seed and reproducibility

All audits seeded at `AUDIT_SEED = 4242` for sample selection. LightGBM training uses `random_state=42, deterministic=True, force_row_wise=True` (same as Pipeline E retry / Step 5). The audit script can be re-run to byte-equivalence by any independent engineer.

## Files

- `AUDIT_REPORT.md` — this file
- `audit_summary.json` — machine-readable summary
- `audit_1_4h_feature_timestamps.csv` — 3,920 (feature, sample) checks
- `audit_2_d1_lag_code_review.md` — code excerpts + pattern equivalence narrative
- `audit_2_d1_lag_samples.csv` — 560 D1 lag samples
- `audit_2_boundary_cases.csv` — 323 Monday + early-UTC boundary samples
- `audit_3_session_features.csv` — 80 session/hour feature checks
- `audit_4_feature_list_review.md` — 28-feature classification
- `audit_4_correlation_matrix.csv` — 28 × N path-shape correlation matrix
- `audit_5_fold_overlap_table.csv` — TSS-CV(5) + KH-24-WFO(7) fold dates
- `audit_6_label_flow.md` — label provenance code trace
- `audit_7_execution_semantics.csv` — 20 admitted-trade execution checks
- `audit_8_e2e_reproduction.csv` — 100 trades with recorded/reproduced/diff probabilities
