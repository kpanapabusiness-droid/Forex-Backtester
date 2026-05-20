# Arc 9 Deployment Candidate A — LGBM Pipeline E at threshold 0.40

> **Status: LOCKED for ONNX export + Python parity verification.**
> Standing closure context: Arc 9 closed `STEP_4_KILL` under v2.3 §16a; the post-kill experiment series (Pipeline E retry → Step 5 LGBM Pipeline E → lookahead audit → SCALED_RISK) produced a validated classifier and a measured per-trade risk profile. KILL stands in the protocol audit trail; deployment work proceeds on the strength of post-kill experiments. Closure-reversal authorship deferred to post-EA-validation.
> Source: Arc 9 Step 5 LGBM Pipeline E dispatch, commit `44de1ca`; lookahead audit commit `9dc4f8a`; scaled-risk measurement commit `5ce39d6`; all on `claude/bold-brattain-d79817`.

---

## Pre-deployment status (this dispatch's resolutions)

| Blocker | Status | Resolved by |
|---|---|---|
| 1. Lookahead verification audit | **RESOLVED — 8/8 GREEN** | `results/l_arc_9/deployment/audit_status_summary.md` (Phase 0.1) |
| 2. v2.x §3 grid amendment OR locked-grid acceptance | **LOCKED-GRID ACCEPTED** — Candidate A operates at threshold 0.40 ∈ {0.40, 0.50, 0.60, 0.70} (no amendment needed for this candidate) | this spec |
| 3. Step 5 fold-1 warmup convention OR data-window backfill | **DEFERRED** — Candidate A's strict 7-fold §10 fails on F1 data-availability artifact; folds 2-7 pass; closure-reversal will document the chosen disposition | post-EA-validation closure |
| 4. Formal Step 4 re-pass under amended protocol | **DEFERRED** — Pipeline E retry's 28-feature expansion (§8 feature-budget extension) is post-kill experimental; closure-reversal will document if this is ratified | post-EA-validation closure |
| 5. Per-trade risk parameter | **LOCKED at 1.0%** per SCALED_RISK measurement (commit `5ce39d6`) | `configs/arc9_canda_exit_policy.yaml` (Phase 0.4) |

---

## Signal definition

| Field | Value |
|---|---|
| Signal name | IB-trend (compression geometry → directional break, long) |
| Source spec | `docs/signal_spec_inside_bar_break_trend_long_v0.1.md` |
| Source module | `signals/lchar_inside_bar_break_trend_long.py` |
| Family | Compression-break (volatility expansion from constricted range) |
| Direction | Long only |
| Signal timeframe | 4H |
| Pair set | 28 FX (KH-24 set) |
| Pool size | 2153 trades |
| Population builder | `build_ex_ante_bounded_population` |

### IB-trend entry trigger — mechanical definition

Locked parameters (per `signals/lchar_inside_bar_break_trend_long.py:47-50`):

```python
TREND_WINDOW: int = 30           # search window t-30..t-1 inclusive
SWING_HALF: int = 3              # k-3..k-1 backward, k+1..k+3 forward
SWING_RIGHT_EDGE_LAG: int = 4    # most recent swing-low at most bar t-4
REFRACTORY_BARS: int = 20        # spacing between fires (signal bars)
```

Signal fires at bar `t` close (entry at bar `t+1` open) iff ALL four conditions hold:

```
1. Trend filter (no MA, pure structural):
     - Identify 3-bar swing-lows in window [t-30, t-4]:
         is_swing_low(k) iff low[k] < min(low[k-3..k-1])
                            AND low[k] < min(low[k+1..k+3])
     - Require >= 1 qualifying swing-low in the window.
     - Require close[t-1] > min(qualifying swing-lows in the window).
2. Inside bar at t-1 (strict nest in mother bar t-2):
     high[t-1] < high[t-2]  AND  low[t-1] > low[t-2]
3. Break trigger at bar t:
     close[t] > high[t-1]   AND  close[t] > open[t]
4. Spacing: (t - last_signal_idx) >= REFRACTORY_BARS (=20 4H bars)
```

Lookahead invariance: swing-low identification uses `k+1..k+3`, but the right-edge constraint `k <= t-4` (=> `k+3 <= t-1`) guarantees ALL bars consumed by the swing-low validation are strictly before bar `t`. No future-bar leak. Verified by Lookahead Audit 1 (3920/3920 (feature, sample) pairs match truncated-series recompute).

Quoted verbatim from `signals/lchar_inside_bar_break_trend_long.py:121-169` (core loop):

```python
earliest_t = TREND_WINDOW
for t in range(earliest_t, n):
    # --- Cond 2: inside bar at t-1 (strict nest in mother bar t-2) ---
    ib_ok = (high[t - 1] < high[t - 2]) and (low[t - 1] > low[t - 2])
    ib_passed[t] = int(ib_ok)

    # --- Cond 3: break trigger at bar t ---
    br_ok = (close[t] > high[t - 1]) and (close[t] > open[t])
    break_passed[t] = int(br_ok)

    # --- Cond 1: trend filter via swing-lows in window t-30..t-1 ---
    window_lo = max(0, t - TREND_WINDOW)
    window_hi_excl = t - SWING_RIGHT_EDGE_LAG + 1
    if window_hi_excl <= window_lo:
        qualifying_sl = np.empty(0, dtype=np.int64)
    else:
        seg = is_sw_array[window_lo:window_hi_excl]
        rel_idx = np.where(seg)[0]
        qualifying_sl = rel_idx + window_lo
    n_qual = int(qualifying_sl.size)
    ...
    if n_qual == 0:
        continue
    most_recent_sl[t] = int(qualifying_sl[-1])
    sl_min = float(low[qualifying_sl].min())
    trend_ok = close[t - 1] > sl_min
    prefilter_ok = bool(ib_ok and br_ok and trend_ok)
    if not prefilter_ok:
        continue
    # --- Cond 4: spacing ---
    if (t - last_signal_idx) < REFRACTORY_BARS:
        continue
    sig[t] = 1
    last_signal_idx = t
```

Volume veto: NONE. Arc 9 signal does not consult any volume gate (verified in Lookahead Audit 7: `volume_veto_note = "Arc 9 signal has no volume veto condition; check N/A."`).

## Cohort identification

| Field | Value |
|---|---|
| Target cluster | `cluster_0_individual` (Step 2 K=3, cid=0) |
| Cluster size in pool | 365 trades (16.95% of pool) |
| Cluster centroid (monotonicity_ratio_in_profit / local_peaks_count / pullback_magnitude_median / time_to_peak_mfe_relative) | **0.5337 / 31.44 / 0.5795 / 0.7713** |
| Automatic §11 archetype label | `Unclassified` (boundary case — see routing) |
| Routed-to §11 archetype | `Stepwise climber` (Step 5 oracle dispatcher decision) |
| `shape_tag` | `tight_unimodal` |
| Capturability composite score | `0.6122971688` |
| Selected SL | `2.0 × ATR(14)` on 4H |

§11 routing basis: 3 of 4 Stepwise climber criteria match (monotonicity 0.534 ≥ 0.50; local_peaks 31.44 ∈ post-v2.x ceiling [5, 50]; time_to_peak_mfe_relative 0.771 ≥ 0.50). Pullback median 0.580 fails the 0.5 ceiling by 0.08 → automatic match = `Unclassified`. Step 5 oracle dispatcher routed to Stepwise climber and accepted the routing on capturability economics (oracle Step 5: +39.45% worst-fold ROI, 0.01% DD).

Step 3 capturability metrics (locked):

| Metric | Value |
|---|---|
| `fwd_mfe_p50` | 6.176R |
| `frac_reach_1R` | 1.000 |
| `frac_wrong_way_pre_peak` | 0.000 |
| `final_r_mean` | +4.423R |
| `t-stat` | +21.81 |
| `shape_tag` | `tight_unimodal` |
| `shape_tag_pass` | 1 (PASS) |
| `bimodal_separated_pass` | 0 |
| `pre_t_sl_atr_multiplier` (v2.3 §4) | 2.0 |

## Admission filter

| Field | Value |
|---|---|
| Pipeline | Pipeline E (entry-time admission, rejected trades never enter book) |
| Classifier | LightGBM (`lgb.LGBMClassifier`, version 4.6.0) |
| Feature count | 28 features |
| Hyperparameters | `n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced', random_state=42, deterministic=True, force_row_wise=True, verbose=-1, n_jobs=-1` |
| Training validation | 5-fold `TimeSeriesSplit` CV |
| Training mean CV AUC | 0.7508 |
| Reproduction parity | byte-identical per-fold AUCs: **0.848380 / 0.735746 / 0.717263 / 0.714402 / 0.738041** |
| **Admission threshold** | **0.40** |
| Feature catalogue (load-bearing) | `configs/arc9_canda_features.yaml` |
| Exit policy + risk | `configs/arc9_canda_exit_policy.yaml` |

### Feature composition (28 features, ordered = ONNX input tensor positions)

**BASELINE_16 (Step 4 original Pipeline E feature set; 4H signal-bar features):**

Cross-dataset base 8 (indices 0-7):
`body_to_range_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, `range_to_atr_14`, `ret_5bar_atr`, `ret_20bar_atr`, `pos_in_20bar_range`, `rsi_14`

Arc 9 specific 8 (indices 8-15):

| Index | Feature | One-line description |
|---|---|---|
| 8 | `n_swing_lows` | Count of 3-bar swing-lows in window `[t-30, t-4]` (from signal trigger) |
| 9 | `most_recent_sl_lag` | Bars between `t` and the most recent qualifying swing-low (min 4, max 30) |
| 10 | `swing_low_dist_atr` | `(close[t] - swing_low_used) / atr14[t]` (distance to reference swing-low in 4H ATR units) |
| 11 | `mother_bar_range_atr` | `(high[t-2] - low[t-2]) / atr14[t]` (mother-bar range in ATR units) |
| 12 | `inside_bar_range_atr` | `(high[t-1] - low[t-1]) / atr14[t]` (inside-bar range in ATR units) |
| 13 | `ib_range_ratio` | `inside_range / mother_range` (compression ratio) |
| 14 | `break_bar_body_atr` | `abs(close[t] - open[t]) / atr14[t]` (break-bar body in ATR units) |
| 15 | `break_close_above_high_atr` | `(close[t] - high[t-1]) / atr14[t]` (break magnitude over inside-bar high) |

**EXPANDED_12 (Pipeline E retry additions):**

D1-lagged context 8 (indices 16-23, one-day backward lag via `merge_asof`):

| Index | Feature | One-line description |
|---|---|---|
| 16 | `d1_trend_state` | bool — D1 close > D1 open AND D1 close > D1 close[-1] |
| 17 | `d1_atr_ratio_to_4h` | `d1_atr14_lag1 / atr14_at_4h_signal_bar` |
| 18 | `d1_pos_in_20d_range` | D1 close position in 20-D1-bar high/low range |
| 19 | `d1_ret_5d_atr` | D1 5-day return in D1 ATR units |
| 20 | `d1_rsi_14` | D1 Wilder RSI(14) |
| 21 | `d1_close_above_kijun` | bool — D1 close > D1 Kijun(26) |
| 22 | `d1_bars_since_swing_high` | bars since most recent confirmed 20-bar D1 swing-high |
| 23 | `d1_bars_since_swing_low` | bars since most recent confirmed 20-bar D1 swing-low **(top feature; dominates importance by 3× gain margin)** |

Session / time 4 (indices 24-27):

| Index | Feature | One-line description |
|---|---|---|
| 24 | `session_london` | `int((hour >= 8) and (hour < 16))` (UTC) |
| 25 | `session_ny_overlap` | `int((hour >= 12) and (hour < 16))` (UTC) |
| 26 | `hour_sin` | `sin(2π · hour / 24)` |
| 27 | `hour_cos` | `cos(2π · hour / 24)` |

Full feature formulas: see `configs/arc9_canda_features.yaml` (load-bearing source of truth).

Structural reading: cluster-0 trades concentrate at the early phase of D1 uptrends, identifiable at entry by recency-to-confirmed-D1-swing-low. The 4H IB-trend bar is the timing trigger; the D1 swing-low context is the cohort discriminator.

## Operating point characteristics (Candidate A @ threshold 0.40)

| Metric | Value |
|---|---|
| Admission threshold | 0.40 |
| Per-fold precision range (F2-F7) | 0.41 - 0.51 |
| Cluster-0 recall (training CV) | ~0.34 |
| Cluster-0 recall (deployment, Step 5 OOS folds 2-7) | ~30% (109 true positives of 365 cluster-0 in pool) |
| Total admits across F2-F7 | 236 trades |
| False-positive rate | ~54% of admit set (127 of 236 admits are non-cluster-0) |

False positives are tolerated because the §11 archetype exit policy gracefully degrades on complement trades — false-positive trades hit SL or trail without catastrophic damage. This is the deployment thesis for tolerating <50% precision.

## Exit policy (§11 Stepwise climber Pipeline E)

§11 row verbatim (`L_ARC_PROTOCOL.md:600`):
> `monotonicity ≥ 0.50, local_peaks 5-50, pullback ≤ 0.5R, time_to_peak_rel ≥ 0.50` → **Stepwise climber** → SL prior 1.3R → Exit (Pipeline E): **MFE-lock at 1R, trail 0.75R from new high**

Concrete parameters (load-bearing source: `configs/arc9_canda_exit_policy.yaml`):

| Field | Value |
|---|---|
| Stop loss | `2.0 × ATR(14)` on 4H, anchored at entry_fill, locked from Step 3 SL selection |
| `1R` reference | = SL distance = `2.0 × ATR(14)` |
| MFE-lock trigger | intrabar MFE ≥ 1R favorable → SL moves to entry (breakeven) |
| Trail | post-MFE-lock only; SL = peak_close − 0.75R; ratchet on each new bar-close peak |
| Time exit | bar `t+1+240` open (4H × 240 ≈ 40 calendar days) |
| Partial take | none |
| Per-trade risk (deployment) | **1.0%** of account equity (locked per SCALED_RISK; was projected 2.0%, falsified) |
| Per-trade risk (research/parity) | 0.5% — used for Step 5 WFO and Dispatch 3 admit-set reproduction |
| Forward window | 240 4H bars |

Reference implementation: `scripts/l_arc_9/experiments/step5_validation.py:_resimulate_trade` (re-used directly by Step 5 LGBM Pipeline E script; no re-implementation drift).

### Risk-parameter rationale (SCALED_RISK, commit `5ce39d6`)

The earlier projection-based 2.0% deployment risk was **falsified** by measurement. Per-trade risk reduced to **1.0%** based on intraday-resolution worst-day DD measurement of the fixed Candidate A admit set:

| Risk | Worst-fold ann ROI | Worst-fold max DD | Worst-day DD | Full-data ann ROI | In-system (≤8% fold / ≤4% day)? | 5ers hard (≤10% / ≤5%)? |
|---|---|---|---|---|---|---|
| 0.5% | +9.63% | 1.26% | 1.01% | +19.06% | PASS / PASS | PASS / PASS |
| **1.0%** | **+20.07%** | **2.52%** | **2.01%** | **+41.45%** | **PASS / PASS** | **PASS / PASS** |
| 1.5% | +31.39% | 3.76% | 3.00% | +67.68% | PASS / PASS | PASS / PASS |
| 2.0% | +43.65% | 5.00% | 3.99% | +98.36% | PASS / PASS (0.01pp margin) | PASS / PASS (1.01pp margin) |
| 2.5% | +56.91% | 6.23% | 4.98% | +134.17% | PASS / FAIL (4.98% > 4%) | PASS / PASS (0.02pp margin) |

Key finding: **worst-day DD is the binding constraint, not worst-fold max DD**. Worst-fold max DD has 5+pp headroom under in-system 8% target at every candidate risk level; worst-day DD reaches the in-system 4% target at exactly 2.0% risk with effectively zero margin. The recommendation rule (in-system target + 1pp safety margin → worst-fold DD ≤ 6%, worst-day DD ≤ 3%) yields **1.0% as the highest risk level satisfying both**. DD scales essentially linearly with risk (DD ratio observed / linear ≈ 0.99 across all multiples); ROI scales supra-linearly via compounding.

Driver: a single observed day (2024-08-06) — two same-day SL hits across `GBP_CHF` and `NZD_USD` — accounts for the binding worst-day DD at every risk level. With only one such event in 5+ years OOS, the 1pp safety margin is sized for ~one additional cluster of equivalent magnitude.

## Step 5 WFO economics (folds 2-7, 0.5% research baseline)

| Metric | Pass-deployable gate | Candidate A actual | Pass? |
|---|---|---|---|
| Worst-fold annualised ROI | ≥ 5% | **+9.63%** (fold F3) | PASS |
| Mean fold annualised ROI | ≥ 8% | **+22.92%** | PASS |
| Worst-fold max DD | ≤ 8% | **1.32%** | PASS |
| All folds positive | required | TRUE (F2-F7) | PASS |
| Trade count per fold | ≥ 15 | min 24 (F3) | PASS |
| Full-data annualised ROI | ≥ 5% | **+19.06%** | PASS |
| Full-data max DD | ≤ 10% | **1.32%** | PASS |

Strict 7-fold §10 fails on F1 = 0 admits — Arc 9 signal data start (2020-10-01) coincides with KH-24 fold 1 OOS_start, leaving no anchored-expanding training data for the classifier on F1. Data-window artifact, not a signal failure. Deployment proceeds on the restricted §10 (F2-F7) basis pending the fold-1 warmup convention.

## Deployment surface position (folds 2-7 captured fraction of floor → ceiling)

| Axis | Floor (raw) | Ceiling (oracle) | Candidate A | Capture |
|---|---|---|---|---|
| Worst-fold ann ROI | −29.64% | +39.45% | +9.63% | 57% |
| Mean fold ann ROI | +6.10% | +61.40% | +22.92% | 30% |
| Worst-fold DD reduction | from 43.61% | to 0.01% | 1.32% | 97% |
| Full-data ann ROI | +0.80% | +60.50% | +19.06% | 31% |
| Full-data DD reduction | from 62.99% | to 0.01% | 1.32% | 98% |

Candidate A captures most available DD reduction (97-98%) and meaningful but conservative ROI capture (30-57%). Compared to Candidate B (sub-grid threshold 0.05), this candidate trades ROI for tighter DD discipline.

## Lookahead audit status (`results/l_arc_9/deployment/audit_status_summary.md`)

| # | Audit | Verdict |
|---|---|---|
| 1 | 4H feature timestamps | GREEN |
| 2 | D1 lag | GREEN |
| 3 | Session features | GREEN |
| 4 | Label leak | GREEN |
| 5 | Fold disjointness | GREEN |
| 6 | Cluster label flow | GREEN |
| 7 | Execution semantics | GREEN |
| 8 | E2E reproduction | GREEN (`max_abs_diff = 4.82e-11`) |

**Overall: GREEN.** Spec pre-deployment blocker #1 resolved at audit commit `9dc4f8a`.

## What deploying this means in plain terms

At every 4H bar across the 28 FX pair set, the IB-trend entry trigger is evaluated. When it fires, the LightGBM classifier reads 28 features (8 cross-dataset baseline + 8 arc-specific 4H + 8 D1-lagged + 4 session/time) and outputs a probability. If probability ≥ 0.40, the trade is taken at the next bar's open with SL = 2.0 × ATR(14). The trade exits per the §11 Stepwise climber Pipeline E policy: SL moves to entry when MFE hits 1R intrabar, then trails 0.75R behind peak close until SL or time-exit (240 bars) fires. Per-trade risk is 1.0% of current equity, compounded; this targets ~+20% worst-fold annualised ROI with worst-day DD ≈ 2% under measurement.

Approximately 11% of all IB-trend signals admitted; of those admitted, ~46% are cluster-0 (true positives); the rest are complement trades that the §11 exit policy handles without catastrophic damage.

## Provenance

| Artefact | Path | Commit |
|---|---|---|
| Signal pool | `results/l_arc_9/step1_verbatim/trades_all.csv` (2153 trades, 28 pairs) | `71438f0` |
| Path features | `results/l_arc_9/step1_verbatim/trades_paths.csv` | `71438f0` |
| Cluster assignment | `results/l_arc_9/step2_clustering/clusters_K3.csv` (cid=0) | `5548cd2` |
| Capturability | `results/l_arc_9/step3_capturability/capturability_pass_list.csv` | `9b2f9fe` |
| Classifier training | `results/l_arc_9/experiments/pipeline_e_retry/` | `0193334` |
| Step 5 evaluation | `results/l_arc_9/experiments/step5_lgbm_pipeline_e/candidate_A_thr0.40/` | `44de1ca` |
| Lookahead audit | `results/l_arc_9/experiments/lookahead_audit/audit_summary.json` | `9dc4f8a` |
| Scaled-risk measurement | `results/l_arc_9/experiments/scaled_risk/SCALED_RISK_RESULT.md` | `5ce39d6` |
| Feature catalogue (canonical) | `configs/arc9_canda_features.yaml` | this dispatch |
| Exit policy (canonical) | `configs/arc9_canda_exit_policy.yaml` | this dispatch |
| Audit status summary | `results/l_arc_9/deployment/audit_status_summary.md` | this dispatch |
| Worktree branch (source) | `claude/bold-brattain-d79817` | HEAD `5ce39d6` |
| Worktree branch (this dispatch) | `feature/arc9-canda-spec-onnx` | — |

## Document control

| Field | Value |
|---|---|
| Candidate identifier | A (locked-grid 0.40) |
| Status | **LOCKED** for ONNX export + Python parity (Phase 1 of this dispatch) |
| Step 5 disposition | PASS-DEPLOYABLE on F2-F7 (strict §10 FAIL on F1 data-availability artifact) |
| Drafted | 2026-05-18 |
| Locked | 2026-05-19 |
| Locked at branch | `feature/arc9-canda-spec-onnx` (base `claude/bold-brattain-d79817@5ce39d6`) |
| Companion candidate | B (recall-floor 0.05) — separate doc, separate dispatch |
| Closure | pending EA validation (Dispatch 2 + Dispatch 3); closure-reversal authored thereafter |
| Anchor preservation | KH-24 K=4 archetype 3 — preserved, no interaction |
