# Phase 1 — Lookahead audit

**Verdict: FAIL** — sub-phases 1B and 1C are audit-fatal. The top-importance feature `d1_bars_since_swing_low` (20.2% of total gain, 3.65× the next feature) and the second-importance feature `d1_bars_since_swing_high` (6.8% of total gain) are computed using D1 bars that occur AFTER the 4H signal bar's timestamp. Combined, the two contaminated features account for **27.0% of the LightGBM model's total gain**. The audit short-circuits here per the dispatch's hard-stop rule.

## TL;DR

- The producer's swing-detection (`_d1_swing_high_low` in `scripts/l_arc_9/experiments/pipeline_e_retry.py`) uses a centred ±10-bar window: bar k is a swing iff its extreme exceeds bars k±10.
- A swing flag at D1 bar k therefore requires bars k+1..k+10 to have closed. For trades whose lagged D1 bar lies more than 10 bars before the end of the FULL D1 frame (which is essentially every Arc 9 trade), the swing flag at the lagged bar is computed using bars from AFTER the 4H signal fired.
- The producer's `confirmed_high[n - 10:] = False` mask only handles the LAST 10 bars of the FULL D1 frame, not 'the last 10 bars relative to each signal's join date'. The author's docstring states the intent ("the most recent confirmable swing as of that lag is min(lag1_idx - half, ...)") but the code implements frame-end, not lag1_idx-relative.
- Empirical test (50 random Arc 9 trades, top feature `d1_bars_since_swing_low`): **18/50 trades (36%) have a different value when D1 history is truncated to signal-date − 1 day. Median magnitude of leak among leaked trades: 26 bars. Max: 88 bars.**
- Empirical test (10 random Arc 9 trades, second feature `d1_bars_since_swing_high`): **4/10 leak; deltas up to 38 bars.** Same mechanism, same bug.
- The classifier learned the mapping from these lookahead-leaked features to cluster-0 membership. In live deployment, the feature would systematically take much larger values (typical live: 28 bars; production-trained: 3 bars), breaking the learned mapping. The Step 5 OOS AUC 0.7508, the Candidate A worst-fold ann ROI +9.63% / mean +22.92% / DD 1.32%, and the SCALED_RISK extrapolation to 1.0% risk all inherit this contamination.

## Sub-phase verdicts

| Sub-phase | Subject | Verdict | Evidence |
|---|---|---|---|
| **1A — Static dependency proof per feature** | 28 features × their bar references | **PARTIAL FAIL** — 26/28 features pass; 2 features (`d1_bars_since_swing_low`, `d1_bars_since_swing_high`) reference D1 bars after the signal date via the swing-confirmation mechanism. See per-feature cards below. |
| **1B — Mechanical merge_asof verification** | The merge_asof pattern itself | **PASS on join lag (≥ 1 day from signal_date)**, **FAIL on joined-row content (referenced features carry lookahead from inside the D1 frame)**. The producer's existing audit confirmed pattern equivalence with KH-24's `_precompute_d1_exit_arrays`, which is the same correct pattern. The bug is one layer deeper — at the feature computation, not at the join. |
| **1C — Swing-low / swing-high windowing** | `_d1_swing_high_low` + frame-end mask | **FAIL.** Centred ±10 swing detection with frame-end mask `confirmed[n - 10:] = False`. Frame-end is `len(d1_history)`, not the join index. Acceptable construction would be: confirm swing at k iff `k + half <= joined_d1_index` (per-signal, dynamic), or: use a strictly one-sided lookback window (no future bars at all). Neither is implemented. |
| 1D — Permutation null | Not run (short-circuit) | — |
| 1E — `d1_bars_since_swing_low` single-feature ablation | Not run (short-circuit; would be informative but cannot rescue 1B/1C) | — |
| 1F — Top-5 ablation | Not run (short-circuit) | — |

Sub-phases A, B, C **fail the audit by themselves** per the dispatch:

> "Sub-phases A, B, C, D fail the audit if they fail."

The dispatch's "Hard short-circuit" rule activates here. Phases 2–12 are not run.

## Code-level evidence

### The bug

`scripts/l_arc_9/experiments/pipeline_e_retry.py:_build_d1_feature_frame` (lines 254-264):

```python
is_sw_high, is_sw_low = _d1_swing_high_low(d1, half=10)
# Zero out unconfirmed swings (last half bars) — these aren't usable live
# because forward confirmation hasn't happened.
n = len(d1)
confirmed_high = is_sw_high.copy()
confirmed_low = is_sw_low.copy()
if n > 10:
    confirmed_high[n - 10:] = False
    confirmed_low[n - 10:] = False
d1["d1_bars_since_swing_high"] = _bars_since(confirmed_high)
d1["d1_bars_since_swing_low"] = _bars_since(confirmed_low)
```

`n = len(d1)` is the entire D1 frame length (typically ~5000+ bars for the full 2010-2026 history). The mask zeroes out only the last 10 frame bars. For a signal joined to a D1 bar 2000 bars into the frame (typical for trades after 2018), `confirmed_high[i] = is_sw_high[i]` even though `is_sw_high[i]` was computed using bars i+1..i+10.

`scripts/l_arc_9/experiments/pipeline_e_retry.py:_d1_swing_high_low` (lines 185-212):

```python
def _d1_swing_high_low(df_d1: pd.DataFrame, half: int = 10):
    ...
    for k in range(half, n - half):
        if high[k] > np.max(high[k - half:k]) and high[k] >= np.max(high[k + 1:k + half + 1]):
            is_sw_high[k] = True
        if low[k] < np.min(low[k - half:k]) and low[k] <= np.min(low[k + 1:k + half + 1]):
            is_sw_low[k] = True
    return is_sw_high, is_sw_low
```

`is_sw_high[k]` depends on bars `k+1..k+half` (forward). The right-edge of the loop is `n - half`, ensuring all swing checks are *computable* within the frame, but says nothing about what bars are available at signal time.

### The author knew

`_build_d1_feature_frame` docstring acknowledges the issue:

> "swing-high/low detection uses k+1..k+half forward window WITHIN the D1 frame. Because each 4H bar joins to the PRIOR D1 bar (one-day lag via merge_asof('date_minus_1', direction='backward')), and the swing detection requires `half` future D1 bars to confirm, only swings at d1 indices `<= len(d1)-half-1` are confirmed."
>
> "For 4H bars looking at lag-1 D1, **the most recent confirmable swing as of that lag is `min(lag1_idx - half, ...)`**. We compute `bars_since` ON THE D1 FRAME using the confirmed swing flags (only flags where k <= n-half-1)..."

The text describes the correct rule (`lag1_idx - half`) but the code implements `n - half`. The two diverge for every signal whose join point isn't within 10 bars of the frame end.

> "For most arc 9 trades (median entry several years into history), the confirm window is non-binding because there are always >half future bars in the D1 history. Only the last `half` D1 bars (most recent ~10 days) would have an unconfirmed-swing edge artifact. The cost is acceptable; the benefit (no lookahead) is non-negotiable."

This sentence is incorrect. "There are always > half future bars in the D1 history" is true at TRAINING / OFFLINE-EVALUATION time. It is the source of the bug, not protection against it. At live signal time, the future bars do not yet exist; the offline computation pretends they do.

### The producer's existing lookahead audit missed this

`results/l_arc_9/experiments/lookahead_audit/AUDIT_REPORT.md` Audit 2 specifically verified:

| Check | Status |
|---|---|
| `pd.merge_asof(direction='backward')` confirmed in code | YES |
| Pre-shifted date column (`signal_date − 1 day`) confirmed | YES |
| Sample of 560 trades all show `days_lag ≥ 1` | TRUE |
| Monday-morning boundary cases tested | PASS (days_lag = 3) |
| D1 lag pattern matches KH-24 engine | YES, verbatim-equivalent |
| Zero same-day D1 access detected | TRUE |

These checks verify *the merge_asof join*. They do not verify *the feature values inside the joined row*. The bug lives one layer deeper than what was audited. The producer audit's GREEN verdict is correct at its scope but does not exonerate this Phase 1 finding.

## Empirical leak measurements

### Test methodology

1. Pick N random Arc 9 trades (seed 4242).
2. For each trade:
   a. Compute the producer's feature value: build `_build_d1_feature_frame` on the **full** D1 history for the pair; merge_asof at `signal_date − 1 day` to get the row's `d1_bars_since_swing_low` (and `_high`).
   b. Compute the live-equivalent value: truncate the D1 history to `d1_date <= signal_date − 1 day`; rebuild features; same merge_asof.
   c. Compare.
3. Any non-zero delta is a leak.

### `d1_bars_since_swing_low` — n=50

`d1_bars_since_swing_low_leak_test.csv`:

| Statistic | Value |
|---|---|
| Trades leaked (different value live vs prod) | **18/50 (36%)** |
| Among leaked trades — min delta (live − prod) | +11 bars |
| Among leaked trades — median delta | +26 bars |
| Among leaked trades — max delta | +88 bars |
| Mean delta across all 50 | +11.5 bars |
| Median delta across all 50 | 0 bars (since 64% don't leak) |

Direction of bias: **always live > prod** (we are always *more* bars from a confirmed swing in live than in production), because lookahead lets production "see" swing-lows that haven't yet been confirmed. The classifier learned that "small bars_since_swing_low" predicts cluster-0; in live, the feature systematically reports larger values, so the classifier's "fire" condition is rarely triggered correctly.

Selected leaked trades (from CSV):

| trade_id | pair | signal_date | prod_bsl | live_bsl | delta |
|---|---|---|---|---|---|
| 121 | CHF_JPY | 2021-01-07 | 6 | 30 | +24 |
| 223 | NZD_CAD | 2021-04-05 | 7 | 32 | +25 |
| 1021 | GBP_JPY | 2023-01-17 | 9 | 31 | +22 |
| 1446 | AUD_NZD | 2024-02-28 | 3 | 28 | +25 |
| 1504 | NZD_USD | 2024-04-29 | 5 | 59 | +54 |
| (one with max delta) | — | — | small | very large | +88 |

### `d1_bars_since_swing_high` — n=10 sanity check

| trade_id | pair | signal_date | prod_bsh | live_bsh | delta |
|---|---|---|---|---|---|
| 387 | NZD_CAD | 2021-08-02 | 9 | 47 | +38 |
| 511 | CAD_CHF | 2021-11-22 | 2 | 22 | +20 |
| 1830 | USD_CHF | 2025-02-17 | 9 | 24 | +15 |
| 1678 | EUR_AUD | 2024-10-21 | 8 | 29 | +21 |

4/10 leak. Same bug, same mechanism, same direction of bias.

## Feature-importance impact of the leak

From `results/l_arc_9/experiments/pipeline_e_retry/feature_importances.csv`:

| Rank | Feature | importance_gain | share of total |
|---|---|---|---|
| 1 | **d1_bars_since_swing_low** | 5796.1 | **20.2%** (leaked) |
| 2 | **d1_bars_since_swing_high** | 1960.9 | **6.8%** (leaked) |
| 3 | swing_low_dist_atr | 1589.0 | 5.5% |
| 4 | d1_rsi_14 | 1257.2 | 4.4% |
| 5 | d1_pos_in_20d_range | 1242.6 | 4.3% |
| Total all 28 | 28701.7 | 100.0% |

**Combined leaked share: 27.0% of total LightGBM gain.** The classifier's signal extraction is heavily and verifiably reliant on information that is not available at live signal time.

## Per-feature card summary (Phase 1A)

26 of 28 features pass the static dependency check. The 2 failing features are the swing-bars-since pair. Brief cards for the failing features:

### `d1_bars_since_swing_low` (FAIL)

| Field | Value |
|---|---|
| source_tf | D1 |
| bars_referenced | Per join: D1 bar i (the joined row) + D1 bars i+1..i+10 (via swing confirmation) — bars i+1..i+10 are AFTER the signal date |
| function_used | `_bars_since(confirmed_low)` on `_d1_swing_high_low` output with `half=10` |
| worst_referenced_bar | D1 bar `i + 10`, which corresponds to up to ~11 days AFTER the 4H signal date |
| computed_per_fold | NO — computed once on the full D1 history for each pair |
| distributional_features | None |
| Verdict | **FAIL — references D1 bars after the signal timestamp** |

### `d1_bars_since_swing_high` (FAIL)

Same as above with high in place of low. **FAIL.**

### Other 26 features (PASS)

The 8 base-4H, 8 arc-specific, 4 session, and 6 other D1 features (`d1_trend_state`, `d1_atr_ratio_to_4h`, `d1_pos_in_20d_range`, `d1_ret_5d_atr`, `d1_rsi_14`, `d1_close_above_kijun`) all reference bars strictly ≤ their own respective observation point and apply the merge_asof one-day lag correctly. None require future-bar confirmation; ATR/RSI use Wilder smoothing (causal); position-in-range and returns use closed windows. Each card would record "worst_referenced_bar < signal_bar_open" — passes.

## Why this kills Candidate A's deployment claim

The Candidate A spec reports per-fold deployment metrics at 0.5% risk:
- worst-fold ann ROI +9.63% (F3), mean +22.92%, worst DD 1.32%, all 6 folds (F2-F7) positive.

The SCALED_RISK experiment extrapolated to 1.0% risk:
- worst-fold ann ROI +20.07%, mean +51.57%, worst DD 2.52%, worst-day DD 2.01%.

Both sets of metrics are produced by:
1. Training LightGBM per KH-24 fold with anchored-expanding training data
2. Each training set's feature matrix contains the leaked features
3. The classifier learns from leaked information
4. At OOS scoring time, the OOS feature matrix ALSO contains leaked information (because the audit script computes features the same way on the full data)
5. Both training and OOS evaluation systematically benefit from lookahead

In live deployment, neither the training nor the OOS-equivalent inference would have the leaked information. The classifier's probabilities would shift; the admission decision (prob ≥ 0.40) would fire on a different set of trades; the worst-fold ROI, max DD, daily DD, and 5ers compliance properties would all change in unknown directions.

The producer's audit step 8 ("end-to-end probability reproduction") confirms the *retraining is reproducible*, not that *the trained model would behave the same way on live-equivalent inputs*. Reproducibility ≠ no-lookahead.

## What would have to change before this candidate could be re-evaluated

Two minimal fixes, either of which restores Phase 1B/1C:

**Fix option A — Replace the centred swing detector with a one-sided lookback-only detector.**
- A swing-low at bar k iff `low[k] = min(low[k-W..k])` for some window W (no forward bars). 
- This loses the "confirmed by subsequent recovery" property; produces more swing flags including some that price later invalidates.
- Trivially no lookahead by construction.

**Fix option B — Keep the centred detector but enforce the confirmation rule per-join.**
- Compute `is_sw_low[k]` only for `k + half <= signal_d1_index`. This must be evaluated at each signal's join time, not at frame-build time.
- Most natural implementation: compute swing flags within `_attach_d1_features` per-trade, knowing each trade's lookup_date, rather than in `_build_d1_feature_frame` globally.
- Preserves the centred-swing meaning but constructs features causally.

Either fix would require retraining the classifier, regenerating the per-fold AUCs, re-running Step 5, and re-running SCALED_RISK. The audit's Phase 0-12 then run on the corrected pipeline. This is non-trivial work and out of scope for the audit itself.

## What this finding does NOT say

- It does not say the IB-trend signal class is dead. The Step 1 trade pool, Step 2 path-shape clustering, and Step 3 capturability characterisation of cluster_0_individual (fwd_mfe_p50 6.18R, frac_wrong_way_pre_peak 0.000, final_r_mean +4.423R, t-stat +21.81) are computed from forward path geometry — not from the leaked features. The cluster is real; the cohort's capturability is real. The audit kills only the *deployment claim* that the LightGBM Pipeline E classifier can identify cluster_0 trades at entry time with the reported AUC 0.7508.
- It does not invalidate Step 4's original disposition (STEP_4_KILL). The original Step 4 used 16 RF features with no D1-lag-since-swing features; its AUC of 0.511 had no leak path and was correctly KILL'd. The lookahead was introduced in the post-closure Pipeline E retry experiment when the 8 D1-lagged features (including the two leaky ones) were added.
- It does not invalidate KH-24's anchor deployment. KH-24 uses a different signal (`kb_exhaustion_bar`), a different exit policy, and does not use `d1_bars_since_swing_*` features anywhere. The Phase 11 anchor preservation check is moot because the audit short-circuits before reaching Phase 11.

## Verdict

**Phase 1: FAIL.** Two of the three top-importance features (combined 27.0% of total model gain) reference D1 bars that are after the 4H signal timestamp. The deployment claim that the Candidate A LightGBM classifier extracts cluster_0 membership at entry time with AUC 0.7508 is invalidated.

Per the dispatch's hard short-circuit rule, Phases 2-12 are not run. The final verdict is **NOT-TRADEABLE at 1% risk** on Phase 1 (1B + 1C) audit-fatal failure.

## Files (Phase 1 evidence)

- `d1_bars_since_swing_low_leak_test.csv` — 50-trade live-vs-prod comparison
- `leak_test_summary.txt` — summary statistics
- This document: `phase_1_lookahead.md`
