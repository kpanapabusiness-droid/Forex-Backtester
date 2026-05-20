# L_ARC_DEFERRED_CANDIDATES — Trials cleared by the L0 threshold but capped out of the L5 registry

**Source:** mechanically generated from `results/lchar/atlas/layer4_conditional.csv` and `results/lchar/_l5_registry_data.json`.
**Generation script:** `scripts/lchar/extract_deferred_candidates.py`
**Generation config:** `configs/lchar/deferred_candidates.yaml`
**L0 reference:** `docs/L0_METHODOLOGY_LOCK.md` §4 (ranking + tie-break), §5 (N=5 threshold semantics)
**L5 registry reference:** `LCHAR_TOPN_REGISTRY.md` (the 5 trials excluded here)
**Sort key (locked, identical to L5):** descending DSR, then descending Sharpe, then descending `n_obs_pooled` (stable mergesort)
**Threshold (locked):** DSR > 0.95 (strict `>`).

---

## What this document is

Across the L4 trial population, **12** trials cleared the L0-locked threshold of DSR > 0.95 by the L0-locked metric. The L5 registry holds the top 5 of them by the L0-locked sort key. This document enumerates the remaining **7** trials.

These deferred trials are not "failed". They cleared the same threshold by the same metric. They were truncated only by the operational N=5 cap that L0 §5 sets on the registry. They are documented here so the L6+ planning conversation has a complete view of what cleared the bridge before the cap was applied.

## What this document is NOT

- **Not a registry.** The L5 registry is `LCHAR_TOPN_REGISTRY.md`. That document is the locked output of the L arc and is unchanged.
- **Not a promotion.** No trial enumerated here is approved for L6+ signal testing. Per L0 §1, modifications to the L-arc methodology — including any expansion of N or any reclassification of these trials as candidates — require an explicit re-planning phase.
- **Not a re-evaluation.** No new statistical computation is performed. DSR, Sharpe, mean/std/skew/kurtosis, and per-pair diagnostics are read directly from `layer4_conditional.csv`. The selection-bias correction (`N_trials = 300`) used in DSR is unchanged.
- **Not predictive.** Per L0 §3, every number here is a descriptive statistic of the historical conditional return distribution under a given trial specification. No claim is made or implied about predictability or tradeable edge.

## How it was produced

1. Load `results/lchar/atlas/layer4_conditional.csv` (single read).
2. Load `results/lchar/_l5_registry_data.json` and extract the set of trial_ids in the L5 registry.
3. Pivot `stat_family == "l4_trial_metric"` rows to one row per `(trial_id, signal_tf)` with columns for each stat.
4. Filter to `value__dsr > 0.95` AND `trial_id NOT IN registry`.
5. Sort by `[value__dsr, value__sharpe, value__n_obs_pooled]` ascending=`[False, False, False]`, kind="mergesort" — identical to L5.
6. For each row, parse the trial_id (`TRIAL__<family>__<base>__<sub>__h_<HHH>`), build the corresponding `DIAG__...` prefix, and aggregate the `l4_trial_diag` rows for that prefix at the matching timeframe to produce the per-pair Sharpe distribution (n_pairs, median, p25, p75, min, max). This logic is imported directly from `scripts/lchar/run_layer5_selection.py` to guarantee identical behaviour.
7. Emit entries ranked 6 through 12, immediately following the L5 registry's ranks 1..5.

---

## Entry 6: `TRIAL__volatility_regime__d1_atr_top_decile__any__h_024`

- **Family:** `volatility_regime`
- **Base condition:** D1 ATR(14) in top decile of trailing 100 D1 bars (lower-TF signals use L2 most-recently-completed lookback)
- **Direction / sub-spec:** `any` — no direction sub-filter
- **Signal TF:** `1H`
- **Horizon:** 24 bars (24 bars = 1 day)
- **DSR:** 0.999424  (95% CI [0.894262, 1])
- **Raw Sharpe:** 0.0190014  (95% CI [0.012865, 0.0247481])
- **Pooled return mean (ATR-normalised):** 0.0538772
- **Pooled return std (ATR-normalised):** 2.83543
- **Skew, excess kurtosis:** -0.070546, 7.0465
- **n_obs_pooled:** 106,560
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.033082, p25 -0.033377, p75 0.10326, range [-0.12198, 0.21684]
- **Cap-deferred from L5 registry:** rank 6 by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.

## Entry 7: `TRIAL__mtf_alignment__3_down__sma__h_024`

- **Family:** `mtf_alignment`
- **Base condition:** All three timeframes trending down under the trend definition
- **Direction / sub-spec:** `sma` — SMA-sign trend definition
- **Signal TF:** `1H`
- **Horizon:** 24 bars (24 bars = 1 day)
- **DSR:** 0.997459  (95% CI [0.77858, 0.999999])
- **Raw Sharpe:** 0.0137486  (95% CI [0.00887013, 0.018267])
- **Pooled return mean (ATR-normalised):** 0.0464389
- **Pooled return std (ATR-normalised):** 3.37772
- **Skew, excess kurtosis:** -0.66643, 8.5713
- **n_obs_pooled:** 175,771
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.02799, p25 -0.010002, p75 0.051188, range [-0.03503, 0.10012]
- **Cap-deferred from L5 registry:** rank 7 by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.

## Entry 8: `TRIAL__univariate_extreme__bar_range_top_decile__pos__h_001`

- **Family:** `univariate_extreme`
- **Base condition:** Bar range (high − low) in top decile of trailing 100 bars (per pair × signal_TF)
- **Direction / sub-spec:** `pos` — signal bar close > open
- **Signal TF:** `1H`
- **Horizon:** 1 bars (1 bar = 1 hour)
- **DSR:** 0.992599  (95% CI [0.656522, 0.999993])
- **Raw Sharpe:** 0.0239943  (95% CI [0.0149158, 0.0325358])
- **Pooled return mean (ATR-normalised):** 0.0192527
- **Pooled return std (ATR-normalised):** 0.802388
- **Skew, excess kurtosis:** -0.16216, 10.162
- **n_obs_pooled:** 50,517
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.028161, p25 0.0063454, p75 0.044728, range [-0.035613, 0.12351]
- **Cap-deferred from L5 registry:** rank 8 by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.

## Entry 9: `TRIAL__mtf_alignment__3_down__sma__h_001`

- **Family:** `mtf_alignment`
- **Base condition:** All three timeframes trending down under the trend definition
- **Direction / sub-spec:** `sma` — SMA-sign trend definition
- **Signal TF:** `1H`
- **Horizon:** 1 bars (1 bar = 1 hour)
- **DSR:** 0.992218  (95% CI [0.648824, 0.999995])
- **Raw Sharpe:** 0.0128028  (95% CI [0.00794317, 0.0175364])
- **Pooled return mean (ATR-normalised):** 0.00912349
- **Pooled return std (ATR-normalised):** 0.712618
- **Skew, excess kurtosis:** 0.0045644, 22.185
- **n_obs_pooled:** 175,863
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.011441, p25 0.0049941, p75 0.017239, range [-0.020457, 0.04173]
- **Cap-deferred from L5 registry:** rank 9 by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.

## Entry 10: `TRIAL__mtf_alignment__3_down__kijun__h_001`

- **Family:** `mtf_alignment`
- **Base condition:** All three timeframes trending down under the trend definition
- **Direction / sub-spec:** `kijun` — Kijun-sign trend definition
- **Signal TF:** `1H`
- **Horizon:** 1 bars (1 bar = 1 hour)
- **DSR:** 0.988961  (95% CI [0.659503, 0.999988])
- **Raw Sharpe:** 0.0126153  (95% CI [0.00809195, 0.0172675])
- **Pooled return mean (ATR-normalised):** 0.00921322
- **Pooled return std (ATR-normalised):** 0.730322
- **Skew, excess kurtosis:** 0.084163, 24.149
- **n_obs_pooled:** 172,392
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.013526, p25 0.0061711, p75 0.017901, range [-0.021655, 0.038462]
- **Cap-deferred from L5 registry:** rank 10 by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.

## Entry 11: `TRIAL__mtf_alignment__3_down__kijun__h_024`

- **Family:** `mtf_alignment`
- **Base condition:** All three timeframes trending down under the trend definition
- **Direction / sub-spec:** `kijun` — Kijun-sign trend definition
- **Signal TF:** `1H`
- **Horizon:** 24 bars (24 bars = 1 day)
- **DSR:** 0.987134  (95% CI [0.581823, 0.999984])
- **Raw Sharpe:** 0.0124959  (95% CI [0.00760335, 0.0171577])
- **Pooled return mean (ATR-normalised):** 0.0432324
- **Pooled return std (ATR-normalised):** 3.45972
- **Skew, excess kurtosis:** -0.5084, 8.9714
- **n_obs_pooled:** 172,280
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.026237, p25 -0.0027739, p75 0.049636, range [-0.045623, 0.10287]
- **Cap-deferred from L5 registry:** rank 11 by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.

## Entry 12: `TRIAL__mtf_alignment__2_down_mixed__sma__h_024`

- **Family:** `mtf_alignment`
- **Base condition:** Extremes both down, 4H_mr up — mixed-down state
- **Direction / sub-spec:** `sma` — SMA-sign trend definition
- **Signal TF:** `1H`
- **Horizon:** 24 bars (24 bars = 1 day)
- **DSR:** 0.969536  (95% CI [0.417159, 0.999913])
- **Raw Sharpe:** 0.0231251  (95% CI [0.0131081, 0.0321506])
- **Pooled return mean (ATR-normalised):** 0.0782874
- **Pooled return std (ATR-normalised):** 3.38539
- **Skew, excess kurtosis:** -0.3655, 6.5505
- **n_obs_pooled:** 43,658
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.040983, p25 -0.0038921, p75 0.077647, range [-0.071608, 0.17261]
- **Cap-deferred from L5 registry:** rank 12 by the L0-locked sort key — above the L0-locked DSR threshold but outside the N=5 cap.

---

## Family concentration (deferred set only)

- The deferred set spans **3** condition families (mtf_alignment (5), univariate_extreme (1), volatility_regime (1)) and **4** unique base conditions.
- Repeated bases (parameter variations): `3_down` × 4.
- Signal-TF distribution in deferred set: `1H` (7).
- Horizons present: 1, 24.

## Cross-entry observations

Per L0 §3 these are descriptive observations only. The same caveats that apply to the L5 registry apply here:

- The entries surface from L4 trials run on the full 2020-10-01 → 2025-12-31 window without out-of-sample partition. Whether any entry survives walk-forward gating is a separate question and would be the explicit job of any L6+ signal arc that adopts a deferred entry as a hypothesis (which itself requires re-planning).
- DSR magnitudes near 1.0 reflect very large pooled sample sizes. With T this large, even small Sharpe deviations from zero saturate the deflated Sharpe under the locked `N_trials = 300` selection-bias correction. This is structural to the metric, not an interpretive claim.
- Per-pair Sharpe diagnostics characterise dispersion across the 28-pair universe; consumers should inspect them before treating any entry as universe-wide.

---

## Reproducibility

All numerical content in this document is sourced from `layer4_conditional.csv` (the locked L4 atlas) and `_l5_registry_data.json` (the locked L5 registry). No new statistical computation is performed.

**Inputs (sha256 of byte content at generation time):**

- `results/lchar/atlas/layer4_conditional.csv` — `c32b13432ed340ec6044f7d51be7d88efbd4f72a4965170e60ff9dbb14e36f56`
- `results/lchar/_l5_registry_data.json` — `975c58c5837b5265740601b160363a410738794752dbcf62a6ff77eaa07d1216`

**Sibling output:**

- `results/lchar/_l_arc_deferred_candidates.json` — `e15c46874631b47b71e1201caf8cd6abb173416a210d8c2a0985db6d0f54140c`

**Body sha256 (this document up to but not including this Reproducibility section):**

- `8de48b39099710d47b2783bda93107a33003477573a82ed9f236683111e7899c`

**Determinism check:**

```
sha256sum docs/L_ARC_DEFERRED_CANDIDATES.md > before.sha
py scripts/lchar/extract_deferred_candidates.py -c configs/lchar/deferred_candidates.yaml
sha256sum docs/L_ARC_DEFERRED_CANDIDATES.md > after.sha
diff before.sha after.sha   # must be empty
```
