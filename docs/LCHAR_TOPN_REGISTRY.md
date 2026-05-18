# LCHAR_TOPN_REGISTRY — L Characterization Arc Top-N Candidate Registry

**Source:** mechanically generated from `results/lchar/atlas/layer4_conditional.csv`
**Selection script:** `scripts/lchar/run_layer5_selection.py`
**L0 reference:** `docs/L0_METHODOLOGY_LOCK.md` §5
**Sort key (locked):** descending DSR, then descending Sharpe, then descending `n_obs_pooled` (stable mergesort)
**Threshold (locked):** DSR > 0.95 for L6+ proceed-flag (strict `>`).
**N (locked):** 5 entries.

Per L0 §3 the registry is descriptive. Trials flagged `PROCEEDS_TO_L6` are **candidates for signal testing**, not confirmed tradeable signals. They are hypotheses to be validated in subsequent L6+ arcs under the project-permanent KH-arc methodology (ex-ante population, WFO worst-fold, etc.).

---

## Entry 1: `TRIAL__univariate_extreme__abs_return_top_decile__neg__h_001`

- **Family:** `univariate_extreme`
- **Base condition:** |log return| in top decile of trailing 100 bars
- **Direction / sub-spec:** `neg` — signal bar close < open
- **Signal TF:** `1H`
- **Horizon:** 1 bars (1 bar = 1 hour)
- **DSR:** 1  (95% CI [0.999148, 1])
- **Raw Sharpe:** 0.0365173  (95% CI [0.0275685, 0.04511])
- **Pooled return mean (ATR-normalised):** 0.0306303
- **Pooled return std (ATR-normalised):** 0.83879
- **Skew, excess kurtosis:** 0.055824, 15.457
- **n_obs_pooled:** 48,814
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.028068, p25 0.0017883, p75 0.058041, range [-0.03274, 0.16009]
- **Threshold disposition:** `PROCEEDS_TO_L6`

## Entry 2: `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120`

- **Family:** `mtf_alignment`
- **Base condition:** Extremes both down, 4H_mr up — mixed-down state
- **Direction / sub-spec:** `kijun` — Kijun-sign trend definition
- **Signal TF:** `1H`
- **Horizon:** 120 bars (120 bars = 5 trading days = ~1 week)
- **DSR:** 0.999999  (95% CI [0.99769, 1])
- **Raw Sharpe:** 0.0383145  (95% CI [0.0288244, 0.0476128])
- **Pooled return mean (ATR-normalised):** 0.277317
- **Pooled return std (ATR-normalised):** 7.23792
- **Skew, excess kurtosis:** -0.45584, 5.8073
- **n_obs_pooled:** 40,572
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.056475, p25 -0.017012, p75 0.22231, range [-0.17325, 0.40173]
- **Threshold disposition:** `PROCEEDS_TO_L6`
- **Arc disposition:** **KILL at Step 3** (Arc 2 redo — 2026-05-16) under v2.0. Cluster 2 (Stepwise climber) carried strong magnitude (fwd_mfe_p50 5.83R, t-stat +52.17, n=2,278) but failed §2 monotonicity floor by 0.0086 + frac_wrong_way by 0.0051 + shape_tag. See `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`.
- **Trigger overlap note (2026-05-17):** **Shares byte-identical entry trigger with Entry 5** (`TRIAL__mtf_alignment__2_down_mixed__kijun__h_024`); only horizon differs. Arc 5 at h=120 (chat override of registry h=24) produced an Arc 2 redo2-byte-identical Step 1 trade pool. Arc 2 redo's KILL disposition (v2.0) and Arc 5's SHELVED disposition (v2.1.1 + PR 2) together establish the signal trigger's full picture: the signal carries genuine admit-set edge under v2.1.1's anchor-rescue, but Pipeline D1 cannot extract it due to rejected-pool adverse selection (mean R −0.46). No further work on this entry trigger without a pipeline architecture change (Pipeline E retrofit with richer features, or new pipeline shape that avoids the rejected-pool cost).

## Entry 3: `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120`

- **Family:** `volatility_regime`
- **Base condition:** D1 ATR(14) in top decile of trailing 100 D1 bars (lower-TF signals use L2 most-recently-completed lookback)
- **Direction / sub-spec:** `any` — no direction sub-filter
- **Signal TF:** `1H`
- **Horizon:** 120 bars (120 bars = 5 trading days = ~1 week)
- **DSR:** 0.999964  (95% CI [0.977818, 1])
- **Raw Sharpe:** 0.0212461  (95% CI [0.0152087, 0.0269815])
- **Pooled return mean (ATR-normalised):** 0.128978
- **Pooled return std (ATR-normalised):** 6.07068
- **Skew, excess kurtosis:** -0.29263, 4.8731
- **n_obs_pooled:** 106,560
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.060905, p25 -0.037157, p75 0.24839, range [-0.36951, 0.48977]
- **Threshold disposition:** `PROCEEDS_TO_L6`

## Entry 4: `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001`

- **Family:** `univariate_extreme`
- **Base condition:** Bar range (high − low) in top decile of trailing 100 bars (per pair × signal_TF)
- **Direction / sub-spec:** `neg` — signal bar close < open
- **Signal TF:** `1H`
- **Horizon:** 1 bars (1 bar = 1 hour)
- **DSR:** 0.999951  (95% CI [0.973678, 1])
- **Raw Sharpe:** 0.0313529  (95% CI [0.0224146, 0.0406223])
- **Pooled return mean (ATR-normalised):** 0.0267845
- **Pooled return std (ATR-normalised):** 0.85429
- **Skew, excess kurtosis:** 0.084388, 15.07
- **n_obs_pooled:** 47,690
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.021786, p25 -0.014172, p75 0.050645, range [-0.034016, 0.19239]
- **Threshold disposition:** `PROCEEDS_TO_L6`

## Entry 5: `TRIAL__mtf_alignment__2_down_mixed__kijun__h_024`

- **Family:** `mtf_alignment`
- **Base condition:** Extremes both down, 4H_mr up — mixed-down state
- **Direction / sub-spec:** `kijun` — Kijun-sign trend definition
- **Signal TF:** `1H`
- **Horizon:** 24 bars (24 bars = 1 day) — *Arc 5 chat override set time_exit_bars=120; with that override Arc 5's Step 1 pool became byte-identical to Arc 2 redo2 (Entry 2 trigger).*
- **DSR:** 0.999769  (95% CI [0.931629, 1])
- **Raw Sharpe:** 0.0320725  (95% CI [0.0220306, 0.0415913])
- **Pooled return mean (ATR-normalised):** 0.105752
- **Pooled return std (ATR-normalised):** 3.29727
- **Skew, excess kurtosis:** -0.16201, 6.7699
- **n_obs_pooled:** 40,619
- **Per-pair Sharpe diagnostic** (across 28 pairs): median 0.051242, p25 -0.01527, p75 0.12302, range [-0.1385, 0.20096]
- **Threshold disposition:** `PROCEEDS_TO_L6`
- **Arc disposition:** **SHELVED (Arc 5 — 2026-05-17)** at Step 6 FAIL under PR 2 + per-pair p50 spread floors + full-pool WFO accounting.
- **Reason:** Admit-set edge confirmed — v2.1.1 anchor-rescue cleared c1 §2 (composite 0.593); D1 RF AUC 0.636/0.640 at t=1 anchor-parity to KH-24; admit-set mean R +0.14 (c1) / +0.21 (c3) positive across all 7 folds under PR 2's §11 row 2 exit (MFE-lock + trail). PR 2 also rescues c1's §9 DD ratio (2.34 → 1.17). **But Pipeline D1's rejected pool (~78% of trades, classifier P < threshold → close at bar 2) has mean R −0.46 (vs +0.025 unconditional bar-2 R) — adverse-selected by construction (a competent classifier rejects trades whose path-so-far looks bad, which correlates with continued drift).** All three candidate strategies (c1 alone, c3 alone, tiered ensemble) FAIL §10 at every risk level. The signal-pipeline mismatch is the failure mode, not the signal itself.
- **Reopen condition:** Pipeline E re-evaluation with richer feature set (current E AUC 0.566 is feature-set bound, RF−LR gap ≤ 0.014 — E filters at entry, avoids rejected-pool cost entirely if it clears 0.65); OR alternative pipeline shape (e.g. E pre-filter + D1-on-survivors hybrid) that doesn't bear the rejected-pool cost; OR full Steps 2-4 retrain under new spreads with re-trained classifiers + re-selected thresholds. See `docs/arc_results/ARC_5_RESULT.md` §"Conditions for reopening".

---

## Family concentration

- Top 5 spans **3** condition families (mtf_alignment (2), univariate_extreme (2), volatility_regime (1)) and **4** unique base conditions.
- Repeated bases (parameter variations): `2_down_mixed` × 2.
- Signal-TF distribution in top 5: `1H` (5).
- Horizons present: 1, 24, 120.

## Registry exhaustion note (2026-05-17, post-Arc-5)

With Arc 5 SHELVED, the registry's h≥120 entries are now exhausted across the L arc programme:

- Entry 1 (h=1) — SKIPPED per chat decision (h=1 trade horizons outside active scope).
- Entry 2 (h=120) — **KILL** at Step 3 (Arc 2 redo, 2026-05-16) under v2.0.
- Entry 3 (h=120) — **CLEAN-NULL** at Step 3 (Arc 3, 2026-05-16) under v2.0.
- Entry 4 (h=1) — SKIPPED per chat decision (h=1 outside active scope); separately Arc 4 took the same trigger and CLOSED CLEAN-NULL on transaction-cost truth (2026-05-17).
- Entry 5 (h=24, run at h=120 chat override) — **SHELVED** at Step 6 FAIL (Arc 5, 2026-05-17).

Remaining h=1 registry entries are outside the L arc programme's active scope (chat-level decision: h=1 horizons not pursued under current registry framing). Further signal exploration moves out-of-registry pending P0 backlog work (`PROTOCOL_IMPROVEMENT_BACKLOG.md`, items P-§9-FRAMING / P-D1-VIABILITY / P-D1-REJECT-BIAS).

---

## Cross-entry observations

Per L0 §3 these are descriptive observations only — no claim is made or implied about predictability or tradeable edge. The registry says where the atlas's mechanical bridge points; it does not endorse what is there. Specifically:

- The entries surface from L4 trials run on the full 2020-10-01 → 2025-12-31 window without out-of-sample partition. Whether they survive walk-forward gating is a separate question and is the explicit job of any L6+ signal arc that adopts a registry entry as a hypothesis.
- DSR magnitudes near 1.0 reflect very large pooled sample sizes (n_obs_pooled ≥ 40,572 across the listed entries). With T this large, even small Sharpe deviations from zero saturate the deflated Sharpe under the locked `N_trials = 300` selection-bias correction. This is structural to the metric, not an interpretive claim.
- Per-pair Sharpe diagnostics characterise dispersion across the 28-pair universe; consumers of the registry should inspect them before treating any entry as universe-wide.

