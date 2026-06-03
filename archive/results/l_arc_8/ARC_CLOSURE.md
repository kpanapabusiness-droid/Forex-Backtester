# ARC_8_CLOSURE — l_arc_8

> **Closed:** 2026-05-22T11:00:00Z
> **Branch:** arc/l_arc_8
> **Closure doc path:** results/l_arc_8/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:

  template_version: v1.2

  # ────── Identity ──────
  arc_name: l_arc_8
  signal: pullback_resume_hhhl_long_v0.1 (HH/HL uptrend, pullback >=0.5xATR, bullish-close break of prior bar)
  tf: 4H
  sub_protocol: vanilla
  closed_timestamp: 2026-05-22T11:00:00Z
  closure_doc_link: results/l_arc_8/ARC_CLOSURE.md

  # ────── Verdict ──────
  verdict: FAIL
  one_line: Search WFO worst-fold ratio 1.749 (gate >=2.0) on top A6 config; oracle 999 confirms V-shape capturable but Step 4 AUC 0.53 entry-feature ceiling caps real WFO.
  failed_at_step: 5
  primary_failure_mode: entry_feature_auc_ceiling

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: 6757
    window_start: 2010-01-01
    window_end: 2026-04-10
    kh24_co_fire_pct: 0.0
    configs_evaluated_step5: 72
    search_scope_flag: normal

  # ────── Best architecture (top of search-WFO ranking; FAIL so candidate not deployable) ──────
  best_architecture:
    name: A6 meta_labeling
    cluster: 2
    archetype: V-shape
    config: A6_SL1.5_TP3R_unlimited_thr_0.3_0.5
    sl_atr: 1.5
    exit_policy: sl_plus_tp_3r
    exposure_cap: unlimited
    worst_fold_ratio: 1.749
    worst_fold_roi_pct: 1.7486
    worst_fold_dd_pct: 1.9826
    mean_fold_ratio: 7.468
    mean_fold_roi_pct: 5.902
    sign_pos_folds: "11/11"
    n_trades_total: 4434
    holdout_roi_pct: 42.7803
    holdout_dd_pct: 1.7369
    holdout_passed: true
    oracle_worst_ratio: 999.0
    oracle_real_gap_sharpe: null
    features_in_winning_config: [swing_low_distance_14, prior_session_low_distance, atr_vs_trailing_100, kijun_26_distance, dollar_bloc_state, atr_percentile_100, spread_vs_trailing_100, spread_percentile_100, usd_strength_index, distance_to_round_number]

    # ── v1.2 deployment-spec fields (added 2026-05-23 retrofit; FAIL arc — for documentation parity) ──
    config_artefact_path: configs/l_arc_8/winning_config.yaml
    deployment_spec_section_present: true

  # ────── Cost decomposition ──────
  # A6 is sizing-based, not binary admit/reject — the admit/reject framing
  # doesn't capture (0x / 0.5x / 1.0x) tiers cleanly. Set null per template
  # ambiguity; the (lo, hi) threshold pair + per-tier R lives in Step 5's
  # per_fold_metrics.csv.
  cost_decomposition: null

  # ────── Per-cluster results ──────
  clusters:
    c0:
      n: 1657
      archetype: Monotonic_down
      sl_atr: 4.0
      step3_composite: 0.393
      mfe_p50_r: 0.069
      ww_pp: 0.0284
      reach_1r: 0.0
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c1:
      n: 3560
      archetype: Choppy
      sl_atr: 1.5
      step3_composite: 0.562
      mfe_p50_r: 1.246
      ww_pp: 0.4003
      reach_1r: 0.5978
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c2:
      n: 1540
      archetype: V-shape
      sl_atr: 1.5
      step3_composite: 1.000
      mfe_p50_r: 7.527
      ww_pp: 0.0
      reach_1r: 1.0
      step4_e_auc: 0.5300
      step4_d1_auc: null
      outcome: dies_step5

  # ────── Architecture results ──────
  architectures_tested: [A1, A6]
  architecture_results:
    A1: {tested: true, won: false, worst_fold_ratio: 0.171}
    A6: {tested: true, won: true,  worst_fold_ratio: 1.749}
    A3: {tested: false, won: false, worst_fold_ratio: null}

  # ────── Archetypes observed ──────
  archetypes_observed: [Monotonic_down, Choppy, V-shape]

  # ────── Cross-arc observation tags ──────
  cross_arc_tags:
    - v_shape_auc_ceiling
    - step4_classifier_at_chance
    - oracle_real_gap_massive
    - multi_tf_features_absent_at_step1
    - search_wfo_fail_holdout_pass_pattern
```

---

## §2 Why failed

The arc died at the L_PROTOCOL §3 PASS-DEPLOYABLE gate on Step 5 — best A6 worst-fold ratio 1.749 against a 2.0 threshold (margin −0.251). The proximate cause is the gate miss; the structural cause is **Step 4's entry-feature AUC ceiling at 0.53** across all three classifiers (RF 0.530, LGBM 0.521, Logistic 0.523), which means the meta-labeling sizing in A6 has almost no separation power. With AUC ≈ chance, A6 effectively sizes all trades alike and the architecture collapses toward a baseline-with-noise — which produces 11/11 positive folds with bounded magnitude (worst ROI +1.75% over a 1.98% DD) but cannot lift the worst-fold ratio above the gate.

The Step 5 **oracle** WFO at TRUE cluster-2 labels hits worst-fold ratio 999 (effectively infinite — every fold clean +60-135% ROI on the 638-980 cluster-2 trades, DD ≤ 1%) — the cluster is genuinely capturable, the failure is at the entry-time identification step. This is the same V-shape AUC-ceiling pattern that closed Arc 10 (DLR, 2026-05-18): Step 4 near-miss on the disjunctive gate, oracle WFO comfortable, real WFO blocked at the classifier.

The A1 (no-filter) family is much worse: best A1 worst-fold ratio 0.171 (config 4: SL=1.5, sl_plus_tp_3r, unlimited). A1 admits the full 6,757-trade pool where 77% are losers (C0 + C1) — the small cluster-2 edge gets washed out. So A6's edge over A1 (1.749 vs 0.171, ~10×) is real even at AUC 0.53 — sizing differentiation matters — but the absolute ratio is still below gate.

Note on what the holdout shows: the top-3 holdout (2021-04 → 2026-04, one-shot) is **PASS-DEPLOYABLE on all three** (ROI +19% to +43%, DD ≤ 1.7%, ratios 19-35). L_PROTOCOL §2 Step 5 holdout-decision-rule requires BOTH the 11-fold search WFO and the holdout to pass the §3 gate; search WFO doesn't, so the arc verdict is FAIL despite the holdout strength. Per §2 Step 5: "Re-evaluating the same candidate after parameter tweaks is forbidden (selection bias laundering)." The asymmetric pattern (search-WFO worst-ratio just below gate, holdout comfortably above) is itself a documented cross-arc tag — see §3.

Two methodology caveats noted in the closure for cross-arc context: (1) the pool-level WFO approximation does NOT use the bar-by-bar multipair backtester (`MultiPairBacktester`); per-trade R outcomes from Step 1 carry through with exit-policy approximation. This understates intraday DD timing and ignores cross-pair concurrency interactions; (2) Step 1 multi_tf features (`d1_close_slope_*`, `w1_close_slope_sign`, `d1_atr_percentile_100`) came back all-NaN because `compute_feature_matrix` requires `panel.aux={"d1":..., "w1":...}` but the Step 1 driver only passed an H4 panel; the fix is staged but not re-run. Effective Step 4 feature catalogue was 23 features (the 27-feature v3 catalogue minus those four multi_tf entries). This is the load-bearing engine-integration gap of this arc.

---

## §3 Cross-arc observations

- **Third documented V-shape near-miss in 6 months** — Arc 6 (failed_breakout_reversal, Step 4 E AUC 0.600), Arc 10 (DLR c1, E AUC 0.6296 / D1 0.5897), Arc 8 (this arc, Step 4 RF AUC 0.5300). All three have clean capturability (Step 3 composite ≥ 0.49) and fail at the entry-time AUC barrier. Arc 8 is the lowest AUC of the three — confirms the pattern is a function of the v3 27-feature envelope, not signal-specific. EXP-05 cross-arc pool framing (Arc 10's c1 + Arc 7's c3 pooled, AUC 0.6348 with shared 17-feature subset) is now joinable with Arc 8 c2 — three-arc V-shape clusterifier pool would add +1,540 trades to that envelope.
- **Oracle/real WFO ratio gap is the largest documented yet** — Oracle worst ratio 999.0 vs real worst ratio 1.749 (gap factor >500×). Arc 10 showed Sharpe 4.61 vs −1.29 (gap +5.90). Arc 8's gap is wider because cluster-2 is more homogeneous (100% reach_1R, 0% ww_pp) — the upside is whoppingly capturable; the entry classifier just can't find it. Confirms that "envelope expansion" (richer entry-time features) is the deployable lever for V-shape archetypes, not architecture variation.
- **Multi-TF feature absence amplifies AUC ceiling** — Step 4 ran on 23 features (D1 + W1 slope/percentile features absent due to engine-integration gap). The top permutation-importance features came from price geometry, distance, and spread regime classes; cross-asset features (`dollar_bloc_state`, `usd_strength_index`) ranked top-5/9. Re-running Step 1 with the aux-panel fix would restore 4 features, two of which (D1 slope sign, D1 ATR percentile) are HTF context that historically lifts V-shape classification (Arc 10 EXP-05 found `L1_minus_L0_atr`, an HTF magnitude feature, carried 116% of the cross-arc LOO drop). Engine-PR scope.
- **Search WFO FAIL + holdout PASS-DEPLOYABLE is a new pattern** — Top-3 holdout all clear the deployable gate (worst holdout ratio 19.4) while search WFO worst ratio is 1.749. Per §2 Step 5 the arc verdict is FAIL (both gates required). But the asymmetry is worth tagging: 2021-2025 holdout had a more favourable market regime for V-shape pullback-resume than 2010-2020 search. Cross-arc question: do other v3 arcs show this asymmetry? If yes, the 11-fold 2010-2020 search WFO may be biased toward a different regime than the deployment-window holdout.
- **A6 with chance-AUC classifier still adds value** — A6 best (ratio 1.749) is 10× A1 best (ratio 0.171). Even at AUC 0.53, sizing differentiation moves the needle. This argues for A6-style sizing being a useful default architecture regardless of classifier strength, provided sizes are bounded (here: 0x/0.5x/1.0x mapping). Worth a v2.4 calibration thought.
- **KH-24 co-fire on PR-HHHL: 0.000% across 6,757 signals** — Confirms the spec's "independence expected" prior. PR-HHHL bullish-resume and KH-24 bearish-exhaustion are structurally orthogonal. This frees PR-HHHL from any portfolio-overlap accounting if it's ever resurrected.

---

## §4 deployment_spec (FAIL arc — included for consistency, NOT for deployment; portfolio-eligibility flagged for Arc 8)

> Abbreviated per dispatch — sub-sections 4.1-4.3 and 4.5-4.10 written from artefacts; §4.4 and §4.11 marked "FAIL arc — not applicable for deployment."
>
> FAIL verdict (closure §1 + §10). Config retained for portfolio-eligibility audit only (see closure §10 `portfolio_eligibility_pending_a5_spec`).

### 4.1 Pair set

- **Pairs:** AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY (28 FX pairs, KH-24 set)
- **Timeframe:** H4 primary
- **Higher-TF references:** D1 anchor (one-bar-lagged); W1 aux (Step 1 feature engineering only — Step 1 d1/w1 features came back all-NaN per closure §2 caveat, so the deployed-version feature catalogue was 23 features rather than the v3 27-feature default)

### 4.2 Signal definition

Pullback-resume HHHL long (PR-HHHL v0.1) — pullback within a confirmed HH/HL uptrend, with bullish close breaking the prior bar's high.

Pseudocode:

```
# Maintain a rolling list of confirmed swing-highs (k=3, right_edge_lag=4) on H4.
# Uptrend test: within the last trend_window_bars=30, the swing-highs are ascending
# (each higher than the prior); same for swing-lows.
#
# At each H4 bar close on pair P:
#   t       = bar close timestamp (UTC)
#   atr14   = ATR(14) on H4 at bar t (Wilder)
#
#   # Trend filter: HH/HL uptrend confirmed
#   if not hhhl_uptrend_confirmed(h4_bars, trend_window_bars=30,
#                                  swing_lookback=3, right_edge_lag=4): skip
#
#   # Pullback: prior bar's close is at least 0.5×ATR below the most recent swing-high
#   if close[t-1] > most_recent_swing_high - pullback_atr_mult=0.5 * atr14: skip
#
#   # Resume: current bar's close breaks prior bar's high AND closes in upper half of range
#   if close[t] <= high[t-1]: skip
#   bar_range = high[t] - low[t]
#   if bar_range == 0: skip
#   if (close[t] - low[t]) / bar_range < upper_half_threshold=0.5: skip
#
#   # Refractory: at least 20 H4 bars since last signal on this pair
#   if h4_bars_since_last_signal[P] < spacing_bars=20: skip
#
#   EMIT signal (long) at t. Entry fills at next H4 bar's open.
```

Locked params per signal spec v0.1: swing_lookback=3, trend_window_bars=30, right_edge_lag=4, pullback_atr_mult=0.5, upper_half_threshold=0.5, spacing_bars=20, atr_period=14.

### 4.3 Feature computation specs

A6 meta-labeling uses the Step 1 27-feature default catalogue minus the 4 multi-TF features that came back all-NaN under the closure §2 aux-panel-integration gap. The 10 features in the winning config (closure §1 `best_architecture.features_in_winning_config`):

- **`swing_low_distance_14`** — H4 bars, distance from current close to the most recent 14-bar swing-low low, as a fraction of ATR(14). Computed at signal-bar close.
- **`prior_session_low_distance`** — H4 bars, distance from current close to the prior trading session's low, in ATR units. Session boundary is broker-day UTC.
- **`atr_vs_trailing_100`** — current ATR(14) divided by the trailing-100-bar mean ATR(14). Vol-regime context.
- **`kijun_26_distance`** — distance from current close to the H4 Ichimoku kijun-sen (26-bar midpoint), in ATR units.
- **`dollar_bloc_state`** — categorical state of the dollar bloc (AUD/CAD/NZD vs USD). Computed cross-pair at signal-bar close.
- **`atr_percentile_100`** — percentile rank of current ATR(14) within the trailing 100 bars.
- **`spread_vs_trailing_100`** — current bid/ask spread relative to trailing 100-bar mean.
- **`spread_percentile_100`** — percentile rank of current spread within trailing 100 bars.
- **`usd_strength_index`** — cross-pair USD strength score at signal-bar close (composite over the 7 USD pairs in the set).
- **`distance_to_round_number`** — H4 close distance to the nearest round-number level (00 / 50 pips), in ATR units.

All features tagged with provisional causal lineage at Step 1; the A6 winning config does NOT trigger Step 6 (lazy — not dispatched because no PASS candidate).

### 4.4 Filter chain (A1 / A2 / A6 architectures only)

**FAIL arc — not applicable for deployment.** A6 meta-labeling uses the Step 4 RF classifier (AUC 0.530) to map per-signal classifier probability → sizing tier (0× / 0.5× / 1×) with lower/upper thresholds (0.3, 0.5). At AUC ≈ chance, the sizing differentiation does some work (A6 worst-ratio 1.75 vs A1 0.17) but cannot lift worst-fold ratio above the 2.0 PASS-VIABLE gate. Closure §10 finds the same FAIL under Amendment 3 with `step5_ratio_below_gate_after_scaling` as primary mode. Section retained for documentation parity only.

### 4.5 Entry mechanics

- **Trigger bar:** H4 bar satisfying §4.2.
- **Fill bar:** next H4 bar (N+1).
- **Fill price:** open of bar N+1.
- **Order type:** market.
- **Slippage assumption:** real bid/ask spreads (HistData M1 in backtest; broker feed in live).

### 4.6 Exit mechanics

- **Initial SL anchor:** entry price.
- **Initial SL distance:** `1.5 × ATR(14)_H4` at signal bar (closure §1 `best_architecture.sl_atr: 1.5`).
- **SL update rule:** static.
- **Trail activation condition:** N/A (exit policy `sl_plus_tp_3r` has no trail).
- **Trail distance / reference / frequency:** N/A.
- **TP:** +3R take-profit (closure §1 `exit_policy: sl_plus_tp_3r`). Triggers when running PnL reaches 3× initial SL distance — closes 100% at next H4 bar's open (bar-close evaluation).
- **Time exit:** 240 H4 bars after entry, force-close at bar N+241 open.
- **Bar-by-bar evaluation order:** (1) SL hit → exit; (2) TP hit at +3R close → exit at next bar open; (3) time exit at bar 240.

### 4.7 Exposure cap

- **Type:** unlimited (closure §1 `best_architecture.exposure_cap: unlimited`).
- **Per-pair invariant:** max 1 open per pair.
- **Per-currency / global caps:** none.
- **Behaviour at cap:** N/A.

### 4.8 Risk sizing

- **`r_safe` value:** 2.0176% — **above** locked `r_max = 2.0%` ceiling by 0.018pp (closure §10: `scalable_to_safe: false`).
- **`r_hard` value:** 2.5219% — also above ceiling.
- **Risk basis:** reset-floor (L arc convention).
- **Position size formula:** standard reset-floor × r_pct sizing; capped at broker minimums.
- **Note (closure §10):** the `r_safe > r_max` scenario is the symmetric inverse of Arc 11's `r_safe < r_min` failure — both are scalability failures, opposite ends of the DD distribution. Documented as `scalability_cap_low_dd_ceiling` cross-arc tag.
- **Starting balance:** $100,000.

### 4.9 Session / time-of-day rules

- **Trading hours:** 24/5 standard FX session.
- **Day-of-week filter:** none beyond weekend break.
- **News-window filter:** none.
- **Holiday handling:** broker calendar.

### 4.10 Discrepancies and caveats

- **Config YAML reconstruction.** `configs/l_arc_8/winning_config.yaml` was reconstructed retrospectively from closure §1, `step_5/wfo_results.csv` row 2 (config_id=30), and `scripts/l_arc_8/run_step5_wfo.py` + `scripts/l_arc_8/shared.py`. Original engine ran from the inline driver script. Reconstructed file is the source of truth going forward but pre-retrofit deployment would require source verification against the driver script.
- **Step 1 multi-TF features all-NaN.** `compute_feature_matrix` requires `panel.aux={"d1":..., "w1":...}` but the Step 1 driver only passed an H4 panel. The 4 multi_tf features (`d1_close_slope_*`, `w1_close_slope_sign`, `d1_atr_percentile_100`) came back NaN. Effective Step 4 feature catalogue was 23 features instead of 27. Fix is staged but not re-run (engine PR scope).
- **Pool-level WFO approximation.** Per-trade R outcomes from Step 1 carry through with exit-policy approximation. NOT a bar-by-bar multipair backtester. Understates intraday DD timing; ignores cross-pair concurrency interactions.
- **Per-day max-DD series + chained DD not measured.** PROVISIONAL on those constraints, but the §10 verdict is DEFINITIVE because the ratio (0.882) and scalability (2.018% > 2.0%) failures are independent of those gaps.
- **Engine ratio vs §3 ratio convention divergence.** Engine reports `worst_fold_ratio: 1.749` (per-fold ROI/DD inside the worst-ROI fold); §3 math reads 0.882 (cross-fold worst_ROI / worst_DD). Both fail the 2.0 gate. Same outcome.
- **Holdout PASS-DEPLOYABLE vs Search FAIL asymmetry.** Top-3 holdout (2021-04 → 2026-04) shows ROI +19% to +43%, DD ≤ 1.7%, ratios 19-35 — comfortably above the gate. §2 Step 5 requires BOTH gates pass; search WFO doesn't, so arc verdict is FAIL despite holdout strength. Worth tracking as a cross-arc pattern.

### 4.11 Deployment readiness checklist

**FAIL arc — not applicable for deployment.**

- [ ] N/A — FAIL verdict; not for deployment.
- [ ] N/A — config retained for portfolio-eligibility audit only; A5 spec pending.
- [ ] N/A
- [ ] N/A
- [ ] N/A
- [ ] N/A (Step 6 not run — lazy dispatch only on PASS candidates)

---

## §10 Amendment 3 re-evaluation (added 2026-05-22)

**Original verdict:** FAIL (primary_failure_mode: `entry_feature_auc_ceiling`, failed_at_step: 5)
**Re-evaluated verdict:** FAIL
**Re-evaluation status:** definitive (two independent constraints fail; scalability is structural, ratio is invariant under scaling)
**Re-evaluated primary_failure_mode:** `step5_ratio_below_gate_after_scaling`

> Per Amendment 3 priority order, `step5_not_scalable` would precede `step5_ratio_below_gate_after_scaling`. Chat directive (2026-05-22) overrides priority: the ratio failure (0.88 < 2.0, invariant under linear scaling) is the *real* binding constraint; the scalability ceiling at `r_safe = 2.018% > 2.0%` is a secondary consequence of unusually low worst-fold DD (1.98%). Reporting both, primary set to ratio.

### Scaling derivation
- `worst_fold_dd_base_pct`: **1.9826%** (config 30, fold 7 — 2016 — has the worst DD; closure §1 `worst_fold_dd_pct`)
- `worst_fold_roi_base_pct`: **+1.7486%** (config 30, fold 9 — 2018 — has the worst ROI; closure §1 `worst_fold_roi_pct`)
- `k_safe = 8.0 / 1.9826 = 4.0351`
- `k_hard = 10.0 / 1.9826 = 5.0439`
- `r_safe_pct = 0.5 × 4.0351 = 2.0176%`
- `r_hard_pct = 0.5 × 5.0439 = 2.5219%`
- `scalable_to_safe`: **false** (2.0176% exceeds the locked `r_max = 2.0%` ceiling by 0.018pp — narrow margin)
- `scalable_to_hard`: **false** (2.5219% > 2.0%)

### Amended DEPLOYABLE gate evaluation

| # | Constraint | Threshold | Value at r_safe | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 1 | Scalable to safe | `r_safe ∈ [0.15%, 2.0%]` | 2.0176% | ✗ | exceeds ceiling by 0.018pp |
| 2 | Worst-fold ROI/DD ratio | ≥ 2.0 | **0.882** | ✗ | invariant; binding constraint per chat directive |
| 3 | Worst-fold ROI | > 0 | +7.05% (= 1.7486 × 4.0351; or +6.99% at r_max=2.0%) | ✓ | sign-positive at any scaling |
| 4 | Per-fold positivity | all 11 positive, 0 negative | 11/11 | ✓ | sign does not scale |
| 5 | Worst-fold DD | ≤ 8% | 8.0% (= 1.9826 × 4.0351) | ✓ | by construction of `k_safe` |
| 6 | Daily DD breaches | = 0 | 0 (at `r_base`; per-day series not built) | ? | engine-emitted per-fold count = 0 across all 11 folds at `r_base`; under upward scaling `k_safe = 4.04` the per-day re-evaluation could surface new breaches if any single day's DD at `r_base` was ≥ 1.24%. PROVISIONAL: per-day max-DD series not measured under v3.0 pre-amendment |
| 7 | Chained max DD | ≤ 10% | unknown × 4.04 | ? | PROVISIONAL: `chained_max_dd_base_pct` not measured; per-fold equity reset only |
| 8 | Trades per fold | ≥ 25 | min ≈ 361 (fold 5, 2014); max 453 (fold 3, 2012) | ✓ | from `step_5/per_fold_metrics.csv` config 30 |
| 9 | Holdout at r_safe | clears prior §3 holdout gate | proxy: ROI ≈ +172.6%, DD ≈ 7.01% (linear scale of base 42.78%/1.7369% × 4.0351; at r_max=2.0% i.e. k=4.0: ROI +171.1%, DD 6.95%) | ✓ | PROXY: not re-run. Holdout passes its scaled DD bound regardless of scaling cap |
| 10 | Step 6 clean | clean | not run | ? | Step 6 lazy — would only run on a Top-1 candidate clearing DEPLOYABLE / VIABLE; this arc didn't, so no Step 6 dispatch under prior protocol. Moot — constraint 2 fails. |

### Amended VIABLE gate evaluation

| # | Constraint | Threshold | Value at r_hard | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 1 | Hard-scalable | `r_hard ∈ [0.15%, 2.0%]` | 2.5219% | ✗ | exceeds ceiling more decisively |
| 2 | Worst-fold ROI/DD ratio | ≥ 2.0 | 0.882 | ✗ | invariant |
| 3 | Mean-fold ROI/DD ratio | ≥ 2.5 | mean_roi 5.902 / mean_dd 0.929 = **6.35** | ✓ | invariant (well above threshold) |
| 4 | Per-fold positivity | ≤ 1 negative fold | 11/11 positive | ✓ | |
| 5 | Worst-fold DD | ≤ 10% | 10.0% (= 1.9826 × 5.0439) | ✓ | by construction |
| 6 | Daily DD breaches | = 0 | 0 at `r_base` | ? | PROVISIONAL — same reason as DEPLOYABLE constraint 6, scaling factor 5.04 |
| 7 | Chained max DD | ≤ 10% | unknown × 5.04 | ? | PROVISIONAL |
| 8 | Trades per fold | ≥ 25 | min 361 | ✓ | |
| 9 | Holdout at r_hard | clears prior §3 holdout gate | proxy: ROI ≈ +215.8%, DD ≈ 8.76% | ✓ | PROXY |
| 10 | Step 6 clean | clean | not run | ? | Moot — constraint 2 fails |

### Engine vs §3 ratio discrepancy (flagged per chat Q1)

The closure §1 reports `worst_fold_ratio: 1.749`. Inspection of `step_5/per_fold_metrics.csv` config_id=30 shows this value is the per-fold `roi_dd_ratio` of fold 9 (worst-ROI fold: ROI 0.01749, DD 0.01000, ratio 1.7487). The engine's `worst_fold_ratio` convention is **"the ratio inside the fold with the lowest ROI"**, not the §3 definition `worst_fold_roi / worst_fold_dd = 1.7486% / 1.9826% = 0.882` (which combines fold-9's ROI with fold-7's DD).

Amendment 3 §3 explicitly defines the gate quantity as ROI/DD invariant under linear scaling. Per chat directive (Q1 answer: option a), the §10 evaluation uses **0.882** as the gate value. The engine's 1.749 reading is retained in §1 for historical fidelity. Both fail the 2.0 gate.

### Final assessment

Arc 8 fails the Amendment 3 DEPLOYABLE gate on two independent constraints: (a) scalability — the worst-fold DD at base risk is so low (1.98%) that scaling up to fill the 8% DD allowance requires `r_safe = 2.018%`, just above the locked 2.0% ceiling; (b) the worst-fold ROI/DD ratio under §3 mathematical reading is 0.882 — well below the 2.0 gate, and invariant under any linear scaling. The VIABLE tier fails the same constraints. The original FAIL verdict stands; the failure-mode taxonomy shifts from `entry_feature_auc_ceiling` (a Step 4 diagnostic finding) to `step5_ratio_below_gate_after_scaling` (the Amendment 3 binding constraint).

What's structurally interesting is that the low-DD-low-ROI profile (worst-fold ROI +1.75% on DD 1.98%, 11/11 sign-consistency, daily breaches = 0 at base) is *not* a typical failure signature. The strategy is operating in a sub-2% DD regime with positive expectancy across every fold — but the ratio sits below 1.0 in the worst fold. Sizing differentiation under A6 lifts ROI without proportionally lifting DD, suggesting the strategy has *some* signal that just doesn't compound into a sufficiently leveraged edge for solo deployment.

**Portfolio-eligibility note (per chat directive):** Arc 8 may be portfolio-eligible under future A5 composition work despite single-strategy FAIL — the low-DD profile (worst 1.98%, mean 0.93%) combined with 11/11 sign-consistency and 0 daily breaches across all folds at `r_base` could contribute uncorrelated edge to a multi-strategy portfolio whose combined account-wide DD/ROI characteristics clear PASS-DEPLOYABLE. Deferred to A5 follow-up (the L_PROTOCOL §3 A5 spec is itself open — see §3 "A5 follow-up flag").

### Missing data flags

- Constraint #6 (daily DD breach count at `r_safe` / `r_hard`): per-day max-DD series not built under v3.0 pre-amendment. Engine-emitted per-fold breach count at `r_base` = 0 across all 11 folds; under `k_safe = 4.04` upward scaling, this could change. PROVISIONAL.
- Constraint #7 (chained max DD): `chained_max_dd_base_pct` not measured; per-fold equity reset (each fold starts at $100k) per Step 6 caveat. Without the chained measurement at `r_base`, the scaled value is unknown. PROVISIONAL.
- Constraint #10 (Step 6): not run — Step 6 lazy dispatch only fires on a PASS-VIABLE/DEPLOYABLE Top-1 candidate; this arc did not produce one under prior protocol. Moot because constraint 2 fails definitively.
- **No engine re-run required for definitive verdict.** The two binding failures (ratio + scalability ceiling) are independent of the missing data — both flagged constraints would need to PASS, AND the ratio + scalability constraints would need to PASS, for an upgrade. Ratio is invariant; scalability has 0.018pp of headroom. Not re-runnable.

### Cross-arc tags (additions to closure §3 tags)

- `low_dd_low_roi` — worst-fold DD 1.98% with ROI 1.75%; sub-2% DD regime, invariant ratio below 1.0
- `portfolio_eligibility_pending_a5_spec` — potential A5 component pending L_PROTOCOL A5 follow-up spec
- `scalability_cap_low_dd_ceiling` — first arc to hit the `r_safe > r_max` ceiling failure mode (DD too low to fill the 8% allowance at allowed leverage)
- `engine_ratio_vs_amendment_ratio_divergence` — engine's `worst_fold_ratio` convention (per-fold roi/dd of worst-ROI fold) ≠ §3 definition (cross-fold worst_roi / worst_dd). Discrepancy: 1.749 vs 0.882. Same FAIL outcome.
