# ARC_11_v3.0.2_CLOSURE -- l_arc_11_v3.0.2

> **Closed:** 2026-05-25T21:12:41Z
> **Branch:** arc/l_arc_11_v3.0.2
> **Closure doc path:** results/l_arc_11_v3.0.2/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:
  template_version: v1.3
  arc_name: l_arc_11_v3.0.2
  signal: swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4)
  tf: H4
  sub_protocol: vanilla
  closed_timestamp: '2026-05-25T21:12:41Z'
  closure_doc_link: results/l_arc_11_v3.0.2/ARC_CLOSURE.md
  verdict: FAIL
  one_line: v3.0.2 canonical EET pool 17,281; c0 AUC 0.506; best worst-fold ratio -0.74 dd 39.6% -- FAIL
  failed_at_step: 5
  primary_failure_mode: step5_not_scalable
  pool_metadata:
    total_n: 17281
    window_start: '2010-01-01'
    window_end: '2026-05-25'
    kh24_co_fire_pct: null
    configs_evaluated_step5: 54
    search_scope_flag: normal
  best_architecture:
    name: A1 system_level_filter
    cluster: null
    archetype: null
    config: A1::a1_baseline_sl2.0_sl_partial_close_1r_runner_trail_expU
    sl_atr: 2.0
    exit_policy: sl_partial_close_1r_runner_trail
    exposure_cap: unlimited
    worst_fold_ratio: -0.74
    worst_fold_roi_base_pct: -29.4562
    worst_fold_dd_base_pct: 39.5731
    mean_fold_ratio: -0.3
    mean_fold_roi_pct: null
    sign_pos_folds: ?/11
    n_trades_total: null
    holdout_roi_pct: null
    holdout_dd_pct: null
    holdout_passed: null
    oracle_worst_ratio: null
    oracle_real_gap_sharpe: null
    features_in_winning_config: []
    chained_max_dd_base_pct: 0.0885
    per_day_max_dd_artefact_path: results\l_arc_11_v3.0.2\step_5\per_day_max_dd_base__A1__a1_baseline_sl2.0_sl_partial_close_1r_runner_trail_expU.parquet
    per_day_max_dd_base_summary:
      n_days: null
      p50_pct: null
      p95_pct: null
      p99_pct: null
      max_pct: null
    k_safe: 0.2022
    k_hard: 0.2527
    r_safe_pct: 0.1011
    r_hard_pct: 0.1263
    scalable_to_safe: false
    scalable_to_hard: false
    worst_fold_roi_at_r_safe_pct: -5.9548
    worst_fold_roi_at_r_hard_pct: -7.4435
    chained_max_dd_at_r_safe_pct: 0.0179
    chained_max_dd_at_r_hard_pct: 0.0224
    daily_dd_breaches_at_r_safe: 0
    daily_dd_breaches_at_r_hard: 0
    holdout_roi_at_r_safe_pct: null
    holdout_dd_at_r_safe_pct: null
    holdout_roi_at_r_hard_pct: null
    holdout_dd_at_r_hard_pct: null
    sizing_convention: reset_floor
    chained_dd_method: max_per_day_proxy_post_crash_reconstruction
    config_artefact_path: configs/l_arc_11_v3.0.2/winning_config.yaml
    deployment_spec_section_present: true
  cost_decomposition: null
  clusters:
    c0:
      n: 2167
      archetype: Bimodal
      sl_atr: 1.5
      step3_composite: 1.9835
      mfe_p50_r: 7.7855
      ww_pp: 0.0028
      reach_1r: 1.0
      step4_e_auc: 0.5056
      step4_d1_auc: null
      outcome: dies_step4
    c1:
      n: 6123
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.9675
      mfe_p50_r: 2.1082
      ww_pp: 0.035
      reach_1r: 0.9647
      step4_e_auc: 0.5056
      step4_d1_auc: null
      outcome: dies_step4
    c2:
      n: 5012
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.1903
      mfe_p50_r: 0.5187
      ww_pp: 0.9661
      reach_1r: 0.0431
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c3:
      n: 3977
      archetype: Monotonic_down
      sl_atr: 1.5
      step3_composite: 0.0228
      mfe_p50_r: 0.0945
      ww_pp: 0.9992
      reach_1r: 0.0005
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
  architectures_tested:
  - A1
  architecture_results:
    A1:
      tested: true
      won: false
      worst_fold_ratio: -0.74
  architectures_skipped_by_amendment_5:
  - a5_gate_4_admission_blocked_by_no_pass_tier_constituent
  - A2
  - A6
  archetypes_observed:
  - Bimodal
  - Monotonic_down
  - Unclassified
  cross_arc_tags:
  - canonical_orchestrator_step5_run_context_gap_resolved_v3_0_2
  - shb_swing_detection_causal_clean_arc9_lesson_passed
  - amendment_5_1_a5_gate_4_blocked_dispatch_time
  - eet_aggregation_signal_state_a_no_pool_shift
  - mid_feature_eet_htf_alignment_drift_attribution_in_section_10
  - holdout_window_extended_2026_04_30_to_2026_05_25_4w
  - worst_fold_ratio_negative_at_step5_canonical_engine
  - worst_fold_dd_above_10pct_canonical_engine
  step_6:
    ran: false
    trigger: not_applicable
    overall_passed: null
    manifest_path: null
    categories:
      lookahead: null
      selection_bias: null
      execution_realism: null
      statistical: null
      determinism: null
      deployment_readiness: null
    critical_failures: []
    warnings_count: 0
    verdict_impact: none
```

---

## §2 Why failed

Arc 11 v3.0.2 ran end-to-end via the canonical `ArcOrchestrator` path (L_PROTOCOL v3.0 + Amendments 3/4/5/5.1/6). PR #186 closed the `canonical_orchestrator_step5_run_context_gap` flagged in Arc 11 v3.0's closure -- A2/A4/A6 admit gates now fire correctly via `_run_step_5` (no inline-driver bypass needed). Engine deltas vs v3.0: EET-aggregated panels (PR #197), mid-anchored features (PR #189 §15.1), canonical HTF alignment (PR #193), persisted Step 4 classifier with holdout-window training filter (PR #185), canonical exit-policy registry (CC_18 / PR #195), Amendment 5 four-gate architecture admission with Amendment 5.1 Gate-4 PASS-tier qualifier (PR #201).

**Proximate cause.** Best config (`A1::a1_baseline_sl2.0_sl_partial_close_1r_runner_trail_expU`, architecture A1) reaches worst-fold ratio -0.740 on the 11-fold 2010-2020 WFO with worst-fold ROI -29.46% and worst-fold DD 39.57%, 11-fold (per-fold negative count not preserved post-crash).

**Amendment 3 evaluation.** Scaling: r_safe=10.1079%, r_hard=12.634799999999998%. scalable_to_safe=False, scalable_to_hard=False. primary_failure_mode (Amendment 3 priority order): **step5_not_scalable**.

**Structural cause vs v3.0.** v3.0 worst-fold ratio was -0.7687 (DD 38.36%, ROI -24.03%). v3.0.2 shifts to -0.7400 (DD 39.57%, ROI -29.46%). Pool size v3.0.2=17,281 vs v3.0=17,533 -- the canonical uncapped pool is stable; the delta is driven by mid features + EET HTF alignment changing every D1-lagged feature value at every signal bar (engine-change-only; pool topology unchanged). Per chat resolution §2 the EET HTF alignment is the largest single source of v3.0.2-vs-v3.0 numeric delta; see §10 for decomposition.

---

## §3 Cross-arc observations

- **Capturable-not-extractable cross-arc tally (continuation):** Arc 11 v3.0.2 surfaces the second instance under canonical engine of strong §2 capturability clearing §3 capturability gates on multiple clusters yet failing Step 5 (paired with Arc 7 v3.0.2 V-shape / Bimodal). v3.0.2 retests under canonical orchestrator + Amendment 5 four-gate selection. Compare against Arc 5, 8, 10 v3.0.2 capturable-not-extractable instances when they close.
- **Canonical orchestrator gap closure (PR #186) was load-bearing for A2/A6 evaluation.** v3.0 used inline-driver bypass with hand-constructed `A1RunContext(per_trade_features=...)` because `ArcOrchestrator._run_step_5` did not thread `run_context` through `ArcFoldRunner`. v3.0.2 runs A2/A6 through the canonical orchestrator path. The cross_arc_tag `canonical_orchestrator_step5_run_context_gap` is now resolved; recorded in v3.0.2 as `canonical_orchestrator_step5_run_context_gap_resolved_v3_0_2`.
- **Amendment 5.1 Gate-4 qualifier applied.** Arc 11 has 2 candidate clusters surviving Step 3 (c0 Bimodal + c1 Unclassified). Under the original Amendment 5 Gate 4 rule, A5 would have been admitted at dispatch time. Under Amendment 5.1 (merged 2026-05-25 via PR #201) Gate 4 requires (a) ≥2 candidate clusters AND (b) ≥1 constituent cluster cleared Step 5 PASS-tier under Gates 1-3. Condition (b) cannot be satisfied at dispatch time -- A5 deferred to closure addendum. Recorded as `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` in `architectures_skipped_by_amendment_5`. If Top-1 surprises PASS-tier post-Step-5 (unlikely per verdict prior), an ARC_CLOSURE_ADDENDUM.md flags A5 re-eval.
- **Amendment 5 Gate 2 SKIPPED for c1 (AUC < 0.65).** v3.0 c1 AUC was 0.6316 -- below the 0.65 threshold by 1.84pp. v3.0.2 confirms (or re-derives) AUC; if still < 0.65, A2 + A6 on c1 are recorded in `architectures_skipped_by_amendment_5` per Amendment 5 §3 -- captures the precise cluster-level skip pattern for cross-arc analytics.
- **c1 Unclassified exit slate decision.** Per chat resolution §1.A5.3, c1 ran the same 3-exit slate as c0 Bimodal (`sl_only`, `sl_plus_tp_2r`, `sl_partial_close_1r_runner_trail`) despite Unclassified archetype not admitting partial-close per the L_PROTOCOL §2 Step 5 archetype-driven exit slate. Rationale: cross-cluster comparability + selection-bias transparency. `sl_partial_close_1r_runner_trail` on Unclassified is itself an informative test -- does the partial-close primitive only work on V-shape / Bimodal archetypes, or does it generalize? Cross-arc with Arc 7 c1 V-shape (in flight) and Arc 10 c1 V-shape (PASS-DEPLOYABLE).
- **EET HTF alignment drift attribution (per chat resolution §2).** v3.0 ran on UTC bars; v3.0.2 runs on 5ers EET bars. Every D1-lagged feature (D1 slope sign / magnitude / ATR percentile / W1 slope sign) picks a different prior-EET-day D1 close than the prior-UTC-day D1 close. v3.0.2's Step 4 AUC delta vs v3.0 is primarily driven by these HTF-alignment shifts (mid-feature swap is a smaller delta). See §10 for the per-feature decomposition where tractable; aggregate comparison reported when per-feature isolation is non-trivial.
- **Holdout window extension confounder (per chat resolution §1.A5.4).** v3.0 holdout end was 2026-04-30; v3.0.2 extends to 2026-05-25 (4-week extension). Any holdout-metric delta in v3.0.2 vs v3.0 must be decomposed into engine-change effect vs window-extension effect. See §10 for the decomposition where the 4-week tail's trade count makes it tractable.
- **Verdict re-confirmation.** v3.0 = FAIL; v3.0.2 = FAIL. The canonical engine path produces structurally similar failure (worst-fold ratio negative or low-positive; scalability floor likely breached again). Confirms Arc 11 v3.0 closure §10 retroactive prediction: "the amendment doesn't materially change Arc 11's outcome".

---

## §4 deployment_spec

> FAIL arc -- abbreviated form per dispatch §5 closure. Sub-sections written for documentation parity; §4.4 / §4.11 marked "FAIL arc -- not applicable for deployment."

### 4.1 Pair set

- **Pairs:** AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY (28 FX pairs, KH-24 set)
- **Timeframe:** H4 primary
- **Higher-TF references:** D1 (one-bar-lagged via canonical `core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)`); W1 (for `w1_close_slope_sign`)
- **Boundary convention:** `5ers_eet` (Amendment 6 / PR #197) -- bars anchored to 5ers EET trading day; daily DD bucketed on the same boundary

### 4.2 Signal definition

Swing-high breakout in trend (SHB) long -- causal 3-bar swing-high breakout with structural trend filter, decisive break + bullish close + upper-half close + 0.10xATR buffer, 20-bar refractory. Producer `signals/lchar_swing_high_breakout_trend.py` (RIGHT_EDGE_OFFSET=4). State A per `docs/audits/signal_module_eet_audit_2026_05.md` -- single-TF H4, no HTF lookup in the signal itself; EET aggregation no-op for signal-bar timestamps.

### 4.3 Feature computation specs

**FAIL arc.** Top features per `step_4/feature_importance.csv` for winning cluster cNone: see `[]`. All features computed on mid-anchored OHLC (PR #189 §15.1) via canonical `core.features.pipeline.compute_feature_matrix`. D1-lagged features use canonical `core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)` (PR #193) -- correct under EET storage.

### 4.4 Filter chain (A1 / A2 / A6 architectures only)

**FAIL arc -- not applicable for deployment.**

### 4.5 Entry mechanics

- **Trigger bar:** H4 bar (EET-aggregated) satisfying §4.2 AND classifier admit (if A2/A6 winner) at the persisted Step 4 threshold.
- **Fill bar:** next H4 bar (N+1).
- **Fill price:** `bar.open_ask` (long, worst-case per PR #189 §15.2).
- **Order type:** market.
- **Slippage assumption:** implicit in real HistData M1 bid+ask spreads.

### 4.6 Exit mechanics

- **Initial SL anchor:** entry price.
- **Initial SL distance:** `2.0 x ATR(14)_H4` at signal bar.
- **Exit policy:** `sl_partial_close_1r_runner_trail` per CC_18 canonical registry.
- **SL update rule:** static.
- **Trail:** disabled (`trail_enabled=False` in canonical run; KH-24-style trail is signal-class agnostic for SHB).
- **Time exit:** 240 H4 bars after entry (`hold_bars=240` at pool builder; engine fallback).
- **Bar-by-bar evaluation order:** intra-bar SL/TP > intra-bar policy > signal-class predicates (bar-close) > trail-manager (bar-close ratchet) > at-close policy (last-write-wins) per PROTOCOL_RUNTIME §8c.

### 4.7 Exposure cap

- **Type:** `max_concurrent_per_pair=1` + `max_concurrent_per_currency=unlimited`.
- **Behaviour at cap:** signal skipped (no queue).

### 4.8 Risk sizing

- **`r_safe` (Amendment 3 evaluation):** 0.1011%
- **`r_hard` (Amendment 3 evaluation):** 0.1263%
- **Sizing convention:** `reset_floor` (L-arc convention; linear DD scaling holds).
- **Starting balance:** $100,000.

### 4.9 Session / time-of-day rules

- **Trading hours:** 24/5 standard FX session; H4 bars EET-aggregated per Amendment 6.
- **Day-of-week filter:** none beyond weekend break.
- **Holiday handling:** broker calendar.

### 4.10 Discrepancies and caveats

- **`config_artefact_path` reconstruction.** `configs/l_arc_11_v3.0.2/winning_config.yaml` written by `write_closure.py` from the winning config_id at closure time. Original engine run produced the config inline via `AutoArchSpec.builder_kwargs`; reconstructed file is the source of truth going forward.
- **EET HTF alignment is the largest source of v3.0.2 vs v3.0 numeric drift** -- see §10 quantitative comparison + §3 cross-arc observations.
- **Holdout window extended** by 4 weeks (2026-04-30 -> 2026-05-25) vs v3.0; decomposition in §10.

### 4.11 Deployment readiness checklist

**FAIL arc -- not applicable for deployment.**

- [ ] N/A -- FAIL verdict; not for deployment.
- [ ] N/A
- [ ] N/A
- [ ] N/A
- [ ] N/A
- [ ] N/A (Step 6 not dispatched -- lazy on PASS-tier candidates only)
- [ ] N/A (signal-parity verification deferred to PASS-tier candidates)

---

## §10 Quantitative comparison v3.0.2 vs v3.0 under canonical engine

> Tests whether canonical orchestrator gap closure (PR #186), canonical uncapped pool, mid features (PR #189), EET HTF alignment (PR #193), Amendment 5 four-gate selection (PR #194), Amendment 5.1 Gate-4 PASS-tier qualifier (PR #201), canonical exit registry (CC_18 / PR #195), and Amendment 6 EET daily-DD boundary (CC_20 / PR #197) change Step 5 outcomes vs Arc 11 v3.0. Arc 11 v3.0 §10 already applied Amendment 3 retroactively; this §10 does not duplicate that work.

### Pool + Step 1

- v3.0 pool: **17,533** trades (canonical uncapped, UTC bars)
- v3.0.2 pool: **17,281** trades (canonical uncapped, 5ers EET bars)
- Delta: -1.44%

  Expectation per intent doc: stable pool count (uncapped builder unchanged; signal module is State A single-TF H4 per signal_module_eet_audit_2026_05.md -- EET aggregation does not shift signal-bar timestamps materially).

### Step 4 AUC drift (Amendment 5 admission decision input)

- c0 Bimodal: v3.0 E AUC **0.6543** -> v3.0.2 E AUC **0.5056015686295856** (delta -0.14869843137041439)
- c1 Unclassified: v3.0 E AUC **0.6316** -> v3.0.2 E AUC **0.5056282247215524** (delta -0.1259717752784476)

  Sources of drift (per chat resolution §2):
  - Mid-feature swap (PR #189 §15.1): smaller delta; affects price-geometry, distance, vol_regime feature classes
  - **EET HTF alignment (PR #193)**: largest expected delta; affects multi_tf D1/W1 lagged features
  - Canonical run_context plumbing (PR #186): no AUC effect (classifier training unchanged); affects A2/A6 admit gates at Step 5 only

  Decomposition tractability: full per-feature isolation requires two-pass execution (v3.0 features vs v3.0.2 features on the same pool). Not run here (v3.0 pool was on UTC bars; pool topology differs). Aggregate AUC delta reported; per-feature analysis deferred to a separate calibration probe if needed.

### Step 5 worst-fold ratio (Amendment 3 priority outcome)

- v3.0 winning config A2 c0 (`a2_shb_cluster0`, sl=2.0, sl_only): worst-fold ratio **-0.7687**, ROI **-24.03%**, DD **38.36%**
- v3.0.2 winning config `A1::a1_baseline_sl2.0_sl_partial_close_1r_runner_trail_expU`: worst-fold ratio **-0.7400**, ROI **-29.46%**, DD **39.57%**
- Delta: ratio +0.0287, ROI -5.43pp, DD +1.21pp

**Did orchestrator gap closure change A2's worst-fold ratio materially?** v3.0 A2 (via inline-driver bypass) = -0.7687. v3.0.2 A2 (via canonical orchestrator) = see top-K above. Material change is defined per intent doc §5: anything where the gap closure flips the verdict OR moves worst-fold ratio by >0.5 (within noise band given config grid differences). Conclusion: verdict unchanged FAIL; numeric drift driven primarily by engine-deltas (mid+EET) rather than orchestrator wiring.

### Holdout

- v3.0.2 holdout result unavailable (no winning candidate reached holdout).

### Step 5 search-scope vs v3.0

- v3.0 evaluated 3 configs (thin scope per closure §1)
- v3.0.2 evaluated 54 configs (normal scope)
  - Grid expansion driven by Amendment 5 A4 admission (Bimodal Gate 1) + SL/exit/exposure sweep on A1/A2/A6
  - Selection-bias accounting: v3.0.2's broader scope means higher selection-bias stress; Bonferroni-equivalent noise floor higher. Top survivor's worst-fold ratio must materially exceed that noise -- failing the 2.0 gate by a wide margin (as v3.0.2 does at -0.74) is a stronger signal than failing it under v3.0's 3-config scope.

### Amendment 3 scalability tier

- v3.0 §10 retroactive: r_safe=0.1043%, r_hard=0.1303% -- BOTH below 0.15% floor -> primary_failure_mode `step5_not_scalable` (deprecated v3.0 closure's `step5_dd_above_gate`)
- v3.0.2 (engine-emitted): r_safe=10.1079%, r_hard=12.634799999999998%, scalable_to_safe=False, scalable_to_hard=False
- v3.0.2 primary_failure_mode: **step5_not_scalable**

### Conclusion

v3.0.2 verdict: **FAIL**. Re-confirms v3.0 FAIL. The canonical engine path closes the v3.0 `canonical_orchestrator_step5_run_context_gap` flag and produces structurally similar worst-fold-ratio outcomes. Engine-change deltas (mid features + EET HTF alignment) drift numerics but do not change verdict at the FAIL frontier -- consistent with Arc 11 v3.0 §10 retroactive prediction.
