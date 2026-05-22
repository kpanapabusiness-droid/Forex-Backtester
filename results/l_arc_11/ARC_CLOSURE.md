# ARC_11_CLOSURE — l_arc_11

> **Closed:** 2026-05-22T12:33:35Z
> **Branch:** arc/l_arc_11
> **Closure doc path:** results/l_arc_11/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:
  template_version: v1.2
  arc_name: l_arc_11
  signal: swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4)
  tf: H4
  sub_protocol: vanilla
  closed_timestamp: '2026-05-22T12:33:35Z'
  closure_doc_link: results/l_arc_11/ARC_CLOSURE.md
  verdict: FAIL
  one_line: Canonical pool 17,533; cluster 0 composite 1.98 + RF AUC 0.654; best ratio -0.77 dd 38.4% — FAIL
  failed_at_step: 5
  primary_failure_mode: step5_dd_above_gate
  pool_metadata:
    total_n: 17533
    window_start: '2010-01-01'
    window_end: '2026-04-30'
    kh24_co_fire_pct: null
    configs_evaluated_step5: 3
    search_scope_flag: thin
  best_architecture:
    name: A2 classifier_filter
    cluster: 0
    archetype: Bimodal
    config: a2_shb_cluster0
    sl_atr: 2.0
    exit_policy: sl_only
    exposure_cap: 2
    worst_fold_ratio: -0.7687
    worst_fold_roi_pct: -24.0273
    worst_fold_dd_pct: 38.3593
    mean_fold_ratio: 0.4305
    mean_fold_roi_pct: null
    sign_pos_folds: 6/10
    n_trades_total: 380
    holdout_roi_pct: 1.3004
    holdout_dd_pct: 57.4969
    holdout_passed: false
    oracle_worst_ratio: null
    oracle_real_gap_sharpe: null
    features_in_winning_config:
    - w1_close_slope_sign
    - d1_atr_percentile_100
    - prior_session_low_distance
    - day_of_week
    - session_london
    - d1_close_slope_magnitude
    - distance_to_round_number
    - atr_percentile_100
    - usd_strength_index
    - spread_vs_trailing_100
    # ── v1.2 deployment-spec fields (added 2026-05-23 retrofit; FAIL arc — for documentation parity) ──
    config_artefact_path: configs/l_arc_11/winning_config.yaml
    deployment_spec_section_present: true
  cost_decomposition:
    admit_pool:
      n_fraction: 0.125
      mean_r: 5.4272
    reject_pool:
      n_fraction: 0.875
      mean_r: -0.8789
    early_exit_pool:
      n_fraction: 0.0
      mean_r: 0.0
  clusters:
    c0:
      n: 2192
      archetype: Bimodal
      sl_atr: 1.5
      step3_composite: 1.9803
      mfe_p50_r: 7.7664
      ww_pp: 0.0018
      reach_1r: 1.0
      step4_e_auc: 0.6543
      step4_d1_auc: null
      outcome: dies_step5
    c1:
      n: 6287
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.9678
      mfe_p50_r: 2.1087
      ww_pp: 0.035
      reach_1r: 0.9648
      step4_e_auc: 0.6316
      step4_d1_auc: null
      outcome: dies_step4
    c2:
      n: 5032
      archetype: Unclassified
      sl_atr: 1.5
      step3_composite: 0.1899
      mfe_p50_r: 0.5111
      ww_pp: 0.968
      reach_1r: 0.0425
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
    c3:
      n: 4022
      archetype: Monotonic_down
      sl_atr: 1.5
      step3_composite: 0.0221
      mfe_p50_r: 0.0901
      ww_pp: 0.9993
      reach_1r: 0.0007
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3
  architectures_tested:
  - A1
  - A2
  - A6
  architecture_results:
    A1:
      tested: true
      won: false
      worst_fold_ratio: -0.8899
    A2:
      tested: true
      won: false
      worst_fold_ratio: -0.7687
    A6:
      tested: true
      won: false
      worst_fold_ratio: 0.0
  archetypes_observed:
  - Bimodal
  - Monotonic_down
  - Unclassified
  cross_arc_tags:
  - step4_auc_above_065_v3_first
  - shb_swing_detection_causal_clean_arc9_lesson_passed
  - canonical_orchestrator_step5_run_context_gap
  - step1_pool_uncapped_canonical_vs_capped_handrolled_2_5x_delta
```

---

## §2 Why failed

Arc 11 ran end-to-end via the canonical v3 infrastructure (`core/arc/arc_pool_builder.py`, `core/steps/step_{2,3,4}_*.py`, `core/architectures/a{1,2,6}.py`, `core/runners/arc_fold_runner.py`, `core/wfo/orchestrator.py`). Verdict: **FAIL**.

**Proximate cause.** Best config (`a2_shb_cluster0`, architecture A2) reaches worst-fold ratio -0.769 on the 11-fold 2010-2020 WFO with worst-fold ROI -24.03% and worst-fold DD 38.36%, 4/10 negative folds.
The worst-fold ratio -0.769 is below the §3 PASS-VIABLE/DEPLOYABLE threshold of 2.0, the dispositive failure.

**Structural cause.** The canonical Step 1 pool is **17,533** trades (2.5× the hand-rolled pool's 7,149) because the canonical builder correctly applies per-pair / per-currency exposure caps at the Step 5 architecture level, not at Step 1. With the full uncapped pool flowing through, the cluster topology shifts: Step 2 selects K=4 (vs hand-rolled K=2) and Step 3 surfaces **two** candidate clusters (vs hand-rolled one). Cluster 0's capturability composite climbs to 1.98 (vs hand-rolled 0.99) — the cohort is markedly stronger than the hand-rolled analysis reported. The §3 DD failure persists at the canonical pool level, but for a different structural reason than the hand-rolled analysis claimed.

**What this tells us about methodology.** Step 1 exposure-capping conflates characterization with deployment. The canonical convention (no cap at Step 1; cap at architecture level in Step 5) is correct — it lets the same pool feed multiple architecture/cap configurations without re-running Step 1 per combination, and produces unbiased cluster geometry. Any arc that hand-rolled exposure caps into Step 1 (including this arc's prior hand-rolled run) is structurally biased toward whichever signals the cap admitted first.

**Holdout consistency.** On the one-shot 2021-01-01 → 2026-04-30 holdout, `a2_shb_cluster0` produced ROI +1.30% / DD 57.50%. Holdout verdict: fail. WFO + holdout combined verdict: FAIL.

---

## §3 Cross-arc observations

- Step 1 exposure-capping bias surfaced (hand-rolled vs canonical 2.5× pool delta). Hand-rolled Arc 11 closure under-reported cohort strength by half. Any arc that uses a Step 1 simulator with per-pair / per-currency caps applied at pool-build time is similarly biased; the canonical `core/arc/arc_pool_builder.py` is the correct reference.
- Canonical orchestrator (`core/arc/arc_orchestrator.py::_run_step_5`) does not plumb `run_context` through `ArcFoldRunner`. Result: A2 / A3 / A4 / A6 — all architectures requiring `per_trade_features` — silently produce 0-trade folds when invoked via `ArcOrchestrator.run()`. This driver bypassed `_run_step_5` and constructed `A1RunContext(per_trade_features=...)` manually before `ArcFoldRunner`. Surface this gap to master chat as a v3 infra blocker for any arc using classifier-based architectures via the orchestrator. Fix is a one-line change in `_run_step_5` to thread `run_context` through; the runner already accepts it.
- First v3.0 arc to clear Step 4 entry-feature gate (RF AUC ≥ 0.65) on 1 candidate cluster(s). Confirms the v3 27-feature default envelope CAN extract for the SHB signal class with the right cluster geometry.
- Swing-detection producer-level causal audit (Arc 9 lesson) PASS by construction: the producer `signals/lchar_swing_high_breakout_trend.py` uses `RIGHT_EDGE_OFFSET=4` to constrain 3-bar swing consumption to k ≤ t-4, making right-side detection bars k+1..k+3 ≤ t-1 — strictly prior to signal-bar open. Confirmation-lag idiom is causally clean; whitelisted by dispatch.

---

## §4 deployment_spec (FAIL arc — included for consistency, NOT for deployment)

> Abbreviated per dispatch — sub-sections 4.1-4.3 and 4.5-4.10 written from artefacts; §4.4 and §4.11 marked "FAIL arc — not applicable for deployment."
>
> FAIL verdict (closure §1 + §10).

### 4.1 Pair set

- **Pairs:** AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY (28 FX pairs, KH-24 set)
- **Timeframe:** H4 primary
- **Higher-TF references:** D1 anchor (one-bar-lagged); W1 aux (for `w1_close_slope_sign` feature). D1 + W1 panels were built and passed via the manual A1RunContext path (driver bypassed orchestrator due to the `canonical_orchestrator_step5_run_context_gap` flagged in closure §3).

### 4.2 Signal definition

Swing-high breakout in trend (SHB) long — causal 3-bar swing-high breakout with structural trend filter, decisive break + bullish close + upper-half close + 0.10×ATR buffer, 20-bar refractory.

Pseudocode:

```
# Maintain rolling 3-bar swing-highs on H4 with causal right-edge offset.
# A swing-high at bar k is "confirmed" when bar k satisfies k <= t_eval - 4
# (RIGHT_EDGE_OFFSET=4) — meaning 3 confirmation bars must close strictly
# below H[k] AND those confirmation bars must close strictly prior to t_eval.
# This makes the right-side detection causal: k+1..k+3 are all at t_eval-1 or earlier.
#
# At each H4 bar close on pair P:
#   t       = bar close timestamp (UTC)
#   atr14   = ATR(14) on H4 at bar t (Wilder)
#
#   # Trend filter: structural uptrend (HH/HL with right-edge causal lag)
#   if not structural_uptrend(h4_bars, t, swing_window_k=3): skip
#
#   # Find the most recent confirmed swing-high prior to t
#   Hs = most_recent_confirmed_swing_high(h4_bars,
#                                          k=3, right_edge_offset=4)
#   if Hs is null: skip
#
#   # Decisive break: bar close strictly above Hs.value + 0.10×ATR buffer
#   if close[t] < Hs.value + 0.10 * atr14: skip
#
#   # Upper-half close: (close - low) / (high - low) >= 0.5
#   bar_range = high[t] - low[t]
#   if bar_range == 0: skip
#   if (close[t] - low[t]) / bar_range < 0.5: skip
#
#   # Refractory: 20 H4 bars since last signal on this pair
#   if h4_bars_since_last_signal[P] < 20: skip
#
#   EMIT signal (long) at t. Entry fills at next H4 bar's open.
```

Causal-clean per closure §3: confirmation-lag idiom whitelisted by dispatch.

### 4.3 Feature computation specs

The 10 features in the winning A2 config (closure §1 `best_architecture.features_in_winning_config`):

- **`w1_close_slope_sign`** — sign of the W1 close-slope (linear-fit slope over the last N W1 bars). Categorical {-1, 0, +1}. Computed at signal-bar close from the W1 panel.
- **`d1_atr_percentile_100`** — percentile rank of current D1 ATR(14) within the trailing 100 D1 bars. One-bar-lagged.
- **`prior_session_low_distance`** — H4 close distance to prior session low, in ATR units.
- **`day_of_week`** — categorical {0..4}, signal-bar UTC weekday.
- **`session_london`** — boolean: is the signal bar within the London session (08:00-16:00 UTC, broker-day convention)?
- **`d1_close_slope_magnitude`** — magnitude of the D1 close-slope over the last N D1 bars. One-bar-lagged.
- **`distance_to_round_number`** — H4 close distance to nearest round-number level (00/50 pips), in ATR units.
- **`atr_percentile_100`** — percentile rank of current H4 ATR(14) within trailing 100 bars.
- **`usd_strength_index`** — cross-pair USD strength composite at signal-bar close.
- **`spread_vs_trailing_100`** — current bid/ask spread relative to trailing 100-bar mean.

Step 1 used the L_PROTOCOL §2 Step 1 27-feature default; Step 4 RF picked these 10 as top by permutation importance.

### 4.4 Filter chain (A1 / A2 / A6 architectures only)

**FAIL arc — not applicable for deployment.** A2 classifier_filter uses Step 4's best classifier (RF, AUC 0.6543) with AUC-best threshold 0.119 to admit/reject signals for cluster-0 (Bimodal, n=2,192 in Step 1 pool). Section retained for documentation parity only. Winning-fold metrics: ratio -0.769, ROI -24%, DD 38%, 4 negative folds — closure §10 finds FAIL on 7+ independent constraints under Amendment 3.

### 4.5 Entry mechanics

- **Trigger bar:** H4 bar satisfying §4.2 AND classifier admit at threshold 0.119.
- **Fill bar:** next H4 bar (N+1).
- **Fill price:** open of bar N+1.
- **Order type:** market.
- **Slippage assumption:** real bid/ask spreads.

### 4.6 Exit mechanics

- **Initial SL anchor:** entry price.
- **Initial SL distance:** `2.0 × ATR(14)_H4` at signal bar (closure §1 `sl_atr: 2.0`).
- **SL update rule:** static.
- **Trail activation:** N/A (`trail_enabled: false`).
- **Trail distance / reference / frequency:** N/A.
- **TP:** none (`sl_only` exit policy).
- **Time exit:** 240 H4 bars after entry.
- **Bar-by-bar evaluation order:** (1) SL hit → exit at SL price; (2) time exit at bar 240.

### 4.7 Exposure cap

- **Type:** per-pair-1 + per-currency-2 (closure §1 `exposure_cap: 2`; driver's `max_concurrent_per_pair: 1`, `max_concurrent_per_currency: 2`).
- **Counter logic:** independent counters per pair (max 1) and per currency (max 2 — counted across both base and quote currency exposure). New signal admitted only if both counters below cap.
- **Behaviour at cap:** signal skipped (no queue).

### 4.8 Risk sizing

- **`r_safe` value:** 0.1043% — **below** locked `r_min = 0.15%` floor (closure §10: `scalable_to_safe: false`).
- **`r_hard` value:** 0.1303% — also below floor.
- **Risk basis:** reset-floor.
- **Note (closure §10):** the `r_safe < r_min` scenario is the symmetric inverse of Arc 8's `r_safe > r_max` failure — both are scalability failures. Documented as `scalability_floor_failure_high_dd`.
- **Starting balance:** $100,000.

### 4.9 Session / time-of-day rules

- **Trading hours:** 24/5 standard FX session. Signal evaluation per H4 bar close; classifier ingests `session_london` feature (boolean) so the model learns session preference rather than the engine hard-filtering by session.
- **Day-of-week filter:** none beyond weekend break (model uses `day_of_week` as a categorical feature).
- **News-window filter:** none.
- **Holiday handling:** broker calendar.

### 4.10 Discrepancies and caveats

- **Config YAML reconstruction.** `configs/l_arc_11/winning_config.yaml` was reconstructed retrospectively from closure §1, `step_5/wfo_results.csv` row 3, `run_summary.json`, and `scripts/l_arc_11/run.py` constants. Original engine ran from the inline driver script. Reconstructed file is the source of truth going forward but pre-retrofit deployment would require source verification against `scripts/l_arc_11/run.py`.
- **`canonical_orchestrator_step5_run_context_gap`.** `ArcOrchestrator._run_step_5` did not thread `per_trade_features` through `ArcFoldRunner`; A2/A3/A4/A6 architectures requiring `per_trade_features` silently produce 0-trade folds when invoked via `ArcOrchestrator.run()`. This driver constructed `A1RunContext(per_trade_features=...)` manually before `ArcFoldRunner`. Surface as v3 infra blocker for any arc using classifier-based architectures via the orchestrator. Fix is a one-line change in `_run_step_5` to thread `run_context` through; the runner already accepts it.
- **Fold count 10 not 11.** Same orchestrator gap consequence — `sign_pos_folds: "6/10"` in §1. The full 11-fold sweep was not completed; downstream §10 evaluation treats the 10 reported folds as the basis.
- **Pool size canonical vs hand-rolled.** Canonical Step 1 pool is 17,533 trades vs hand-rolled 7,149 (2.5× delta). Canonical builder correctly applies exposure caps at Step 5, not Step 1 — hand-rolled was biased toward signals the cap admitted first. Closure §3 cross-arc tag.
- **Per-day max-DD series + chained DD not measured.** PROVISIONAL on those constraints; verdict DEFINITIVE because multiple independent constraints fail (ratio, scalability, sign-consistency, trade count, holdout DD).
- **Step 4 AUC ≥ 0.65 was met (0.6543).** First v3.0 arc to clear the entry-feature gate. Architecture-side failures dominate — the classifier extraction is genuinely capturing some signal but the WFO outcome under chained-fold realism is FAIL.

### 4.11 Deployment readiness checklist

**FAIL arc — not applicable for deployment.**

- [ ] N/A — FAIL verdict; not for deployment.
- [ ] N/A
- [ ] N/A
- [ ] N/A
- [ ] N/A
- [ ] N/A (Step 6 not run — lazy dispatch only on PASS candidates)

---

## §10 Amendment 3 re-evaluation (added 2026-05-22)

**Original verdict:** FAIL (primary_failure_mode: `step5_dd_above_gate`, failed_at_step: 5)
**Re-evaluated verdict:** FAIL
**Re-evaluation status:** definitive (four independent constraints fail; missing data flags don't affect the verdict)
**Re-evaluated primary_failure_mode:** `step5_not_scalable` (per Amendment 3 §3 priority order)

### Scaling derivation
- `worst_fold_dd_base_pct`: **38.36%** (A2 cluster-0, worst fold; closure §1 `worst_fold_dd_pct`)
- `worst_fold_roi_base_pct`: **−24.03%** (closure §1 `worst_fold_roi_pct` — negative)
- `k_safe = 8.0 / 38.36 = 0.2085`
- `k_hard = 10.0 / 38.36 = 0.2607`
- `r_safe_pct = 0.5 × 0.2085 = 0.1043%`
- `r_hard_pct = 0.5 × 0.2607 = 0.1303%`
- `scalable_to_safe`: **false** (`r_safe = 0.1043%` < locked `r_min = 0.15%` floor by 0.046pp)
- `scalable_to_hard`: **false** (`r_hard = 0.1303%` also below 0.15% floor)

### Amended DEPLOYABLE gate evaluation

| # | Constraint | Threshold | Value at r_safe | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 1 | Scalable to safe | `r_safe ∈ [0.15%, 2.0%]` | 0.1043% | ✗ | below floor — base DD too large to scale risk down to 8% while staying ≥ 0.15% per-trade |
| 2 | Worst-fold ROI/DD ratio | ≥ 2.0 | **−0.626** (§3 math: −24.03 / 38.36) | ✗ | invariant; engine reading −0.769 also < 2.0 |
| 3 | Worst-fold ROI | > 0 | −5.01% (= −24.03 × 0.2085) | ✗ | sign-negative at any scaling |
| 4 | Per-fold positivity | all 11 positive, 0 negative | 6/10 (NB: 10 folds, not 11 — see closure §3 `canonical_orchestrator_step5_run_context_gap`) | ✗ | 4 negative folds — DEPLOYABLE requires 0 |
| 5 | Worst-fold DD | ≤ 8% | 8.00% (= 38.36 × 0.2085) | ✓ | by construction of `k_safe` (but moot — scalability fails first) |
| 6 | Daily DD breaches | = 0 | per-fold counts at r_base: folds 6/9/10 = 1 each, fold 11 = 2 (A2 winner) | ✗ | non-zero at `r_base` already; under downward scaling `k_safe = 0.21` count could drop but not below 0 if any single day's DD was ≥ ~24% at base. PROVISIONAL on the proper per-day recount; DEFINITIVE that base count > 0. |
| 7 | Chained max DD | ≤ 10% | unknown × 0.21 | ? | PROVISIONAL: `chained_max_dd_base_pct` not measured |
| 8 | Trades per fold | ≥ 25 | min n_trades = **15** (A2 winner, fold 5) | ✗ | below 25 floor — `step_5/per_fold_metrics.csv` confirms |
| 9 | Holdout at r_safe | clears prior §3 holdout gate | proxy: ROI ≈ +0.27% (= 1.30 × 0.2085), DD ≈ 11.99% (= 57.50 × 0.2085) | ✗ | DD 11.99% > 8% gate; holdout failed at base anyway |
| 10 | Step 6 clean | clean | not run | ? | Step 6 lazy — not dispatched because arc didn't produce a PASS-VIABLE/DEPLOYABLE candidate. Moot. |

### Amended VIABLE gate evaluation

| # | Constraint | Threshold | Value at r_hard | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 1 | Hard-scalable | `r_hard ∈ [0.15%, 2.0%]` | 0.1303% | ✗ | below floor |
| 2 | Worst-fold ROI/DD ratio | ≥ 2.0 | −0.626 | ✗ | invariant |
| 3 | Mean-fold ROI/DD ratio | ≥ 2.5 | engine reports 0.4305; closure §1 `mean_fold_roi_pct: null` | ✗ | well below 2.5 even at engine reading |
| 4 | Per-fold positivity | ≤ 1 negative fold | 4 negative folds | ✗ | exceeds VIABLE tolerance |
| 5 | Worst-fold DD | ≤ 10% | 10.00% (= 38.36 × 0.2607) | ✓ | by construction (moot) |
| 6 | Daily DD breaches | = 0 | non-zero at base | ✗ | same as DEPLOYABLE |
| 7 | Chained max DD | ≤ 10% | unknown × 0.26 | ? | PROVISIONAL |
| 8 | Trades per fold | ≥ 25 | min 15 | ✗ | |
| 9 | Holdout at r_hard | clears prior §3 holdout gate | proxy: ROI ≈ +0.34%, DD ≈ 14.99% | ✗ | DD > 10% VIABLE gate |
| 10 | Step 6 clean | clean | not run | ? | Moot |

### Engine vs §3 ratio discrepancy (flagged per chat Q1)

The closure §1 reports `worst_fold_ratio: −0.7687`. §3 mathematical reading: `worst_fold_roi / worst_fold_dd = −24.03 / 38.36 = −0.626`. Per chat directive (Q1: option a), the §10 evaluation uses **−0.626**. Both readings fail the 2.0 gate. Verdict outcome unchanged.

### Final assessment

Arc 11 fails the Amendment 3 DEPLOYABLE *and* VIABLE gates on **at least seven independent constraints**: scalability (both tiers; `r_safe = 0.10%`, `r_hard = 0.13%` — both below the 0.15% floor by ~0.02–0.05pp because worst-fold DD 38.36% is too large to compress to 8%/10% within allowed per-trade risk), ratio (−0.63 vs 2.0 gate, invariant), worst-fold ROI sign (negative), per-fold positivity (4 negative folds), trade count (min 15 < 25), holdout DD at any scaling, and (informationally) daily breaches non-zero at base. Per Amendment 3 §3 failure-mode priority order, `step5_not_scalable` precedes the other failures and becomes the primary failure mode — replacing the original closure's `step5_dd_above_gate` (which is itself deprecated in the amendment's taxonomy per §3 "Failure-mode taxonomy / Deprecated but retained for historical closures").

The original FAIL verdict stands. The amendment doesn't materially change Arc 11's outcome; it does provide a cleaner failure-mode taxonomy. Missing data flags (chained DD, per-day series) don't affect the verdict — multiple definitive failures are independent of those gaps.

### Missing data flags

- Constraint #6 (daily DD breaches at `r_safe`): per-day max-DD series not built. Per-fold count at `r_base` IS available (non-zero — already FAIL the constraint at base). PROVISIONAL on the recounted value at `r_safe`, DEFINITIVE on the base-count being non-zero.
- Constraint #7 (chained max DD): `chained_max_dd_base_pct` not measured. PROVISIONAL — but moot, scalability and ratio fail definitively independent of this gap.
- **No engine re-run recommended.** Multiple independent definitive FAILs; the amended gate's strictest reading would require re-running with continuous-equity + per-day emission to upgrade *missing flags* but no flag-upgrade path produces a different verdict.

### Cross-arc tags (additions to closure §3 tags)

- `scalability_floor_failure_high_dd` — first documented case of `r_safe < r_min` failure mode (mirror of Arc 8's `r_safe > r_max`: same Amendment 3 scalability mechanism, opposite end of the DD distribution)
- `multi_independent_failure_amendment3` — failure mode count: 6+ constraints fail independently at the amended gate, vs the original protocol's single `step5_dd_above_gate` framing. Amendment 3's failure-mode taxonomy surfaces more diagnostic detail without changing the verdict.
- `engine_ratio_vs_amendment_ratio_divergence` — engine `worst_fold_ratio: −0.769` vs §3 math `−0.626`; same FAIL outcome.
