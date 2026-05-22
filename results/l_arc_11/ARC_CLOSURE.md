# ARC_11_CLOSURE — l_arc_11

> **Closed:** 2026-05-22T12:33:35Z
> **Branch:** arc/l_arc_11
> **Closure doc path:** results/l_arc_11/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:
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
