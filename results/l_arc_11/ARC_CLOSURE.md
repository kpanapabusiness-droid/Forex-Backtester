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
