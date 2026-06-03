# Arc 11 — Dispatch Log

Run completed: 2026-05-22T12:33 UTC (canonical-infra run)

## Steps executed

| Step | Source | Key result |
|---|---|---|
| 1 Plumbing | `core/arc/arc_pool_builder.py` | pool n=17,533; pool_sha256=b1d1a5604db835e8... |
| 2 Clustering | `core/steps/step_2_clustering.py` | best K=4 (silhouette 0.4934); silhouettes {2: 0.396, 3: 0.395, 4: 0.493, 5: 0.476, 6: 0.459} |
| 3 Capturability | `core/steps/step_3_capturability.py` | candidate clusters: (0, 1). c0: composite 1.98, reach_1R 1.0, mfe_p50 7.77R, ww_pp 0.002; c1: composite 0.97, reach_1R 0.96, mfe_p50 2.11R, ww_pp 0.035 |
| 4 Extraction | `core/steps/step_4_extraction.py` | c0 best=rf AUC 0.6543 threshold 0.119; c1 best=rf AUC 0.6316 threshold 0.321 |
| 5 WFO | `core/runners/arc_fold_runner.py` + `core/wfo/orchestrator.py::run_search`/`run_holdout` | A1: ratio -0.89 / DD 40.6%; A2: ratio -0.77 / DD 38.4%; A6: 0 trades; all FAIL |

## Deviations from dispatch

- **Hand-rolled scripts deleted** per chat decision 2026-05-22. The first iteration of Arc 11 used a from-scratch `scripts/l_arc_11/step_{1..5}_*.py` simulator that biased the Step 1 pool (applied per-pair exposure cap at pool-build time → 7,149 trades). Canonical re-run produces 17,533 trades (2.5x larger) because `core/arc/arc_pool_builder.py` correctly applies caps at Step 5 architecture level only.
- **Branch `arc/l_arc_11`** created by rename from worktree auto-branch.
- **Signal spec doc** reconstructed from producer docstring at `docs/archive/signal_specs/signal_swing_high_breakout_trend_long_v0.1.md` per chat ack on intent doc Flag A.
- **Inter-step end-turn for chat review** overridden per chat instruction; arc ran continuously.
- **v3 KH-24 anchor reproduction** PARTIAL per Path B (CC_06); divergence inherited but not blocker per intent doc Flag C.
- **`ArcOrchestrator._run_step_5` wiring gap** — see "Flags for master chat" below. Driver bypassed the orchestrator's Step 5 helper and invoked `run_search` / `run_holdout` directly on `ArcFoldRunner(run_context=A1RunContext(per_trade_features=...))` to allow A2 / A6 to evaluate.
- **A3 / A4 architectures skipped** — A4 would require per-bar classifier inference + exit decisions, out of scope for this arc's time budget. A3 (deferred entry) was not wired in this canonical run; the dispatch's archetype mapping (V-shape → A3) doesn't fire for Bimodal cluster 0. Could be added in follow-up.
- **KH-24 co-fire integrity check** at Step 1 DEFERRED (KH-24 strategy not wired in canonical pool builder; informational only per dispatch).
- **Oracle WFO** not run in canonical infra — the canonical `OracleFoldRunner` requires manual invocation outside the orchestrator's Step 5 helper (same `run_context` gap as A2/A6). Could be added in follow-up; closure `oracle_worst_ratio` field set to null.

## Flags for master chat

### 1. v3 canonical infra bug — `_run_step_5` does not plumb `run_context`

`core/arc/arc_orchestrator.py::_run_step_5` constructs `ArcFoldRunner` without passing the `run_context` argument:

```python
r = ArcFoldRunner(
    architecture=arch,
    signal_evaluation=signal_eval,
    panels=self.panels,
)
```

`ArcFoldRunner` accepts `run_context` (see `core/runners/arc_fold_runner.py:40`), and the architectures A2/A3/A4/A6 ALL require `ctx.per_trade_features` to be non-None to admit any trades (see `core/architectures/a2_classifier_filter.py:111`, `a6_meta_labeling.py:122`, etc.). Silent failure mode: every classifier-based architecture admits zero trades when invoked via `ArcOrchestrator.run()`.

**Fix:** one-line change in `_run_step_5` to pass `run_context=` to `ArcFoldRunner`. The `run_context` source could be a new field on `ArcConfig` (`run_context_factory: Callable[[], A1RunContext] | None = None`) that the orchestrator invokes during Step 5 wiring.

**Verified during this arc:** Arc 11 driver bypassed the orchestrator helper and constructed `per_trade_features` from `feature_matrix` manually; A2 then admitted 380 trades (still FAIL but at least evaluable). Without this workaround Arc 11's classifier-based architectures would be unevaluable.

### 2. Hand-rolled Step 1 pool cap convention vs canonical

Arc 11's pre-canonical iteration applied `max_concurrent_per_pair=1` at Step 1 pool-build time. Canonical builder does NOT cap at Step 1 — exposure caps are applied at Step 5 architecture level. The hand-rolled approach biased pool composition by keeping only the FIRST signal in each pair-overlap window, depressing pool size 2.5x (7,149 vs 17,533) and shifting cluster topology (K=2 vs K=4 selected).

Cohort-level finding: canonical c0 composite is 1.98 (vs hand-rolled 0.99). The SHB cluster 0 cohort is materially stronger than the hand-rolled closure reported.

Any arc that hand-rolls a Step 1 simulator with per-pair caps applied at pool-build time will be similarly biased.

### 3. Empty-IS F01 handling (informational)

`core/wfo/folds.py::build_v3_folds` produces 11 nominal folds. With `min_is_days=365` in `run_search`, fold F01 (OOS=2010, IS=empty) is skipped — 10 evaluable folds remain. This is the canonical behaviour and matches my Arc 11 closure's `sign_pos_folds: X/10` convention. Any chat-side language around "11-fold WFO" should clarify whether 11 = nominal-builder count or evaluable count.

### 4. Verdict + cross-arc tag emissions

- Arc 11 verdict: FAIL (`step5_dd_above_gate`). Best A2 worst-fold DD 38.36% >> 10% PASS-VIABLE gate.
- Strong cohort signal: c0 step3_composite 1.98 + step4_e_auc 0.65. Cohort is real; the classifier extraction at AUC 0.65 isn't strong enough to filter out the contaminating losses from clusters 1/2/3.
- See closure `§3 Cross-arc observations` for the full list (4 tags).
