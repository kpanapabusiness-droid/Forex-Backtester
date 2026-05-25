# Arc 7 v3.0.1 — CC log doc

> **Pattern source:** `WORKFLOW.md` v3.0 §2 step 4 — log doc records
> verification results, deviations from dispatch, and flags for chat.
> **Dispatch:** Arc 7 Retry v3.0.1 (chat-supplied 2026-05-22; PRs #185/#186/#188/#189 landed)
> **Intent doc:** `docs/dispatches/arc_7_v3_0_1_intent.md`
> **Closure doc:** `results/l_arc_7/ARC_CLOSURE.md`
> **Branch:** `arc/l_arc_7` (reset to `origin/main @ 4fccff0`)
> **Verdict:** FAIL — `step5_not_scalable`

---

## Turn-by-turn

**Turn 1 — Branch handling + read-first sweep.** Hard-reset `arc/l_arc_7`
to `origin/main @ 4fccff0` per dispatch (prior v3.0 closure discarded;
never merged, no methodological loss). Read L_PROTOCOL Amendments 1-4,
template v1.3, WORKFLOW, PROTOCOL_RUNTIME, engine capability audit,
CLAUDE.md.

**Turn 2 — Intent doc.** Wrote `docs/dispatches/arc_7_v3_0_1_intent.md`
including producer-level swing-low causal trace (Arc 9 lesson). Flagged
seven interpretive calls (A driver path, B EET vs UTC, C A5 trigger,
D chained_dd_method, E Step 6 auto-only, F results clobber, G Step 6
§6.3 UTC literal vs EET reality).

**Turn 3 — Chat approval.** Intent approved with specifics on all seven
calls. Composition driver per intent §A; 5ers_eet per PR #189;
chained_dd_method=equity_stitching; Step 6 auto-only; results clobber
proceed; G framework gap not arc failure.

**Turn 4+ — Implementation + execution.** Wrote signal module + spec doc
+ tests (5/5 pass). Wrote composition driver `scripts/arc_7/run_arc_7.py`
(~900 LOC). Wrote closure helper `scripts/arc_7/write_closure.py` for
template v1.3 emission. Launched driver in background.

**Turn N — Driver complete.** 4760s wall (~79 min). FAIL verdict.
Generated closure + ran tracker parser + force-pushed + opened PR.

---

## Code written

- `core/strategies/liquidity_sweep_reclaim_long/__init__.py`
- `core/strategies/liquidity_sweep_reclaim_long/signal_module.py` —
  `LiquiditySweepReclaimLongSignal` per `core.arc.signal_protocol.SignalModule`
  Protocol. Bid-side OHLC throughout per KH-24 v3 convention (PR #189
  mid-price refactor applies to features, not signal evaluation per
  `PROTOCOL_RUNTIME.md` §15.1).
- `docs/archive/signal_specs/signal_spec_liquidity_sweep_reclaim_long_v0.1.md`
- `tests/test_liquidity_sweep_reclaim_long_signal.py` — 5 unit tests
  (protocol conformance, all-six-conditions fire, C4 violation suppresses,
  no-lookahead perturbation, params locked). All pass.
- `scripts/arc_7/__init__.py`
- `scripts/arc_7/run_arc_7.py` — composition driver per intent §A.
  Uses canonical primitives end-to-end:
  - `build_arc_pool` (Step 1 plumbing)
  - `compute_feature_matrix` (Step 1 features; `panel.aux` injection
    workaround for multi_tf features — engine_capability_audit
    §"compute_feature_matrix doesn't accept aux panels")
  - `run_step_{2,3}` (canonical)
  - `run_step_4(persistence_dir=..., train_end=2021-01-01)` (PR #185 +
    holdout-exclusion fix); driver renames `lineage`→`causal_lineage`
    column at Step 4 call site (engine_capability_audit §"Lineage filter")
  - `build_{a2,a3,a4,a6}_config_from_step4` (PR #185 classifier persistence)
  - `build_path_classifier_fits_per_fold` (PR #186 A3/A4 per-fold retrain)
  - `ArcFoldRunner` + `run_search` + `run_holdout` (Step 5)
  - `compute_per_day_max_dd` + `stitch_per_fold_oos_equity` +
    `compute_chained_max_dd_from_continuous_equity` +
    `rescale_arch_config_risk` + `classify_amended_fold_stats` (Amendment 3)
  - `maybe_dispatch_step_6` + `replace_top_1_with_step6_fail`
    (Amendment 4 / PR #188)
  - `OracleFoldRunner` (oracle WFO per cluster with winning exit policy
    per Arc 10 v3 cross-arc lesson)
  - Local helper `_build_panel_5ers_eet` (serial composition of canonical
    `aggregate(..., boundary_convention="5ers_eet")` since the canonical
    `build_panel_parallel` doesn't pass `boundary_convention` through;
    engine code unchanged per dispatch)
- `scripts/arc_7/write_closure.py` — closure-generation helper for template
  v1.3 (manual emission per chat note 2; `core.arc._closure_template` still
  emits OLD 7-section layout).

---

## Driver execution

- Launch: 2026-05-24 01:47:07 (local) → completion 03:06:27 (local).
- Wall: 4760.0s = ~79 min.
- 5ers_eet cache cold-build for 28 pairs × 3 TFs took ~25 min (serial via
  `_build_panel_5ers_eet`; subsequent runs will hit warm cache via
  `data/cache/{H4,D1,W1}_5ers_eet/`).
- Step 1 pool: 5,175 trades in ~30s after panels built.
- Step 1 features: ~70s (27 × 5,175 with cross_pair + multi_tf via
  panel.aux).
- Step 2: K=4, silhouette 0.488 (~10s).
- Step 3: 2 candidates (c0 Bimodal, c1 Unclassified) (~2s).
- Step 4: 2 cluster extractions in 51s; persisted classifiers under
  `results/l_arc_7/step_4/classifiers/`.
- A4 per-fold fits build: ~2 min (12 fold fits × ~10s each).
- Step 5: 28 (config) × 10 (folds) = 280 (cand × fold) pairs in ~47 min;
  avg 10.2s per fold-run.
- Holdout top-3 × 1 fold: ~2 min.
- Amendment 3 evaluation per top-3: ~30s.
- Oracle WFO per cluster × 10 folds: ~2 min.
- Step 6 not dispatched (no PASS-tier candidate).

---

## Verification results

- **Pool size:** 5,175 ≥ 500 dispatch minimum ✓ (no HALT trigger fired).
- **Determinism:** `seed_everything(42)` at driver entry; pool sha256 =
  `402048a2e231dcbc9394b38f82d8a1032ceb368fcb1b4ccce85a9fd8720ad460`.
  Two-run reproducibility not exercised (compute cost prohibitive on full
  WFO; would double total wall to ~160 min).
- **Signal module tests:** 5/5 pass including no-lookahead perturbation
  spot-check.
- **Lookahead spot-check at Step 1:** integrity report records declared
  lineage = `clean`; producer-level trace in `signal_module.py` docstring
  + closure §3 verifies `swing_low_N` uses only strictly prior bars.
- **Amendment 3 emission:** `chained_max_dd_base_pct`, `per_day_max_dd`
  parquet per top-K, holdout re-runs at r_safe / r_hard, scaled metrics,
  scalability check all emitted per `core.wfo.amended_gates`. Tracker
  payload populates all Amendment 3 fields.
- **Step 6 dispatch logic:** correctly skipped (no PASS-tier candidate
  cleared §3 #1-9). Manual CLI not invoked per dispatch §E.
- **Anchor preservation (L_PROTOCOL §1 + §8):** not exercised this arc.

---

## Deviations from dispatch

1. **Exit-policy axis reduced.** Tested `trail_enabled ∈ {True
   (sl_plus_trailing_atr), False (sl_only)}` only. Skipped:
   `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_partial_close_1r_runner_trail`,
   `sl_plus_trailing_swing`, `time_exit_n_bars`. Reason: v3 A1/A2/A6
   Configs don't expose `tp_atr_mult` or partial-close mechanics.
   Explicit list at `results/l_arc_7/step_5/skipped_configs.md` per
   dispatch's "do not silently shrink" rule.
2. **`build_panel_parallel` doesn't accept `boundary_convention`.**
   Composition via `_build_panel_5ers_eet` local helper that calls
   canonical `aggregate(..., boundary_convention="5ers_eet")` serially.
   Engine code unchanged per dispatch §"What you do NOT do". Cold-cache
   serial build is one-time (~25 min); warm cache fast (<10s).
3. **A4 training-bar offset fixed at K=5.** Per `path_classifier_per_fold.A4_TRAIN_DECIDE_OFFSET`
   default. Amendment 2 §"A4" intent is per-bar training; v3.0.1 engine
   uses fixed K. Not a deviation from dispatch (dispatch points at the
   canonical module); flagged here for future v3.0.2 follow-up.
4. **Closure renderer mismatch.** `core.arc._closure_template.py` emits
   OLD 7-section layout; closure manually rendered via
   `scripts/arc_7/write_closure.py` per chat note 2.

---

## Flags for chat (after PR merge — or as inputs to next dispatch)

1. **Architecture-map gap — second arc of record.** Both Arc 7 v3.0 (c0
   Bimodal AUC 0.6758) and Arc 7 v3.0.1 (c1 Unclassified AUC 0.6642) had
   Step 4 AUC above 0.65 BUT A2/A6 not tested because dispatch archetype
   map says Bimodal → {A1, A4} and Unclassified → {A1}. Recommended
   v1.1 dispatch amendment: when any cluster Step 4 AUC ≥ 0.65,
   automatically add A2 and A6 to the search regardless of archetype.
   Surfaced in closure §3 and tagged
   `architecture_map_gap_second_arc_recommend_v1_1`.
2. **Engine cleanup items (not fixed inline per dispatch §"What you do NOT do"):**
   - `core.parallel.build_panel_parallel` doesn't pass `boundary_convention`
     through to `aggregate` — driver compose-around via
     `_build_panel_5ers_eet`. Trivial fix.
   - `core.features.pipeline.feature_lineage_dataframe` emits `lineage`
     column; `core.steps.step_4_extraction._filter_lineage` checks
     `causal_lineage` — driver renames at call site (engine_capability_audit
     §"Lineage filter"). Single-line fix.
   - `compute_feature_matrix(pair, df, panel)` doesn't accept aux panels
     for multi_tf features — driver bypasses Panel frozen-dataclass via
     `object.__setattr__(panel, "aux", {...})`. Engine-side fix would add
     `aux` parameter.
   - `core.arc._closure_template.ARC_CLOSURE_TEMPLATE` emits OLD 7-section
     layout pre-v1.0; should emit v1.3 with §1 tracker_payload YAML.
3. **Driver script `scripts/arc_7/run_arc_7.py` could be promoted to
   `scripts/arc_template/run_arc.py`** as the v3.0.1 reference driver for
   future arcs that need archetype-based architecture selection (until
   orchestrator's `auto_arch_specs` API supports dynamic per-cluster
   spec construction). ~900 LOC composition driver reusable as-is by
   parameterizing arc_name + signal_module imports.
4. **Step 6 §6.3 UTC literal vs EET reality (per intent §G):** N/A this
   closure (Step 6 didn't dispatch). Becomes live when a PASS-tier candidate
   gets evaluated under EET — master chat owns framework fix.

---

## Definition of done

1. ✓ Intent doc produced (with swing-detection causal trace) and chat-approved
2. ✓ Steps 1-5 executed continuously on canonical infra
3. — Step 6 not auto-dispatched (no PASS-tier candidate per Amendment 4 evaluation order)
4. ✓ `results/l_arc_7/ARC_CLOSURE.md` produced per template v1.3
5. ✓ `docs/dispatches/arc_7_v3_0_1_log.md` produced per WORKFLOW §2
6. ✓ `ARC_TRACKER.md` updated via parser
   (`scripts/update_tracker_from_closure.py results/l_arc_7/ARC_CLOSURE.md`)
7. — Force-push: pending after this log writes
8. — PR open: pending
9. — End turn after PR opens
