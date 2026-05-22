# step4_classifier_persistence_log.md

> **Dispatch:** `CC Dispatch — Engine PR: Expose Step 4 Fitted Classifiers for A2/A6 Architectures`
> **Intent doc:** [`step4_classifier_persistence_intent.md`](step4_classifier_persistence_intent.md)
> **Branch (worktree):** `claude/dreamy-meitner-4d18ac` (PR opens from this; dispatch name `engine/step4-classifier-persistence` recorded in PR title only)
> **Status:** Tasks 1-10 + 6.5 complete. 58 protocol_runtime tests pass (43 prior + 13 new persistence + 2 new e2e). ruff clean on touched files. Awaiting full-suite green.

---

## Chat resolutions used

Re-sent five answers from the dispatch follow-up:

- **Q1: Option A** — persisted classifier refits on the full lineage-filtered Step 1 pool (same data Step 4's CV iterated over).
- **Q1 critical condition surfaced:** Step 4 today does train on the full pool **including holdout window**. Chat acknowledged; bug deferred to a separate dispatch. This PR documents the inheritance explicitly in `manifest.json` (via `trained_on_pool_size`) and in `docs/PROTOCOL_RUNTIME.md` §7 "Data-scope inheritance". When the holdout-exclusion fix lands, the persistence helpers retrofit unchanged.
- **Q2:** L_PROTOCOL line 282 applies to A3 / A4 only. Added "Architecture-specific retraining policy" table to L_PROTOCOL §2 Step 5; rewrote the shared-discipline bullet to scope per-fold-retrain to A3 / A4 explicitly. A2 / A6 reuse the single Step-4-fit classifier across folds.
- **Q3:** Builders co-located with `load_classifier` in `core/steps/classifier_persistence.py`.
- **Q4:** Persisted classifier trains on same data Step 4 cross-validated against. No special train/holdout window for persistence. Documented in §7 and in the `_persist_best_classifier` docstring.
- **Q5:** Orchestrator wired (Task 6.5). `ArcOrchestrator._run_step_5` now constructs `A1RunContext(per_trade_features=…)` from `cfg.feature_matrix` and threads it to every `ArcFoldRunner`. New `AutoArchSpec` dataclass + `auto_arch_specs` field on `ArcConfig` lets callers declare "build A2 / A6 from Step 4 result"; orchestrator dispatches to `build_a2_config_from_step4` / `build_a6_config_from_step4` at Step 5 time. Integration test `test_a2_runs_end_to_end_via_orchestrator` exercises the full orchestrator path (not a bypass).

---

## Files created

| Path | Purpose |
|---|---|
| `core/steps/classifier_persistence.py` | `load_classifier`, `ClassifierIntegrityError`, `build_a2_config_from_step4`, `build_a6_config_from_step4`. SHA256-verified loads with `UserWarning` on joblib / sklearn / lightgbm version drift. |
| `tests/protocol_runtime/test_step_4_classifier_persistence.py` | 13 unit tests covering persistence file placement, manifest SHA, round-trip predictions, builders, threshold sweep, integrity errors, determinism. |
| `tests/protocol_runtime/test_a2_end_to_end.py` | 2 integration tests exercising A2 via `ArcOrchestrator.run()` — the test that would have caught the Arc-11 gap. |
| `docs/dispatches/step4_classifier_persistence_intent.md` | Read-first intent doc with 5 chat questions. |
| `docs/dispatches/step4_classifier_persistence_log.md` | This file. |

## Files modified

| Path | Change |
|---|---|
| `core/steps/step_4_extraction.py` | `ClusterExtraction` extended with `fitted_classifier_path` / `_type` / `_feature_order`. New `persistence_dir` + `arc_name` kwargs on `run_step_4` (both default such that existing callers see no change). `_persist_best_classifier` refits best-AUC algorithm on full lineage-filtered pool; `_write_manifest` writes SHA256 + provenance. |
| `core/arc/arc_orchestrator.py` | New `AutoArchSpec` dataclass; new `auto_arch_specs` field on `ArcConfig`. `_run_step_4` invokes persistence via `output_dir / "step_4" / "classifiers"`. `_run_step_5` now receives `pool` + `s4`, builds `A1RunContext` from `cfg.feature_matrix`, dispatches `AutoArchSpec` entries to `_AUTO_BUILDERS`, threads run_context through `ArcFoldRunner`. Holdout block also receives run_context. `_build_per_trade_features` helper mirrors `scripts/l_arc_11/run.py:build_per_trade_features`. |
| `L_PROTOCOL.md` | Amendment 4 added (top). §2 Step 4 output list extended with `classifiers/` artefacts. §2 Step 5 ML mechanics gains "Architecture-specific retraining policy" table; shared-discipline bullet scoped to A3 / A4. |
| `docs/PROTOCOL_RUNTIME.md` | §7 Step 4 documents the persistence artefacts, manifest schema, helper usage, data-scope-inheritance caveat, and orchestrator wiring. |
| `docs/BACKTESTER_ARCHITECTURE.md` | "Out of scope" list updated — points readers to PROTOCOL_RUNTIME §7 for the new persistence pattern. No new section added (this file does not document step internals by design). |

---

## Decisions deviating from / clarifying the dispatch

1. **`BACKTESTER_ARCHITECTURE.md` Step 4 section.** Dispatch Task 9 asks to update this file's "Step 4 section". The file is the *backtester* reference and explicitly leaves step internals to PROTOCOL_RUNTIME (see lines 367-380, "Out of scope"). Updating the "Out of scope" pointer rather than duplicating §7 content avoids the parallel-doc divergence trap. Flagged in intent doc §"File paths" T8 row.
2. **`pyproject.toml` version pinning.** Dispatch Risk #1 asks to pin joblib / sklearn in `pyproject.toml` "if not already". There is no `pyproject.toml` in the repo (`requirements-dev.txt` carries top-level deps with no pins). Per intent doc Interpretive call 4 recommendation B+C: manifest records `joblib_version`, `sklearn_version`, `lightgbm_version` at write time; loader emits `UserWarning` on drift but does not refuse. Pinning is a separate concern for a broader dependency-review PR.
3. **Builder location.** Dispatch says "arc config layer". `core/arc/arc_config.py` does not exist (`ArcConfig` lives in `arc_orchestrator.py`). Per chat Q3, builders went into `core/steps/classifier_persistence.py` next to `load_classifier`. Orchestrator imports them from there.
4. **`auto_arch_specs` instead of overloading `architecture_configs`.** Existing `architectures` × `architecture_configs` zip pattern preserved for callers with pre-built configs (Arc 11 driver still works). New `auto_arch_specs: tuple[AutoArchSpec, ...]` is the orchestrator-driven path. Callers can mix the two (explicit A1 + auto-spec A2 in the same `ArcConfig`). Avoids a backwards-incompatible signature change.
5. **`persistence_dir` defaults to `None`.** Existing 4 Step 4 tests, plus the `test_arc_orchestrator_e2e` smoke (which does not set `feature_matrix`), continue to pass unchanged. Orchestrator-driven runs that set `output_dir` AND `feature_matrix` get persistence automatically. Drivers that prefer the canonical pattern (e.g. Arc 11) can call `run_step_4(..., persistence_dir=...)` directly.

---

## Verification

```
$ python -m pytest tests/protocol_runtime -x -q
58 passed, 235 warnings in 99.31s
```

Breakdown:
- 43 pre-existing protocol_runtime tests — all pass
- 13 new persistence tests — all pass
- 2 new A2 e2e tests — all pass

```
$ python -m ruff check core/steps/step_4_extraction.py core/steps/classifier_persistence.py \
    core/arc/arc_orchestrator.py tests/protocol_runtime/test_step_4_classifier_persistence.py \
    tests/protocol_runtime/test_a2_end_to_end.py
All checks passed!
```

Full-suite green: TBC (background job in flight at time of writing this log).

---

## Open follow-ups (separate dispatches)

1. **Bug surfaced under Q1 critical condition: Step 4 CV trains on full pool including holdout window.** Persistence helpers are data-agnostic — fix retrofits cleanly. Recommendation: open separate dispatch `engine/step4-holdout-exclusion` adding a `train_window_end` parameter to `run_step_4`, threaded from `WfoStructure.holdout.oos_start`. Arc 5 / 7 retries should wait for this before deployment-credible verdicts.
2. **Arc 5 v3.0.1 + Arc 7 v3.0.1 A2/A6 retries** per dispatch "After this lands". Unblocked once this PR merges and bug #1 above is at least scoped.
3. **Wave 2** per dispatch — unblocked once Arcs 5/7 retry green.
4. **`requirements-dev.txt` pin pass** — separate dependency-review PR.

End of log.
