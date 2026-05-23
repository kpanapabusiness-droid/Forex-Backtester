# step4_classifier_persistence_log.md

> **Dispatch:** `CC Dispatch — Engine PR: Expose Step 4 Fitted Classifiers for A2/A6 Architectures` (+ Option III holdout-exclusion fix per chat HOLD on PR #183)
> **Intent doc:** [`step4_classifier_persistence_intent.md`](step4_classifier_persistence_intent.md)
> **Branch:** `engine/step4-classifier-persistence-v2` (supersedes #183 — Path R per chat direction; PR #183 reverted on main at commit 192e01e)
> **Status:** Tasks 1-10 + 6.5 + R1-R9 complete. Persistence + holdout-exclusion landed atomically per chat Option III. ruff clean on touched files.

---

## Chat resolutions used

Re-sent five answers from the dispatch follow-up:

- **Q1: Option A** — persisted classifier refits on the IS-only lineage-filtered Step 1 pool (`entry_time < train_end`). When no holdout is configured (`train_end=None`), the refit pool is the full lineage-filtered pool — same data Step 4's CV iterated over.
- **Q1 critical condition resolved IN THIS PR (Option III per chat HOLD on #183):** Step 4 now respects a `train_end` parameter. Both the 5-fold TimeSeriesSplit CV evaluation AND the persisted-classifier refit restrict to `entry_time < train_end`. The orchestrator threads `train_end` from `WfoStructure.holdout.oos_start`. Manifest declares the cutoff explicitly via a new top-level `train_end` field (ISO timestamp or `null`).
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
| `core/steps/step_4_extraction.py` | `ClusterExtraction` extended with `fitted_classifier_path` / `_type` / `_feature_order`. New `persistence_dir`, `arc_name`, **`train_end`** kwargs on `run_step_4` (all default such that existing callers see no change). When `train_end` is supplied, trades/cluster_assignments/feature_matrix are filtered to `entry_time < train_end` BEFORE CV. `_persist_best_classifier` refits best-AUC algorithm on the IS-only lineage-filtered pool; `_write_manifest` writes SHA256 + provenance + the top-level `train_end` declaration. |
| `core/arc/arc_orchestrator.py` | New `AutoArchSpec` dataclass; new `auto_arch_specs` field on `ArcConfig`. `_run_step_4` invokes persistence via `output_dir / "step_4" / "classifiers"` and threads `train_end` from `WfoStructure.holdout.oos_start` (new `_resolve_train_end` helper). `_run_step_5` now receives `pool` + `s4`, builds `A1RunContext` from `cfg.feature_matrix`, dispatches `AutoArchSpec` entries to `_AUTO_BUILDERS`, threads run_context through `ArcFoldRunner`. Holdout block also receives run_context. `_build_per_trade_features` helper mirrors `scripts/l_arc_11/run.py:build_per_trade_features`. |
| `L_PROTOCOL.md` | §2 Step 4 output list extended with `classifiers/` artefacts. §2 Step 5 ML mechanics gains "Architecture-specific retraining policy" table per Q2; shared-discipline bullet scoped to A3 / A4. **No top-of-doc Amendment 4** — chat HOLD direction (the previous PR's unauthorised "Amendment 4 (2026-05-23)" line was removed). |
| `docs/PROTOCOL_RUNTIME.md` | §7 Step 4 documents the persistence artefacts, manifest schema (incl. new `train_end` field), helper usage, **holdout-exclusion contract** (replaces the prior "Data-scope inheritance" note), and orchestrator wiring. |
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

Final superseding-PR test counts (after R6 / R7 / R8 additions for Option III):
- 43 pre-existing protocol_runtime tests — all pass
- 17 persistence tests (13 from PR #183 scope + 4 new holdout-exclusion regression tests under R6) — all pass
- 3 A2 / orchestrator e2e tests (2 from PR #183 scope + 1 new orchestrator-threads-train-end regression test) — all pass

```
$ python -m ruff check core/steps/step_4_extraction.py core/steps/classifier_persistence.py \
    core/arc/arc_orchestrator.py tests/protocol_runtime/test_step_4_classifier_persistence.py \
    tests/protocol_runtime/test_a2_end_to_end.py
All checks passed!
```

Full-suite green: TBC (background job in flight at time of writing this log).

---

## Open follow-ups (separate dispatches)

1. **Arc 5 v3.0.1 A2/A6 retry** — separate dispatch. Unblocked once this PR merges. Tests A2/A6 architecture viability against the cleanly-trained classifier; gate evaluation under v3.0 (pre-Amendment-3-engine-PR) gate logic per chat direction — acceptable since this is an architecture-viability retry, not an amended-gate verdict.
2. **Arc 7 v3.0.1 A2/A6 retry** — separate dispatch. Same posture as #1.
3. **Amendment 3 engine implementation PR** — separate, multi-day. Chained DD, per-day max-DD, scaled holdout, sizing-convention check, scaled gate logic, failure-mode taxonomy emission.
4. **Wave 2 dispatches** — blocked until items 1-3 close cleanly.
5. **`requirements-dev.txt` pin pass** — separate dependency-review PR (joblib / sklearn / lightgbm version pins, currently only declared via manifest-recorded versions + loader UserWarning on drift).

## Path R execution log (chat HOLD on PR #183)

PR #183 was merged before the HOLD landed. Path R per chat direction:

1. Reverted `3441d86` on main → new commit `192e01e` ("Revert ... (#183)").
2. Created `engine/step4-classifier-persistence-v2` from the prior worktree tip `7ca0727` (which still held the PR #183 changes locally).
3. Removed unauthorised "Amendment 4 (2026-05-23)" top-of-doc line in L_PROTOCOL.md.
4. Added Option III: `train_end` param on `run_step_4`, orchestrator threads from `WfoStructure.holdout.oos_start`, manifest top-level `train_end` field, 4 new regression tests under R6.
5. Replaced PROTOCOL_RUNTIME §7 "Data-scope inheritance" with "Holdout exclusion" contract.
6. Pushed new branch; opened superseding PR. Original PR #183 = effectively never-shipped.

End of log.
