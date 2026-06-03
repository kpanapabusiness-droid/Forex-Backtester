# heavy_ml_probe PR-E — Log Doc (Gate PR)

> **Branch:** `infra/heavy_ml_probe_pr_e` → `infra/heavy_ml_probe_build` (cut from PR-D locally; chat's "PR-D merged" signal honoured at the head level — origin build branch still at main pre-merge per `git ls-remote`).
> **Dispatch:** chat prompt 2026-05-25 with PR-D flag dispositions + gate-PR scope.
> **Intent doc:** [`docs/dispatches/heavy_ml_probe_build_intent.md`](heavy_ml_probe_build_intent.md).
> **PR-D log:** [`docs/dispatches/heavy_ml_probe_pr_d_log.md`](heavy_ml_probe_pr_d_log.md).
> **Sub-protocol spec:** [`docs/sub_protocols/heavy_ml_probe.md`](../sub_protocols/heavy_ml_probe.md) v1.0 (PR-E added Invocation section + scope clarification).
> **Authoring CC session:** worktree `elegant-ride-ad9a57`.

---

## §1 Determinism gate (dispatch §5 — load-bearing)

**Gate verdict: PASS. No HALT.**

`test_determinism_gate_all_artefacts_byte_identical` in `tests/heavy_ml_probe/test_integration.py` runs the full pipeline twice on identical input + seed and asserts byte-identical sha256 for every single artefact:

| Artefact | sha256 stable? |
|---|---|
| `stub_summary.md` | ✓ |
| `compute_budget_used.md` (PR-E aggregate) | ✓ |
| `automl_leaderboard.csv` | ✓ |
| `automl_feature_importance.csv` | ✓ |
| `meta_label_results.csv` | ✓ |
| `meta_label_classifier_manifest` (sidecar JSON) | ✓ |
| `survival_model_results.csv` | ✓ |
| `survival_classifier_manifest` (sidecar JSON) | ✓ |
| `step_5/heavy_ml_augmented/heavy_ml_manifest.json` | ✓ |
| top-level `step_4/heavy_ml/manifest.json` (stable payload) | ✓ |

10/10 artefacts byte-identical. Top-level manifest's `created_at` is the only field that varies; `stable_payload_sha256` excludes it for the comparison. All sidecar manifests (per-stage classifier manifests + Step 5 manifest) emit without timestamps per the PR-C convention so direct sha256 comparison works.

Real determinism issues found + fixed during PR-E development:

1. **`os.path.relpath` for sibling-directory paths.** First implementation used `Path.relative_to` for the Step 5 manifest's `classifier_manifest_path` entries; that requires the path to be a descendant. Step 4 and Step 5 dirs are siblings under `output_root`, so the relative path needs `..` segments. Switched to `os.path.relpath` which handles up-traversal cleanly.

---

## §2 Adapter contract verification (dispatch §2 + §8 — mandatory)

**Confirmed all three adapter contracts match what `core/architectures/{A2,A4,A6}*.py` expects.** No HALT.

### §2.1 A2 (`build_a2_from_heavy_ml`)

A2's runtime per `core/architectures/a2_classifier_filter.py` consumes `A2Config(classifier=..., threshold=float, classifier_feature_order=tuple[str, ...], ...)` where `classifier.predict_proba(X)` returns `(n_samples, 2)` ndarray. PR-C's FLAML meta-label inner estimator (e.g. `LGBMClassifier`, `XGBClassifier`) IS sklearn-compatible and exposes `predict_proba` directly — drop-in. Test `test_build_a2_from_heavy_ml_returns_a2config` asserts:

- Returns an `A2Config` instance
- `cfg.classifier` exposes `predict_proba`
- `cfg.classifier_feature_order` includes the lineage-gate-accepted columns

### §2.2 A6 (`build_a6_from_heavy_ml`)

Identical contract to A2. A6's `A6Config(classifier, lower_threshold, upper_threshold, classifier_feature_order, ...)` consumes the same predict_proba shape. Test `test_build_a6_from_heavy_ml_returns_a6config` passes.

### §2.3 A4 (`build_a4_from_heavy_ml`) — the load-bearing one

A4's runtime per `core/architectures/a4_pipeline_d_exits.py:_A4ExitPredicate.__call__` consumes a `PathClassifierFit` via `predict_admit(fit, feature_row) → (admit, proba)` from `core/architectures/_path_classifier.py`. `predict_admit` calls `fit.model.predict_proba(X)[0, 1]` where X is `(1, len(fit.feature_order))`.

**PR-E's CoxPHAdapter implements that contract.** The adapter:

- Exposes `hazard_predict(features_dict, bars_survived_at_N) → P` (dispatch §2.2 primary contract)
- Exposes `predict_proba(X) → (n, 2)` that reads `bars_survived` from the last column (named `BARS_SURVIVED_FEATURE = "_bars_survived"`) and delegates to `hazard_predict`
- Returns `feature_order = (*cox_features, BARS_SURVIVED_FEATURE)` so A4's feature-row assembly can splice in the bars-survived column

Test `test_build_a4_predict_admit_handshake` is the **end-to-end contract test** — it loads the emitted manifest, builds the A4 bundle, constructs a feature_row with `BARS_SURVIVED_FEATURE` populated, and asserts `predict_admit(fit, feature_row)` returns `(bool, float in [0, 1])`. PASS.

### §2.4 Hazard math correctness (dispatch §2.2)

`test_coxph_adapter_hazard_predict_hand_computed` traces the dispatch §2.2 formula step-by-step:

```
coefficients   = [0.5, -0.3]
features       = [1.0, 2.0]
linear_pred    = 0.5*1 + (-0.3)*2 = -0.1
relative_risk  = exp(-0.1) ≈ 0.9048
baseline times = [2, 5, 10]
baseline H_0   = [0.10, 0.40, 0.90]
bars_survived  = 2 → H_0(2) = 0.10
k_horizon      = 3 → t=5 → H_0(5) = 0.40
integrated_hz  = (0.40 - 0.10) * 0.9048 = 0.27144
P              = 1 - exp(-0.27144) ≈ 0.2378
```

Adapter returns `0.2378` (matches to 1e-9). PASS.

Plus 6 boundary-case tests: step-function-flat region → P=0; pre-first-event → P=0; past-last-event clamps; NaN propagation on missing feature; predict_proba shape; feature_order includes BARS_SURVIVED_FEATURE.

---

## §3 Wall-clock for the synthetic end-to-end run

Gate test scale (n=500 trades, n_folds=5, max_iter_per_fold=10, 4 features):

```
GATE-test scale (n=500, n_folds=5, max_iter=10): 5.47s wall
  overall_status: all_ok
  automl AUC:        0.9745
  meta-label AUC:    0.9052
  survival c-idx:    0.7880
```

5.47s wall-clock — well under the dispatch §5 target of ~30s. The full PR-E test file (`test_integration.py` — 9 tests) completes in **39s** total because most tests reuse the pipeline_run fixture (which executes once per call).

Full test suite (149 tests = 31 PR-A + 19 PR-B + 30 PR-C + 31 PR-D + 9 integration + 29 adapter) completes in **241s** end-to-end. Lint: clean.

---

## §4 Step 5 augmented scope decision (dispatch §6)

**Narrow scope selected and implemented.** Per dispatch §6's default recommendation: PR-E ships the adapters + emits `step_5/heavy_ml_augmented/heavy_ml_manifest.json` describing what's adapter-buildable. PR-E does **NOT** invoke `core/architectures/{A2,A4,A6}*.py` itself — that's the overseer Step 5 loop's job when it runs against an arc declaring `sub_protocol: heavy_ml_probe`.

Documented in:
- `docs/sub_protocols/heavy_ml_probe.md` — new "Scope clarification (PR-E)" paragraph in §"Output" section
- `core/heavy_ml_probe/adapters.py` module docstring
- `core/heavy_ml_probe/pipeline.py::_emit_step5_manifest` docstring + schema

Wide scope (PR-E runs A2/A4/A6 directly) was NOT pursued — would cross dispatch §3's "do not touch `core/architectures/**`" boundary. If chat wants wide scope, surface in a follow-up PR; the adapters are already in place to make it a small extension.

---

## §5 Scope landed (verbatim against dispatch §8)

| Path | Status | Purpose |
|---|---|---|
| `core/heavy_ml_probe/adapters.py` | new | A2/A4/A6 adapter builders + `CoxPHAdapter` (Cox PH wrapper with `hazard_predict` + sklearn `predict_proba` shim) + `FoldSelectionStrategy` enum + manifest reader + per-stage status taxonomy |
| `core/heavy_ml_probe/pipeline.py` | extended | `_compute_budget_aggregate_markdown` (cross-stage budget); `_emit_step5_manifest` (Step 5 augmentation manifest); `_pipeline_overall_status` (drives exit code 3); `PipelineResult` gains `step5_manifest_path` + `overall_status` |
| `scripts/heavy_ml_probe/run_probe.py` | extended | Exit code 3 partial-success handling; constants `EXIT_OK / EXIT_RUNTIME_ERROR / EXIT_ARGPARSE_ERROR / EXIT_PARTIAL_SUCCESS`; stdout surface adds `overall status` line + step5 manifest path |
| `docs/sub_protocols/heavy_ml_probe.md` | extended | Operator invocation section + adapter manifest format + scope clarification (narrow) |
| `tests/heavy_ml_probe/test_adapters.py` | new | 29 tests — hand-computed Cox PH math, manifest schema validation, partial-success error handling, fold strategies |
| `tests/heavy_ml_probe/test_integration.py` | new | 9 tests — end-to-end, byte-identical determinism gate, adapter handshake, CLI exit codes (0 / 0 / 3 / 1) |

Artefact outputs per dispatch §47:

- All PR-B/C/D artefacts already landing under `step_4/heavy_ml/` — PR-E ensures they're emitted in a single CLI call (the orchestrator already does this; PR-E added the cross-stage `compute_budget_used.md` writer at pipeline end)
- `step_5/heavy_ml_augmented/heavy_ml_manifest.json` — NEW — locked schema version `1.0`
- `compute_budget_used.md` — extended to cover all three stages (was AutoML-only in PR-B; PR-E moved the writer to pipeline end)

PR-E does NOT modify `core/architectures/**`, `L_PROTOCOL.md`, or any arc folders.

---

## §6 Verification

### §6.1 Test suite

```
py -3 -m pytest tests/heavy_ml_probe -q -W ignore::UserWarning
........................................................................ [ 48%]
........................................................................ [ 96%]
.....                                                                    [100%]
149 passed in 241.35s (0:04:01)
```

149 tests: 31 PR-A + 19 PR-B + 30 PR-C + 31 PR-D + **9 integration + 29 adapter** (new in PR-E). All pass first run.

PR-E test breakdown:

| Group | Count | Coverage |
|---|---|---|
| CoxPH adapter math (dispatch §2.2) | 12 | hand-computed; step-function flat; pre-first-event; past-last clamp; NaN on missing feature; predict_proba shape; feature_order; rejects FULL_REFIT; rejects empty payloads; rejects k_horizon ≤ 0; ENSEMBLE_MEAN coefficient averaging; ENSEMBLE_MEAN feature-mismatch rejection |
| `_interp_cumulative_hazard` step semantics | 1 | exhaustive boundary checks at + between + outside observed event times |
| `_coxph_payload_from_pickle` | 2 | minimal dict round-trip; raises on missing key |
| `EnsembleMeanProbaClassifier` | 2 | averages member predictions; empty members raises |
| Manifest emission | 1 | schema version + sub_protocol + stages + adapters block + used_features |
| A2/A6 builders | 2 | returns correct config type + predict_proba + feature_order |
| A4 builder + predict_admit handshake | 2 | returns bundle; predict_admit + BARS_SURVIVED_FEATURE path |
| Fold-selection strategies | 2 | LAST_FOLD vs ENSEMBLE_MEAN both work; FULL_REFIT raises |
| Manifest schema validation | 4 | missing manifest; bad schema_version; wrong sub_protocol; stage != ok |
| Integration end-to-end | 9 | full artefact set; **determinism gate**; adapter handshake; ensemble strategy; CLI exit 0 (all_ok); CLI exit 0 (all_skipped); CLI exit 3 (partial); CLI exit 1 (holdout); diagnostic-message smoke |

### §6.2 Sibling regression

```
py -3 -m pytest tests/discovery -q
..........................................                               [100%]
42 passed in 54.34s
```

42/42 — `signal_discovery_probe` untouched.

### §6.3 Lint

```
py -3 -m ruff check core/heavy_ml_probe scripts/heavy_ml_probe tests/heavy_ml_probe
All checks passed!
```

7 initial unused-import warnings auto-fixed via `--fix`.

### §6.4 CLI exit code semantics

Verified in `test_integration.py`:

| Pool shape | Expected exit | Actual |
|---|---|---|
| Full schema + monkey-patched lineage → all stages succeed | 0 (`EXIT_OK`) | ✓ |
| Full schema + real-registry lineage (rejects all) → all stages skip on `no_clean_features` | 0 (`EXIT_OK` — clean skip, not failure) | ✓ |
| Schema missing `bars_held` + `final_r` → AutoML ok, meta-label + survival skip | 3 (`EXIT_PARTIAL_SUCCESS`) | ✓ |
| Pool entry_time inside holdout window → HoldoutGuardViolation raised | 1 (`EXIT_RUNTIME_ERROR`) | ✓ |

---

## §7 Deviations from dispatch

### §7.1 A2 sources from PR-C's meta-label classifier (not separate AutoML output)

Dispatch §2.1 calls for A2 to wrap "the FLAML classifier(s) from PR-B's AutoML stage." PR-B currently persists only the AutoML leaderboard + importance — **not** the inner classifier objects. PR-C persists the FLAML inner estimators for the meta-label target. Since both are FLAML AutoML outputs (same wrapper, different target), PR-E points A2 at the PR-C artefact set.

This is a small contract gap that came up only at PR-E integration. A future enhancement (PR-F or post-PR-F) could add separate AutoML-target persistence to PR-B; PR-E's adapter would change one path lookup. Surfaced in `build_a2_from_heavy_ml` docstring + log §7.1.

The semantic implication: A2 in heavy_ml_probe is effectively "the meta-label classifier used as a hard filter rather than a sizing signal." If chat wants A2 to consume a true `y` (cluster-membership) classifier, PR-B needs the persistence extension. Documented but not blocking PR-E.

### §7.2 `compute_budget_used.md` writer moved from `_run_automl_stage` to pipeline end

Necessary to cover all 3 stages (was AutoML-only in PR-B). PR-B tests asserted the file's existence after AutoML ran; still passes because the file is written before pipeline returns. Test assertions updated to allow the file to land even when AutoML skips (it now contains skip-reason narrative for every stage).

### §7.3 Exit code 3 only on `partial` overall status

Dispatch §4 says exit 3 when "at least one stage succeeded, at least one stage failed cleanly." Implemented as: `partial = mix of ok + skipped`. `all_skipped` → exit 0 (per dispatch §4 wording "all three cleanly skipped"). Tests cover both branches.

### §7.4 `_emit_partial_manifest` does NOT emit Step 5 manifest

When an exception (HoldoutGuardViolation, AllFeaturesRejected) is being re-raised, the partial-manifest path writes the Step 4 stub_summary + Step 4 manifest but NOT the Step 5 manifest. Rationale: an exception path means the pipeline didn't complete cleanly; adapter consumers reading a missing Step 5 manifest get a clear "manifest not found" error from `HeavyMLManifestError`. Could be revisited if chat prefers a stub Step 5 manifest with all stages marked failed.

---

## §8 Flags for chat

### §8.1 Non-blocking

- **A2 sources from PR-C's meta-label classifier** (§7.1 above). If chat wants A2 to consume a vanilla-target classifier, PR-B needs a persistence extension. Documented in `build_a2_from_heavy_ml` docstring.
- **`build_a4_from_heavy_ml` returns a tuple `(A4Config, CoxPHAdapter)`** rather than mutating A4Config. The dispatch §2.2 says "A4Config object exposing a callable hazard_predict" — ambiguous between "A4Config with new attribute" (would require touching `core/architectures/A4.py`) and "bundle exposing both." Chose the bundle because it respects dispatch §3's "do not touch core/architectures/**." If chat wants A4Config to directly carry `hazard_predict`, a one-line subclass would do it; flag for PR-F or post-merge.
- **Test wall-clock at 241s** (was 138s after PR-D). 100s growth this PR — 9 integration tests + 29 adapter tests with hand-computed Cox PH on synthetic data. Acceptable; integration tests are the load-bearing determinism gate so reducing coverage there isn't a fair trade.
- **`FULL_REFIT` fold strategy is `NotImplementedError`.** Dispatch §3 marks it as "not implemented in PR-E; raise NotImplementedError. Future enhancement." Implemented as such; tested.

### §8.2 Surfaced for PR-F intent

- **README / operator notes** in `docs/sub_protocols/heavy_ml_probe.md` got an Invocation section in PR-E — PR-F can extend with full chat-side analysis examples (sweeping `k_horizon`, comparing LAST_FOLD vs ENSEMBLE_MEAN, etc.).
- **Lineage gate doc polish** — the existing `causal_lineage.py` docstrings are stable but a top-level reader on what "lineage gate is binding" means in practice could land in PR-F.
- **Minimum-N spec doc paragraph** — PR-D added the warning machinery; PR-F can promote it to a dedicated subsection in `heavy_ml_probe.md`.

### §8.3 No HALT

Determinism gate passes (§1). All adapter contracts verified (§2). No `core/architectures/**` modifications (dispatch §3 boundary respected). WORKFLOW §6 unused.

---

## §9 Branch state at PR-time

- Build branch (`infra/heavy_ml_probe_build`) on origin still at `7c238e8` (main's tip at PR-A time). PR-A/B/C/D's GitHub merges have not happened yet at PR-E-open time — chat's "merged" signals interpreted at the head level (each PR's branch contains all upstream).
- PR-E branch (`infra/heavy_ml_probe_pr_e`) cut from `infra/heavy_ml_probe_pr_d` locally so the head includes A+B+C+D+E commits. When chat merges A/B/C/D → build on GitHub, the PR-E diff will recompute and show only PR-E's changes.

---

## §10 What lands next (per dispatch §9)

After PR-E merges into `infra/heavy_ml_probe_build`:

- **PR-F (Documentation + polish — final PR)** — README extensions, inline doc cleanup surfaced during build, possible spec-doc polish (minimum-N subsection, lineage-gate-explanation), final readiness check. ≤ 0.5 day.
- **Build-branch merge to main** — after PR-F lands.

Per chat directive: I will not start PR-F work during the PR-E review window.

---

End of PR-E log.
