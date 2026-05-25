# heavy_ml_probe PR-A — Log Doc

> **Branch:** `infra/heavy_ml_probe_pr_a` → `infra/heavy_ml_probe_build` (cut from `main` @ `7c238e8`).
> **Dispatch:** chat reply confirming PR-A scope + Q1/Q2/Q5b resolutions (2026-05-24).
> **Intent doc:** [`docs/dispatches/heavy_ml_probe_build_intent.md`](heavy_ml_probe_build_intent.md).
> **Sub-protocol spec:** [`docs/sub_protocols/heavy_ml_probe.md`](../sub_protocols/heavy_ml_probe.md) v1.0.
> **Authoring CC session:** worktree `elegant-ride-ad9a57`.

---

## §1 Scope landed (verbatim against dispatch §8 PR-A)

| Path | Purpose | Lines |
|---|---|---|
| `requirements-dev.txt` | Append heavy-ML deps (flaml, lifelines, scikit-survival, xgboost, catboost, lightgbm) | +12 |
| `configs/heavy_ml_probe/default.yaml` | Locked invocation parameters; PR-B/C/D YAML schema frozen from this PR | new |
| `core/heavy_ml_probe/__init__.py` | Package marker + version | new |
| `core/heavy_ml_probe/io.py` | Deterministic CSV / parquet / text + sha256 manifest writer | new |
| `core/heavy_ml_probe/metrics.py` | AUC + concordance + IBS wrappers (lazy lifelines / sksurv imports) | new |
| `core/heavy_ml_probe/causal_lineage.py` | Pre-evaluation lineage gate (Q3-resolved schema: `FeatureSpec`-backed) | new |
| `core/heavy_ml_probe/pipeline.py` | Orchestration skeleton: config → pool → lineage gate → stub manifest | new |
| `scripts/heavy_ml_probe/__init__.py` | Empty package marker for CLI parity with sibling | new |
| `scripts/heavy_ml_probe/run_probe.py` | CLI: `--arc --pool --cluster-id --config --output-root` | new |
| `tests/heavy_ml_probe/__init__.py` | Package marker | new |
| `tests/heavy_ml_probe/test_causal_lineage.py` | 12 tests — gate behaviour + summary determinism + case-insensitivity + purity | new |
| `tests/heavy_ml_probe/test_io_manifest.py` | 19 tests — IO determinism + manifest schema + two-run end-to-end | new |

End-to-end stub semantics per dispatch: CLI runs, reads pool parquet, applies lineage gate via the registry-backed `feature_lineage_dataframe()`, writes `step_4/heavy_ml/stub_summary.md` + `step_4/heavy_ml/manifest.json` with sha256 of the summary. AutoML / meta-label / survival code paths are explicitly not implemented (PR-B/C/D).

---

## §2 Verification results

### §2.1 Test suite

```
py -3 -m pytest tests/heavy_ml_probe -x -q
...............................                                          [100%]
31 passed in 0.64s
```

All 31 tests pass on the first run. Coverage breakdown:

- `test_causal_lineage.py` — 12 tests: accepts clean; rejects suspect / unverified / unknown / excluded-class; mixed pool accounting; dedup; purity (no input mutation); summary determinism; case-insensitive lineage values; raises on missing required columns.
- `test_io_manifest.py` — 19 tests: write_text appends single newline; LF (no CR) on Windows; CSV deterministic sort; CSV column-mismatch raises; parquet round-trip; sha256 matches hand-computed digest; manifest schema includes `schema_version` + `sub_protocol`; nested forward-slash paths; artefact sort determinism; extras collision raises; extras recorded; end-to-end stub writes both artefacts; lineage gate filters suspect; two-run byte-identical artefacts; two-run stable-payload manifest sha256; FileNotFoundError on missing pool; ValueError on bad sub_protocol name; ValueError on missing top-level config sections.

### §2.2 Sibling regression

```
py -3 -m pytest tests/discovery -q
..........................................                               [100%]
42 passed in 49.91s
```

42/42 — sibling sub-protocol untouched.

### §2.3 Lint

```
py -3 -m ruff check core/heavy_ml_probe scripts/heavy_ml_probe tests/heavy_ml_probe
All checks passed!
```

One ruff warning surfaced during initial write (`dataclasses.field` unused import in `causal_lineage.py`) and was fixed before commit. Final pass clean.

### §2.4 CLI smoke

End-to-end smoke against the default config + a 10-row synthetic pool:

```
lineage gate: 2 / 2
manifest written: True
stub summary written: True
```

The CLI surface (`scripts/heavy_ml_probe/run_probe.py`) exits 0 on a clean run and exits 1 with a clear stderr message on `FileNotFoundError` / `ValueError`. Both branches exercised by the test suite.

### §2.5 Determinism (dispatch §6 #7)

Two-run sha256 equality verified two ways in `test_io_manifest.py`:

1. **Artefact bytes** — `stub_summary.md` is timestamp-free, so its sha256 matches byte-identically across two pipeline runs against the same input pool.
2. **Stable-payload manifest** — `pipeline.stable_payload_sha256` zeroes out the `created_at` field and hashes the canonicalised JSON. The remaining payload (schema version, arc, cluster, pool path, pool size, lineage gate counts, artefacts block) matches across runs.

The manifest's `created_at` timestamp intentionally differs across runs (UTC ISO of write time). This matches the sibling discovery convention; the determinism test harness excludes it explicitly.

### §2.6 sksurv Windows wheel (intent doc PR-A risk #1)

`py -3 -m pip install --dry-run flaml lifelines scikit-survival xgboost catboost lightgbm` resolved cleanly on this Windows machine (Python 3.14, x64):

```
scikit_survival-0.27.0-cp314-cp314-win_amd64.whl (822 kB)
```

No Cython-from-source fallback required. The intent-doc-flagged risk did not materialise on this environment. Other contributors on different Python versions may still hit wheel-availability gaps — surface in PR-B if so.

`lightgbm` was already installed (4.6.0) from the vanilla Step 4 path; listing it explicitly in `requirements-dev.txt` makes the dependency contract explicit (vanilla Step 4 currently imports it defensively, but heavy_ml_probe needs it as a first-class learner in the FLAML search space).

---

## §3 Deviations from intent doc

None of substance. Two micro-additions explicitly flagged in the intent doc landed as planned:

- `tests/heavy_ml_probe/test_io_manifest.py` (added per intent §3 "Two minor additions").
- `scripts/heavy_ml_probe/__init__.py` (added per intent §3).

Intent doc §11 listed both as flag-only; both included in PR-A as documented.

One internal refactor not flagged in the intent doc but worth recording:

- `core/heavy_ml_probe/pipeline.py::_build_lineage_dataframe()` is a thin shim around `core.features.pipeline.feature_lineage_dataframe()`. The indirection exists so tests can pass a hand-built lineage DataFrame via `apply_lineage_gate(..., lineage_df=...)` without importing the full v3 features stack (which triggers panel + cache imports — slow in CI). Same pattern that the sibling tests use; documented inline.

---

## §4 Flags for chat

### §4.1 Non-blocking

- **`scripts/heavy_ml_probe/run_probe.py` exit codes diverge slightly from intent doc.** The intent doc didn't specify exit codes. PR-A uses `0` for success, `1` for config / pool errors (FileNotFoundError / ValueError), and lets unhandled exceptions propagate to stderr with exit code 2 (the default `sys.exit(1)` from argparse error path is reserved for arg-parse failures). Chat may want a stricter spec when CI starts consuming exit codes; for PR-A this is documented in the module docstring.

- **YAML schema is locked from PR-A forward.** PR-B/C/D add behaviour but should NOT add new top-level YAML sections without bumping `schema_version` in `configs/heavy_ml_probe/default.yaml`. The pipeline asserts the `schema_version` + `sub_protocol.name` fields on load. If chat wants per-section schema validation (e.g. via pydantic), flag at PR-B intent — for PR-A the light required-keys check is the only enforcement.

- **`pipeline.stable_payload_sha256()` exposes a determinism-test helper as part of the public module surface.** It's narrow-purpose. Could move to `tests/heavy_ml_probe/_helpers.py` later if it stays unused outside tests. Not blocking; keeping it on `pipeline.py` matches the sibling's `core/discovery/io.py::write_manifest` location convention.

### §4.2 Surfaced for PR-B intent

- **FLAML's `max_iter` semantics on this Python 3.14 / FLAML 2.6.0 install** to be confirmed at PR-B's intent doc with a quick smoke (10-iter run, verify the budget counter increments by exactly 10). Q2 resolution accepted the 1-trial-per-eval reading at face value; PR-B's empirical check protects against API drift in the FLAML 2.6.0 line.

- **`auc_roc` returns NaN on single-class folds** (a `roc_auc_score` would raise). This matches vanilla Step 4's per-fold AUC handling (`core/steps/step_4_extraction.py:182-184`). PR-B's FLAML wrapper should propagate the NaN — not coerce to 0.5 — because the FLAML leaderboard's metric column must distinguish "model could not be scored" from "model performed at chance."

### §4.3 None

No HALT conditions encountered during PR-A. WORKFLOW §6 unused.

---

## §5 Branch state at PR-time

```
git status
On branch infra/heavy_ml_probe_pr_a
Your branch is ahead of 'origin/main' by 1 commit  (parent: 7c238e8)
nothing to commit, working tree clean
```

(State as of the commit + push that opens this PR. Files-touched list in §1 above is the authoritative diff.)

PR-A target: `infra/heavy_ml_probe_build`. The build branch was published from `main` (`7c238e8`) in the same chat turn as PR-A development; it has no commits of its own — it exists only as the merge target for the per-PR series A → F.

---

## §6 What lands next (per dispatch §8)

After PR-A merges into `infra/heavy_ml_probe_build`:

- **PR-B (AutoML)** — `core/heavy_ml_probe/automl.py` + 11-fold TS-split + `max_iter=1000` enforcement + leaderboard + permutation importance + `test_automl.py`. Largest single PR (1-2 days). Intent doc lands first per WORKFLOW §2.
- **PR-C (Meta-labeling)** — `core/heavy_ml_probe/meta_labeling.py` reusing the AutoML path with the reach-1R-before-SL target. ≤ 1 day.
- **PR-D (Survival)** — Cox PH + RSF + the Q5b-resolved adapter pattern wrapping survival predictions into A4's `predict_admit` shape. ≤ 1 day.
- **PR-E (Integration + Step 5 hook)** — gate PR. End-to-end determinism on synthetic data + `compute_budget_used.md` writer + `build_a2_config_from_heavy_ml` / `build_a6_config_from_heavy_ml` wrappers pointing at `step_4/heavy_ml/classifiers/`. ≤ 1 day.
- **PR-F (Docs + polish)** — README / inline doc cleanup. ≤ 0.5 day.

Per chat directive: I will not start PR-B work during the PR-A review window.

---

End of PR-A log.
