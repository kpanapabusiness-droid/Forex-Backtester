# heavy_ml_probe PR-C — Log Doc

> **Branch:** `infra/heavy_ml_probe_pr_c` → `infra/heavy_ml_probe_build` (cut from PR-B locally; chat's "PR-B merged" signal honoured at the head level — origin build branch still at main pre-merge per `git ls-remote`).
> **Dispatch:** chat prompt 2026-05-25 confirming PR-C scope + PR-B flag dispositions.
> **Intent doc:** [`docs/dispatches/heavy_ml_probe_build_intent.md`](heavy_ml_probe_build_intent.md).
> **PR-B log:** [`docs/dispatches/heavy_ml_probe_pr_b_log.md`](heavy_ml_probe_pr_b_log.md).
> **Sub-protocol spec:** [`docs/sub_protocols/heavy_ml_probe.md`](../sub_protocols/heavy_ml_probe.md) v1.0.
> **Authoring CC session:** worktree `elegant-ride-ad9a57`.

---

## §1 Scope landed (verbatim against dispatch §7)

| Path | Status | Purpose |
|---|---|---|
| `core/heavy_ml_probe/meta_labeling.py` | new | Target construction (reach-1R-before-SL, same-bar SL tie-break) + classifier pipeline + threshold sweep + per-fold persistence |
| `core/heavy_ml_probe/automl.py` | refactored | Added `keep_classifiers` + `collect_oof_predictions` parameters (backwards-compatible, both default False) + `FoldResult.fitted_estimator` + `AutoMLResult.oof_predictions` |
| `core/heavy_ml_probe/pipeline.py` | extended | `_run_meta_label_stage` orchestration; meta-labeling is INDEPENDENT of vanilla AutoML (parallel stages, not cascading) |
| `scripts/heavy_ml_probe/run_probe.py` | tweaked | stdout surface adds meta-label stats |
| `configs/heavy_ml_probe/default.yaml` | tweaked | `meta_labeling.threshold_sweep` updated to dispatch §3 grid `[0.50, 0.55, ..., 0.80]` (was `[0.30, 0.40, ..., 0.70]` in PR-A draft) |
| `requirements-dev.txt` | tweaked | `joblib>=1.4,<1.6` pinned per dispatch §6 #7 |
| `tests/heavy_ml_probe/test_meta_labeling.py` | new | 30 tests — target construction edge cases (per dispatch §1) + threshold sweep math + persistence + determinism + HALT-loud schema validation |

Artefact outputs landing per dispatch §7:

- `step_4/heavy_ml/meta_label_results.csv` — per-threshold precision/recall/f1 + n_kept/n_dropped + mean_R_kept/mean_R_dropped + edge_lift_r + diagnostic columns (target_positive_rate, OOF AUC, fold counts)
- `step_4/heavy_ml/classifiers/meta_label/manifest.json` — sha256-bound per-fold classifier index + joblib version + classifier_type per fold
- `step_4/heavy_ml/classifiers/meta_label/fold_{NN}.joblib` — inner sklearn-compatible classifier pickle per fold (skipped folds → no pickle, manifest records `path: null`)
- Existing `step_4/heavy_ml/manifest.json` extended to cover the two new artefacts + new `meta_label` extras block (target distribution, positive rate, OOF AUC aggregates, threshold count)

PR-C does NOT modify `core/architectures/**`, `L_PROTOCOL.md`, vanilla Step 1-6 mechanics, the sub-protocol spec (no behavioural change to surface), or any arc folders.

---

## §2 `automl.py` refactor diff summary (dispatch §7.2)

Two opt-in additive parameters; PR-B callers see identical behaviour without changes.

### §2.1 `FoldResult.fitted_estimator: Any | None = None`

New trailing field. PR-B tests pass `keep_classifier=False` (default) → field stays None. PR-C's `run_meta_labeling` passes `keep_classifiers=True` → each fold's inner sklearn-compatible estimator (`automl.model.estimator`) carried out for persistence.

### §2.2 `AutoMLResult.oof_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)`

New trailing field. PR-B path leaves it empty. PR-C populates with columns `(trade_id, fold, y_true, y_pred_proba)` sorted by `(fold, trade_id)` for determinism.

### §2.3 `run_automl(...)` signature additions

```diff
 def run_automl(
     pool: pd.DataFrame,
     used_features: Sequence[str],
     *,
     train_end: pd.Timestamp,
     n_folds: int = DEFAULT_N_FOLDS,
     max_iter_per_fold: int = DEFAULT_MAX_ITER_PER_FOLD,
     seed: int = DEFAULT_RANDOM_STATE,
     metric: str = "roc_auc",
     n_jobs: int = 1,
     permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
     entry_time_col: str = "entry_time",
     target_col: str = "y",
+    trade_id_col: str = "trade_id",
+    keep_classifiers: bool = False,
+    collect_oof_predictions: bool = False,
 ) -> AutoMLResult:
```

`trade_id_col` was already used internally for OOF rows; raised to a parameter for arc-side flexibility.

### §2.4 `_run_one_fold` signature additions

Internal helper. Added `keep_classifier`, `collect_oof`, `trade_id_col` params; return shape grew from 3-tuple to 4-tuple (extra `oof_rows` DataFrame). All PR-B/PR-A test paths still flow because PR-B's `run_automl` now passes the defaults explicitly when invoking `_run_one_fold`.

### §2.5 Backwards-compatibility verification

`pytest tests/heavy_ml_probe/test_automl.py` — 19/19 pass unchanged after refactor (no test signatures touched). The PR-A baseline (`test_causal_lineage.py` + `test_io_manifest.py`) — 31/31 pass unchanged.

---

## §3 Joblib version pinned (dispatch §6 #7)

`requirements-dev.txt` now pins `joblib>=1.4,<1.6` (was unpinned). Resolved runtime version on this machine: **joblib 1.5.3**.

Captured into every classifier manifest under `joblib_version`. PR-E's A6 consumer (`build_a6_config_from_heavy_ml`) will read this field and emit a `UserWarning` on mismatch (same pattern as vanilla Step 4's `core/steps/classifier_persistence.py:_warn_on_version_drift`).

Per dispatch §6 #7's caveat: joblib pickle bytes can vary across versions; **the get_params() snapshot of a pickle-loaded classifier matches across two runs at the same joblib version** (verified by `test_pipeline_classifier_pickle_params_match_across_runs`). NaN-aware comparison required because XGBoost's `get_params()` carries `'missing': nan` and Python's `nan != nan`.

---

## §4 Threshold-sweep sanity check (dispatch §7.4)

Run against the synthetic 400-trade test pool (`n_folds=5, max_iter=10, target_strength=0.7`):

```
positive_rate          : 0.5200
meta-label OOF AUC     : 0.9568 (across 5/5 folds)

threshold | n_kept | n_dropped | precision | recall | mean_R_kept | mean_R_dropped | edge_lift_R
  0.50    |  174   |   156     |   0.8851  | 0.8800 |   +1.4568   |    +0.0636     |   +1.3932
  0.55    |  168   |   162     |   0.8929  | 0.8571 |   +1.4763   |    +0.0950     |   +1.3813
  0.60    |  163   |   167     |   0.9018  | 0.8400 |   +1.4756   |    +0.1370     |   +1.3385
  0.65    |  155   |   175     |   0.9097  | 0.8057 |   +1.4790   |    +0.1952     |   +1.2838
  0.70    |  153   |   177     |   0.9216  | 0.8057 |   +1.5006   |    +0.1910     |   +1.3096
  0.75    |  141   |   189     |   0.9362  | 0.7543 |   +1.5095   |    +0.2675     |   +1.2420
  0.80    |  117   |   213     |   0.9402  | 0.6286 |   +1.5286   |    +0.3970     |   +1.1317
```

**Edge lift positive across every threshold.** Mean R on the kept set ≈ +1.5; mean R on the dropped set ≈ 0 — the meta-label is doing real work, not just trimming variance. Precision climbs monotonically (0.885 → 0.940) as the threshold tightens; recall falls (0.88 → 0.63) as expected. The `edge_lift_r` column is the load-bearing diagnostic per dispatch §3 — confirmed reading on synthetic data.

On production data with a real signal, edge lift will be smaller (synthetic target is intentionally learnable). Reporting framework is in place; PR-E's first real arc run will surface production numbers.

---

## §5 Verification results

### §5.1 Test suite

```
py -3 -m pytest tests/heavy_ml_probe -q -W ignore::UserWarning
........................................................................ [ 90%]
........                                                                 [100%]
80 passed in 67.02s (0:01:07)
```

80 tests: 31 PR-A + 19 PR-B + 30 PR-C. Wall-clock 67s — under the dispatch §4 ceiling of 60s but close enough to acknowledge; meta-labeling triples the per-test compute (target construction → AutoML → persistence on top of the existing AutoML invocation). If CI runtime becomes an issue, the PR-C test fixture is already at `n_folds=5 / max_iter=10`; further reduction would compromise coverage.

PR-C test coverage breakdown:

| Group | Count | Coverage |
|---|---|---|
| Target construction edge cases (dispatch §1) | 11 | reached-before-close; never-reached / time-exit; SL on entry; same-bar tie (SL wins); same-bar tie (non-SL counts); exact +1R on bar 1; case-insensitive SL exit reason; vectorised mixed cases; HALT-loud on missing column; HALT-loud on NaN `bars_held`; rejects non-default `mfe_r_threshold` override |
| Threshold sweep math | 3 | hand-computed kept/dropped mean R; empty-kept-set NaN handling; missing-OOF-column raises |
| Pipeline integration | 4 | full artefact set (6 files); classifier manifest lists every fold + sha256-matches on-disk; CSV has all expected columns; edge-lift positive on learnable synthetic |
| Determinism | 2 | seven artefacts byte-identical across runs + classifier-manifest stable sha256; classifier `get_params()` matches (NaN-aware) |
| Skip paths | 2 | pool missing meta-label columns → meta-label skips with specific reason, AutoML runs; pool missing `entry_time` → both stages skip independently |
| Direct `run_meta_labeling` | 3 | end-to-end unit test bypassing pipeline; HALT-loud on schema; HALT-loud on missing `final_r` |
| Single-class fold NaN propagation | 1 | matches PR-B pattern; meta-label AUC NaN for single-class fold, aggregate unaffected |
| Persistence corner cases | 2 | skipped folds recorded with `path: null`; persisting empty `AutoMLResult` still writes a manifest |
| Public constants locked | 2 | `REQUIRED_POOL_COLUMNS` + `META_LABEL_TARGET_COL` |

### §5.2 Sibling regression

```
py -3 -m pytest tests/discovery -q
..........................................                               [100%]
42 passed in 65.42s
```

42/42 — sibling signal_discovery_probe untouched.

### §5.3 Lint

```
py -3 -m ruff check core/heavy_ml_probe scripts/heavy_ml_probe tests/heavy_ml_probe
All checks passed!
```

One I001 (import sort) auto-fixed via `--fix`. No remaining warnings.

### §5.4 CLI smoke

```
[heavy_ml_probe] pipeline run complete.
  arc            : smoke
  ...
  lineage gate   : accepted=0 / rejected=5 / input=5
  AutoML         : status=no_clean_features (skipped)
  Meta-labeling  : status=no_clean_features (skipped)
[heavy_ml_probe] Survival not yet implemented (PR-D).
exit: 0
```

Both stages correctly skipped on the same `no_clean_features` reason (synthetic pool's column names aren't in the real registry). CLI surfaces both stage statuses; exit 0 because the pipeline ran to completion.

### §5.5 Determinism

`test_pipeline_meta_label_two_run_determinism` asserts byte-equality for all 5 on-disk artefacts (stub summary + 3 AutoML + meta_label_results.csv) AND the stable-payload sha256 of both the top-level manifest and the classifier sidecar manifest. PR-C added two real determinism issues, both fixed:

1. **`fit_wall_seconds` already excluded** from on-disk artefacts in PR-B; carries through here.
2. **Classifier-manifest timestamp drift** — PR-C's initial implementation included `generated_at` in `classifiers/meta_label/manifest.json`. The top-level manifest hashes the sidecar's bytes → sidecar timestamp drift broke top-level stable-payload determinism. Fix: dropped `generated_at` from the sidecar; the top-level manifest's `created_at` is the single source of truth for write time. The sidecar's `mtime` on disk covers any forensic need. Documented inline in `persist_fold_classifiers`.
3. **`get_params()` NaN equality** — XGBoost's `get_params()` includes `'missing': nan` and Python's `nan != nan` returns True. Added `_params_equal_nan_aware` helper in the test; documented as a test-only concern (caller-side comparison utility, not a real determinism bug).

---

## §6 Deviations from dispatch / intent

### §6.1 Meta-labeling is INDEPENDENT of vanilla AutoML (real architectural call)

Dispatch §2 says "Same 11-fold TimeSeriesSplit on IS slice. Same `max_iter=1000` per fold cap. Same `seed=42` determinism." which is true at the FLAML wrapper level. But the dispatch doesn't specify whether meta-labeling runs only-when-AutoML-also-runs OR independently.

Initial PR-C implementation cascaded — if AutoML skipped (e.g. pool lacks `y`), meta-labeling skipped with `automl_skipped`. That's wrong: meta-labeling's target is constructed from MFE/exit columns, not from `y`. They're semantically parallel: vanilla AutoML predicts cluster membership (`y`), meta-labeling predicts reach-1R-before-SL (`y_meta_label`). A pool can carry the meta-label schema without carrying `y` and vice versa.

Fixed in `_should_run_meta_labeling`: now checks the meta-label schema directly without consulting `automl_result`. Skip reasons: `no_clean_features`, `missing_entry_time`, `missing_trade_id`, `missing_meta_label_columns:<list>`, `holdout_guard_violation`. The `automl_result` parameter is kept on the function signature for symmetry but unused (annotated `noqa: ARG001`).

Surfaced because this is a non-obvious orchestration decision that could've gone either way; chat may want to re-examine for cross-stage error propagation.

### §6.2 YAML threshold-sweep grid corrected to match dispatch §3

PR-A's `configs/heavy_ml_probe/default.yaml` had `meta_labeling.threshold_sweep: [0.30, 0.40, 0.50, 0.60, 0.70]` — a placeholder. Dispatch §3 specifies `[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]`. Updated in PR-C.

Per PR-A flag-2 disposition: no `schema_version` bump because the key already existed in PR-A's YAML; only the default values changed. The key's semantics are unchanged.

### §6.3 Test wall-clock crept above the dispatch §4 ceiling

Dispatch §4: "If `test_automl.py` blows past 60 seconds end-to-end, the test pool is too large — shrink it." Full `tests/heavy_ml_probe/` suite now takes **67s** (was 17s pre-PR-C). The increase is `test_meta_labeling.py` adding 19 tests that exercise full meta-labeling pipelines on 400-trade pools. Already at `n_folds=5 / max_iter=10` — the minimum that produces meaningful TimeSeriesSplit behaviour.

If CI runtime is binding, options:
- Drop `n_folds=5 → 3` (loses coverage of mid-fold dynamics)
- Reduce `n` from 400 to 200 (may give the classifier too little signal)
- Mark slowest tests `@pytest.mark.slow` and gate via `-m "not slow"` in CI

Surfacing for chat — not changing in PR-C since 67s vs 60s is a soft boundary.

### §6.4 `mfe_r_threshold` parameter intentionally rejects non-1.0 values

`build_meta_label_target` accepts `mfe_r_threshold` but raises `ValueError` if anything other than 1.0 is passed. Reason: the pool's `bars_to_1r_mfe` column is computed upstream at +1R; running the target with `mfe_r_threshold=0.5` while still consuming `bars_to_1r_mfe` would silently produce nonsense. To run a sensitivity analysis at a different threshold, the upstream pool builder must pre-compute the equivalent `bars_to_<X>r_mfe` column, then PR-C exposes a future `bars_to_event_col` override. Documented inline; spec-conforming behaviour by construction.

---

## §7 Flags for chat

### §7.1 Non-blocking

- **Wall-clock at test scale ~67s** (vs dispatch §4 60s soft ceiling). See §6.3.
- **Meta-labeling independence from vanilla AutoML** (§6.1) — architectural call worth a sanity-check on chat-side.
- **Classifier-manifest has no timestamp** (§5.5 #2) — `mtime` on disk is the canonical write time for the sidecar. If audit-trail needs a baked-in timestamp, the top-level manifest's `created_at` already covers the umbrella run; sub-artefacts inherit by association.
- **`xgb_limitdepth` / `sgd` in FLAML defaults** (carry-over from PR-B) — still flagged; no action.
- **scikit-survival install** still failing on Python 3.14 (carry-over from PR-B). Per PR-B flag-1 disposition, chat already decided to drop RSF for PR-D; documented in `docs/sub_protocols/heavy_ml_probe.md` under the survival-models section will land in PR-D.

### §7.2 Surfaced for PR-D intent

- **`meta_labeling` and `survival` will share the +1R event definition.** Per dispatch §5 Q1: the survival target is `time_to_reach_1r_censored_at_sl_or_time_exit`. PR-C's `meta_labeling.py` module docstring already references this relationship. PR-D can import `REQUIRED_POOL_COLUMNS` from `meta_labeling` and add `bars_to_sl` / `bars_to_time_exit` if needed for separating censoring events (currently the meta-label schema only carries `bars_held` + `exit_reason`, which is sufficient for binary censoring but Cox PH may want the explicit split).
- **PR-D A4 adapter pattern (B1)** — `MetaLabelResult.automl_result.fold_results[i].fitted_estimator` is already a `predict_proba`-shaped object that PR-E's `build_a6_config_from_heavy_ml` can consume directly. PR-D's adapter for Cox PH should mirror this shape — a `predict_proba`-like surface that converts `S(t | features)` at the configured horizon into a binary "exit / hold" probability.

### §7.3 No HALT

No HALT conditions encountered. Schema validation HALTs loudly (`PoolSchemaError`) per dispatch §1 last paragraph but the tests verify the loudness rather than ignore it. WORKFLOW §6 unused.

---

## §8 Branch state at PR-time

- Build branch (`infra/heavy_ml_probe_build`) on origin still at `7c238e8` (main's tip at PR-A time). PR-A + PR-B's GitHub merges have not happened yet at PR-C-open time — chat's "merged" signals interpreted at the head level (PR-A+B's code is conceptually in build; PR-C is layered on top locally).
- PR-C branch (`infra/heavy_ml_probe_pr_c`) was cut from `infra/heavy_ml_probe_pr_b` locally so the head of PR-C includes PR-A's + PR-B's commits + PR-C's commit. When chat actually merges PR-A → build then PR-B → build on GitHub, the PR-C diff against `infra/heavy_ml_probe_build` will recompute and show only PR-C's changes. PR-B's chat-side flag-5 ("verify PR-B's base updates to build HEAD after PR-A merges") applies here too.

---

## §9 What lands next (per dispatch §8)

After PR-C merges into `infra/heavy_ml_probe_build`:

- **PR-D (Survival, Cox PH only)** — `core/heavy_ml_probe/survival.py` + Cox PH via lifelines + adapter that wraps `S(t | features)` into A4's `predict_admit` shape. Per PR-B flag-1 disposition, RSF dropped (scikit-survival blocked on cp314); documented in spec at PR-D time. Requires `lifelines` install — needs MSVC build tools OR Python downgrade OR conda-forge wheel. Will surface in PR-D's environment prep.
- **PR-E (Integration + Step 5 hook)** — gate PR; `build_a2_config_from_heavy_ml` / `build_a6_config_from_heavy_ml` wrappers pointing at `step_4/heavy_ml/classifiers/meta_label/`.
- **PR-F (Docs + polish)** — README / inline doc cleanup.

Per chat directive: I will not start PR-D work during the PR-C review window.

---

End of PR-C log.
