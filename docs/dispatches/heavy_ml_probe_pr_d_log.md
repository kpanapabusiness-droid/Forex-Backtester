# heavy_ml_probe PR-D — Log Doc

> **Branch:** `infra/heavy_ml_probe_pr_d` → `infra/heavy_ml_probe_build` (cut from PR-C locally; chat's "PR-C merged" signal honoured at the head level — origin build branch still at main pre-merge per `git ls-remote`).
> **Dispatch:** chat prompt 2026-05-25 with scope adjustments dropping RSF (no cp314 wheel) AND lifelines (also no cp314 wheel); swap to `statsmodels.duration.hazard_regression.PHReg` for Cox PH.
> **Intent doc:** [`docs/dispatches/heavy_ml_probe_build_intent.md`](heavy_ml_probe_build_intent.md).
> **PR-C log:** [`docs/dispatches/heavy_ml_probe_pr_c_log.md`](heavy_ml_probe_pr_c_log.md).
> **Sub-protocol spec:** [`docs/sub_protocols/heavy_ml_probe.md`](../sub_protocols/heavy_ml_probe.md) v1.0 (updated PR-D for RSF deferral + library swap).
> **Authoring CC session:** worktree `elegant-ride-ad9a57`.

---

## §1 statsmodels install verification (mandatory dispatch §1)

**Verified clean on Python 3.14 / Windows x64. No HALT.**

### §1.1 Wheel availability

```
py -3 -m pip install --dry-run statsmodels
...
Downloading statsmodels-0.14.6-cp314-cp314-win_amd64.whl (9.6 MB)
Downloading patsy-1.0.2-py2.py3-none-any.whl (233 kB)
Would install patsy-1.0.2 statsmodels-0.14.6
```

`statsmodels-0.14.6` has a clean `cp314-win_amd64` wheel. Only new transitive dep is `patsy` (pure-Python). No `ecos` involvement (that's what blocks lifelines + scikit-survival). No MSVC build tools required.

### §1.2 Import + fit smoke

```
statsmodels: 0.14.6
PHReg module: statsmodels.duration.hazard_regression
params: [0.51620833 0.05383552 0.1833429 ]
pvalues: [4.96708726e-04 6.66841077e-01 1.74842410e-01]
bse: [0.14822817 0.12505711 0.13512817]
```

True coefficient on first feature was 1.0; recovered 0.52 on a tiny (n=100) noisy sample — directionally correct.

### §1.3 Pickle round-trip

```
reload params match: True
```

`joblib.dump → joblib.load` round-trip preserves `params` exactly. Per dispatch §10 last paragraph's "if persisted PHRegResults object can't be reloaded across processes, HALT" check — passes.

### §1.4 baseline_cumulative_hazard format (deviation note)

Dispatch §8 docstring described it as `results.baseline_cumulative_hazard()` (callable). Actual API: `results.baseline_cumulative_hazard` is a property returning a `list` of strata, where each stratum is `[unique_times, cumulative_hazard, survival_function]` as 3 same-length ndarrays. For single-stratum models (the default), `bch[0]` is the relevant entry. PR-D's `_extract_baseline_cumulative_hazard` normalises this into a plain dict `{stratum, times, cumulative_hazard, survival_function}` for joblib-stable persistence.

§1 outcome: **PROCEED**. statsmodels installs clean, PHReg fits + pickles, baseline hazard extractable. All four §1 acceptance criteria met.

---

## §2 Scope landed (verbatim against dispatch §2)

| Path | Status | Purpose |
|---|---|---|
| `core/heavy_ml_probe/survival.py` | new | Target construction (time-to-+1R censored at SL/time-exit, same-bar SL tie-break) + per-fold PHReg fit + concordance eval + persistence (results + baseline hazard) |
| `core/heavy_ml_probe/metrics.py` | refactored | `concordance` implemented from scratch (Harrell's C-index per dispatch §4); `integrated_brier_score` now raises `NotImplementedError` with RSF-deferral pointer |
| `core/heavy_ml_probe/pipeline.py` | extended | `_run_survival_stage` orchestration; survival is INDEPENDENT of AutoML + meta-labeling (third parallel stage); new artefacts + extras in manifest; PipelineResult extended with survival fields |
| `scripts/heavy_ml_probe/run_probe.py` | tweaked | stdout surfaces survival concordance + status; "Survival not yet implemented" message replaced |
| `docs/sub_protocols/heavy_ml_probe.md` | extended | survival-models section updated: library swap (lifelines → statsmodels.PHReg) + RSF deferral documented |
| `requirements-dev.txt` | tweaked | `statsmodels` added; `lifelines` + `scikit-survival` removed (both blocked on Py 3.14) |
| `tests/heavy_ml_probe/test_survival.py` | new | 31 tests — target construction edge cases + concordance correctness + end-to-end + determinism + min-N warning + convergence-failure handling + pipeline integration |

Artefact outputs landing per dispatch §2:

- `step_4/heavy_ml/survival_model_results.csv` — long-form (fold × feature): `coefficient, p_value, std_err, concordance, n_train, n_train_event, convergence_warning`
- `step_4/heavy_ml/classifiers/survival/manifest.json` — sha256-bound per-fold model index + statsmodels version + joblib version + per-fold (concordance, n_train_event, fit_message)
- `step_4/heavy_ml/classifiers/survival/fold_NN.joblib` — pickled dict `{results, baseline_hazard, used_features, coefficients}` per fold (skipped folds → no pickle, manifest records `path: null`)
- Existing `step_4/heavy_ml/manifest.json` extended with new `survival` extras block

PR-D does NOT modify `core/architectures/**`, `L_PROTOCOL.md`, vanilla Step 1-6 mechanics, or any arc folders.

---

## §3 Concordance implementation cross-check (dispatch §4)

Implemented from scratch in `core/heavy_ml_probe/metrics.py::concordance`. Standard Harrell C-index pairwise definition. Per dispatch §4 fallback: lifelines is blocked → no library reference available locally; cross-checked against a hand-computed synthetic case AND against the boundary behaviours that have closed-form answers.

### §3.1 Test coverage

| Test | Expected | Got |
|---|---|---|
| `test_concordance_perfectly_ranked` (risk inverse-rank to time, all events) | 1.0 | 1.0 ✓ |
| `test_concordance_perfectly_reversed` (risk same-rank as time, all events) | 0.0 | 0.0 ✓ |
| `test_concordance_random_close_to_half` (random risk + time, n=200) | ≈ 0.5 | 0.40–0.60 ✓ |
| `test_concordance_hand_computed_with_censoring` (3 trades, 1 censored, traced) | 2/3 | 0.6667 ✓ |
| `test_concordance_with_ties` (3 trades, tied risks contribute 0.5) | 2.5/3 | 0.8333 ✓ |
| `test_concordance_no_comparable_pairs_returns_nan` (all-censored validation) | NaN | NaN ✓ |
| `test_concordance_non_finite_dropped` (NaN-time row dropped, then 1.0) | 1.0 | 1.0 ✓ |
| `test_concordance_shape_mismatch_raises` (shape mismatch) | ValueError | ValueError ✓ |

The hand-computed case is the load-bearing reference. Trace:
- Trades: `(event=1, time=1, risk=10), (event=0, time=5, risk=8), (event=1, time=3, risk=5)`
- Comparable pairs `(i, j)` where `i` had event AND `t_i < t_j`:
  - `(0, 1)`: t=1 < t=5, risk_i=10 > risk_j=8 → concordant
  - `(0, 2)`: t=1 < t=3, risk_i=10 > risk_j=5 → concordant
  - `(2, 1)`: t=3 < t=5, risk_i=5 < risk_j=8 → discordant
- Result: `(2 + 0) / 3 = 0.6667` ✓

No drift from the spec math. Implementation vectorised via numpy broadcasting (O(n²) memory; fine at validation-fold scale ~ few hundred trades).

---

## §4 Convergence-warning frequency

### §4.1 Test-scale (n=400, n_folds=5, 4 features)

Early folds (n_train=70, 136) trigger `MIN_N_WARN` (n_train < 200) → `convergence_warning=True` flagged on those rows. Later folds clean. Per-fold detail from a representative run:

```
fold 1: n_train= 70 ev=29 concordance=0.8510  WARN  msg=ok
fold 2: n_train=136 ev=52 concordance=0.8021  WARN  msg=ok
fold 3: n_train=202 ev=82 concordance=0.7799        msg=ok
fold 4: n_train=268 ev=113 concordance=0.8788       msg=ok
fold 5: n_train=334 ev=140 concordance=0.8195       msg=ok
```

statsmodels' own `ConvergenceWarning` did NOT fire on any of these — the convergence_warning flag in the output is purely from our min-N rule. PHReg's analytic optimiser handles small-N cleanly within this sample.

### §4.2 Production-scale probe (n=5000, n_folds=11, 8 features)

```
wall-clock: 1.15s
concordance: nanmean=0.7792 std=0.0168
valid folds: 11/11
total events: 10036
folds with convergence_warning: 0/11
fit failures: 0/11
```

Linearly scaling per-fold: 1.15s / 11 folds ≈ 0.1s per Cox PH fit on n_train ~ 2500. Per-arc cost at 8 features is negligible (PR-C's AutoML stage dominates wall-clock). Even at 30+ features this should stay well under 1 minute per arc — orders of magnitude below the dispatch's "could surface as material wall-clock" concern (none materialised).

### §4.3 Pathological synthetic — convergence-failure path covered

`test_run_survival_no_events_fold_skipped` constructs a pool where the first 90% of trades are all-censored (never reach +1R). TimeSeriesSplit's early folds train on slices with zero events. Cox PH refuses to fit on zero-event data; the survival module short-circuits with `fit_message="skipped:no_events_in_training_slice"`, marks `fit_succeeded=False`, sets concordance to NaN, and continues. No crash, full audit trail in manifest. Confirmed at least one fold skipped in the test.

Real PHReg convergence failures (`ConvergenceWarning`, `LinAlgError`) are trapped at `_safe_phreg_fit`. If they fire in production, the corresponding fold's `fit_message` will carry `f"{ExcType}: {msg}"` and `convergence_warning=True` flags it for closure-doc audit.

---

## §5 Verification results

### §5.1 Test suite

```
py -3 -m pytest tests/heavy_ml_probe -q -W ignore::UserWarning
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 137.85s
```

111 tests: 31 PR-A + 19 PR-B + 30 PR-C + 31 PR-D. Wall-clock 138s — increase over PR-C (~67s) is the new survival tests adding ~70s (mostly the determinism test which fits Cox PH twice + reloads pickles). Acceptable trade-off; all individual tests under 30s.

PR-D test coverage breakdown:

| Group | Count | Coverage |
|---|---|---|
| Target construction edge cases (dispatch §3) | 10 | event-before-close; never-reached/censored; SL on entry; same-bar SL tie-break; same-bar non-SL (event=1); 0-bar clipped to 1; vectorised mixed; HALT-loud schema; HALT-loud NaN bars_held; case-insensitive SL |
| Concordance correctness (dispatch §4) | 8 | perfectly ranked (1.0); perfectly reversed (0.0); random ≈ 0.5; hand-computed w/ censoring; ties; no comparable pairs → NaN; shape mismatch raises; non-finite dropped |
| End-to-end `run_survival` | 3 | smoke; positive `a` coefficient recovered on synthetic; baseline-hazard persisted (non-decreasing cumhaz, monotone-decreasing survival, values in [0,1]) |
| Determinism | 1 | two runs → byte-identical CSV + identical pickle params + identical baseline-hazard arrays |
| Min-N + convergence | 2 | n < 200 warning fires + run completes; zero-events fold skipped cleanly |
| Pipeline integration | 4 | full artefact set written; CSV columns; two-run determinism through pipeline; survival skips with reason on missing columns |
| Persistence corner | 1 | empty SurvivalResult still writes a manifest |
| Constants locked | 2 | `SURVIVAL_REQUIRED_POOL_COLUMNS`, `MIN_N_WARN` |

### §5.2 Sibling regression

```
py -3 -m pytest tests/discovery -q
..........................................                               [100%]
42 passed in 84.66s
```

42/42 — `signal_discovery_probe` untouched.

### §5.3 Lint

```
py -3 -m ruff check core/heavy_ml_probe scripts/heavy_ml_probe tests/heavy_ml_probe
All checks passed!
```

Two unused imports + one unused local auto-fixed via `--fix`. No remaining warnings.

### §5.4 CLI smoke

End-to-end CLI run against a 300-trade synthetic pool carrying ALL three schemas (vanilla `y`, meta-label MFE columns, survival columns):

```
[heavy_ml_probe] pipeline run complete.
  ...
  lineage gate   : accepted=0 / rejected=11 / input=11
  AutoML         : status=no_clean_features (skipped)
  Meta-labeling  : status=no_clean_features (skipped)
  Survival (Cox) : status=no_clean_features (skipped)
[heavy_ml_probe] PR-E (Step 5 augmentation hook) not yet implemented.
exit: 0
artefacts: ['stub_summary']
automl skip: no_clean_features
meta_label skip: no_clean_features
survival skip: no_clean_features
```

All three stages cleanly skipped on the same reason (synthetic columns aren't in the real feature registry — gate rejected every column). Exit 0 because the pipeline ran to completion; the skip is informational.

### §5.5 Determinism

`test_pipeline_survival_two_run_determinism` asserts byte-equality on the survival CSV + the survival classifier manifest + the top-level manifest's stable payload across two consecutive pipeline runs. **Cox PH is deterministic** (analytic optimisation, no random seed needed) — confirmed empirically: per-fold coefficients + baseline-hazard arrays match exactly across runs. Two separate determinism layers covered:

1. **CSV bytes** — sorted by `(fold, feature)`, written with `lineterminator='\n'`, NaN-safe formatting.
2. **Classifier-manifest** — no embedded timestamp (carried PR-C convention forward); sha256 of the on-disk JSON matches across runs.
3. **Pickled coefficients + baseline hazard** — `numpy.testing.assert_array_equal` passes across runs at the pickle-load layer.

---

## §6 Deviations from dispatch

### §6.1 `integrated_brier_score` stubbed, not removed

Dispatch §2 lists IBS as deferred (RSF dropped). PR-D keeps the function signature on the public API surface but the body now raises `NotImplementedError` with a pointer to the deferral rationale. This preserves backwards compatibility for any future caller that imports it — they get a clear error rather than `AttributeError`. Re-enabling it is a one-PR change when sksurv ships its cp314 wheel.

### §6.2 `predict_proba` consciously NOT used

Dispatch §4 hints at using `results.predict()`. PR-D's `_predicted_risk` instead computes `exp(X @ params)` directly. Rationale: `results.predict()` returns a `statsmodels.duration.hazard_regression.PHReg.predict.<locals>.bunch` object (not an ndarray) that's harder to keep stable across statsmodels versions. Computing the linear predictor manually is one line, dependency-free, and matches the textbook definition of Cox PH hazard ratio.

### §6.3 baseline_cumulative_hazard format (already noted §1.4)

Dispatch §8 said `results.baseline_cumulative_hazard()` — actual API is a property returning `list[list[ndarray]]`. Documented in `_extract_baseline_cumulative_hazard` docstring; the function normalises to a plain dict for joblib portability.

### §6.4 Survival is INDEPENDENT of AutoML AND meta-labeling

Same architectural pattern as PR-C's meta-labeling (PR-C log §6.1). The dispatch left this orchestration choice unspecified for survival; PR-D follows the meta-labeling precedent — survival's preconditions are evaluated independently of the other two stages. Skip reasons mirror PR-C's vocabulary.

---

## §7 Flags for chat

### §7.1 Non-blocking

- **Production-scale Cox PH is fast** (1.15s for n=5000 / 11 folds / 8 features). FLAML AutoML still dominates the heavy_ml_probe wall-clock; survival is a rounding error on top of it. No optimisation needed.
- **Test wall-clock at 138s** (PR-A: 1s, PR-B: 17s, PR-C: 67s, PR-D: 138s). 70s of growth this PR — most of it the survival determinism test which fits Cox PH twice across 5 folds and reloads 10 pickles. If CI runtime tightens, the determinism test can drop to 3 folds.
- **statsmodels version recorded in classifier manifest** for PR-E version-drift detection (same convention as joblib version in PR-C).

### §7.2 Surfaced for PR-E intent

- **`build_a4_adapter_from_heavy_ml` consumer surface.** PR-D's persistence format is the joblib pickle dict `{results, baseline_hazard, used_features, coefficients}`. PR-E's adapter (Q5b option B1) needs to:
  - Load the per-fold pickle
  - Compute `lp_at_N = features_at_N @ coefficients`
  - Compute `hazard_ratio_at_N = exp(lp_at_N)`
  - Look up `baseline_cumulative_hazard` at time `N` and `N+K` to get `H_0(N)` and `H_0(N+K)`
  - Compute `S(N+K | N) = exp(-(H_0(N+K) - H_0(N)) * hazard_ratio)`
  - Return `P(reach +1R in K bars) = 1 - S(N+K | N)` as A4's predict_admit signal
  - The `baseline_hazard` dict stores time/cumhaz arrays; PR-E should bilinear-interpolate when N falls between observed event times.
- **Which fold to deploy?** PR-D persists all 11 folds. PR-E decides: last fold (most data), full-IS refit, or ensemble of N best by concordance. Dispatch §7 explicitly defers this to PR-E.

### §7.3 No HALT

§1 install check passed; pickle round-trip works; no production-scale convergence failures observed. WORKFLOW §6 unused.

---

## §8 Branch state at PR-time

- Build branch (`infra/heavy_ml_probe_build`) on origin still at `7c238e8` (main's tip at PR-A time). PR-A/B/C merges have not happened yet at PR-D-open time — chat's "merged" signals interpreted at the head level (each PR's branch contains all upstream).
- PR-D branch (`infra/heavy_ml_probe_pr_d`) cut from `infra/heavy_ml_probe_pr_c` locally so the head of PR-D includes A+B+C+D commits. When chat merges A/B/C → build on GitHub, the PR-D diff will recompute and show only PR-D's changes.

---

## §9 What lands next (per dispatch §11)

After PR-D merges into `infra/heavy_ml_probe_build`:

- **PR-E (Integration + Step 5 augmentation hook)** — final integration PR. Wires AutoML + meta-labeling + survival into a single CLI invocation (already there in PR-D, but PR-E adds the consumer adapters). Provides:
  - `build_a2_config_from_heavy_ml(cluster_id, step4_dir)` — wraps PR-C meta-label classifiers into A2Config
  - `build_a6_config_from_heavy_ml(cluster_id, step4_dir)` — same, for A6
  - `build_a4_adapter_from_heavy_ml(cluster_id, step4_dir, fold_choice, horizon_K)` — Q5b adapter wrapping PR-D Cox PH `S(t | features)` into A4's `predict_admit` shape
  - Step 5 hook documentation explaining how an arc invokes these
- **PR-F (Docs + polish)** — README / inline cleanup. Final PR.

Per chat directive: I will not start PR-E work during the PR-D review window.

---

End of PR-D log.
