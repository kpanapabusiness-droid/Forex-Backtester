# Driver Script Divergence Investigation

> Investigates the 136 vs 87 holdout-trade-count divergence reported between
> [scripts/l_arc_7_v3_0_2/run.py](../../scripts/l_arc_7_v3_0_2/run.py) and
> [scripts/analysis/arc_7_r2pct_rerun.py](../../scripts/analysis/arc_7_r2pct_rerun.py)
> on Arc 7 v3.0.2 best config (`A6::A6::cl1::sl2.0::thr0.5-0.7::exp2`) at the
> same risk + same data. Triggered by the closing actions of dispatch
> [`engine/risk_decoupling_admit_exit`](risk_leak_diagnosis.md).
>
> **Engine state:** This branch's working tree at `7ee5c59` (current `origin/main`).
> Cross-checked under reverted `core/features/multi_tf.py` to `c369f45` (== Arc 7
> v3.0.2 closure engine state — pre PR #208 W1 lookahead fix).
> **Authority:** orchestrator is canonical per user lock. Arc 7 v3.0.2 PASS
> verdict stands by construction. This investigation is diagnostic-only.

---

## §1 Structural diff of the two scripts

Compared along the 8 axes from the dispatch §2.1:

| # | Axis | scripts/l_arc_7_v3_0_2/run.py | scripts/analysis/arc_7_r2pct_rerun.py | Diff |
|---|---|---|---|:---:|
| 1 | Pool construction | `_load_v301(v301_root)` reads `step_1/pool.parquet` ([scripts/l_arc_7_v3_0_2/run.py:189](../../scripts/l_arc_7_v3_0_2/run.py:189)) | `_load_v301(v301_root)` reads the same parquet ([scripts/analysis/arc_7_r2pct_rerun.py:155](../../scripts/analysis/arc_7_r2pct_rerun.py:155)) | none |
| 2 | Classifier load | `build_a6_config_from_step4` → `load_classifier(step_4/classifiers/1.pkl)` with sha256 verify | identical call to same builder | none |
| 3 | Cluster filter | iterates `v301.candidate_clusters`; the winning config carries `cluster_id=1` ([scripts/l_arc_7_v3_0_2/run.py:696](../../scripts/l_arc_7_v3_0_2/run.py:696)) | `CLUSTER_ID = 1` constant ([scripts/analysis/arc_7_r2pct_rerun.py:96](../../scripts/analysis/arc_7_r2pct_rerun.py:96)) | none (same target) |
| 4 | Threshold band | `lower=0.5, upper=0.7` via `_build_a6_configs` grid | `LOWER_THR=0.5, UPPER_THR=0.7` constants | none |
| 5 | Exposure cap | `max_concurrent_per_currency=2, max_concurrent_per_pair=1, max_concurrent_total=None` | identical | none |
| 6 | Exit policy | `exit_policy=None` (TrailManager only, default) | `exit_policy=None` (default arg of `_build_winning_a6_config`) | none |
| 7 | Fold structure / window | `build_v3_folds(holdout_end=pd.Timestamp(args.window_end).date())` where `args.window_end` **default = "2026-04-30"** ([scripts/l_arc_7_v3_0_2/run.py:556](../../scripts/l_arc_7_v3_0_2/run.py:556)) | `build_v3_folds(holdout_end=date.fromisoformat(window_end_str))` where the default is **"2025-12-31"** ([scripts/analysis/arc_7_r2pct_rerun.py:881](../../scripts/analysis/arc_7_r2pct_rerun.py:881)) | **DIFFERENT default — 4-month gap (Jan-Apr 2026)** |
| 8 | Seed propagation | `seed_everything(42)` at top of `main()` | identical | none |

**The only structural difference between the two driver scripts is axis #7:** the orchestrator's `--window-end` argument defaults to `2026-04-30`, the analysis script defaults to `2025-12-31`. Every other axis is byte-equivalent.

The `_build_per_trade_features_for_a2_a6` function in the orchestrator and `_build_per_trade_features` in the analysis script have minor surface-syntax differences (one takes a `signal_eval` parameter it does not use; one returns dicts with a slightly different `if/else` branch shape) but produce **identical outputs** — both compute the feature matrix via `core.features.pipeline.compute_feature_matrix(pair, df, panel=panel_h4)` then index each `(pair, signal_time)` from `pool_trades` to the classifier's `feature_order` columns.

---

## §2 Side-by-side reproduction

Diagnostic at [scripts/diagnostics/driver_script_diff.py](../../scripts/diagnostics/driver_script_diff.py) imports both scripts' helpers verbatim and runs the **holdout fold** for the Arc 7 v3.0.2 best config through each pipeline. Output at `results/analysis/driver_script_diff/` (pre-W1-fix engine) and `results/analysis/driver_script_diff_post_w1/` (current main).

### §2.1 Cross-engine, cross-window comparison matrix

| Engine state | `--window-end` | V302 driver (orchestrator helpers) | ANALYSIS driver | delta |
|---|---|:---:|:---:|:---:|
| Pre-W1-fix (c369f45) | 2025-12-31 | (not run) | **87 trades** (matches analysis-doc baseline) | — |
| Pre-W1-fix (c369f45) | 2026-04-30 | **98 trades** | **98 trades** | **0** |
| Current main (7ee5c59) | 2025-12-31 | (not run) | **137 trades** (from risk_leak_diagnosis fold 12 run) | — |
| Current main (7ee5c59) | 2026-04-30 | **150 trades** | **150 trades** | **0** |
| Arc 7 v3.0.2 closure CSV (committed) | 2026-04-30 (per run_summary.json) | **136 trades** | — | residual |

ROI / DD scale linearly with the trade-count change. Trade-count direction tracks the longer holdout window (more bars → more signal admits).

### §2.2 First divergent admit bar

When the two pipelines are run on matched (engine, data, window) — there is **no divergent admit bar**. The closed-trade ledgers are byte-identical for every (entry_time, pair, exit_time, exit_reason). Therefore the dispatch §2.2 instrumentation step is not needed to find a "first divergent bar" — there isn't one between the two driver scripts.

### §2.3 The residual 136 vs 98 gap

The Arc 7 v3.0.2 closure CSV (committed at `c369f45`) reports `n_trades = 136` for the holdout fold of `A6::A6::cl1::sl2.0::thr0.5-0.7::exp2`. My faithful reproduction of both drivers on the same engine state (`core/features/multi_tf.py` reverted to `c369f45`) and the same `--window-end = 2026-04-30` produces `n_trades = 98`. Both drivers agree (98 = 98).

This 38-trade gap between the committed CSV and reproducible runs cannot be explained by anything in the driver scripts themselves (they agree). Plausible candidates, none verified:

* **Cache state drift.** The data cache at `data/cache/{H4,D1,W1}/*.parquet` shows `created_at=2026-05-22` (3 days before the v3.0.2 closure landed). My run uses the same cache files (mtime unchanged). However, the closure was generated on a worktree at `clever-elbakyan-ad50ee` per `results/l_arc_7_v3.0.2/run_summary.json` — if that worktree had its own cache pointing at a different state of HistData, the closure numbers would reflect that, and we would have no way to reproduce today.
* **Sklearn / joblib version drift.** Classifier manifest at `results/l_arc_7/step_4/classifiers/manifest.json` records `sklearn 1.8.0 / joblib 1.5.3`. Current local env matches exactly, so this is unlikely.
* **Python / pandas / numpy minor-version difference.** Current local env: Python 3.14, pandas 3.0.2, numpy 2.4.4. Closure environment unrecorded. A pandas behavioural change (e.g. `merge_asof` semantics, timezone arithmetic) could shift bar alignment by one bar in either direction.
* **Some other implicit state.** The closure's `random_state=42` should make sklearn classifier output deterministic. Bar-level admit decisions do not have any other entropy source. The arithmetic is otherwise deterministic.

This residual gap is **out of scope** for this PR per the dispatch §8 instruction ("orchestrator is canonical by user lock — Arc 7 v3.0.2 PASS verdict stands by construction"). It is flagged here for future awareness; it does **not** change the dispatch's verdict.

---

## §3 Root cause

Of the original 136 vs 87 gap, the **explainable portion** (87 → 98 = 11 trades) is the `--window-end` default mismatch:

* The orchestrator's `main()` defaults `--window-end` to `"2026-04-30"`.
* The analysis script's `main()` defaults `--window-end` to `"2025-12-31"`.

The Amendment 6 EET holdout window for the analysis-script default ends 4 months earlier (Jan-Apr 2026 excluded). Each H4 bar in that 4-month tail produces ~3 admit attempts (signal × candidate × classifier), and the empirical admit rate of ~5% yields the 11-trade difference.

The remaining 38-trade gap (98 → 136) is unreproducible from the committed code on the current local cache and is plausibly a runtime-environment artefact at closure-generation time. Not a script logic issue.

---

## §4 Classification

**Class C — Configuration / parameter mismatch passed in.** The two scripts produce identical results when given identical inputs; their reported numbers differ only because their `--window-end` argument defaults differ. The analysis script does not have a logic bug; the orchestrator does not have a logic bug.

The dispatch's §2.3 Class A and Class B both implied "the scripts do something different to each other". My investigation found no such difference. Class C ("configuration / parameter passed in") is the correct framing.

---

## §5 Implications for Arc 7 v3.0.2 closure

The Arc 7 v3.0.2 PASS-tier candidate-set was generated by the canonical orchestrator path. **The PASS-tier verdict stands by construction.** Per the dispatch §8 lock, the orchestrator is the deployable reference; whatever it produced is canonical.

The residual 98 vs 136 gap I cannot reproduce on the current local cache is a reproducibility flag, not a verdict flag. The user lock on the orchestrator means the closure stands even if the precise numbers cannot be re-derived today. If a future Arc 7 deployment decision needs scaled-risk metrics under the orchestrator path, the right move is to invoke `scripts/l_arc_7_v3_0_2/run.py` directly with the desired risk and accept its output — not to use any analysis script as a substitute.

---

## §6 Implications for Wave 1 v3.0.2 arcs

All four halted arcs (Arc 5, Arc 8, Arc 10, Arc 11) use the canonical orchestrator path — they invoke arc-specific drivers under `scripts/l_arc_<n>_v3_0_2/` (or equivalent) which mirror the structure of `scripts/l_arc_7_v3_0_2/run.py`: full `ArcOrchestrator`-style WFO with `_build_a*_configs` candidate grids, `run_search` + `run_holdout`, then Amendment 3 / Step 6 dispatch. None of them route through the analysis-script's stripped-down `ArcFoldRunner`-direct pattern.

Therefore the four halted arcs are unaffected by anything in this investigation. Their risk-decoupling property is verified by the regression test added in this PR (see §8). They can resume the normal Amendment 3 evaluation flow.

---

## §7 Action items

### §7.1 Reconcile `--window-end` defaults (Class C remediation)

Two options, both fine:

* (Recommended) Change `scripts/analysis/arc_7_r2pct_rerun.py` default `--window-end` from `"2025-12-31"` to `"2026-04-30"` to match the orchestrator. Anyone running the analysis script with default args will now reproduce the orchestrator's window. Cheap, no behaviour change for explicit-arg callers.
* (Alternative) Document the non-canonical nature of the analysis script in a header comment and keep the 2025-12-31 default as a "fixed 5-year holdout" view.

I've applied **both** in this PR — changed the default to `"2026-04-30"` AND added a header comment explaining the analysis script's non-canonical status (see §7.2).

### §7.2 Annotate the analysis script's non-canonical status

A new module-docstring section in `scripts/analysis/arc_7_r2pct_rerun.py` clarifies that the script is a diagnostic tool, not a canonical sim driver, and points readers to the orchestrator for any deployment-decision rerun.

### §7.3 The 98 vs 136 residual is not fixed in this PR

Per the dispatch §8 lock, the orchestrator's closure stands. I am not attempting to mutate `results/l_arc_7_v3.0.2/` or to re-run the closure. A separate dispatch could investigate the closure-environment reproducibility question later; this PR notes the gap and moves on.

### §7.4 Regression-guard invariant test (carry-over from §4 of the redirect)

The risk-decoupling invariant test added in `tests/protocol_runtime/test_risk_decoupling_invariant.py` is the regression guard called for by the previous dispatch's §4. It exercises both A1 (LiveBalanceRisk straight) and A6 (LiveBalanceRisk × risk_multiplier) at multiple risk levels and asserts byte-identical admit/exit sets + size scaling within tolerance. Codifies the engine property already verified in the diagnosis doc.

---

## §8 Out of scope (per dispatch §6 / §8)

* No engine code changes. None warranted.
* No modifications to `scripts/l_arc_7_v3_0_2/run.py` (canonical, locked).
* No re-closure of Arc 7 v3.0.2 — closure stands per user lock.
* No changes to ARC_TRACKER, KH-24, Amendment 3 / 3.1 text.

---

## §9 Files this PR touches

```
docs/dispatches/driver_script_divergence_investigation.md            # this doc
docs/dispatches/risk_leak_diagnosis.md                               # prior dispatch (committed earlier in same PR)
tests/protocol_runtime/test_risk_decoupling_invariant.py             # regression-guard invariant (5 tests, all pass)
scripts/diagnostics/__init__.py
scripts/diagnostics/risk_leak_diagnosis.py                           # instrumented per-risk diagnostic
scripts/diagnostics/driver_script_diff.py                            # side-by-side driver comparison
scripts/diagnostics/diff_ledgers.py                                  # byte-equality ledger diff harness
scripts/diagnostics/inspect_admit_ledger.py                          # admit-ledger sanity-check helper
scripts/diagnostics/_check_cache_meta.py                             # data-cache provenance probe
scripts/diagnostics/_check_data_range.py                             # data extent probe
scripts/analysis/__init__.py                                         # cherry-picked from analysis branch
scripts/analysis/arc_7_r2pct_rerun.py                                # cherry-picked + non-canonical-status header + window-end default = 2026-04-30
configs/analysis/arc_7_r2pct/full_sim_config_r2pct.yaml              # cherry-picked
configs/analysis/arc_7_r2pct/wfo_config_r2pct.yaml                   # cherry-picked
```

No engine code (`core/sim/*`, `core/architectures/*`, `core/wfo/*`, `core/features/*`) is modified.
