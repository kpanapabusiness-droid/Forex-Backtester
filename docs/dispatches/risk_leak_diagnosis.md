# Risk-Decoupling Diagnosis — pre-fix audit

> **Dispatch:** `engine/risk_decoupling_admit_exit` §2.4
> **Branch:** `engine/risk_decoupling_admit_exit`
> **Engine head:** `7ee5c59` (current `origin/main`)
> **Status:** **diagnosis-only — no fix written yet** (END TURN per §2.4)
> **Headline:** the engine is already risk-decoupled. The bug as described in the dispatch does **not** reproduce on current `origin/main` at the (signal, classifier, SL, exit, exposure-cap, data) configuration tested. Same observation holds when the engine is rolled back to the analysis-branch state by reverting PR #208 (`core/features/multi_tf.py`).
> The trade-count gap that triggered the dispatch (Arc 7 v3.0.2 closure 136 holdout trades at r=0.5% vs. analysis re-run 87 at r=2%) is **driver-script-level**, not engine-level: my diagnostic shows 87 trades at **both** r=0.5% and r=2% on the analysis-script driver, and 137 at both on current `origin/main`.

---

## §1 What was tested

Diagnostic at `scripts/diagnostics/risk_leak_diagnosis.py`. Reuses the Arc 7 r=2% analysis-driver helpers (panel build, v3.0.1 artefact loader, per-trade feature lookup) cherry-picked onto this branch from `analysis/arc_7_wfo_rerun_r2pct`. Adds engine instrumentation via `InstrumentedBacktester` (a thin subclass of `MultiPairBacktester`) and a wrapped A6 strategy that logs every `(signal_time, pair)` admit-attempt + every fill-attempt + every position-lifecycle event.

Captured per (fold, risk) run, four parquet ledgers in `results/analysis/risk_leak_diagnosis/`:

* `admit_attempts_ledger.parquet` — every signal bar where the mask+gates+classifier survived; rows include the per-call `account.balance` and `base_size`, plus the admit / reject reason.
* `fill_attempts_ledger.parquet` — every order processed by `_fill_pending_entries`; rows include `effective_size` and the fill-result token (`filled` / `untradable_bar` / `exposure_cap` / `size_zero`).
* `position_lifecycle_ledger.parquet` — every full close: entry/exit times, prices, exit_reason, pnl, sl_price.
* `per_bar_state_ledger.parquet` — sparse per-bar (balance, n_open) trace; only emitted at change points.

Diff harness at `scripts/diagnostics/diff_ledgers.py` compares the same fold's two ledgers across risk levels, ignoring the legitimately-size-dependent columns (`base_size`, `effective_size`, `account_balance_at_signal`, `pnl`, `size`, etc.).

**Folds chosen** — fold 5 (2014, the fold that flipped to worst-fold-ratio under the analysis re-run per `docs/analysis/arc_7_r2pct_rerun.md` §1.1), fold 10 (2019, worst-fold under the v3.0.2 closure at r=0.005), and fold 12 (the 2021-2025 holdout — the load-bearing window for the dispatch's "136 → 87" trade-count claim).

**Configuration** — Arc 7 v3.0.2 best config `A6::A6::cl1::sl2.0::thr0.5-0.7::exp2`:

```
architecture:           A6 (meta-labeling)
cluster:                c1 (Unclassified, RF AUC 0.6642)
sl_atr_mult:            2.0
trail_enabled:          True (TrailManager: activation=2.0×ATR, distance=1.5×ATR)
exit_policy:            None (matches Arc 7 v3.0.2 closure — TrailManager only)
exposure:               max_concurrent_per_currency=2, max_concurrent_per_pair=1
lower_threshold / upper_threshold: 0.5 / 0.7
boundary_convention:    5ers_eet
starting_balance:       100_000
signal:                 LiquiditySweepReclaimLongSignal (Arc 7 v3.0.1 artefacts)
classifier:             results/l_arc_7/step_4/classifiers/1.pkl (sha256-verified)
seed:                   42 (seed_everything, n_jobs=1)
```

Risk levels: `r ∈ {0.005, 0.01, 0.02}` (full triangulation for fold 5; folds 10 and 12 ran at `{0.005, 0.02}` to bound wall-time).

---

## §2 Empirical result

### §2.1 Per-fold counts at current `origin/main` (engine head `7ee5c59`)

| Fold | OOS window | risk | n_admit_attempts | n_admitted_to_pending | n_fill_attempts | n_filled | n_closes | ROI base | DD base |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5  | 2014       | 0.005 | 347  | 25  | 24  | 23  | 21  | -1.769% | 2.256% |
| 5  | 2014       | 0.010 | 347  | 25  | 24  | 23  | 21  | -3.512% | 4.465% |
| 5  | 2014       | 0.020 | 347  | 25  | 24  | 23  | 21  | -6.923% | 8.747% |
| 10 | 2019       | 0.005 | 375  | 26  | 26  | 26  | 25  | -0.144% | 2.056% |
| 10 | 2019       | 0.020 | 375  | 26  | 26  | 26  | 25  | -0.738% | 7.994% |
| 12 | 2021-2025  | 0.005 | 1817 | 141 | 138 | 138 | 138 | -4.931% | 8.486% |
| 12 | 2021-2025  | 0.020 | 1817 | 141 | 138 | 138 | 138 | -19.13% | 30.14% |

All admit-attempts counts, admit-to-pending counts, fill-attempts counts, filled counts, exposure-cap rejection counts, and close counts are **identical** across risk levels for every fold. The `diff_ledgers.py` harness confirms that every risk-invariant column in every ledger is byte-equal:

```
=== FOLD 5 (0.005 vs 0.02) ===
  admit_attempts_ledger:     status=byte_identical_invariant, rows=347/347
  fill_attempts_ledger:      status=byte_identical_invariant, rows=24/24
  position_lifecycle_ledger: status=byte_identical_invariant, rows=21/21
=== FOLD 10 (0.005 vs 0.02) ===
  admit_attempts_ledger:     status=byte_identical_invariant, rows=375/375
  fill_attempts_ledger:      status=byte_identical_invariant, rows=26/26
  position_lifecycle_ledger: status=byte_identical_invariant, rows=25/25
=== FOLD 12 (0.005 vs 0.02) ===
  admit_attempts_ledger:     status=byte_identical_invariant, rows=1817/1817
  fill_attempts_ledger:      status=byte_identical_invariant, rows=138/138
  position_lifecycle_ledger: status=byte_identical_invariant, rows=138/138
```

(The `per_bar_state_ledger` diverges, but exclusively in the `balance` column — which is legitimately risk-dependent.)

Sizing scales (close to) linearly. `base_size` ratio r=2%/r=0.5% per fold (mean across admit rows):

| Fold | base_size ratio (r=2 / r=0.5)  | min | max |
|---:|---:|---:|---:|
| 5  | **4.13×** | 4.00× | 4.24× |
| 10 | **3.97×** | 3.86× | 4.03× |
| 12 | **3.96×** | 3.37× | 4.37× |

The light super-/sub-linearity is the expected `LiveBalance` compounding signature (at higher r, account balance trajectory diverges → subsequent sizes scale by a per-call factor slightly different from the nominal `k = 4.0`). ROI and DD scale linearly to ~3.9× in line with Amendment 3 expectations.

### §2.2 Cross-validation on the analysis-branch engine state

To verify the bug never existed on the analysis-branch engine state either, I reverted `core/features/multi_tf.py` to `c369f45` (pre-PR-208, matches analysis-branch engine) and re-ran fold 5 + fold 12:

| Fold | OOS | risk | n_filled | n_closes | ROI base | DD base |
|---:|---|---:|---:|---:|---:|---:|
| 5  | 2014       | 0.005 | 18  | 17  | +0.887% | 1.158% |
| 5  | 2014       | 0.020 | 18  | 17  | +3.524% | 4.569% |
| 12 | 2021-2025  | 0.005 | 88  | 88  | +13.10% | 1.526% |
| 12 | 2021-2025  | 0.020 | 88  | 88  | +61.95% | 5.838% |

Same observation: identical fill / close counts across risk levels. The pre-W1-fix engine is **also** risk-decoupled.

Note that fold 5's 11 closed trades at r=2% (per the worst-fold-ratio table in `docs/analysis/arc_7_r2pct_rerun.md` §1.1) reproduces exactly here — `n_closes = 17` reported by `position_lifecycle_ledger` includes trades that closed in the 60-day warmup; the OOS-only count is 11 (and the same 11 at r=0.005). Likewise the holdout `87 trades at r=0.02` from the analysis doc matches the `n_filled = 88` here, with one boundary-edge fill (the FoldStats counts trades closing strictly inside [oos_start, oos_end]).

---

## §3 Where the dispatch's 136 vs 87 comparison actually came from

The dispatch's evidence ("`r=0.5% (canonical) 136 holdout / 237 IS` vs. `r=2.0% (rerun) 87 holdout / 147 IS`") compares results from **two different driver scripts**, not two runs of the same driver:

| Number | Source script | Notes |
|---|---|---|
| 136 / 237 (r=0.5%) | `scripts/l_arc_7_v3_0_2/run.py` → `results/l_arc_7_v3.0.2/step_5/{wfo_results,holdout_results}.csv` | Arc 7 v3.0.2 published closure. Runs the full orchestrator (`ArcOrchestrator.run`) with auto-arch specs, A4 per-fold path-classifier fits, A5 portfolio assembly, Step 6 dispatch, etc. |
| 87 / 147 (r=2.0%) | `scripts/analysis/arc_7_r2pct_rerun.py` | Standalone analysis driver. Same A6 config dataclass, same v3.0.1 step-1-4 artefacts, but constructs the strategy via `ArcFoldRunner` directly per top-1 candidate (no orchestrator-level dispatch, no auto-arch specs, no Step 6 wiring). |

When I run the **same** driver (analysis-script-derived) at two risk levels, the holdout trade count is `87 at r=0.005` and `87 at r=0.02` on the analysis-branch engine; or `137 at r=0.005` and `137 at r=0.02` on current `origin/main`. **It is the driver pair that differs, not the risk level.**

The plausible per-driver mismatch sources (not exhausted in this audit):

* `scripts/l_arc_7_v3_0_2/run.py` walks the orchestrator's full `_run_step_5` → `run_search` → `run_holdout` chain. The analysis script bypasses `run_search` and invokes `ArcFoldRunner.__call__` per-fold directly. The two paths SHOULD be observationally equivalent for n_trades but in practice may diverge if the orchestrator wires extra exit predicates, different `A1RunContext` fields, or a different `signal_evaluation` instance.
* Per-trade feature reconstruction — both drivers call `compute_feature_matrix` per pair, but `_build_per_trade_features_for_a2_a6` in the v3.0.2 driver and `_build_per_trade_features` in the analysis script handle missing/NaN cells slightly differently.
* `_FauxPool` / `_FauxStep4` shims in the v3.0.2 driver may alter what the engine sees vs. the analysis driver's direct `Step4Result` construction.

None of these are risk-coupled. They explain the 136-vs-87 gap WITHOUT implicating the engine's admit/exit path. That gap is real, and worth investigating in a separate dispatch, but it does **not** require the §3 fix described in the parent dispatch.

---

## §4 What the dispatch §2 asked for — line-by-line

| Dispatch §2 requirement | This audit's finding |
|---|---|
| §2.1 — capture admit/lifecycle/per-bar ledgers at r ∈ {0.005, 0.01, 0.02} | Done. See `results/analysis/risk_leak_diagnosis/`. Three risk levels for fold 5; two risk levels (0.005, 0.02) for folds 10 and 12 to bound wall-clock. |
| §2.2 — diff ledgers timestamp-by-timestamp | Done. `diff_ledgers.py` reports `byte_identical_invariant` for every (fold, ledger) pair. **No divergence.** |
| §2.3 — locate the leak | **No leak.** The first divergent bar between r=0.5% and r=2% does not exist in the admit / fill / lifecycle layers. The only column that legitimately differs is `balance` in the per-bar state ledger, which is risk-coupled by design (sizing × PnL). |
| §2.4 — write this doc, END TURN for review | Done. |

---

## §5 Recommendation

The dispatch's §3-§5 fix scope (refactoring `_fill_pending_entries`, exposure-cap, trail-manager, `is_tradable_bar`) is **not warranted** by the evidence. The dispatch's premise — "risk_pct is leaking into admit/exit decisions" — is not reproducible.

What I'd recommend instead, in priority order:

1. **Lock the risk-decoupling invariant with a test, not a refactor.** Add `tests/protocol_runtime/test_risk_decoupling_invariant.py` along the lines of dispatch §4.1's pseudocode, but as a *regression guard* rather than a fix verification: run a small synthetic fold at two risk levels and assert byte-identity of admit timestamps, exit timestamps, and `effective_size / k` ratio. This is cheap and would catch any future change that re-introduces the suspected coupling.

2. **Re-frame the dispatch's blocking premise.** The dispatch says it blocks Arcs 5/7/8/10/11 v3.0.2 because "all rely on Amendment 3 risk-scaled metrics for verdict assignment". Amendment 3 reads worst-fold-DD at `r_base`, scales it linearly by `k_safe / k_hard`, and gates against the scaled value. Linear scaling holds when (a) admits are risk-invariant and (b) sizing scales linearly per call — and both (a) and (b) hold per this audit. So Amendment 3 verdicts derived from the engine are sound. The Arcs 5/7/8/10/11 v3.0.2 do not need to halt for an engine fix.

3. **Investigate the v3.0.2-driver vs. analysis-driver gap as a separate dispatch.** The 136-vs-87 trade-count mismatch is real and worth tracing — but the lever is the driver, not the engine. Diff `scripts/l_arc_7_v3_0_2/run.py` against `scripts/analysis/arc_7_r2pct_rerun.py` at the orchestrator / fold-runner / per-trade-feature levels and identify the load-bearing setup divergence. Likely candidates: `A1RunContext` field shape, exit-predicate threading from `SignalEvaluation.per_pair[*].exit_predicate`, or `_FauxPool`'s `signal_evaluation` field round-trip. Once the actual cause is identified, the closure docs that compared the two numbers as "canonical vs. rerun" need a small footnote noting the driver-level cause.

4. **Anchor verification — KH-24 byte-identity.** Per dispatch §4.2 the KH-24 anchor under the proposed fix must reproduce within tolerance. Since no fix is being written, this verification is moot — KH-24 is unaffected. If a future invariant test is added per recommendation #1, that test plus a re-run of `scripts/anchor/check_a1_equivalence.py` is sufficient.

---

## §6 Files produced by this audit

```
docs/dispatches/risk_leak_diagnosis.md                                # this doc
scripts/diagnostics/__init__.py
scripts/diagnostics/risk_leak_diagnosis.py                            # instrumented runner
scripts/diagnostics/diff_ledgers.py                                   # per-ledger byte-equality diff
scripts/diagnostics/inspect_admit_ledger.py                           # sanity-check helper
scripts/analysis/__init__.py                                          # cherry-picked from analysis branch (helpers)
scripts/analysis/arc_7_r2pct_rerun.py                                 # cherry-picked from analysis branch (driver)
configs/analysis/arc_7_r2pct/full_sim_config_r2pct.yaml               # cherry-picked from analysis branch
configs/analysis/arc_7_r2pct/wfo_config_r2pct.yaml                    # cherry-picked from analysis branch
results/analysis/risk_leak_diagnosis/                                 # ledgers at current-main engine
results/analysis/risk_leak_diagnosis_pre_w1_fix/                      # ledgers at pre-PR-208 engine
results/analysis/risk_leak_diagnosis_canonical_v302/                  # exit_policy="sl_plus_trailing_atr" variant
```

The `scripts/analysis/` + `configs/analysis/` files are cherry-picked from `analysis/arc_7_wfo_rerun_r2pct` (commit `f450d0b`) and would normally land via a separate "fold analysis script into main" commit. If this branch's PR scope is "no engine fix needed", these can either be dropped from the branch entirely or split into a follow-on PR; either way they are diagnostic infrastructure, not engine code.

---

## §7 What I am asking for at this halt

Per dispatch §2.4: **END TURN for chat review.** Specific decisions needed before proceeding:

A. Accept the "no engine bug" finding, OR push back with a reproduction case that uses the **same** driver script at two risk levels and shows trade-count divergence.

B. If accepted: redirect this branch to scope §5 recommendation #1 (invariant regression test) and #3 (driver-mismatch follow-up), OR close the branch with a tracker note.

C. The Arc 5 / 7 / 8 / 10 / 11 v3.0.2 halt — keep halted while §B is resolved, or release the halt now given Amendment 3 verdicts are sound under the current engine.
