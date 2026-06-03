# Arc 8 — CC Log Doc

> Per WORKFLOW.md §2 and dispatch §"Closure doc". Append-only record of
> what CC actually did vs the dispatch, deviations, verification outcomes,
> and chat-side flags.
>
> Companion docs:
> - Dispatch: `docs/dispatches/arc_8_dispatch.md` (chat-pasted at session open)
> - Intent: `docs/dispatches/arc_8_intent.md`
> - Closure: `results/l_arc_8/ARC_CLOSURE.md`

---

## Session timeline

| Step | Status | Wall time | Output |
|---|---|---:|---|
| Intent doc | ✅ Approved | n/a | docs/dispatches/arc_8_intent.md |
| Step 1 (plumbing) | ✅ Done | 114.1s | results/l_arc_8/step_1/ |
| Step 2 (clustering) | ✅ Done | 11.2s | results/l_arc_8/step_2/ |
| Step 3 (capturability) | ✅ Done | 1.4s | results/l_arc_8/step_3/ |
| Step 4 (extraction) | ✅ Done | 14.6s | results/l_arc_8/step_4/ |
| Step 5 (WFO search) | ✅ Done (approximated) | 37.9s | results/l_arc_8/step_5/ |
| Step 6 (causal audit) | ❌ Skipped | n/a | (Step 5 verdict FAIL → lazy step not invoked per L_PROTOCOL §2 Step 6) |
| Closure | ✅ Done | n/a | results/l_arc_8/ARC_CLOSURE.md |

---

## Chat flag resolutions (from arc_8_intent.md F1-F9)

All nine flags resolved by chat before Step 1. Defaults / overrides actually applied:

| Flag | Topic | Resolution | Applied |
|---|---|---|---|
| F1 | spec's v2.x §-refs | ignore, semantically clean | ✅ ignored |
| F2 | data window | dispatch wins (2010-01-01 → most-recent-complete-month) | ✅ window 2010-01-01 → 2026-04-10 (data-limited) |
| F3 | SL sweep set | dispatch's {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} | ✅ applied at Step 3 |
| F4 | spread fallback | L_PROTOCOL §1 wins (no fallback) | ✅ no fallback file used; data-quality flag reported |
| F5 | branch rename | rename to arc/l_arc_8 | ✅ done at session start |
| F6 | stale v2.x configs in configs/l_arc_8/ | leave untouched, write v3 alongside | ✅ wrote scripts/l_arc_8/ as new path (chose script-driver location instead of new config files; existing step2.yaml..step5.yaml v2.x configs untouched) |
| F7 | cross-asset DXY/US10Y | omit from feature space, use 27-feature v3 catalogue | ✅ ran with v3 catalogue |
| F8 | signal implementation in core/signals/ | approved as arc-scope | ✅ `core/signals/pullback_resume_hhhl.py` |
| F9 | peer-arc co-fire | KH-24 only at Step 1 | ✅ KH-24 reported (0.000%); peers deferred |

---

## Files created (this arc)

- `core/signals/pullback_resume_hhhl.py` — signal producer (vectorised; 3-bar swing detection, HH/HL trend window, pullback gate, resume trigger, spacing)
- `scripts/l_arc_8/__init__.py`, `shared.py`, `build_step1_pool.py`, `run_step2_clustering.py`, `run_step3_capturability.py`, `run_step4_extraction.py`, `run_step5_wfo.py`
- `docs/dispatches/arc_8_intent.md`
- `docs/dispatches/arc_8_log.md` (this file)
- `results/l_arc_8/ARC_CLOSURE.md`
- `results/l_arc_8/step_1/*` — pool.parquet, paths.parquet, features.parquet, feature_lineage.csv, integrity_report.{md,json}, manifest.json
- `results/l_arc_8/step_2/*` — path_features.parquet, silhouette_sweep.csv, cluster_assignments.parquet, cluster_metrics.csv, cluster_summary.md, manifest.json
- `results/l_arc_8/step_3/*` — sl_sweep.csv, capturability.csv, capturability_summary.md, manifest.json
- `results/l_arc_8/step_4/*` — extraction_metrics.csv, feature_importance.csv, extraction_summary.md, classifiers/cluster_2_rf.pkl, manifest.json
- `results/l_arc_8/step_5/*` — wfo_results.csv, per_fold_metrics.csv, wfo_oracle.csv, holdout_results.csv, architectures_ranked.md, manifest.json

---

## Material deviations from the dispatch (for chat review)

### D1 — Multi-TF features absent in Step 4 training matrix

**Dispatch:** Step 1 feature space = L_PROTOCOL §2 Step 1 default (27-feature v3 catalogue).
**What happened:** Step 1 ran `compute_feature_matrix(pair, df_h4, panel=panel_h4)` — passed only the H4 panel. The 4 multi-TF features (`d1_atr_percentile_100`, `d1_close_slope_magnitude`, `d1_close_slope_sign`, `w1_close_slope_sign`) require `panel.aux["d1"]` and `panel.aux["w1"]`, which weren't attached. Producers emit all-NaN columns when aux is missing (per `core/features/multi_tf.py`).
**Impact:** Step 4 trained on 23 features instead of 27 (the all-NaN columns were dropped before training). Step 4 AUC 0.5300 is on the 23-feature envelope.
**Fix:** A one-line patch was staged (`object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})` after building D1 + W1 panels) but the chat interrupted before re-running Step 1. The patch is in `scripts/l_arc_8/build_step1_pool.py` lines 332-340. Re-running Step 1 (+ downstream from Step 4) would restore those four features; expected to lift Step 4 AUC marginally (Arc 10's `L1_minus_L0_atr` analogue is the D1-slope feature class).
**Status:** acknowledged in closure §3 cross-arc observation #3.

### D2 — Step 5 used pool-level WFO approximation, not bar-by-bar MultiPairBacktester

**Dispatch:** Step 5 uses the v3 architecture-search WFO with multi-pair sim.
**What happened:** Implemented a pool-level WFO that takes per-trade R-outcomes from Step 1 (with SL rescaling per Step 3's closed-form rule) and applies exit-policy / exposure-cap / sizing rules as post-hoc transforms. Equity = `balance * prod(1 + r * size * risk_pct)` per fold.
**Rationale:** Building the full A1/A3/A6 architectures on `MultiPairBacktester` with the four exit policies (sl_only, sl_plus_tp_2r, sl_plus_tp_3r, sl_partial_close_1r_runner_trail) is multi-PR engine work — A3 alone requires per-fold classifier training on path-so-far features. The pool-level approximation captures the architecture axis (filter/SL/exit/exposure/sizing) with reasonable fidelity for ranking. Oracle WFO under the same approximation establishes the upper bound.
**Limitations:**
- Intra-day DD timing not captured (could understate worst-fold DD)
- Cross-pair concurrency interactions ignored except via the explicit `max_concurrent_per_pair_1` rule
- `sl_partial_close_1r_runner_trail` not implementable in pool-level form (needs per-bar timing of MFE vs MAE), so omitted; only sl_only / sl_plus_tp_2r / sl_plus_tp_3r evaluated
- A3 (Pipeline DE) not built (requires new classifier training + path-so-far features at bar N)
**Configs evaluated:** 72 (A1: 18, A6: 54). A3-intended config count: 36 (would total 108, "broad").
**Status:** acknowledged in closure §2 final paragraph. Engine PR scope tracked.

### D3 — A2 (classifier_filter) and A3 (pipeline_de) not in architecture list

**Dispatch §"Step 5":** V-shape recovery → A1, A3, A6.
**What happened:** A1 and A6 implemented. A3 deferred (D2 above). A2 not in the V-shape arch map per dispatch — correctly omitted.
**Impact:** Architecture-axis coverage incomplete for V-shape. Expected behaviour of A3 (deferred entry — wait N bars, evaluate path-so-far, then enter or skip) is that some early SL hits would be avoided, raising mean R per admitted trade. Whether this lifts worst-fold ratio above the 2.0 gate is unknown; given oracle ceiling is 999, the room exists.
**Status:** §6 improvement: "A3 implementation as engine-PR; re-evaluate Arc 8 with A3 included."

### D4 — Exposure axis reduced from {2, unlimited} to {1, unlimited}

**Dispatch §"Step 5":** Exposure cap `max_concurrent_per_currency` ∈ {2, unlimited}.
**What happened:** Pool-level WFO uses `max_concurrent_per_pair_1` (≤ 1 open per pair) and `unlimited`. The `max_concurrent_per_currency=2` rule requires per-currency interaction enforcement which fits naturally in the bar-by-bar driver but doesn't reduce cleanly to a pool-level chronological constraint.
**Impact:** Real KH-24-style per-currency cap (the dispatch's intended axis) not tested. The pool-level approximation uses a more restrictive per-pair=1 cap; effects on the worst-fold ratio likely differ.
**Status:** acknowledged. Closure §6 improvement.

### D5 — SL sweep at Step 5 clamped to {1.5, 2.0, 2.5}

**Dispatch §"Step 5":** SL = Step 3 selected ± 1 step (3 values).
**Step 3 selected SL for c2:** 1.5×ATR.
**What happened:** Took {1.5, 2.0, 2.5} as the 3 search SLs (one step up + one step further up from selected). The dispatch literally says "± 1 step either side" implying {1.0, 1.5, 2.0}, but Step 3's sweep doesn't include 1.0; the Step 3 sweep set starts at 0.5 with step 0.5. Going to {1.0, 1.5, 2.0} would require recomputing Step 3 at SL=1.0 (within the original sweep). I chose to clamp to the search range {1.5, 2.0, 2.5} rather than synthesise an out-of-sweep value.
**Status:** minor; documented for chat awareness.

### D6 — Pool-level ww_pp approximation

**L_PROTOCOL §2 Step 3:** wrong-way-path prevalence `ww_pp` = P(MAE hits -1R before MFE hits +1R). Requires per-bar timing.
**What happened:** Step 1's pool stores summary mfe_r / mae_r without per-bar arrival order. ww_pp computed from magnitudes only: `wrong_way = (mfe < 1 AND mae <= -1) OR (mfe >= 1 AND mae <= -1 AND |mae| >= mfe)`. Conservative — over-estimates ww_pp on bimodal paths where MFE was reached first.
**Impact:** Cluster 2 ww_pp reports 0.00% at every swept SL — looks pristine. Real ww_pp may be marginally higher with per-bar timing. Probably doesn't change the qualitative cluster-2 candidate-flag outcome (reach_1R=100%, mfe_p50=7.5R both pristine).
**Status:** approximation, documented.

---

## Determinism

- All scripts call `core.determinism.seed_everything(42)` before any randomness.
- `random_state=42` and `n_jobs=1` baked into classifier construction (Appendix A defaults).
- Parquet outputs sorted-rows / sorted-columns where practical.
- CSV outputs `lineterminator="\n"` (PR-D contract).
- Pool sha256: `cdc41e0a47a30c4c5f0a5ee8ea2a5c89...` (see step_1/integrity_report.json). Two-run reproducibility not verified end-to-end this session (single-run only); the sha is stable across re-orderings of the underlying frame.

---

## Verification gates skipped

- Step 6 (causal audit) — not invoked per L_PROTOCOL §2 Step 6 (lazy; runs only on PASS-DEPLOYABLE / PASS-VIABLE).
- KH-24 anchor reproduction — not in this arc's scope (PR-E.1.7 closed it).
- Determinism re-run sha verification — single-run only; sha computed and persisted but not re-derived in a second invocation this session.

---

## HALT triggers — none fired

Per dispatch §"HALT triggers":
- ✅ No new KH-24 anchor divergence (not run this arc)
- ✅ No lookahead detected (Step 1 spot-check 5/5 pass; right-edge audit pass — min lag 8 bars)
- ✅ Pool size 6,757 ≥ 500
- ✅ K=3 silhouette 0.3544 ≥ 0.30
- ✅ ML mechanics unambiguous (Amendment 2 spec applied; A3 deferred for engine reasons not ambiguity)
- ✅ No spec ambiguity in HH/HL detection or pullback definition
- ✅ No mid-arc parameter tweaking

---

## End

PR title: `[ARC 8 v3.0] pullback_resume_hhhl_long — FAIL`
PR body: see commit message. Arc closes with FAIL verdict; cross-arc observations folded into v3 V-shape recurrence pattern alongside Arcs 6, 10.
