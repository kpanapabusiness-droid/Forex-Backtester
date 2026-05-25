# Arc 10 v3.0.2 — Intent Doc

> **Arc:** `l_arc_10_v3.0.2`
> **Branch:** `arc/l_arc_10_v3.0.2` (FRESH — cut from `origin/main@a8c02b4` 2026-05-25)
> **Signal:** D1 swing-low rejection long (DLR v0.1) — bullish rejection of confirmed ascending D1 swing-low, 4H entry
> **Sub-protocol:** vanilla
> **Boundary convention:** `5ers_eet` — LOCKED, NON-NEGOTIABLE, end-to-end at every subsystem
> **Closure target:** `results/l_arc_10_v3.0.2/ARC_CLOSURE.md`
> **Pre-execution status:** intent doc landed; awaiting chat approval before Step 1 compute

---

## §0 Why this arc exists

Arc 10 v3.0 closed PASS-VIABLE → Amendment-3 re-evaluated PASS-DEPLOYABLE under `boundary_convention="utc"`. Two reasons that verdict isn't a deployment decision:

1. **PR #193 audit (signal_module_eet_audit_2026_05.md):** Arc 10's `signals/lchar_dlr_long.py::_date_to_d1_index` was State B (silent same-EET-day D1 lookahead) under EET storage. Safe under the UTC verdict, but the deployment venue is 5ers MT5 — EET. The "would actually be deployed" methodology is EET.
2. **Prior `arc/l_arc_10_v3.0.2` (deleted 2026-05-25):** was run under UTC, produced byte-identical numbers to Arc 10 v3.0 (worst-fold ROI 26.49%, DD 9.22%, ratio 5.4185, holdout 59.07%/5.03%, n=3301). That confirms the engine refactor + signal-module canonicalisation are no-op under UTC. It does **not** test the production convention.

This arc is the wildcard of Wave 1 v3.0.2 — the cross-arc V-shape archetype hypothesis under canonical 5ers_eet.

The corrected dispatch's "byte-identical UTC result is methodologically irrelevant to a 5ers deployment decision" framing stands. The calibration doc `docs/calibration/arc_10_signal_parity_rerun_2026_05.md` ±2pp tolerances vs UTC original are obsolete under EET methodology.

---

## §0.5 Path A — Step 6 deferred

Step 6 framework still treats `features_in_winning_config: []` as critical failure (`core/step_6/lookahead.py:57-68` + `:119-128`). Arc 10's expected winning architecture is A1 with empty `features_in_winning_config` by design (rule-based; no classifier features). Auto-dispatch would false-flag.

Per resolution continuation: Step 6 patch (`engine/step_6_a1_vacuous_pass`) is in flight on a separate chat. This arc proceeds **under Path A**:
- Steps 1-5 complete under canonical 5ers_eet now
- Provisional closure with Step 6 deferred
- Manual CLI Step 6 run post-patch-merge via `python scripts/run_step_6.py results/l_arc_10_v3.0.2/`
- Addendum `ARC_CLOSURE_ADDENDUM.md` ships separately ~30 min after patch lands

L_PROTOCOL Amendment 4 explicitly sanctions manual CLI: *"manual CLI invokable on any closure"*. Deferring on patch-availability grounds is a legitimate use of the manual path, not a methodology variance.

---

## §1 Pre-flight gate (executed)

Verified on this worktree against `origin/main@a8c02b4`:

| Requirement | Status |
|---|---|
| PR #185 (Step 4 classifier persistence + holdout-window training filter) | ✓ on main |
| PR #186 (Amendment 3 risk-normalised gates) | ✓ on main |
| PR #188 (Step 6 framework + Amendment 4) | ✓ on main (framework wired; A1 vacuous-pass patch separate) |
| PR #189 (mid-price features + 5ers EET bar boundaries + worst-case fills) | ✓ on main |
| PR #193 (signal-level EET timezone alignment + canonical `htf_alignment.py`) | ✓ on main |
| PR #194 (Amendment 5 AUC-gated A2/A6 dispatch-time selection) | ✓ on main |
| PR #195 (canonical exit-policy registry + `sl_partial_close_1r_runner_trail`) | ✓ on main |
| PR #197 (EET session semantics — Amendment 6 daily-DD boundary) | ✓ on main |
| PR #201 (Amendment 5.1 — Gate 4 PASS-tier-constituent qualifier) | ✓ on main |
| PR #204 (backfill AMENDMENT_5 + 5.1 cutoff ISOs) | ✓ on main |
| PR #205 (Spread P&L decomposition diagnostic) | ✓ on main |
| Step 6 A1 vacuous-pass patch | **NOT landed — Path A: Step 6 deferred to manual CLI** |
| CC_20 follow-up rename (`core/utils/` → `core/time_utils/`) | ✓ on main (`88d44de`) |

Step 6 framework capability check intentionally skipped per resolution §2.1 — Step 6 will not auto-dispatch this turn.

---

## §2 Branch hygiene (executed)

- Prior `arc/l_arc_10_v3.0.2` (local + remote, head `ddc6420` UTC-contaminated WIP) **deleted** under user confirmation (2026-05-25). The branch had not been deleted prior to this dispatch despite both dispatches asserting otherwise — surfaced and resolved.
- Fresh `arc/l_arc_10_v3.0.2` cut from `origin/main@a8c02b4` and pushed with upstream tracking.
- Current worktree (`ecstatic-meninsky-bf1aa4`) is on the new branch.
- `arc/l_arc_10` (v3.0 original, UTC, PASS-DEPLOYABLE per Amendment 3 §10) preserved on main; informational reference only.

---

## §3 Files this dispatch will touch

### Net-new artefacts under `results/l_arc_10_v3.0.2/`
- `ARC_OPEN.md` — arc open record (per L_PROTOCOL §6)
- `step_1/pool.parquet` + `paths.parquet` + `feature_matrix.parquet` + `integrity_report.md` + `manifest.json`
- `step_2/cluster_assignments.parquet` + `cluster_summary.md` + `manifest.json`
- `step_3/capturability.csv` + `capturability_summary.md` + `manifest.json`
- `step_4/extraction_metrics.csv` + `feature_importance.csv` + `extraction_summary.md` + `classifiers/<cluster_id>.pkl` (per candidate cluster) + `classifiers/manifest.json` + `manifest.json`
- `step_5/wfo_results.csv` + `wfo_oracle.csv` + `architectures_ranked.md` + `best_candidate.md` + `per_day_max_dd_base__<safe_cid>.parquet` (Amendment 3 emission) + `manifest.json`
- `ARC_CLOSURE.md` — provisional, Step 6 deferred (see §6 of this intent)
- Cache directories `data/cache/4H_5ers_eet/`, `data/cache/D1_5ers_eet/`, `data/cache/W1_5ers_eet/` populated for the 28-pair set (build if cold)

### Modified
- `ARC_TRACKER.md` — closed-arc row appended via parser
- `scripts/tracker_parser/rolling_state.json` + `scripts/tracker_parser/parsed.log` — parser side-effects

### Net-new arc-runner scripts (if pattern matches Arc 5 v3.0.2 / Arc 11 v3.0.2)
- `scripts/l_arc_10_v3_0_2/step_1.py` … `step_5.py` (concrete shape decided at execution time; may instead invoke `ArcOrchestrator` directly)
- `configs/l_arc_10_v3.0.2/arc_open.yaml` — arc config (boundary_convention, pair_set, window, signal module, SL grid, exit policy grid)

---

## §4 Steps 1-5 plan with explicit `boundary_convention="5ers_eet"`

### Step 1 — Plumbing

- **Signal:** `signals/lchar_dlr_long.py` (DLR v0.1 — D1 swing-low rejection long), evaluated at H4 bar close on 28 pairs
- **Window:** 2010-01-01 → 2026-04-10 (matches Arc 10 v3.0 closure)
- **Boundary:** `boundary_convention="5ers_eet"` passed to `core.data.aggregator.aggregate` and threaded into `Panel.boundary_convention` for all subsequent slicing. Caches written to `data/cache/<TF>_5ers_eet/<PAIR>.parquet`.
- **D1 alignment:** canonical `core.signals.htf_alignment.get_htf_index_at(..., require_fully_closed=False)` for DLR's `_date_to_d1_index` (post-PR-#193). Under 5ers_eet this lands on different D1 closes than the UTC version for the same H4 timestamp — that is the load-bearing methodology change.
- **Pool:** `core.arc.arc_pool_builder.build_arc_pool` canonical, ex-ante construction, uncapped exposure at Step 1
- **Features:** `core.features.pipeline.compute_feature_matrix` — 27 features across 7 classes. All mid-price-anchored per PR #189 §15.1. D1-derived features (`L1_minus_L0_atr` family, D1 slope sign/magnitude, D1 ATR percentile) re-route through `get_htf_value_at(..., require_fully_closed=True)`.
- **Integrity report:** 5 of 6 protocol-listed checks via `core.arc.integrity` (pool_size, per_pair, coverage, lookahead-declared-lineage, determinism two-run). D1-lag NaN-perturbation + spread-floor + KH-24 co-fire by-hand at arc-script level if needed.

**Expected delta vs prior UTC run:** every D1-lookup-derived feature changes; pool composition may shift (different signal firings under EET D1 alignment); cluster geometry may shift. Methodology, not a bug.

### Step 2 — Clustering

- K ∈ {2, 3, 4, 5, 6}, KMeans (`random_state=42`, `n_init=10`)
- Silhouette selection
- Shape-tag assignment per `core.steps._shape_tags`
- Arc 10 v3.0 found K=3 with c1 V-shape n=1528 (46.3% of pool); under 5ers_eet, K and cluster geometry may differ. Report what emerges.

### Step 3 — Capturability

- Per-cluster reach_1R/2R/3R, MFE p25/p50/p75/p90, ww_pp, ttp distribution, mean R
- Capturability composite + candidate-cluster flag (`reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R`)
- SL multiplier sweep over {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR per cluster
- Arc 10 v3.0 c1 was selected SL=4.0 at Step 3 (UTC); under 5ers_eet may differ

### Step 4 — Extraction

- Per candidate cluster: RF + LGBM + LR at L_PROTOCOL Appendix A defaults
- 5-fold TimeSeriesSplit; per-fold AUC; threshold sweep (AUC-best via Youden's J)
- Permutation importance per feature
- Classifier persistence: `step_4/classifiers/<cluster_id>.pkl` with SHA256 + provenance manifest. `train_end` set to holdout `oos_start` (2021-01-01) per PR #185.
- **Arc 10 v3.0 c1 mean OOS AUC was 0.5199 under UTC.** Under 5ers_eet may be higher, lower, or the same. Architecture admission depends on this number per §5.

### Step 5 — WFO architecture search

**Architecture admission per L_PROTOCOL Amendment 5 four-gate (dispatch-time, Amendment 5.1 Gate 4 qualifier applies):**

Tentative plan (finalised once Step 2-4 produce actual cluster set + AUC):

| Cluster (expected) | Archetype | Step 4 AUC under EET | Gates that fire | Architectures admitted |
|---|---|---|---|---|
| c1 (V-shape, dominant) | V-shape recovery | TBD | Gate 1 V-shape → A3; Gate 3 → A1; **Gate 2 if AUC ≥ 0.65 → A2, A6** | Minimum: A1, A3. If AUC ≥ 0.65 also A2, A6. |
| c0 | TBD (Arc 10 v3.0: monotonic_down, dies_step3) | n/a (not candidate) | Gate 3 → A1 only if candidate-flag fires (which under v3.0 it didn't) | Likely skip. |
| c2 | TBD (Arc 10 v3.0: monotonic_down outlier n=2) | n/a | n/a | Skip. |
| Gate 4 (A5) | Portfolio | requires ≥2 candidate clusters AND ≥1 PASS-tier constituent | Almost certainly does not fire (Arc 10 expected single candidate c1) | Not admitted; record `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` only if (a) holds but (b) doesn't. If (a) doesn't hold, the field is empty. |

**Skipped-architecture accounting under Amendment 5.1:**
- Arc 10 v3.0 baseline tested {A1, A3, A6} per archetype rule. Under Amendment 5 Gate 2, A6 admission is now conditional on AUC ≥ 0.65 (not archetype). If Step 4 c1 AUC < 0.65 under EET, A6 is **skipped under Amendment 5** and `architectures_skipped_by_amendment_5` records `[A6]`. A2 was never admitted under Amendment 1 (archetype rule didn't fire it for V-shape) and isn't admitted under Amendment 5 either when AUC < 0.65.

**Search grid per admitted architecture:**

- **SL multiplier:** centred on Step 3 cluster optimum ±1 step. Arc 10 v3.0 selected SL=4.0 at Step 3 (UTC); winning A1 ran at SL=3.5. Under 5ers_eet Step 3 may pick a different SL — grid recentred accordingly.
- **Exit policy (V-shape canonical slate per L_PROTOCOL §2 Step 5):** `{sl_only, sl_plus_tp_2r, sl_partial_close_1r_runner_trail}`. Add `sl_plus_trailing_atr` only if Step 3 selection suggests (Arc 10 v3.0 did not).
- **Exposure cap:** {2, unlimited}
- **A3 deferred bars:** {3, 5}
- **A6 threshold pairs (if admitted):** {(0.3, 0.5), (0.4, 0.6), (0.5, 0.7)}

**WFO:**
- 11-fold IS 2010-2020, anchored expanding
- Holdout 2021-01-01 → 2026-04-10, one-shot per top-K candidate
- `r_base = 0.5%`
- Amendment 3 emissions: per-day max-DD parquet, chained DD via equity-stitching (v3.0.1 default), holdout re-runs at `r_safe` / `r_hard`
- **Amendment 6 EET daily-DD boundary NATIVE** — `compute_per_day_max_dd(boundary_convention="5ers_eet")` matches the panel boundary (this is the coherent gate Amendment 6 was designed for)
- **Oracle WFO** per cluster — locked to `sl_only` at Step-3 best SL for cross-arc consistency (the Arc 10 v3.0 oracle/winner exit-policy mismatch is documented; preserved here for cross-arc table integrity)

**Top-K decision:**
- Top-3 by worst-fold ratio across all admitted architectures × clusters → 2021-2025 holdout one-shot

### Step 6 — Causal audit

- **Auto-dispatch DEFERRED** under Path A.
- Closure §1 `tracker_payload.step_6` block records `ran: false`, `trigger: deferred_pending_framework_patch_engine_step_6_a1_vacuous_pass`, `manual_cli_pending: true`.
- Post-patch-merge addendum (separate task ~30 min): run `python scripts/run_step_6.py results/l_arc_10_v3.0.2/`, write `ARC_CLOSURE_ADDENDUM.md` with full §6.1-§6.6 result, update step_6 block to `ran: true`, verdict amendment per Amendment 4 if critical failure.

---

## §5 Architecture admission per Amendment 5 / 5.1 — dispatch-side preamble

Arc 10 v3.0.2 dispatch preamble (this intent doc, satisfying L_PROTOCOL §2 Step 5 "Architecture selection (Amendment 5)" implementation discipline item #3):

**Cluster c1 (V-shape recovery, expected dominant cluster) — admission rationale:**
- Gate 1 (Shape-required, archetype-driven, V-shape → A3): admit **A3**
- Gate 2 (Classifier-driven, AUC-driven): admit **A2 + A6 iff Step 4 c1 mean OOS AUC ≥ 0.65**. Citation locked dispatch-time once Step 4 lands.
- Gate 3 (Universal): admit **A1**
- Gate 4 (Portfolio): does not fire on single candidate cluster (Arc 10 expected single c1)
- **Choppy skip:** N/A — c1 is V-shape, not Choppy

**Step 4 AUC summary (preamble — final values backfilled at Step 4 close):**
- c1 mean OOS AUC under 5ers_eet: TBD (Arc 10 v3.0 UTC reference: 0.5199 LGBM)

**Architectures-skipped accounting (preamble):**
- If c1 AUC < 0.65 under EET: `architectures_skipped_by_amendment_5: [A6]` (A6 was admitted under Amendment 1's V-shape rule but is now AUC-gated). Note: Arc 10 v3.0 closure explicitly tested A6 under the prior rule and reports it as Top-2 (not Top-1) — same architecture would be skipped under Amendment 5 if EET AUC remains at chance.
- If c1 AUC ≥ 0.65: `architectures_skipped_by_amendment_5: []`. Methodologically interesting if EET D1 alignment lifts AUC above the UTC ceiling.

---

## §6 Compute estimate

- **Step 1 cache-cold build** for 5ers_eet panels: ~15-30 min wall-clock for 28-pair × M5/M15/M30/H1/H4/D1/W1 × 2010-2026 if caches don't exist. Mitigated if some pairs/TFs already cached.
- **Steps 2-4:** ~30 min combined on the 28-pair pool (Arc 10 v3.0 reference ≈ 25 min).
- **Step 5 WFO:** dominant cost. Per-architecture × per-config × 11-fold IS + 1 holdout. Arc 10 v3.0 evaluated 96 configs in ~3-4 hr; under Amendment 5 admission (A1 + A3, plus A2 + A6 if AUC clears) and 4-6 SL × 3-4 exits × 2 exposure × cluster set, expect ~50-150 configs depending on admission. **Estimated 4-10 hr.**
- **Total wall-clock:** ~6-12 hr including cache-cold panel build.

Cache-cold 5ers_eet panel build is a one-time cost and is methodologically required — UTC caches don't substitute.

---

## §7 Closure plan — provisional, Step 6 deferred

### §7.1 §1 tracker_payload.step_6 block (deferred shape)

```yaml
step_6:
  ran: false
  trigger: deferred_pending_framework_patch_engine_step_6_a1_vacuous_pass
  manual_cli_pending: true
  overall_passed: null
  manifest_path: null
  categories:
    lookahead: null
    selection_bias: null
    execution_realism: null
    statistical: null
    determinism: null
    deployment_readiness: null
  critical_failures: []
  warnings_count: 0
  verdict_impact: none
  notes: |
    Step 6 auto-dispatch deferred. Framework patch
    (engine/step_6_a1_vacuous_pass) in flight separately;
    addresses the false-positive on A1's by-design empty
    features_in_winning_config. Step 6 will be run via
    manual CLI post-patch-merge; closure addendum will
    update this block + §10.
```

### §7.2 §2 prose verdict — PROVISIONAL framing

Whatever Step 5 produces (PASS-DEPLOYABLE / PASS-VIABLE / FAIL), tag PROVISIONAL:

> Verdict provisional pending Step 6 manual CLI audit (deferred per framework patch in flight). Per L_PROTOCOL Amendment 4 §"Discipline rules", manual CLI does not modify verdict but produces audit record. If Step 6 surfaces critical failure post-merge, verdict subject to amendment via closure addendum.

For PASS verdicts, the template-v1.3.1 `verdict` field uses the `-PROVISIONAL` suffix variant (e.g. `PASS-DEPLOYABLE-PROVISIONAL`); for FAIL, no suffix needed (Step 6 cannot upgrade FAIL).

### §7.3 §3 Cross-arc observations — V-shape archetype hypothesis MANDATORY

The cross-arc V-shape archetype hypothesis is THE wildcard test for this arc:
- Arc 7 c1/c3 V-shape; Arc 10 v2.3 c1 V-shape; Arc 10 v3.0 c1 V-shape — three pre-EET instances
- Arc 10 v3.0.2 c1 V-shape (if it surfaces) — first EET-canonical instance
- Result interpretation framed against this lineage

Explicit note on 5ers_eet vs UTC methodology delta: the prior `arc/l_arc_10_v3.0.2` UTC run was byte-identical to original; this run's deltas are the diagnostic.

### §7.4 §10 retroactive re-evaluation

Quantitative comparison vs Arc 10 v3.0 original (UTC):
- Frame as "the prior UTC run was byte-identical to original; the EET run is the actual canonical test"
- Deltas vs original under EET are the diagnostic finding
- `docs/calibration/arc_10_signal_parity_rerun_2026_05.md` ±2pp tolerances explicitly marked obsolete under 5ers_eet methodology
- §10 conclusion deferred pending manual Step 6 lands (full §10 narrative finalised in addendum)

### §7.5 §1 architectures_skipped_by_amendment_5

Per Amendment 5.1:
- Single candidate cluster expected (c1 V-shape) → Gate 4 doesn't fire → no A5 entry
- A6 admission per actual Step 4 AUC under EET (see §5 above)

### §7.6 Tracker delta + commit + PR

Standard arc-close artefact set per WORKFLOW §2:
1. Write closure + run parser (`python scripts/update_tracker_from_closure.py results/l_arc_10_v3.0.2/ARC_CLOSURE.md`)
2. Atomic commit on `arc/l_arc_10_v3.0.2`: closure doc + ARC_TRACKER.md + rolling_state.json + parsed.log
3. PR to main with description including:
   - Methodology note: Steps 1-5 under canonical 5ers_eet; Step 6 deferred to post-framework-patch manual CLI
   - Reference: `engine/step_6_a1_vacuous_pass` (Step 6 patch branch on parallel chat)
   - Reference: this intent doc

---

## §8 HALT triggers

Standard set, modified per resolution §6:

- **Removed:** "ROI/DD delta >5pp vs UTC original" trigger. Under EET, large deltas are EXPECTED and CORRECT. No HALT on EET-vs-UTC differences.
- **Removed:** Step 6 framework capability check (Path A deferral). Step 6 will not auto-dispatch this turn.
- **Retained:** anchor reproduction failures, ambiguous diagnostic bisects, lookahead detected mid-arc, ex-ante construction violations, sizing-convention violations, determinism failures, parser HALT on closure validation.
- **Soft surface (continue arc):** Step 1 pool size or Step 2 cluster geometry diverging dramatically from any prior reference — surface in respective summary doc but continue, these deltas are methodologically expected under different bar boundaries.

On HALT: no PR; branch pushed; diagnostic doc at `docs/dispatches/arc_10_v3_0_2_diagnostic.md`; end turn.

---

## §9 Definition of done

This turn (intent doc):
- [x] §1 pre-flight (minus Step 6 framework check) PASS
- [x] Fresh `arc/l_arc_10_v3.0.2` cut from current main + pushed
- [x] Intent doc written under 5ers_eet locked, Step 6 deferred
- [ ] END TURN for chat review **← stopping here**

Post intent-doc approval (separate turn):
- [ ] Steps 1-5 under 5ers_eet end-to-end
- [ ] Provisional closure per Step 5 outcome
- [ ] Parser invocation + atomic tracker-delta commit
- [ ] PR opened with methodology + Step 6 deferral note

Post Step 6 patch lands (separate addendum task, ~30 min):
- [ ] Merge main into arc branch (or fresh addendum branch)
- [ ] `python scripts/run_step_6.py results/l_arc_10_v3.0.2/`
- [ ] `results/l_arc_10_v3.0.2/ARC_CLOSURE_ADDENDUM.md` written
- [ ] Updated `step_6` block (`ran: true`, full audit outcome) + verdict amendment if critical failure + updated §10 conclusion if Step 6 changes anything
- [ ] Atomic addendum + tracker delta commit + PR

---

## §10 Interpretive calls flagged for chat

1. **Confirm Step 5 search grid is exactly:** SL ∈ {Step-3-best ±1 step, 3 values} × Exit ∈ {`sl_only`, `sl_plus_tp_2r`, `sl_partial_close_1r_runner_trail`} (+ `sl_plus_trailing_atr` only if Step 3 suggests) × Exposure ∈ {2, unlimited}. Architectures admitted per Amendment 5 four-gate finalised dispatch-time once Step 4 AUC lands. **Question: any expansion or contraction wanted?**
2. **Oracle WFO lock to `sl_only` at Step-3 best SL** — keeps cross-arc oracle table consistent with Arc 10 v3.0 framing (which has the known oracle-vs-winner exit-policy mismatch caveat documented in `results/l_arc_10/ARC_CLOSURE.md` §3). **Question: keep this convention or switch oracle to match the winning architecture's exit?** (Recommendation: keep for table consistency; the asymmetry is already documented as an open methodology item.)
3. **No comparison delta-doc this run.** The corrected dispatch removed the UTC-vs-UTC delta trigger; closure §10 narrates the EET-vs-UTC delta inline rather than via a separate `docs/calibration/arc_10_v3_0_2_eet_vs_utc_delta.md` artefact. **Question: chat want a separate calibration doc anyway, or fold it into §10?**

---

End of intent. Ending turn for chat review.
