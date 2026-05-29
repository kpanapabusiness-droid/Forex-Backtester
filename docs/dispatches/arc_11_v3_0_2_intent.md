# Arc 11 v3.0.2 — Intent Doc

> **Dispatch:** `DISPATCH_arc_11_v3_0_2.md`
> **Arc:** `l_arc_11_v3.0.2`
> **Signal:** swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge `t-4`)
> **Branch:** `arc/l_arc_11_v3.0.2` (cut from `origin/main@103f7d3`; pushed)
> **Sub-protocol:** vanilla
> **Closure target:** `results/l_arc_11_v3.0.2/ARC_CLOSURE.md`
> **Verdict prior:** almost certainly FAIL (re-confirmed below).

---

## §0 Pre-flight + branch hygiene status

All §1 / §2 checks PASS:

| Check | Result |
|---|---|
| PRs on `origin/main`: #185, #186, #188, #189, #190, #193, #194, CC_18 (#195), CC_20 (#197) | ✅ all present (latest `103f7d3`) |
| `core/arc/arc_orchestrator.py` `run_context` plumbed to A2/A6 | ✅ wired (PR #186 — verified via grep: lines 510, 712, 743, 795) + engine_capability_audit_2026_05.md (§"Step 5 — Architectures" A2/A6 status WIRED) |
| `from core.sim.exit_policies import build_exit_policy, available_policies` + `sl_partial_close_1r_runner_trail` present | ✅ CC_18 OK |
| `compute_per_day_max_dd` + `amended_gates.classify_amended_fold_stats` + `core.time_utils.session_boundary.SUPPORTED_CONVENTIONS == ('utc', '5ers_eet')` | ✅ CC_20 OK |
| `core.signals.htf_alignment.{get_htf_value_at, get_htf_row_at, get_htf_index_at}` | ✅ PR-193 OK |
| Branch `arc/l_arc_11_v3.0.2` cut from `origin/main`, pushed to origin | ✅ |
| `arc/l_arc_11` historical branch untouched | ✅ |

Cross-arc tag `canonical_orchestrator_step5_run_context_gap` (Arc 11 original §3) is materially closed by PR #186 per the engine audit. v3.0.2 runs Step 5 through the canonical `ArcOrchestrator._run_step_5` path; the inline-driver bypass that Arc 11 v3.0 used is no longer needed.

---

## §1 File paths to be touched

### New files (worktree write)

- `results/l_arc_11_v3.0.2/ARC_OPEN.md` — arc open, per L_PROTOCOL §6 required fields
- `results/l_arc_11_v3.0.2/step_1/` — `pool.parquet`, `paths.parquet`, `integrity_report.md`, `manifest.json`, plus feature-matrix parquet (orchestrator emits)
- `results/l_arc_11_v3.0.2/step_2/` — `cluster_assignments.parquet`, `cluster_summary.md`, `manifest.json`
- `results/l_arc_11_v3.0.2/step_3/` — `capturability.csv`, `capturability_summary.md`, `manifest.json`
- `results/l_arc_11_v3.0.2/step_4/` — `extraction_metrics.csv`, `feature_importance.csv`, `extraction_summary.md`, `classifiers/<cluster>.pkl` + `classifiers/manifest.json`
- `results/l_arc_11_v3.0.2/step_5/` — `wfo_results.csv`, `wfo_oracle.csv`, `architectures_ranked.md`, `best_candidate.md`, `per_day_max_dd_base__<safe_cid>.parquet` per top-K, `manifest.json`
- `results/l_arc_11_v3.0.2/step_6/` — only if Top-1 clears §3 constraints #1-9 (Amendment 4 auto-dispatch)
- `results/l_arc_11_v3.0.2/ARC_CLOSURE.md` — per template **v1.3.1** with full §1 tracker_payload + §2 prose + §3 cross-arc + §4 deployment_spec (abbrev. on FAIL) + §10 retroactive re-eval (mandatory)
- `configs/l_arc_11_v3.0.2/winning_config.yaml` — best architecture config (FAIL: written for documentation parity; PASS: required)
- `docs/dispatches/arc_11_v3_0_2_intent.md` (this file)
- `docs/dispatches/arc_11_v3_0_2_log.md` — produced at execution end
- Tracker delta: `ARC_TRACKER.md`, `scripts/tracker_parser/rolling_state.json`, `scripts/tracker_parser/parsed.log` (parser-driven; atomic with closure commit)

### Read-only (referenced)

- `signals/lchar_swing_high_breakout_trend.py` — signal module (State A per `docs/audits/signal_module_eet_audit_2026_05.md`; no fix needed)
- `core/arc/arc_orchestrator.py`, `core/steps/`, `core/architectures/`, `core/runners/`, `core/sim/exit_policies/`, `core/time_utils/session_boundary.py`, `core/wfo/amended_gates.py`
- `results/l_arc_11/ARC_CLOSURE.md` — preserved; read for §10 retroactive comparison
- Scripts harness — likely a new `scripts/l_arc_11_v3_0_2/run.py` mirroring Arc 11's `scripts/l_arc_11/run.py` shape but **using the canonical `ArcOrchestrator` path now that the `run_context` gap is closed** (no manual `A1RunContext` construction needed)

### Explicitly NOT touched

- `arc/l_arc_11` (historical branch, absent from remote per audit; not checked out)
- `results/l_arc_11/` (historical closure; read-only)
- `main` (worktree on `arc/l_arc_11_v3.0.2` throughout)
- `time_exit_n_bars` exit policy (not registered per CC_18; not needed — Arc 11's existing `sl_only` + `hold_bars=240` at pool builder satisfies time-exit semantics)

---

## §2 Steps 1-5 plan

### Step 1 — Plumbing

- Signal: `lchar_swing_high_breakout_trend.py` (SHB long, 4H, causal 3-bar swing right-edge `t-4`); State A — no signal-module changes.
- Population: canonical `build_arc_pool` via `core/arc/arc_pool_builder.py`. **Uncapped pool by construction** — confirms the canonical convention that exposure caps are applied at Step 5 architecture level, not Step 1. Arc 11 v3.0's tag `step1_pool_uncapped_canonical_vs_capped_handrolled_2_5x_delta` documented a 2.5× delta vs hand-rolled capped pool; v3.0.2 expects the same uncapped pool size (~17,533 trades) baseline. Pool-size delta vs v3.0 surfaced in `step_1/integrity_report.md` (expected ≤ ±5% from v3.0 canonical 17,533).
- Pairs: 28 (KH-24 set).
- Window: 2010-01-01 → 2026-04-30 (matches v3.0; reuses panel cache).
- SL anchor: `2.0 × ATR(14)` at signal bar (v3.0.2 carries Step 3 SL sweep at architecture level; pool builder anchor remains canonical).
- `hold_bars=240` (time-exit horizon; not a policy registry entry).
- Feature matrix: canonical 27-feature v3.0 default — `core/features/pipeline.py` 7 classes (price_geometry, distance, multi_tf, cross_pair, spread_regime, session, vol_regime).
  - **Mid-anchored features per PR #189 §15.1** — automatic via the canonical feature pipeline (Arc 11 v3.0 ran pre-PR-#189, so feature values will drift from v3.0).
  - **EET HTF-alignment per PR #193** — `core/features/multi_tf.py` `_build_d1_lag1_series` uses canonical `get_htf_value_at(...)` (audit confirms WIRED). For Arc 11 specifically, the signal module itself is single-TF H4 (State A) so signal-bar timestamps are unaffected; the feature-matrix HTF lookups (D1 slope sign/magnitude, ATR percentile, W1 slope sign) benefit from the canonical alignment.
- `boundary_convention`: **`5ers_eet`** for both panels and downstream EET-session bucketing (Amendment 6 / PR #197). Per the orchestrator wiring (PROTOCOL_RUNTIME §15.5), `Panel.boundary_convention` carries the choice through fold-slicing and into `compute_per_day_max_dd`. Cache namespace: `data/cache/<TF>_5ers_eet/<PAIR>.parquet`.
- Integrity report: emits standard 5 checks per the engine audit (`pool_size_min`, `per_pair_below_warn`, `per_pair_zero_trades`, `coverage_window`, `lookahead_declared_lineage`, `determinism_two_run`). The three not-yet-engine-wired checks (D1-lag NaN-perturbation, spread-floor activation, KH-24 co-fire) will be flagged as informational in the closure (matches Arc 11 v3.0).
- Determinism: `seed_everything(42)`; pool sha256 reproducible across two runs.

**Expected delta from Arc 11 v3.0:**
- Pool count: stable (~17,533) — uncapped canonical builder unchanged.
- Feature values: drift expected on every mid-derived feature (PR #189 §15.1 swap from close → mid) and every D1/W1-lagged feature (PR #193 canonical HTF alignment under EET storage).
- Cluster geometry: may shift modestly. Bimodal recurrence to be verified per §7 HALT trigger.

### Step 2 — Clustering

- `core/steps/step_2_clustering.py` — KMeans over K ∈ {2, 3, 4, 5, 6}, silhouette selection, deterministic.
- v3.0 selected K=4 (c0 Bimodal n=2,192; c1 Unclassified n=6,287; c2 Unclassified n=5,032; c3 Monotonic_down n=4,022).
- Verify Bimodal recurrence. If silhouette tie shifts K (or if Bimodal collapses into another archetype), surface in `step_2/cluster_summary.md` and continue — Arc 11's Bimodal had a documented silhouette-tie historically.
- HALT triggers § §7 below.

### Step 3 — Capturability

- `core/steps/step_3_capturability.py` — reach_1R/2R/3R, MFE p25/p50/p75/p90, ww_pp, ttp p25/p50/p75, mean_R, capturability composite (0.40·reach_1R + 0.40·(mfe_p50/3) + 0.20·(1−ww_pp)).
- Candidate-cluster flag: `reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R`.
- Per-cluster SL sweep {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}; argmax composite selects each cluster's preferred SL.
- v3.0 baseline: c0 Bimodal composite 1.98 (mfe_p50 7.77, ww_pp 0.0018, reach_1r 1.00, sl_atr 1.5); c1 Unclassified composite 0.97 (mfe_p50 2.11, ww_pp 0.035, reach_1r 0.965, sl_atr 1.5). Both candidate.
- v3.0.2 expectation: both candidates likely survive given the strong baseline metrics. Step 4 admission for Step 5's A5 portfolio gate depends on **two clusters surviving §3 capturability** — likely fires (see §3 below).

### Step 4 — Extraction

- `core/steps/step_4_extraction.py` — RF + LightGBM + LogisticRegression at L_PROTOCOL Appendix A defaults; 5-fold TimeSeriesSplit; AUC + AUC-best threshold sweep; permutation importance.
- **Lineage filter** WIRED post-PR-#185 — features tagged non-`clean` are excluded from training. Arc 11 v3.0 pre-dated the column-name reconciliation; v3.0.2 enforces canonically.
- **Holdout-window training filter** WIRED post-PR-#185 — `train_end` threaded from `WfoStructure.holdout.oos_start` (2021-01-01) restricts CV + refit to IS-only.
- **Classifier persistence** WIRED post-PR-#185 — best-AUC classifier per candidate cluster pickled to `step_4/classifiers/<cluster_id>.pkl` with sha256 + provenance manifest. A2/A6 consume via `build_a2_config_from_step4` / `build_a6_config_from_step4` with no Step 5 retrain (Amendment 2 lock).
- v3.0 baseline AUCs (likely to drift modestly under v3.0.2 mid-feature + EET-aligned HTF feature space):
  - c0 Bimodal E AUC 0.6543 (clears 0.65 by 0.43pp)
  - c1 Unclassified E AUC 0.6316 (below 0.65 by 1.84pp)
- v3.0.2 expectation: under mid-features + canonical HTF-aligned D1 features, c0 AUC may drift either side of 0.65; c1 unlikely to clear without a material lift. Re-derived numbers populate the actual Amendment 5 admission decision at Step 5 dispatch.

### Step 5 — WFO architecture search

**Architecture admission table (Amendment 5 four-gate, EXPECTATION based on v3.0 AUCs; re-derived from v3.0.2 Step 4 outputs at dispatch time):**

| Cluster | Archetype | E AUC (v3.0) | Gate 1 | Gate 2 (AUC ≥ 0.65) | Gate 3 | Gate 4 (≥2 candidates) | Admitted set |
|---|---|---|---|---|---|---|---|
| c0 | Bimodal | 0.6543 | A4 (Bimodal → A4) | A2, A6 (0.6543 ≥ 0.65) | A1 | (deferred per A5.1) | **A1, A2, A4, A6** |
| c1 | Unclassified | 0.6316 | (none; Unclassified) | (none; 0.6316 < 0.65) | A1 | (deferred per A5.1) | **A1** |
| c2 | Unclassified | n/a (dies §3) | n/a | n/a | n/a | n/a | (skip; not candidate) |
| c3 | Monotonic_down | n/a (dies §3) | n/a | n/a | n/a | n/a | (skip; not candidate) |

**Citations:** L_PROTOCOL §2 Step 5 "Architecture selection (Amendment 5)" + `archive/L_PROTOCOL_v3_0_AMENDMENT_5.md` §2 (four-gate procedure). Gate 1 Bimodal→A4 per the worked-example table; Gate 2 0.65 threshold matches Amendment 1's classifier deployability bar; Gate 3 universal.

**Amendment 5.1 (merged 2026-05-25 via PR #201) revises Gate 4:** A5 admission now requires (a) ≥2 candidate clusters AND (b) ≥1 constituent cleared Step 5 search-WFO at PASS-DEPLOYABLE / PASS-VIABLE under Gates 1-3. Condition (b) is evaluated post-Step-5, so A5 cannot be admitted at dispatch time. Arc 11 v3.0.2's closure records this in `architectures_skipped_by_amendment_5` as `a5_gate_4_admission_blocked_by_no_pass_tier_constituent`. Per A5.1, if Top-1 surprises PASS-tier, A5 admission may be added as a closure addendum without re-running the arc.

**`architectures_skipped_by_amendment_5`:** if c1's v3.0.2 AUC re-derives below 0.65, then A2 and A6 on c1 are skipped per the four-gate rule. Under the *prior* Amendment 1 rule (uniform archetype-driven gating), A2/A6 would have been considered for any cluster — so c1's A2 + A6 are recorded in this field for cross-arc analytics. (For c0: A2/A6 are not skipped if AUC clears; the field captures only the genuinely-skipped-by-A5 set.)

**Step 5 search dimensions (admitting cluster × architecture):**

| Dim | Values | Source |
|---|---|---|
| Architecture set | per admission table above | Amendment 5 |
| Exits | `sl_only`, `sl_plus_tp_2r`, `sl_partial_close_1r_runner_trail` | L_PROTOCOL §2 Step 5 "Exit policy" (Bimodal → partial-close + TP-2R; always `sl_only` baseline). For c1 Unclassified (no archetype-driven exit slate), defaults to `sl_only` + the same two for cross-cluster comparability — surfaced in dispatch for selection-bias accounting. |
| SL multiplier | per-cluster Step 3 optimum ±1 step (3 values total) | L_PROTOCOL §2 Step 5 "SL multiplier" |
| Exposure cap | `max_concurrent_per_currency ∈ {2, unlimited}` | L_PROTOCOL §2 Step 5 "Exposure cap" |
| WFO | 11-fold 2010-2020, `r_base = 0.5%`, holdout 2021-01-01 → arc-close timestamp | L_PROTOCOL §2 Step 5 "WFO structure" |
| Pipeline DE bars (A3, if admitted by future re-derived AUC — currently not admitted) | n/a | L_PROTOCOL §2 Step 5 |
| A6 thresholds | `{(0.3, 0.5), (0.4, 0.6), (0.5, 0.7)}` | L_PROTOCOL Amendment 2 |

**Per-architecture config count estimate (post-Amendment-5.1, assuming v3.0 AUCs hold):**
- A1 (Gate 3 universal; runs once at arc level, NOT per cluster — A1 has no classifier-driven cluster filter): 3 SLs × 3 exits × 2 exposure = **18**
- A2 c0 (Gate 2; auto_arch_specs with builder_kwargs sweeping SL/exit/exposure × default threshold): 3 SLs × 3 exits × 2 exposure = **18**
- A4 c0 (Gate 1 Bimodal; auto_arch_specs with builder_kwargs sweeping SL/exposure × Bimodal exit slate × exit_threshold): 3 SLs × 2 exits × 2 exposure × 3 exit_thresholds = **36**
- A6 c0 (Gate 2; auto_arch_specs with builder_kwargs sweeping SL/exit/exposure × default threshold pair): 3 SLs × 3 exits × 2 exposure = **18**
- A5: deferred per Amendment 5.1 — recorded in `architectures_skipped_by_amendment_5` with reason `a5_gate_4_admission_blocked_by_no_pass_tier_constituent`. Closure addendum if Top-1 surprises PASS-tier.
- **Total per-arc ≈ 90 configs** → `search_scope_flag: normal` (per template Section 4: 50-99 normal).

If c1 v3.0.2 AUC re-derives ≥ 0.65 (currently 0.6316 < 0.65; unlikely but possible), add A2 c1 (18) + A6 c1 (18) = 36 → total ~126 → `broad` scope.

**Amendment 3 emissions per top-K:**
- `per_day_max_dd_base__<safe_cid>.parquet` (EET-bucketed via PR #197 default).
- `chained_max_dd_base_pct` via equity stitching (`chained_dd_method: equity_stitching`; the v3.0.2 full-window sim follow-up is documented as deferred in PROTOCOL_RUNTIME §8b).
- Holdout re-runs at `r_safe` and `r_hard` per top-K (priority-ordered failure mode taxonomy at `core/wfo/amended_gates.py`).

**Oracle WFO** per cluster — Arc 11 v3.0 post-closure off-protocol oracle showed c1 raw PASS at +101%/yr ROI / 2.48% DD. v3.0.2 oracle WFO reproduces this number → re-confirms "capturable-not-extractable" diagnosis (paired with Arc 7 c0 Bimodal v3.0; Arc 7 v3.0.2 c1 Unclassified in flight).

**Top-3 by worst-fold ratio → 2021-2025 holdout** one-shot per L_PROTOCOL §2 Step 5 "Holdout decision rule".

### Step 6 — Causal audit (Amendment 4)

Auto-dispatches only if Top-1 clears §3 constraints #1-9 (`_run_amendment_3_evaluation` produces PASS-DEPLOYABLE / PASS-VIABLE). Given the verdict prior (FAIL), Step 6 is unlikely to trigger. If it does, the framework runs the six categories via `core/step_6/` and writes `results/l_arc_11_v3.0.2/step_6/`.

---

## §3 Architecture-admission table under Amendment 5

(see Step 5 table above for the full per-cluster breakdown including admitting gate citations)

**Specific decision call requested by dispatch §4 item 3:**

- **c0 Bimodal E AUC 0.6543 ≥ 0.65 (canonical Amendment 5 Gate 2 threshold)** → A2 and A6 ADMITTED for c0. Citation: `archive/L_PROTOCOL_v3_0_AMENDMENT_5.md` §2 Gate 2 ("If Step 4 mean OOS AUC ≥ 0.65 for the cluster: add A2 and A6.") + L_PROTOCOL §2 Step 5 "Architecture selection (Amendment 5)" Gate 2 bullet ("Gate 2 — Classifier-driven (AUC-driven): if Step 4 mean OOS AUC ≥ 0.65 for the cluster → add A2 and A6. Fires REGARDLESS of archetype.").
- **c1 Unclassified E AUC 0.6316 < 0.65** → A2 and A6 NOT ADMITTED for c1 under Amendment 5 Gate 2. Citation: same. The dispatch flagged this as "NEAR the Amendment 5 gate" — confirmed below threshold by 1.84pp under v3.0 AUC. v3.0.2 Step 4 may re-derive a different AUC; admission re-evaluated at v3.0.2 Step 4 outputs.

**Compatibility:** these are the *v3.0 Step 4 AUCs*. v3.0.2 mid-feature + EET-aligned HTF feature space will produce *new* AUCs. The admission decision is operationally made at v3.0.2 Step 4 outputs, not from v3.0 numbers. The table above is the projection; the actual `architectures_skipped_by_amendment_5` field in the closure is populated from v3.0.2 actuals.

---

## §4 Step 4 AUC summary plan (re-derived under v3.0.2)

Per dispatch §4 item 4:

1. **Mid features** alter classifier inputs vs Arc 11 v3.0 (which used close-based features pre-PR-#189). Direction unpredictable — some features may sharpen (less bid-ask noise), others may dull (lost asymmetry signal).
2. **EET HTF alignment** is no-op for the SHB signal itself (signal module is single-TF H4, State A per the audit). For the **feature matrix**, multi_tf.py D1-lag-1 features (D1 slope sign / magnitude, D1 ATR percentile, W1 slope sign) now use canonical `get_htf_value_at(..., require_fully_closed=True)`. Under UTC convention these are byte-identical to legacy; under the v3.0.2 EET convention they pick the *correct* prior-EET-day D1 (Arc 11 v3.0 ran on UTC bars; v3.0.2 runs on EET bars — feature values differ at every D1-lag-1 lookup).
3. **Canonical run_context plumbing means A2/A6 train cleanly on c0/c1.** Arc 11 v3.0 surfaced `canonical_orchestrator_step5_run_context_gap` because A2 was wired via inline-driver bypass with manually constructed `A1RunContext`. v3.0.2 runs through `ArcOrchestrator._run_step_5` end-to-end — A2/A6 admit gates fire canonically per PR #186.

**Re-derived AUCs to be reported in closure §1 `clusters.<c>.step4_e_auc` per cluster.** Top-feature importance ranking from `step_4/feature_importance.csv` cited in §3 cross-arc observations (compare against v3.0's top-10 to track feature-space stability).

---

## §5 Verdict prior

**Almost certainly stays FAIL.** Structural drivers from Arc 11 v3.0:

- Worst-fold ratio −0.7687 (well below +2.0 gate; sign-negative on ROI per §3 invariant)
- Scalability floor breach: `r_safe = 0.10% < r_min = 0.15%`; `r_hard = 0.13% < 0.15%` — both tiers fail `scalable_to_safe` / `scalable_to_hard`
- 4 negative folds (DEPLOYABLE requires 0; VIABLE allows 1)
- Min trades/fold = 15 (gate ≥ 25)
- Holdout DD 57.5% (gate ≤ 8% at `r_safe`)
- Daily DD breaches non-zero at `r_base`

Per Amendment 3 priority order, `primary_failure_mode` re-classifies to `step5_not_scalable` (replacing v3.0's `step5_dd_above_gate`, which is deprecated). This matches the original Arc 11 §10 retroactive re-evaluation conclusion — engine improvements don't change the structural deficit.

**Diagnostic value (the actual reason for running v3.0.2):**

1. Does **orchestrator gap closure** (PR #186 plumbed `run_context`) change A2's worst-fold ratio materially? v3.0 A2 was −0.7687 via inline-driver bypass; v3.0.2 A2 runs canonically through `ArcOrchestrator._run_step_5`. Expect small numeric shift, no verdict flip.
2. Does **canonical uncapped pool** (already what Arc 11 v3.0 used post-canonical-builder) interact with v3.0.2 mid-feature + EET HTF alignment to change Step 5 outcomes? Expected: pool count near-identical; feature values drift; Step 5 outcomes shift quantitatively but not categorically.
3. Does **Amendment 5 A2 admission for c0** (now formally gated at AUC ≥ 0.65 with v3.0.2 AUC) reproduce Arc 11 v3.0's "A2 best of FAIL" pattern? Expected: yes if c0 v3.0.2 AUC clears 0.65; A2 still loses on worst-fold ratio.
4. Does **A5 portfolio (newly admitted under Amendment 5 Gate 4 with 2 candidate clusters)** show admit-only-vs-deployment pattern? Cross-arc capturable-not-extractable tally: Arc 7 (V-shape) + Arc 11 (Bimodal + Unclassified) + new Arc 5/8/10 v3.0.2 instances.
5. Does **timing > features** off-protocol finding reproduce under canonical engine? Oracle WFO PASS reproduces if c1 oracle still hits +101%/yr ROI under EET + mid features.

A material improvement in worst-fold ratio (e.g., to a value still FAIL but no longer negative) is a useful data point even at FAIL.

---

## §6 Compute estimate

| Step | Estimate | Driver |
|---|---|---|
| Step 1 (pool build) | ~5-10 min | Cached EET panels; canonical builder iterates 28 pairs × 16 years. Cache namespace `data/cache/<TF>_5ers_eet/<PAIR>.parquet` may need build on first 5ers_eet aggregation pass for this branch (~15-20 min if cache cold; ~2-3 min if warm). |
| Step 2 (clustering) | ~1-2 min | KMeans K∈{2..6} on ~17.5k trades × 5 path features. |
| Step 3 (capturability) | ~1-2 min | Per-cluster reach/MFE/ww_pp + SL sweep. |
| Step 4 (extraction) | ~5-10 min | RF + LGBM + LR × 5-fold CV × 2 candidate clusters; permutation importance with `n_jobs=1`. Holdout-window training filter + classifier persistence on top. |
| Step 5 (WFO search) | **~3-6 hr** | ~162 configs × 11 WFO folds + holdout re-runs at scaled risks. A5 portfolio is admit-only-economics (no new sim). Per-day max-DD parquet + chained DD per top-K. If c1 v3.0.2 AUC also clears 0.65 (adding A2 + A6 for c1), bump to ~234 configs → ~4-8 hr. |
| Step 6 (causal audit) | ~10-20 min | Only if Top-1 PASS — unlikely. |
| Closure + parser | ~30 min | Closure doc, parser run, tracker delta, atomic commit, PR. |

**Total wall-clock estimate: ~4-8 hours single-threaded** (workstation). Determinism contract preserved end-to-end (`random_state=42`, `n_jobs=1`, `lineterminator='\n'`).

---

## §7 HALT triggers (per dispatch §7)

- Post-merge orchestrator still has `run_context` gap to A2/A6 — **CHECKED: not triggered** (PR #186 closed; verified at §0).
- Step 1 canonical uncapped pool produces 2-3× delta vs Arc 11 v3.0 hand-rolled — **NOT EXPECTED to trigger** (Arc 11 v3.0 already used canonical uncapped builder; v3.0.2 expects pool count near-identical to v3.0's 17,533).
- Bimodal cluster fails to recur in Step 2 — **surface in `step_2/cluster_summary.md`, do not halt**. Continue to Step 3.

Additional standard HALT triggers per WORKFLOW.md §6 apply.

---

## §8 Open dispatch interpretive calls for chat

1. **`time_exit_n_bars` registry gap (per dispatch §0.7).** Arc 11 v3.0 used `hold_bars=240` at pool builder + `sl_only` exit policy — *not* a registered `time_exit_n_bars` exit. v3.0.2 mirrors this. The CC_18 registry lacks `time_exit_n_bars`, but Arc 11 does not need it. **No action needed; documenting for clarity.**

2. **A5 portfolio composition admission.** Arc 11 v3.0 did NOT test A5 (architectures_tested was [A1, A2, A6]). v3.0.2 admits A5 under Amendment 5 Gate 4 because Arc 11 has 2 candidate clusters surviving Step 3 (c0 Bimodal + c1 Unclassified). This is a *new* architecture vs v3.0. If A5 surfaces a competitive worst-fold ratio (combining c0's admit-only economics with c1's), this is a v3.0.2-specific finding worth surfacing in §3 cross-arc.

3. **Exit-policy slate per cluster.** Per L_PROTOCOL §2 Step 5 "Exit policy" + dispatch §5 Step 5: Bimodal-archetype c0 admits `sl_partial_close_1r_runner_trail` + `sl_plus_tp_2r` + `sl_only`. Unclassified c1 has no archetype-driven slate. I default c1 to the SAME three exits for cross-cluster comparability + selection-bias transparency. **Confirm at chat review** — alternative is `sl_only` only for c1 (smaller search space; less comparable across clusters).

4. **Holdout window end.** v3.0 closure used `2026-04-30`. v3.0.2 will use `2026-05-25` (= today / arc-close timestamp) per L_PROTOCOL §3 "Holdout window" ("moving target by design"). 4-week extension; marginal effect on holdout metrics.

5. **Re-confirmation that §10 retroactive re-eval is mandatory even though v3.0.2 is itself the re-run.** Per dispatch §5 "Closure": "§10 retroactive re-evaluation MANDATORY — quantitative comparison vs Arc 11 original". Interpreted as: §10 in v3.0.2's closure quantitatively compares v3.0.2 results vs Arc 11 v3.0's results (specifically testing whether canonical orchestrator gap closure + canonical uncapped pool change Step 5 outcomes). NOT a re-application of Amendment 3 to v3.0 numbers (which is what Arc 11 v3.0's §10 already did). **Confirm interpretation at chat review.**

---

## §9 End of intent doc

Waiting for chat review per dispatch §4: "**END TURN AFTER INTENT DOC. Wait for chat review.**"

No code or closure work proceeds until chat confirms the plan and resolves the §8 interpretive calls.
