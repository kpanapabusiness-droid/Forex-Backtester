# Dispatch — Arc 5 v3.0.2 CORRECTED — Intent Doc

> **Arc:** `l_arc_5_v3.0.2`
> **Signal:** `mtf_alignment.2_down_mixed.kijun.h_120` (module: `core/signals/mtf_alignment_2_down_mixed_kijun.py`)
> **Primary TF:** **H1** (signal module hard-codes `primary_tf = "H1"`; auxiliary `H4`, `D1`) — corrected from prior dispatch's 4H claim.
> **Boundary convention:** `5ers_eet` end-to-end (Amendment 6 canonical; no UTC fallback).
> **Sub-protocol:** vanilla.
> **Branch:** `arc/l_arc_5_v3.0.2` — FRESH, cut 2026-05-25 from `0710f18` (current main tip). Prior local + origin branches (containing pool tip `7078739` and intent tip `875caf7`) deleted under chat go-ahead.
> **Closure target:** `results/l_arc_5_v3.0.2/ARC_CLOSURE.md` (template v1.3.1).
> **Dispatch source:** `DISPATCH_arc_5_v3_0_2_CORRECTED.md`.

---

## §A Pre-flight gate (§1 of dispatch) — PASSED

- `git log origin/main --oneline -60`: confirmed all required PRs present —
  - Amendment 5 chain: #185, #186, #188, #189, #190, #193, #194 ✔
  - CC_18 canonical exit registry (#195) ✔
  - CC_20 EET session semantics (#197) ✔
  - Amendment 5.1 Gate 4 PASS-tier-constituent qualifier (#201) + cutoff-ISO backfill (#204) ✔
- Capability smoke tests (Windows / `C:\Users\panap\miniconda3\python.exe`):
  - CC_18 → OK (`sl_partial_close_1r_runner_trail` registered)
  - CC_20 → OK (`SUPPORTED_CONVENTIONS == ('utc', '5ers_eet')`)
  - PR-193 → OK (`core.signals.htf_alignment` exports present)
  - Amendment 5.1 → OK (`AMENDMENT_5_1_CUTOFF_ISO = "2026-05-25T05:29:01Z"` in `scripts/tracker_parser/schema.py:55`)

No HALT triggers fired.

---

## §B Branch hygiene (§2 of dispatch) — DONE

1. Deleted local `arc/l_arc_5_v3.0.2` (was tip `7078739` — prior UTC Step-1 artefacts).
2. Deleted `origin/arc/l_arc_5_v3.0.2` (was `875caf7` — prior intent commit).
3. Renamed this worktree's branch `claude/compassionate-pare-bae095` → `arc/l_arc_5_v3.0.2`.
4. Pushed with upstream tracking; `git branch --show-current` reports `arc/l_arc_5_v3.0.2`; head at `0710f18`.

Worktree path: `C:\Users\panap\Documents\Forex-Backtester\.claude\worktrees\compassionate-pare-bae095\` (path retained from initial worktree-create; branch identity renamed).

---

## §C Read-first phase (§3 of dispatch) — DONE

Read in order:
1. `L_PROTOCOL.md` — Amendment 6 (§3 "Boundary"), Amendment 5 (§2 Step 5), Amendment 5.1 (Gate 4 qualifier) ✔
2. `archive/L_PROTOCOL_v3_0_AMENDMENT_5.md` ✔
3. `docs/PROTOCOL_RUNTIME.md` §8b (Amendment 3 emissions, 5ers_eet daily-DD bucketing) + §15.5 (EET session semantics propagation) ✔
4. `docs/audits/engine_capability_audit_2026_05.md` (post-Phase-1 sprint state — Amendment 3 + 6 + canonical exits all WIRED) ✔
5. `docs/audits/signal_module_eet_audit_2026_05.md` — confirms `mtf_alignment_2_down_mixed_kijun.py` was the prior State-C zero-pool failure; this module **is** the canonical fix restored alongside `core/signals/htf_alignment.py` ✔
6. `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.3.1 (`architectures_skipped_by_amendment_5`, `step_6` block, deployment_spec required for PASS) ✔
7. `core/signals/mtf_alignment_2_down_mixed_kijun.py` — confirmed `primary_tf = "H1"`, `auxiliary_tfs = ("H4", "D1")`, mid-OHLC kijun-26, 2_down_mixed decision-tree, runtime lookahead-invariant via `require_fully_closed=True` ✔
8. `WORKFLOW.md` — dispatch artefact pattern, arc-close artefact set, Amendment 5 dispatch-time selection convention ✔

---

## §D File paths to be touched

```
NEW (this arc):
  scripts/l_arc_5_v3_0_2/build_step1_pool.py        — Step 1 dispatch script (5ers_eet panels)
  scripts/l_arc_5_v3_0_2/run_step_2_3_4.py          — Steps 2-4 dispatch
  scripts/l_arc_5_v3_0_2/run_step_5.py              — Step 5 WFO search (post Step 4 + Amendment-5 gate resolution)
  configs/l_arc_5_v3_0_2/arc.yaml                   — arc config (pairs, window, r_base, sub_protocol)
  results/l_arc_5_v3.0.2/ARC_OPEN.md
  results/l_arc_5_v3.0.2/step_1/...                 — pool.parquet, integrity_report.md, manifest.json
  results/l_arc_5_v3.0.2/step_2/...                 — cluster_assignments.parquet, cluster_summary.md
  results/l_arc_5_v3.0.2/step_3/...                 — capturability.csv, capturability_summary.md
  results/l_arc_5_v3.0.2/step_4/...                 — extraction_metrics.csv, feature_importance.csv,
                                                       classifiers/<cid>.pkl + manifest.json
  results/l_arc_5_v3.0.2/step_5/...                 — wfo_results.csv, wfo_oracle.csv,
                                                       architectures_ranked.md, best_candidate.md,
                                                       per_day_max_dd_base__<cid>.parquet (×top-K)
  results/l_arc_5_v3.0.2/step_6/...                 — auto-dispatched if Top-1 clears §3 #1-9
  results/l_arc_5_v3.0.2/ARC_CLOSURE.md             — template v1.3.1
  docs/dispatches/arc_5_v3_0_2_intent.md            — THIS DOC
  docs/dispatches/arc_5_v3_0_2_log.md               — execution log (post-execution)

MODIFIED (cross-arc shared state, atomic with closure):
  ARC_TRACKER.md                                    — parser delta
  scripts/tracker_parser/parsed.log                 — sha256 entry
  scripts/tracker_parser/rolling_state.json         — rolling registries

CACHE (gitignored; build cost; not committed):
  data/cache/H1_5ers_eet/<28 pairs>.parquet
  data/cache/H4_5ers_eet/<28 pairs>.parquet
  data/cache/D1_5ers_eet/<28 pairs>.parquet
```

No engine code modified. All Amendment-3/4/5/5.1/6 plumbing already on main per pre-flight.

---

## §E Steps 1-5 plan — explicit `boundary_convention="5ers_eet"` callouts

### Step 1 — Plumbing

- **Signal:** `MtfAlignment2DownMixedKijunSignal` at H1 bar close.
- **Pairs:** 28 FX (canonical set; per L_PROTOCOL).
- **Window:** 2010-01-01 → present (closure-time holdout cut at 2021-01-01 per Appendix B).
- **Panel aggregation:** `boundary_convention="5ers_eet"` for H1, H4, D1 panels. Cache namespace `data/cache/<TF>_5ers_eet/<PAIR>.parquet`.
- **Pool builder:** `core.arc.arc_pool_builder.build_arc_pool(...)` — canonical, uncapped, ex-ante.
- **Feature pipeline:** `core.features.pipeline.compute_feature_matrix(...)` — all 7 feature classes (price geometry, distance, multi-TF, cross-pair, spread regime, session, vol regime). Mid-anchored ATR + features per PR #189 §15.1. HTF alignment via PR #193 canonical utility. `Panel.boundary_convention="5ers_eet"` propagates into:
  - `core/features/multi_tf.py` D1-lag-1 series (via `get_htf_value_at`)
  - `core/features/distance.py` prior-session HL bucketing (via `utc_to_eet_trading_day`) — Amendment 6 / §15.5
- **Integrity report:** standard 6 checks per `core.arc.integrity` (pool_size_min, per_pair_below_warn, per_pair_zero_trades, coverage_window, lookahead_declared_lineage, determinism_two_run). The three protocol-listed "informational" checks (D1-lag NaN-perturbation, spread-floor activation, KH-24 co-fire) carried out by-hand at script level per existing arc pattern.
- **Anti-confounder note (per dispatch §5 Step 1):** every feature value WILL differ from prior UTC run — do not investigate as drift.

### Step 2 — Clustering

- `core.steps.step_2_clustering.run_step_2` — KMeans over K ∈ {2,3,4,5,6}, silhouette selection, archetype tagging via `assign_shape_tag`.
- Report whatever surfaces. Prior UTC run had c0 Bimodal + c1 Unclassified — cluster geometry under 5ers_eet may differ; this is the methodology, not a bug.

### Step 3 — Capturability

- `core.steps.step_3_capturability.run_step_3` — per-cluster reach_1R/2R/3R, MFE distribution, ww_pp, mean R, p25/p50, composite, candidate-flag at `reach_1R ≥ 0.50 ∧ ww_pp ≤ 0.30 ∧ mfe_p50 ≥ 1.5R`.

### Step 4 — Extraction

- `core.steps.step_4_extraction.run_step_4` per candidate cluster:
  - RF + LGBM + LR (Appendix A defaults; `random_state=42`).
  - 5-fold TimeSeriesSplit on IS-only pool (`train_end = 2021-01-01`).
  - Per-fold AUC + AUC-best threshold via Youden's J.
  - Permutation importance, `n_repeats=5`, `random_state=42`, `n_jobs=1`.
  - Classifier persistence to `step_4/classifiers/<cid>.pkl` + SHA256 manifest (PR #185).
- Cross-arc flag: Arc 8 v3.0.2 surfaced ~0.16 AUC lift from multi-TF feature restoration. If Arc 5 v3.0 had an equivalent `compute_feature_matrix` panel-passing gap (it did not — Arc 5 v3.0 ran under PR-189-EET-aggregated but pre-#193 signal module, then the module was outright deleted from main for Arc 5 v3.0.1; this run is the first to use the canonical restored module under EET storage), expect comparable lift surfaced via Step 4 AUC vs prior UTC reading. Will report decomposition in §3 cross-arc of closure.

### Step 5 — WFO architecture search

- `core.architectures.{a1..a6}` driven by `core.wfo.orchestrator.run_search` via `ArcOrchestrator`.
- **WFO structure:** 11-fold IS 2010-01-01 → 2020-12-31; holdout one-shot 2021-01-01 → present.
- **`r_base = 0.5%`** (v3.0 arc default).
- **Sizing convention:** `reset_floor` exclusively (Amendment 3 §"Scalability bounds").
- **Daily-DD bucketing:** `compute_per_day_max_dd(equity_curve, boundary_convention="5ers_eet")` — Amendment 6 / §15.5. Load-bearing.
- **Reset-floor ratchet:** `core/sim/risk/reset_floor.py` with `boundary_convention="5ers_eet"` — daily-floor reset matches the 5ers EET broker day.
- **Amendment 3 emissions auto-fired** by `ArcOrchestrator._run_amendment_3_evaluation` per top-K: `per_day_max_dd_base__<cid>.parquet`, `chained_max_dd_base_pct`, holdout re-runs at `r_safe` / `r_hard`, amended gate classification.
- **SL multiplier:** centred on Step 3 per-cluster optimum ± 1 step (3 values).
- **Exit policy slate per archetype** (per L_PROTOCOL §2 Step 5):
  - V-shape recovery → `{sl_only, sl_plus_tp_2r, sl_partial_close_1r_runner_trail}` (the slate cited in the dispatch).
  - Stepwise climber → `{sl_only, sl_plus_trailing_atr, sl_plus_trailing_swing}`.
  - Bimodal → `{sl_only, sl_partial_close_1r_runner_trail, sl_plus_tp_2r}`.
  - Monotonic up/down / Unclassified → `{sl_only, sl_plus_tp_2r}` minimum (do not default to all 6 per dispatch §5 Step 5).
- **Exposure cap:** `{max_concurrent_per_currency=2, unlimited}` (2 values).
- **Top-3 by worst-fold ratio** → 2021-2025 holdout one-shot per top-K.

### Step 6 — Causal audit

- Auto-dispatches if Top-1 clears §3 #1-9 — Amendment 4.
- Patched A1 vacuous-pass framework handles `features_in_winning_config: []` correctly.
- Critical failure → verdict downgrade to FAIL with `primary_failure_mode = step6_causal_audit_fail`.

### Closure

- `results/l_arc_5_v3.0.2/ARC_CLOSURE.md` per template v1.3.1.
- `§1 tracker_payload.tf: H1` (corrected). `boundary_convention: 5ers_eet` recorded.
- `architectures_skipped_by_amendment_5: [...]` — populated per §F below (Phase-2 required for post-cutoff PASS verdicts; closure is post-cutoff `AMENDMENT_5_CUTOFF_ISO`).
- §10 retroactive informational comparison vs PR #172 numbers IF recoverable — caveat: methodologically different test.
- Tracker parser + atomic commit + PR.

---

## §F Architecture-admission table per Amendment 5 four-gate (with Amendment 5.1 on Gate 4)

Resolved per-cluster post-Step-3/Step-4. The dispatch-time rule (not engine-time) per Amendment 5 §9.

**Universal admission (at dispatch time, no dependencies):**
- **Gate 3** → **A1** for every non-Choppy surviving cluster.

**Conditional admissions (resolved after Step 3 archetype + Step 4 AUC land):**

| Gate | Architecture | Admission condition | Resolves after |
|---|---|---|---|
| Gate 1 | A3 | cluster archetype ∈ {V-shape recovery} | Step 2/3 (archetype tag) |
| Gate 1 | A4 | cluster archetype ∈ {Stepwise climber, Bimodal} | Step 2/3 |
| Gate 2 | A2 + A6 | Step 4 mean OOS AUC ≥ 0.65 for cluster | Step 4 |
| Gate 3 | A1 | always (every non-Choppy cluster) | dispatch (universal) |
| Gate 4 | A5 | **(a)** ≥ 2 candidate clusters surviving Step 3 AND **(b)** ≥ 1 constituent reaches PASS-DEPLOYABLE or PASS-VIABLE under Gate 1/2/3 architecture | post-Step-5 (Amendment 5.1) |

**Cluster skip:** Choppy archetype → all architectures skipped for that cluster.

**Dispatch-time admission at arc-open:**
- A1 admitted universally (Gate 3).
- A3/A4 admission **tentative**, contingent on Step 2 archetype tags.
- A2/A6 admission **tentative**, contingent on Step 4 mean OOS AUC ≥ 0.65 per cluster.
- **A5 NOT admitted at dispatch time per Amendment 5.1.** If (a)+(b) both hold post-Step-5, A5 admission lands as a closure addendum without re-running the rest of the arc. If only (a) holds, closure records `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` in `architectures_skipped_by_amendment_5`.

**`architectures_skipped_by_amendment_5` content rule (closure):**
- Empty `[]` if Amendment 5's gate set equals or supersets the Amendment-1 rule for every surviving cluster.
- Architecture codes ({A1..A6}) for any architecture that would have been tested under Amendment 1's archetype-driven rule but skipped under Amendment 5.
- Reason string `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` if Gate 4 condition (b) failed post-Step-5.

A1 is admitted at dispatch and engine receives a fully-resolved spec at Step 5 launch. A3/A4/A2/A6 are scheduled post-Step-2/Step-3/Step-4 per Amendment 5 §9. A5 is scheduled post-Step-5 per Amendment 5.1.

---

## §G Step 4 AUC summary plan

For Gate 2 transparency the closure will report, per candidate cluster:

- Mean OOS AUC across the 5-fold TimeSeriesSplit (all three classifier families).
- Best-of-three (per-cluster `step4_e_auc`).
- Per-fold AUC table (for Step 6 §6.4 statistical-integrity reproducibility).
- Gate 2 result: **ADMIT A2+A6** (mean OOS AUC ≥ 0.65) or **SKIP A2+A6** (below 0.65); per cluster.

Gate-2 admission decision recorded explicitly in `step_4/extraction_summary.md` AND in §1 closure YAML.

---

## §H Compute estimate

| Phase | Wall-clock estimate | Notes |
|---|---:|---|
| 5ers_eet panel cache build (28 pairs × {H1,H4,D1}) | **15-30 min** (cache-cold) | Worktree cache currently empty — see §I.1 below. One-time cost amortised across this arc + any other 5ers_eet work from this worktree. |
| Step 1 pool + feature matrix | 5-15 min | 27 features × 28 pairs × ~10 years of H1 bars; signal-bar-evaluation only. |
| Step 2 clustering | 1-2 min | KMeans K∈{2..6} on path features. |
| Step 3 capturability | <1 min | Closed-form per-cluster metrics. |
| Step 4 extraction (per candidate cluster) | 10-30 min/cluster | RF + LGBM + LR × 5-fold TSCV + permutation importance × refit on lineage-filtered pool. Scales linearly with surviving candidate clusters. |
| Step 5 WFO search | **1-3 hours** | 11 folds × N configs (typically 50-100 per Amendment 5 §6); plus oracle WFO per candidate cluster; plus holdout one-shot. Amendment-3 re-run adds 2 sims per top-K. |
| Step 6 auto-dispatch (if PASS) | 5-15 min | Six-category framework; byte-compare adds the bulk. |
| Closure writing + parser invocation | manual + 1 min | Hand-written closure → `python scripts/update_tracker_from_closure.py`. |
| **Total (excl. cache build)** | **~2-5 hours** | Wall-clock; CPU-bound on `n_jobs=1` determinism contract. |
| **Total (incl. cache build)** | **~2.5-5.5 hours** | First-run-from-cold-worktree case. |

If A5 is admitted post-Step-5 (Amendment 5.1 condition (b) holds), add ~30-60 min for the portfolio recombination + Amendment-3 re-evaluation.

---

## §I Interpretive calls / flags for chat review

These are choices I'd default through if no chat-side correction lands during the review window. Flagging explicitly per WORKFLOW.md §2 intent-doc convention.

### I.1 Worktree data access (NEW — not addressed in dispatch)

The worktree has **no `data/histdata/` or `data/cache/` content** — `.gitignore` excludes `data/*`, so per-worktree these directories are empty even though the main checkout (`C:\Users\panap\Documents\Forex-Backtester\data\`) has both raw HistData and warm 5ers_eet panels (56 files per TF × 3 TFs).

Default plan: configure the arc scripts with `histdata_root=` and `cache_root=` pointing at the absolute path of the main repo's `data/` (`C:/Users/panap/Documents/Forex-Backtester/data/`). This avoids the 15-30 min cache-build cost AND keeps the artefact path under the canonical location. The compute estimate above (§H) reflects the cache-cold case as a conservative upper bound; the warm-cache path drops the line.

Alternative: build a separate worktree cache from scratch (clean isolation, but +15-30 min). Or symlink/junction the main-repo `data/` into this worktree.

**Default if no chat redirect:** point at main-repo `data/` via absolute paths in `configs/l_arc_5_v3_0_2/arc.yaml`. This is a YAML-only choice — no engine work.

### I.2 Exit-policy slate per archetype

The dispatch §5 Step 5 specifies the V-shape slate `{sl_only, sl_plus_tp_2r, sl_partial_close_1r_runner_trail}` and "do NOT default to all 6". I have applied the L_PROTOCOL §2 Step 5 archetype-matched slate (§E Step 5 above) — the V-shape entry matches the dispatch exactly. Stepwise / Bimodal / Monotonic / Unclassified entries use the L_PROTOCOL-prescribed mappings. If the dispatcher intended the V-shape slate to apply to all clusters regardless of archetype, redirect.

### I.3 Prior UTC numerical comparison in §10

Per dispatch §5 Closure, "quantitative comparison vs PR #172 numbers IF recoverable" — PR #172 is closed and its artefacts are off the deleted UTC branch. Recoverable numbers depend on what was promoted to `ARC_HISTORY.md` / `ARC_TRACKER.md`. If only verdict + worst-fold-ratio survive, §10 will compare those and label deltas informational; if per-fold numbers are recoverable, will include those too. The methodological caveat ("EET vs UTC is methodologically different — large deltas expected and correct") will land in §10 regardless.

### I.4 A5 admission addendum mechanics

Per Amendment 5.1, if Gate 4 (b) holds post-Step-5 (a Gate 1/2/3 architecture lands PASS-tier), A5 admission lands as a **closure addendum without re-running the rest of the arc**. I read "addendum" as: write A5 results into the same closure doc + same tracker delta as the rest of the arc, and run the parser once on the final state. Will not commit a partial closure prior to A5 evaluation. If the intended semantics is a two-PR sequence (initial closure without A5, addendum PR with A5), redirect.

---

## §J Comparability acknowledgment (per dispatch §4 #6)

**This arc is NOT directly comparable to PR #172 / the prior `arc/l_arc_5` UTC results.** The original ran under UTC bar boundaries; the corresponding daily-DD measurement boundary was UTC. Under 5ers_eet bar boundaries, every feature value differs (price geometry on different bars, multi-TF lookups against EET-anchored HTF closes, distance features bucketed by EET trading day, daily-DD bucketed by EET broker day, reset-floor ratchet on EET trading day). Step 2 cluster geometry, Step 3 capturability metrics, Step 4 AUC, Step 5 WFO economics — all derive from a different feature substrate. §10 delta vs prior is informational; the comparison is between two different methodological tests, not between two runs of the same test.

The 5ers production deployment substrate is `5ers_eet`. UTC was an artefact of pre-PR-189 engine design and is opt-in only for KH-24 anchor byte-identity. Arc 5 v3.0.2 is the first run of this signal under the deployment substrate.

---

## §K Definition of done (§9 of dispatch)

- [x] Fresh `arc/l_arc_5_v3.0.2` cut from current main (`0710f18`).
- [x] Intent doc written explicitly stating `boundary_convention="5ers_eet"` everywhere.
- [ ] Intent doc chat-approved (this turn ends; awaiting review).
- [ ] Steps 1-5 executed under 5ers_eet per §E.
- [ ] Step 6 auto-dispatched if Top-1 cleared §3 #1-9.
- [ ] `results/l_arc_5_v3.0.2/ARC_CLOSURE.md` written per template v1.3.1.
- [ ] Tracker parser run on arc branch; delta included in same commit as closure.
- [ ] ONE PR opened to main; chat reviews.

---

## §L Awaiting

Chat review of this intent doc per WORKFLOW.md §2.4. Specific asks:

1. Confirm or redirect §I.1 (worktree data access — default: absolute paths to main-repo `data/`).
2. Confirm §I.2 (per-archetype exit slates, V-shape matches dispatch exactly).
3. Confirm §I.3 (§10 prior-UTC comparison scope).
4. Confirm §I.4 (A5 addendum = single-PR sequence with A5 results bundled into the same closure).
5. Anything else — redirect freely.

**End turn.**
