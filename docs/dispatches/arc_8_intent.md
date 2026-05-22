# Arc 8 — Intent Doc

> **Dispatch:** Arc 8 — pullback-resume HH/HL long under L_PROTOCOL v3.0 (vanilla overseer)
> **Author:** CC (orchestrator)
> **Status:** plumbing only — awaiting chat review per WORKFLOW §2 before any code or compute
> **Date:** 2026-05-22

This doc fulfils the dispatch's mandatory "Intent doc" section (six items). A "Flags for chat" section follows the six items: every dispatch-vs-spec-vs-v3-engine mismatch I noticed is surfaced there for chat to resolve before Step 1 starts. No judgement applied — defaults left where they are, decisions deferred.

---

## 1. Resolved signal spec path

**`docs/archive/signal_specs/signal_spec_pullback_resume_hhhl_long_v0.1.md`** — present, locked at arc-open per L_ARC_PROTOCOL §1.8 framing (v2.x phrasing inside the spec; see Flag F1).

The dispatch's reading list said `docs/archive/signal_specs/signal_pullback_resume_hhhl_long_v0.1.md` (no `_spec_` infix). The on-disk filename uses the `signal_spec_` prefix shared by all eight files in that folder. Resolved to the on-disk path. Fallbacks (`ARC_8_RESULT.md`, `ARC_8_CLOSURE.md`) not required — primary spec is present.

## 2. Signal definition — confirmed elements

Spec defines all three required mechanisms (dispatch §"Intent doc" item 2):

- **HH/HL detection:** 3-bar local extreme on `high`/`low` over window `t-30..t-1`. ≥ 2 swing-highs strictly ascending (HH); ≥ 2 swing-lows strictly ascending (HL). Right-edge constraint: most recent identifiable swing at most bar `t-4` (uses k+1..k+3 forward lookahead within detection window — confirmation lag).
- **Pullback:** `close[t-1] ≤ most_recent_swing_high − 0.5 × ATR(14)[t-1]`.
- **Resume trigger (bar t):** bullish close (`close[t] > open[t]`) AND breaks prior bar high (`close[t] > high[t-1]`) AND close in upper half of bar range (`(close − low) / (high − low) ≥ 0.5`).
- **Spacing:** ≥ 20 bars since last signal on the same pair.
- **Entry:** bar `t+1` open per `docs/SPREAD_SEMANTICS_LOCK.md` round-trip convention.

Right-edge swing audit (spec §"Step 1 right-edge swing audit") is mandatory at Step 1: confirm every selected signal's trigger-evaluation uses swings at most bar `t-4`. Engine-touching halt if standard lookahead spot-check shows future-bar dependency.

## 3. Files CC will touch or create

**New (created by this arc):**

- `core/signals/pullback_resume_hhhl.py` — signal producer module (HH/HL detection + pullback + resume trigger). Currently `core/signals/` contains only `__init__.py`; this is the first signal under v3.0.
- `tests/test_signals_pullback_resume_hhhl.py` — unit tests for swing detection, pullback condition, trigger, spacing, right-edge audit.
- `configs/wfo_l_arc_8.yaml` — top-level arc config (signal binding, window, pair set, risk, exposure cap, SL sweep, exit policies). Referenced by signal spec as the target.
- `configs/l_arc_8/step1.yaml` — Step 1 pool-build config (does not currently exist; only step2..step5 v2.x configs do — see Flag F6).
- `scripts/l_arc_8/run_step_1_pool.py` — Step 1 driver (signal → pool → feature matrix → integrity report → manifest).
- `results/l_arc_8/ARC_OPEN.md` — per L_PROTOCOL §6 required fields.
- `results/l_arc_8/step_1/pool.parquet` — Step 1 trade pool with feature matrix.
- `results/l_arc_8/step_1/integrity_report.md` — Step 1 integrity checks (per dispatch §"Step 1").
- `results/l_arc_8/step_1/manifest.json` — sha256 per artefact.
- `docs/dispatches/arc_8_log.md` — append-only log doc per WORKFLOW §2 and dispatch §"Closure doc".
- `docs/dispatches/arc_8_diagnostic.md` — only if a HALT trigger fires (dispatch §"HALT triggers").
- `results/l_arc_8/step_2/*` through `results/l_arc_8/step_5/*` — per L_PROTOCOL §2 step artefacts (created in later turns).
- `results/l_arc_8/ARC_CLOSURE.md` — write-once at arc close per L_PROTOCOL §6 / dispatch §"Closure doc".
- `ARC_TRACKER.md` — append-only update at closure per L_PROTOCOL §6.

**Touched (read-only):** `L_PROTOCOL.md`, `WORKFLOW.md`, `CLAUDE.md`, `docs/BACKTESTER_ARCHITECTURE.md`, `docs/features_reference.md`, the signal spec, `configs/wfo_kh24.yaml` (for anchor-style WFO fold builder reference if needed).

**NOT touched:** `EA/`, `reference/kh24_ea/`, `core/strategies/` (KH-24 logic), live KH-24 deployment, any other arc's `results/l_arc_N/` folder.

## 4. Confirmed window, pair set, risk, TF

| Field | Confirmed value | Source |
|---|---|---|
| Signal TF | 4H | dispatch §header + spec "Signal TF" |
| Pair set | 28 FX (KH-24 set) | dispatch §header + spec "Pair set" |
| Risk per trade | 0.5% × reset floor balance | dispatch §header + spec "Risk per trade" |
| Window (search) | 2010-01-01 → 2020-12-31 (11-fold WFO) | L_PROTOCOL §2 Step 5 + dispatch §header (resolves spec conflict — see Flag F2) |
| Window (holdout) | 2021-01-01 → most-recent-complete-month (one-shot) | L_PROTOCOL §2 Step 5 + dispatch §header |
| Forward window per trade | 240 bars (4H) | spec "Forward window" |
| Exposure cap (Step 1 sim) | unrestricted | dispatch §"Step 1" — Step 5 architecture search varies this |
| Initial SL (Step 1 sim) | 2.0 × ATR(14)_4H from entry price | dispatch §"Step 1" + spec "Initial SL (Step 1 sim)" |
| SL sweep (Step 3) | dispatch's {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} — see Flag F3 | dispatch §"Step 3" (conflicts with spec; deferred to chat) |

## 5. L_PROTOCOL Step 1 feature space

Will be used as-is per the dispatch. Source of truth: `docs/features_reference.md` — 27 features across 7 classes (cross_pair × 4, distance × 3, multi_tf × 4, price_geometry × 5, session × 7, spread_regime × 2, vol_regime × 2). Lineage tags from the reference doc are binding (no `suspect`/`unverified` feature enters any classifier without the Step 6 audit per L_PROTOCOL §2 Step 6).

L_PROTOCOL §2 Step 1 lists "cross-asset (if available): DXY state, US10Y state" — these are not in the 27-feature v3.0 catalogue. Will run Step 1 without them (this is the "as-is" reading: take what the engine has). See Flag F7.

No per-arc signal-specific features added at Step 1. Step 4 may add path-so-far features for A3/A4 per L_PROTOCOL §2 Step 5 ML mechanics; those are not Step 1 features.

## 6. Determinism plan

- `random_state = 42` everywhere (numpy, sklearn, KMeans, RF/LGBM/Logistic, threshold sweeps). Sourced from `core.determinism.RANDOM_STATE`.
- `n_jobs = 1` within each work unit. Parallelism only at the per-pair pool level via `core.parallel.parallel_pair_map` (PR-D guarantees byte-identical output across pool sizes — `tests/test_determinism.py` asserts this).
- Parquet caching enabled. M1 + per-TF cache per PR-A; feature-matrix cache per PR-D at `data/cache/features/l_arc_8/<feature_set_hash>.parquet` with `.meta.json` sidecar.
- `lineterminator="\n"` for every CSV / text artefact (PR-D contract).
- Two-run sha256 pool reproducibility check per dispatch §"Step 1" integrity list. Will be reported in `step_1/integrity_report.md`.

---

## Flags for chat (resolve before Step 1)

These are the deltas I noticed between the dispatch, the signal spec, the current v3.0 protocol, and the v3 engine state. Surfaced without resolution. Each item names a default I'll take if no redirect comes back — chat can override any of them.

### F1 — Signal spec references v2.x sections

The spec mentions "L_ARC_PROTOCOL §1.8" and "§16a Path A" (v2.x). Current protocol is `L_PROTOCOL.md` v3.0. The mechanics in the spec (trigger definition, spacing, entry convention, pool-size prior, right-edge swing audit, co-fire matrix) translate cleanly — they are signal-definition content, not protocol mechanics. No v3.0 contradiction. **Default:** treat the spec's mechanics as canonical for the signal and ignore the v2.x §-refs in the prose. The dispatch is the authoritative protocol wrapper.

### F2 — Data window: dispatch vs spec

- Dispatch §header: `2010-01-01 → most-recent-complete-month`
- Spec "Data window": `2020-10-01 → 2026-01-31`
- L_PROTOCOL §2 Step 5: training 2010-01-01 → 2020-12-31, holdout 2021-01-01 → present (one-shot)

The dispatch + protocol agree; the spec is narrower (KH-24-anchor-era window only). The spec window would collapse to the holdout segment under v3.0 WFO and leave no training folds.

**Default:** dispatch + L_PROTOCOL window (2010-01 → most-recent-complete-month, 11-fold WFO 2010-2020 + one-shot holdout 2021-present). The spec window appears to be carried over from v2.x KH-24 anchor framing.

### F3 — SL sweep set: dispatch vs spec

- Dispatch §"Step 3": `{1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR(14)`
- Spec "SL sweep at Step 3": `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H`

Overlap on {1.5, 2.0, 3.0, 4.0} (4 values). Differ on the tails: dispatch tests 2.5 and 3.5 (intermediate steps); spec tests 0.5 and 1.0 (tight stops).

**Default:** dispatch set. Rationale for default: dispatch is the orchestrator instruction; L_PROTOCOL §2 Step 5 also implies the {1.5..4.0} stepwise range for Step 5 ("centred on Step 3's per-cluster selected SL, ±1 step"). Tight stops (0.5, 1.0) would constrain Step 5's ±1-step search downward in a way the protocol doesn't anticipate. Easy to flip if chat prefers the spec set.

### F4 — Spread fallback policy: spec vs L_PROTOCOL §1 non-negotiable

- Spec "Spread": `Real per-bar MT5 bid/ask; configs/spread_floors_5ers.yaml fallback only when raw = 0`
- L_PROTOCOL §1: *"HistData M1 bid+ask is the canonical spread source. No fallback mechanism — zero-spread bars are a data quality flag, not silently backfilled."*
- v3 engine (PR-A/PR-B): `spread_close = close_ask - close_bid`; zero/negative/NaN flagged in `bid_ask_data_quality`; no fallback file.
- File state: `configs/spread_floors_5ers.yaml` does not exist in the v3 worktree.

L_PROTOCOL §1 is non-negotiable (binding). The spec describes the v2.x convention. The v3 engine implements §1.

**Default:** L_PROTOCOL §1 — no fallback. Zero-spread bars are flagged via `bid_ask_data_quality`. Spread-floor activation rate is reported informationally per dispatch §"Step 1" integrity list (the rate will reflect data-quality flags, not floor activations). Naming in the integrity report will be "spread data-quality flag rate" to avoid confusion.

### F5 — Branch state

Dispatch §header: branch `arc/l_arc_8`, dedicated worktree.
Current state: worktree path `.claude/worktrees/intelligent-wilson-753d2b`, branch `claude/intelligent-wilson-753d2b`, cut from main per recent commit list.

**Default:** continue on the current worktree + branch; rename branch to `arc/l_arc_8` before PR open (or push under the current name and let chat rename at merge). I will NOT cut a new branch or worktree mid-arc. Chat: confirm acceptable, or instruct otherwise before Step 1.

### F6 — Stale v2.x configs under `configs/l_arc_8/`

Folder contains v2.x configs from the prior Arc 8 run (referenced in CLAUDE.md as "Prior v2.x findings for this signal exist in ARC_HISTORY.md; treat them as historical record"):

- `step2.yaml` — references "L_ARC_PROTOCOL v2.3 stack", `results/l_arc_8/step1_verbatim/`, K-sweep {3..7}, archetype centroid pattern matching
- `step3.yaml` — `archetype_groups` with cluster IDs already routed (cluster 0 unassigned, c1+c3 V-shape, c2 Early-peak), capturability composite from v2.x §7
- `step4.yaml` — references v2.3 §§ (8, v2.2 §2/§3, v2.3 §4), `capturability_pass_list.csv` input
- `step5.yaml` — locks `cluster_id: 1`, references `configs/wfo_kh24.yaml` 7-fold KH-24 anchor folds, `pre_t_sl_atr_multiplier: 4.0`, "Pipeline E / Pipeline D1" framing

All four assume v2.x cluster IDs, v2.x WFO fold geometry (7 KH-24 anchor folds, not 11 expanding), v2.x pipeline framing (E/D1, not A1..A6), and v2.x cluster routing. None are usable under v3.0.

**Default:** leave them in place (no deletion) and write fresh v3.0 step configs alongside. Specifically: `configs/l_arc_8/step1.yaml` (new), and I'll choose `configs/l_arc_8/step2_v3.yaml` etc. or overwrite when those steps run (preferring overwrite + git history preservation since v2.x outputs are already in `docs/archive/arc_results/`). Chat: confirm overwrite is preferred, or whether you'd rather I move the v2.x ones to `configs/l_arc_8/archive_v2_3/` first.

### F7 — Feature catalogue: protocol mentions cross-asset (DXY, US10Y); v3.0 catalogue has none

L_PROTOCOL §2 Step 1 lists "Cross-asset (if available): DXY state, US10Y state" as a feature class. `docs/features_reference.md` lists 27 features across 7 classes; no DXY or US10Y producer exists. PR-A/PR-B/PR-C don't load DXY or US10Y data.

**Default:** "as-is" reading per dispatch item 5 — run with the 27-feature v3.0 catalogue. Cross-asset absence is logged in the Step 1 integrity report. Adding DXY/US10Y feature producers is engine work, not within Step 1 scope. Chat: confirm or override.

### F8 — Signal implementation from scratch

`core/signals/` contains only `__init__.py`. No signal exists there yet. The dispatch's "signal application: ex-ante population via `build_ex_ante_bounded_population` (or v3 equivalent)" implies a v3 wrapper exists; grep finds the function name only in archived v2.x protocol docs. The mechanism in v3 is `core.signals.<signal_module>` + `core.signal_adapter` (likely) + `core.sim.multipair_backtester` for the simulation.

**Default:** implement `core/signals/pullback_resume_hhhl.py` as a self-contained producer that emits signal timestamps + entry/SL/exit fields, then drives the v3 multi-pair backtester via the existing `core.sim.multipair_backtester.MultiPairBacktester` to build the trade pool. Compute features via `core.features.pipeline.compute_feature_matrix`. Cache outputs via `core.features.cache.get_or_compute` with `arc_id="l_arc_8"` and `signal_def` = a stable string derived from the spec mechanics. Chat: confirm this composition before code lands.

### F9 — Co-fire matrix scope (informational)

Spec §"Step 1 co-fire matrix" requests co-fire reporting vs Arc 9, Arc 10, Arc 11 *"if Step 1 landed"* for each. Those arcs are in parallel CC sessions per CLAUDE.md and their Step 1 outputs may or may not be present in this worktree.

**Default:** report KH-24 co-fire unconditionally (dispatch §"Step 1" integrity list). Report Arc 9/10/11 co-fire ONLY if those arcs' Step 1 pool parquets exist at known paths (`results/l_arc_9/step_1/pool.parquet` etc.) at the time Step 1 runs. Missing peer-arc pools are reported as absent, not as failures.

---

## What happens next

Per dispatch §"Intent doc" closing line ("Plumbing only. No interpretive judgement. End turn for chat review.") and WORKFLOW §2 step 3:

1. End turn here. Await chat review.
2. Chat resolves any/all of F1..F9 (or accepts defaults).
3. On chat approval, proceed to Step 1: write the signal module + tests, write `configs/wfo_l_arc_8.yaml` + `configs/l_arc_8/step1.yaml`, write `results/l_arc_8/ARC_OPEN.md`, run Step 1, produce `pool.parquet` + `integrity_report.md` + `manifest.json`, end turn for chat review of the integrity report.
4. Step 2..5 proceed turn-by-turn per dispatch.
5. Step 6 invoked only if Step 5 produces PASS-DEPLOYABLE / PASS-VIABLE.
6. ARC_CLOSURE.md + arc_8_log.md + PR titled `[ARC 8 v3.0] pullback_resume_hhhl_long — <verdict>`.
