# TODO — Live Project Tracker

> The operational todo list. Append, check off, delete as work completes.
> Distinct from `ARC_TRACKER.md` (which is auto-updated arc state) and `ARC_HISTORY.md` (frozen pre-v3.0 record).
> Last updated: 2026-05-25 (CC_22 comprehensive docs refresh — post-Phase-1-engine-build sweep)

---

## Status legend
- 🟢 DONE
- 🟡 IN PROGRESS
- 🔴 NOT STARTED
- ⚪ BLOCKED / WAITING
- ⚫ HALTED / CANCELLED / SKIPPED

---

## Current state — quick view

- **Phase 0 — Framework validation:** 🟢 CLOSED (Phase 0a Path B verdict per PR-E.1.7; Phase 0b skipped by chat decision)
- **Phase 1 — Arc 1-11 re-runs:** 🟡 IN PROGRESS — Wave 1 partial (Arcs 8 / 10 / 11 closed; Arcs 5 / 7 closures in flight via PRs #172 + #180); Wave 1 retries under signal-parity engine: 🔴 NOT STARTED (engine prerequisites now met as of PR #197 merge)
- **Phase 2 — Sub-protocols + new signals:** 🔴 NOT STARTED (blocked on Phase 1 closure)
- **Engine state:** **post-PR-#197**. Steps 1-5 + signal parity (mid features + 5ers EET bars + worst-case fills) + Step 6 framework (Amendment 4) + Amendment 3 risk-normalised gates + Amendment 5 four-gate architecture selection + Amendment 6 EET daily-DD boundary + canonical exit-policy registry (incl. `sl_partial_close_1r_runner_trail`) + signal-level EET timezone canonical utility + EET session semantics across distance / reset-floor / per-day-max-DD.
- **Architectures wired:** A1, A2, A3, A4, A6 (A5 not built — deferred until Wave 2 produces ≥ 1 VIABLE candidate; Arc 8 first candidate)
- **Protocol state:** L_PROTOCOL v3.0 + Amendments 1, 2, 3, 4, 5, 6 inline (Amendment 6 inline-only per CC_22 — no archive file). Closure template v1.3.1. Parser v1.3 with Amendment-5 + Amendment-3 cutoff validation.

### Active parallel chat work

- **heavy_ml_probe** sub-protocol build — PR #187 (PR-A) OPEN; PR-B/C/D/E/F to follow per build plan
- **signal_discovery_probe** 10k local run — user-side workstation operation, IN PROGRESS

---

## Reset checklist

### Round 1 — Foundation

| Task | Status |
|---|---|
| Halt in-flight CC sessions (v2.5 coord, KH-24 protocol, D1 SHB diagnostic) | 🟢 |
| Branch & worktree cleanup (`CC_01_CLEANUP.md`) | 🟢 |
| HistData M1 bid+ask download (tick → aggregate, all 28 pairs, 2010-current) | 🟢 |
| Repo inventory + Arc History (`CC_03_INVENTORY_AND_ARC_HISTORY.md`) | 🟢 |
| UNCERTAIN flag resolution (`CC_04_UNCERTAIN_RESOLUTION.md`) | 🟢 |
| Merge consolidation branch `claude/wonderful-ellis-8d3091` → main | 🟢 |

### Round 2 — Protocol redesign

| Task | Status |
|---|---|
| Work through 11 design topics in `REDESIGN_DECISIONS.md` | 🟢 |
| Draft `L_PROTOCOL.md` | 🟢 |
| Finalise `L_PROTOCOL.md` with pushback edits | 🟢 |
| Land `L_PROTOCOL.md` on main | 🟢 |
| Draft `docs/sub_protocols/heavy_ml_probe.md` | 🟢 |
| Draft `docs/sub_protocols/signal_discovery_probe.md` | 🟢 |
| Draft `ARC_TRACKER.md` skeleton (empty schema) | 🟢 |
| Lock Wave 1 / Wave 2 arc composition | 🟢 |
| Backup destination for HistData (separate folder off-repo) | 🟢 |

### Round 3 — Backtester + data foundation

| Task | Status |
|---|---|
| HistData download completes + backup verified | 🟢 |
| Backtester reconfiguration for HistData as primary | 🟢 PR-A |
| Backtester support for new feature classes (cross-pair, currency strength, session, multi-TF) | 🟢 PR-C |
| Backtester WFO logic for 11-fold 2010-2020 + 1-shot 2021-2025 holdout | 🟢 PR-C |
| Backtester multi-pair simultaneous simulation | 🟢 PR-B |
| Backtester determinism (sha256 manifests, lineterminator='\n') | 🟢 PR-D |
| KH-24 strategy + EA-correction rounds 1-3 (anchor reproduction) | 🟢 PR-E.1/E.1.5/E.1.6 |
| `BACKTESTER_ARCHITECTURE.md` audit / update | 🟢 PR-E.1.7 |
| `DATA_FOUNDATION.md` finalised | 🟢 PR-E.1.7 |
| KH-24 anchor reproduction (Mode A) — Path B verdict | 🟢 PR-E.1.7 |

### Round 4 — Claude project folder cleanup

| Task | Status |
|---|---|
| Review `/mnt/project/` against new repo state | 🟢 |
| Remove pre-reset docs (v2.x amendments, dispatch handovers, etc.) | 🟢 |
| Sync currently-active docs (L_PROTOCOL, ARC_HISTORY, etc.) into project | 🟢 (synced through PR #197) |

### Round 5 — Closure infrastructure

| Task | Status | Notes |
|---|---|---|
| Lock `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.0 | 🟢 | PR #169 |
| Extend `ARC_TRACKER.md` schema (cluster registry, cost-decomp registry, tag registry) | 🟢 | PR #169 |
| Land L_PROTOCOL §6 update referencing template | 🟢 | |
| Build tracker parser `scripts/update_tracker_from_closure.py` | 🟢 | PR #179 built parser; PR #181 retrofitted Arcs 8/10/11 to template v1.2; PR #182 CC_11 follow-up backfilled parsed.log + parser registry |
| Closure template v1.3 (Amendment 4 — Step 6) | 🟢 | PR #188 |
| Parser v1.3 (template v1.3 detection + Phase 2 tightening, cutoff 2026-05-23T06:20:59Z) | 🟢 | PR #188 |
| Closure template v1.3.1 (Amendment 5 — `architectures_skipped_by_amendment_5` field) | 🟢 | PR #194 |
| Parser v1.3 Amendment-5 extension (optional field + Phase 2 enforcement, cutoff `AMENDMENT_5_CUTOFF_ISO` placeholder pending PR-merge backfill) | 🟢 | PR #194 |
| `docs/templates/TODO_REFRESH_TEMPLATE.md` committed | 🟢 | CC_22 PR (this) |

### Round 6 — Engine consolidation (initial sprint)

| GitHub PR | Description | Status |
|---|---|---|
| #184 | Engine capability audit 2026-05 (read-only enumeration of WIRED / PARTIAL / MISSING capabilities) | 🟢 merged 2026-05-24 |
| #185 | Step 4 fitted-classifier persistence + holdout-window training fix (supersedes #183 which was reverted) | 🟢 merged 2026-05-23 |
| #186 | L_PROTOCOL Amendment 3 implementation + A3/A4 wiring + Step 4/5 fixes | 🟢 merged 2026-05-23 |
| #188 | Step 6 causal audit framework + Amendment 4 + parser v1.3 + closure template v1.3 | 🟢 merged 2026-05-24 |
| #189 | Signal parity engine (mid features + trail mid + 5ers EET bar boundaries) | 🟢 merged 2026-05-24 |

### Round 7 — Final Engine Consolidation (NEW)

Closes the post-Round-6 engine sprint. All items merged 2026-05-25.

| Item | Status | PR | Notes |
|---|---|---|---|
| Signal-level EET timezone audit + canonical alignment utility | 🟢 | #193 | `core/signals/htf_alignment.py`; 8 modules fixed; 33 new tests |
| Amendment 5 — AUC-gated A2/A6 architecture selection + parser v1.3.1 field | 🟢 | #194 | Dispatch-time enforcement; engine unchanged |
| Canonical exit-policy registry + `sl_partial_close_1r_runner_trail` primitive + per-arc migration | 🟢 | #195 | `core/sim/exit_policies/`; Account partial-fill semantics; 218-test reference parity |
| EET session semantics: distance / reset_floor / `compute_per_day_max_dd` + **Amendment 6** | 🟢 | #197 | `core/time_utils/session_boundary.py`; `Panel.boundary_convention` propagation; daily-DD boundary now EET broker trading day |
| CC_22 docs comprehensive refresh — post-Phase-1-engine-build state alignment | 🟢 | this PR | Tier 1-4 audit; full TODO + engine_capability_audit refresh; Amendment 6 prefatory block in L_PROTOCOL |

---

## Phase 0 — Framework validation

| Task | Status | Notes |
|---|---|---|
| Phase 0a: KH-24 full system anchor re-run on HistData | 🟢 | Closed PR-E.1.7 under Path B verdict. F7 in-band; F2 sign recoverable; F1/F4/F5/F6 attributable to documented residuals (Sections G+H deferred + data-source drift). v3 certified Phase 0 ready. |
| Phase 0b: KH-24 base signal through L_PROTOCOL v3.0 | ⚫ | SKIPPED by chat decision — base signal through v3.0 protocol deemed not worth running given Path A verdict on Phase 0a. |

If a future arc requires tighter anchor reproduction than Path B, the
path is Sections G (news filter) + H (post-fill SL anchor) per
`docs/dispatches/kh24_ea_full_diff.md`.

---

## Phase 1 — Arc 1-11 re-runs

### Wave composition (locked)

**Wave 1 (5 arcs) — LOCKED. Strongest in-cohort edge:**
- Arc 5 — `mtf_alignment.2_down_mixed.kijun` (admit-only +0.14-0.21R; V-shape c1)
- Arc 8 — PR-HHHL long (c1 +2.59R/trade; V-shape FG-weak)
- Arc 10 — DLR long (oracle Sharpe 4.61; V-shape c1 near-miss)
- Arc 11 — SHB long (oracle 101% worst-fold; V-shape c1)
- Arc 7 — liquidity sweep + reclaim (first capturable-not-extractable; 3 V-shape units survive §2)

**Wave 2 (6 arcs) — LOCKED. Standard + low priority:**
- Arc 4 RERUN — `bar_range_top_decile.neg.h_001`
- Arc 4 original — same signal as RERUN (separate due to pre-spread-audit numbers)
- Arc 6 — failed-breakout reversal long (Stepwise cohort)
- Arc 3 — `volatility_regime.d1_atr_top_decile`
- Arc 1 — LCHAR rank 1
- Arc 2 — LCHAR rank 2 (SHELVED under v2.1.2)

**Deferred:** Arc 9 (IB-trend) — only if Wave 2 surfaces nothing. Known lookahead-corrupted under v2.x; low expected value.

### Wave 1 status

| Arc | Verdict (closure) | Re-eval verdict (Amend 3) | Closure PR |
|---|---|---|---|
| Arc 8 | FAIL (step 5) | FAIL | merged (PR #173) |
| Arc 10 | PASS-VIABLE | PASS-DEPLOYABLE | merged (PR #174) — re-eval PR #177 |
| Arc 11 | FAIL (step 5) | FAIL | merged (PR #171) |
| Arc 5 | FAIL (step 5) | — | 🟡 OPEN — PR #172 in flight |
| Arc 7 | FAIL (step 5) | — | 🟡 OPEN — PR #180 in flight |

### Phase 1 task list

| Task | Status | Notes |
|---|---|---|
| Wave 1 composition locked | 🟢 | Arc 5, 8, 10, 11, 7 |
| Wave 2 composition locked | 🟢 | Arc 4 RERUN, 4 original, 6, 3, 1, 2 (+ Arc 9 deferred) |
| Wave 1 closures: Arcs 8, 10, 11 | 🟢 | |
| Wave 1 closures: Arcs 5, 7 | 🟡 | PRs #172 + #180 open |
| Arcs 8/10/11 retroactive Amendment-3 re-evaluation | 🟢 | PR #177 |
| Arc 5 v3.0.1 retry under signal-parity engine + Amendment 3 gates + signal-EET fix | 🔴 | Blocked on PR #172 closure. Engine prerequisites met (PRs #189, #193, #197). |
| Arc 7 v3.0.1 retry | 🔴 | Blocked on PR #180 closure. Engine prerequisites met. |
| Arc 10 signal-parity rerun on UTC convention | 🔴 | Procedure documented at [docs/calibration/arc_10_signal_parity_rerun_2026_05.md](docs/calibration/arc_10_signal_parity_rerun_2026_05.md); expected near-zero delta. PR #193 + Amendment 6 do NOT change Arc 10 UTC-baseline numbers. |
| Wave 1 closes (cross-arc tracker review for patterns) | 🔴 | Gated on retries above |
| Dispatch Wave 2 (6 arcs in parallel) | 🔴 | Gated on Wave 1 close. **Wave 2 dispatches must apply L_PROTOCOL Amendment 5 four-gate architecture-selection rule from first dispatch** (Phase 1 chat-side; out of scope for this PR). |
| Wave 2 closes; cross-arc synthesis | 🔴 | |
| Decision: any deployable found? | 🔴 | |
| Retroactive audit Arc 8 v3.0 + Arc 11 v3.0 under L_PROTOCOL Amendment 5 | ⚪ | Deferred until Arc 7 v3.0.2 closes — verdict informs whether retroactive retries are worth running. Criterion per Amendment 5 §7: "FAIL closed with at least one Step 3 surviving cluster whose Step 4 AUC ≥ 0.65 was not evaluated under A2 AND A6 under prior protocol." Arc 10 v3.0 already evaluated A6 under Amendment 1's V-shape mapping; likely unaffected, verify at audit. |

---

## Phase 2 — Sub-protocols + new signals

Triggered after: Phase 1 closes (all 11 arcs).

### Sub-protocol probes

| Task | Status | Notes |
|---|---|---|
| `heavy_ml_probe` build (PR-A scaffolding + lineage gate + IO) | 🟡 | PR #187 (parallel chat); PR-B/C/D/E/F to follow |
| `signal_discovery_probe` infrastructure | 🟢 | Built via PR #175 (arc_discovery_01 dispatch) |
| `signal_discovery_probe` 10k local run by user | 🟡 | User-side workstation operation, in progress |
| Discovery top-3 follow-up arcs | 🔴 | Post-local-run + signal-parity engine merged (#189 ✓) |
| `heavy_ml_probe` invocation on best Phase 1 cohort-edge arc | 🔴 | Blocked on Phase 1 close + heavy_ml_probe build complete |
| t=x contamination probe (drafted pre-reset) | 🔴 | Low priority; diagnostic only; useful for Pipeline D / DE validation |

### New signal classes (each becomes a Phase 2 arc)

All 🔴 NOT STARTED — defer to Phase 2 trigger.

| Task | Notes |
|---|---|
| Currency strength signal (cross-sectional momentum) | Likely needs currency_strength sub-protocol |
| Asia session range breakout (Arc 13 from old queue) | Session-anchored |
| Failed-breakdown reversal in uptrend (Arc 15 from old queue) | Symmetric to Arc 6 |
| Persistent-momentum continuation (Arc 16 from old queue) | Bar-statistics feature class |
| Three-bar bullish reversal (Arc 12 from old queue) | Multi-bar sequence |
| News-window-anchored signals | Requires economic calendar feed |

---

## Engine consolidation — items the engine still needs

| Item | Status | Notes |
|---|---|---|
| A5 portfolio composition spec + build | 🔴 | Deferred until Wave 2 produces VIABLE candidates. Arc 8 first candidate. |
| Full-window sim for chained DD (`chained_dd_method: full_window_sim`) | 🔴 | v3.0.2 follow-up; replaces equity-stitching default from PR #186. Required post-Wave-2 per closure template v1.3. |
| Audit doc footer (items MISSING → WIRED as they ship) | 🟢 | engine_capability_audit_2026_05.md refreshed in CC_22 with post-PR-#197 footer |
| Arc 8 closure §10 correction | 🔴 | `primary_failure_mode` → `step5_not_scalable`; tracker per-failure-mode adjustment |
| Cross-platform CI | 🔴 | Linux-only currently; user dev is Windows. Non-blocking housekeeping. |
| Closure-writer YAML auto-emission | 🔴 | Currently hand-written; defer until orchestrator gains closure-emit capability |
| Legacy engine retirement (migrate `live/run_daily.py` + 32 importers off `core/backtester.py` + `core/signal_logic.py`) | 🔴 | Gate: KH-24 anchor preservation (±0.5pp ROI / ±1pp DD). Out of scope for current PRs; separate dispatch needed. |
| EA mid-trail update (live MT5 EA still uses bid-side `CopyClose`) | 🔴 | To restore backtest↔EA parity post-PR-#189; separate deployment PR |
| Backfill `AMENDMENT_5_CUTOFF_ISO` in `scripts/tracker_parser/schema.py` with PR #194 merge timestamp | 🔴 | Placeholder pinned at `2026-05-23T00:00:00Z`. Mirror PR-186 / Amendment 3 backfill pattern. Trivial one-line follow-up PR. |
| Backfill `AMENDMENT_5_1_CUTOFF_ISO` in `scripts/tracker_parser/schema.py` with PR Amendment-5.1 merge timestamp | 🔴 | Placeholder pinned at `2026-05-25T00:00:00Z`. Mirror `AMENDMENT_5_CUTOFF_ISO` backfill pattern. Trivial one-line follow-up PR. |
| KH-24 anchor F2/F3 fold-2 anchor-drift bisect investigation | 🔴 | Pre-existing on main; F2/F3 divergence between legacy `KH24FoldRunner` (warmup_days=30) and A1 path (full-history warmup). Documented; queued investigation. |
| Triple-spread sensitivity diagnostic (Step 6 §6.3 enhancement) | 🔴 | Optional Step 6 framework enhancement |
| HistData ↔ 5ers MT5 spread comparison (calibration task) | 🔴 | User-side; procedure at [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md §5](docs/calibration/histdata_mt5_aggregation_parity_2026_05.md) |
| Cross-arc V-shape clusterifier (Phase 2 research) | 🔴 | EXP-05 pool: AUC 0.6348 with `L1_minus_L0_atr` mandatory. Arc 7 c3 + Arc 10 c1 pooled. Leading v2.4-era candidate; carry into Phase 2. |
| Closure-writer YAML auto-emission | 🔴 | Quality-of-life; defer |

---

## Standing items / open questions

- **Signal parity gap:** 🟢 RESOLVED via PR #189 (mid-price feature refactor + 5ers EET bar boundaries). All future engine output is venue-independent. Pre-PR-#189 closures (Arcs 8, 10, 11; Arcs 5, 7 closures in flight) carry venue-specific signal artefacts; retries documented in Phase 1 task list.
- **Signal-module timezone gap:** 🟢 RESOLVED via PR #193 (canonical `core/signals/htf_alignment.py` utility + 8 modules fixed). Audit at `docs/audits/signal_module_eet_audit_2026_05.md`.
- **EET session semantics gap:** 🟢 RESOLVED via PR #197 (Amendment 6; daily-DD boundary aligned to EET broker trading day; `core/time_utils/session_boundary.py` canonical). Three convention-aware consumers wired: `distance.py`, `reset_floor.py`, `compute_per_day_max_dd`.
- **5ers timezone verification:** PR #189 assumes 5ers = EET/EEST (EU DST rules). User to verify against 5ers documentation. If NY-close session rollover, a small follow-up PR adjusts the boundary convention parameter (engine code is convention-parameterised already).
- **HistData ↔ 5ers MT5 4H comparison (5 majors):** User pulls 5ers MT5 H4 closes from VPS for EURUSD/GBPUSD/USDJPY/AUDUSD/USDCAD over a 30-day post-2020 window; comparison procedure in [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md §5](docs/calibration/histdata_mt5_aggregation_parity_2026_05.md). Acceptance: per-pair mean abs diff <5 pips on majors.
- **5ers_eet cache build for 28 pairs × 7 TFs:** One-time workstation operation per PROTOCOL_RUNTIME.md §15.3.
- **Cross-asset data (DXY / US10Y / SPX as features):** on hold per user. Revisit if Phase 1 doesn't produce deployable.
- **KH-24 live VPS health:** confirmed running normally on 5ers MT5 broker feed. Independent of local data state. Untouched by Phase-1-engine-build sprint (legacy engine path preserved).
- **OpenWebUI / OpenClaw:** evaluated and skipped — no immediate value.
- **Obsidian:** skipped — user doesn't search docs themselves.
- **`STATUS.md` and `CHANGELOG.md` and `SESSION_ZERO.md`:** untouched by reset and untouched by CC_22 per chat decision. Pre-v3.0 legacy; separate cleanup decision not in scope.

---

## Doc set state

### Locked / ready
- `L_PROTOCOL.md` (v3.0 + Amendments 1, 2, 3, 4, 5, 6 inline) 🟢
- `ARC_HISTORY.md` (frozen) 🟢
- `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.3.1 🟢
- `docs/templates/TODO_REFRESH_TEMPLATE.md` (NEW, CC_22 PR) 🟢
- `ARC_TRACKER.md` (parser-auto-updated) 🟢
- `scripts/update_tracker_from_closure.py` (parser; supports template v1.0 / v1.1 / v1.2 / v1.2.1 / v1.3 / v1.3.1) 🟢
- `docs/sub_protocols/heavy_ml_probe.md` (spec; engine build in flight via PR #187) 🟢
- `docs/sub_protocols/signal_discovery_probe.md` (spec + engine wired via PR #175) 🟢
- `docs/audits/engine_capability_audit_2026_05.md` (REFRESHED in CC_22 PR with post-PR-#197 footer summarising items resolved across PRs #185-#197) 🟢
- `docs/audits/signal_module_eet_audit_2026_05.md` (landed PR #193) 🟢
- `docs/calibration/histdata_mt5_aggregation_parity_2026_05.md` (from PR #189; PR-# corrections in CC_22) 🟢
- `docs/calibration/arc_10_signal_parity_rerun_2026_05.md` (from PR #189; PR-# corrections + PR #193 / Amendment 6 impact notes in CC_22) 🟢
- `docs/BACKTESTER_ARCHITECTURE.md`, `docs/PROTOCOL_RUNTIME.md` (updated through PR #197 + CC_22 refresh; Amendment 5 architecture-selection note added) 🟢
- `REPO_INVENTORY.md` (classification of pre-reset .md files) 🟢
- `README.md`, `CLAUDE.md`, `project_brief.md`, `WORKFLOW.md` (CC_22 comprehensive refresh) 🟢

### To create
- Step 6 framework auto-dispatch verification on first PASS arc (Wave 2)
- A5 portfolio composition spec
- Full-window-sim implementation for `chained_dd_method: full_window_sim` (v3.0.2)

### Per-arc folders (created when arc opens)
- `results/<arc_name>/ARC_OPEN.md`
- `results/<arc_name>/ARC_CLOSURE.md`
- `results/<arc_name>/step_<N>/`
- `results/<arc_name>/step_6/` or `step_6_manual_<timestamp>/` (per Amendment 4)

---

## Ideas parked (not yet planned, capture so they're not lost)

- **Heavy-ML "long-shot" exploratory budget:** include RL, tabular NN, LSTM/Transformer at very small compute budget for completeness, separate from AutoML core. Triggers only if standard ML doesn't break through. Compute cap to be defined.
- **Cross-asset features (DXY, US10Y, SPX):** Phase 2 candidate if Phase 1 fails. Data source: HistData if it has them, Stooq / Yahoo as alternative.
- **News-windowed sub-protocol:** requires economic calendar feed (NFP, FOMC, CPI dates). Not currently integrated.
- **D1 SHB diagnostic (drafted pre-reset):** original purpose was to test if 4H entry-feature ceiling is TF-specific. Under v3.0 with TF-flexible arcs, naturally tested when any 4H signal arc runs with `tf_mode: flexible`. May not need standalone probe.
- **Old dispatches (system-level reeval, currency-strength, t=x contamination, free-reign):** preserved as references. Each maps to Phase 2 work in this TODO; no separate re-instantiation needed.

---

## Reminder — what gets updated when

| Doc | Update cadence |
|---|---|
| `L_PROTOCOL.md` | Only at major redesign events (Amendments are inline; current: 1-6) |
| `docs/sub_protocols/*` | When sub-protocol is amended |
| `docs/templates/ARC_CLOSURE_TEMPLATE.md` | Only at template version bump (locked at v1.3.1) |
| `docs/templates/TODO_REFRESH_TEMPLATE.md` | Stable — generic placeholders, rarely changed |
| `ARC_TRACKER.md` | Auto on arc open/close via parser (`scripts/update_tracker_from_closure.py`) |
| `ARC_HISTORY.md` | Never (frozen at v3.0 start) |
| `TODO.md` (this file) | Manually as work progresses; full rewrites every 2-3 weeks at major state transitions or after substantial PR clusters |
| `README.md`, `CLAUDE.md`, `project_brief.md` | Rarely; bundled with TODO refresh |
| `WORKFLOW.md` | When operational conventions evolve |
| `BACKTESTER_ARCHITECTURE.md`, `DATA_FOUNDATION.md`, `PROTOCOL_RUNTIME.md` | When backtester / data source / runtime changes |
| Per-arc `ARC_OPEN.md` / `ARC_CLOSURE.md` | Write-once |
| `docs/calibration/*` | Per calibration / cross-broker run; ad-hoc |
| `docs/audits/*` | Periodic capability audit (next due if engine gains a significant new module) |

This file (TODO.md) is the only doc with continuous manual updates. ARC_TRACKER is the only doc with continuous auto-updates. Everything else is rare or write-once.

---

## §6 PR numbering convention

This file uses **GitHub PR numbers** throughout. Earlier internal dispatch markdowns (e.g. `CC_15_SIGNAL_PARITY_ENGINE.md`, `CC_16_TODO_REFRESH.md`) used notional labels that pre-dated the actual GitHub sequence. For reference, the dispatch-to-GitHub mapping for the post-PR-184 sequence:

| Dispatch notional label | GitHub PR | Description |
|---|---|---|
| CC_15 "PR #187 signal parity" | **PR #189** | Signal parity engine (this is the dispatch's notional "PR #187") |
| (not in CC_15 dispatch) | PR #187 | heavy_ml_probe PR-A (parallel chat) |
| (not in CC_15 dispatch) | PR #188 | Step 6 causal audit framework + Amendment 4 + parser v1.3 (this mastermind chat) |
| (not in any dispatch label) | PR #193 | Signal-level EET timezone audit + fix |
| (not in any dispatch label) | PR #194 | Amendment 5 — AUC-gated A2/A6 + parser v1.3.1 field |
| (not in any dispatch label) | PR #195 | Canonical exit-policy registry + `sl_partial_close_1r_runner_trail` |
| CC_20 "EET session semantics" | **PR #197** | EET session semantics + Amendment 6 daily-DD boundary |
| CC_22 "Comprehensive docs refresh" | this PR | Post-Phase-1-engine-build docs sweep |

When referring to historical PR numbers within closure docs or dispatch logs, the dispatch's notional label is the source of truth for that closure's narrative; the table above provides the bridge to current GitHub state.
