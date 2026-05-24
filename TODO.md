# TODO — Live Project Tracker

> The operational todo list. Append, check off, delete as work completes.
> Distinct from `ARC_TRACKER.md` (which is auto-updated arc state) and `ARC_HISTORY.md` (frozen pre-v3.0 record).
> Last updated: 2026-05-24 (PR #187 signal parity engine — Sub-changes A.distance + A.trail + C.EET)

---

## Status legend
- 🟢 DONE
- 🟡 IN PROGRESS
- 🔴 NOT STARTED
- ⚪ BLOCKED / WAITING
- ⚫ HALTED / CANCELLED

---

## Current state — quick view

**Phase 0 — Framework validation:** 🟢 READY (v3.0 backtester closed PR-E.1.7 / 2026-05-22; KH-24 anchor partial-by-attribution per Path B)

### Queued follow-up dispatches (post PR #187 — signal parity)

- 🔴 **Legacy engine retirement.** Migrate `live/run_daily.py` + 32 other importers off `core/backtester.py` + `core/signal_logic.py` onto V3 paths (core/sim/multipair_backtester, core/features, core/architectures). Then delete both legacy files. **Gate:** KH-24 anchor preservation (±0.5pp ROI / ±1pp DD on worst-fold). Out of scope for PR #187 per chat resolution (KH-24 live deployment continuity). Separate dispatch needed.
- 🔴 **5ers timezone verification.** PR #187 assumes 5ers = EET/EEST (EU DST rules). User to verify against 5ers documentation. If NY-close session rollover, a small follow-up PR adjusts the boundary convention parameter — engine code is convention-parameterised already.
- 🔴 **Arc 10 signal-parity re-run.** Workstation execution of `py -m scripts.l_arc_10_v3.step_5` on the tightened engine to quantify the delta. Procedure documented in [docs/calibration/arc_10_signal_parity_rerun_2026_05.md](docs/calibration/arc_10_signal_parity_rerun_2026_05.md). Expected delta near-zero. Not blocking; documented as required follow-up per dispatch Task 4.
- 🔴 **HistData ↔ 5ers MT5 4H comparison (5 majors).** User pulls 5ers MT5 H4 closes from VPS for EURUSD/GBPUSD/USDJPY/AUDUSD/USDCAD over a 30-day post-2020 window; comparison documented per [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md §5](docs/calibration/histdata_mt5_aggregation_parity_2026_05.md). Acceptance: per-pair mean abs diff <5 pips on majors.
- 🔴 **5ers_eet cache build for 28 pairs × 7 TFs.** One-time workstation operation to populate the new `data/cache/<TF>_5ers_eet/` cache directory. Procedure in PROTOCOL_RUNTIME.md §15.3.
- 🔴 **EA mid-trail update.** Live MT5 EA currently uses `CopyClose(PERIOD_H4)` (bid-side) for trail logic per PR-E.1.6. To maintain backtest↔EA parity with PR #187's mid-trail, the EA must be updated to compute mid from bid+ask CopyClose calls. Separate deployment PR.
**Phase 1 — Arc 1-11 re-runs:** 🔴 NOT STARTED (blocked on Phase 0 launch)
**Phase 2 — New signals + sub-protocol probes:** 🔴 NOT STARTED (blocked on Phase 1 closure)

**HistData download:** 🟢 DONE (28 pairs, 52 GB tick + 18 GB M1 derived; 2026-05-21 backup verified)
**v3.0 backtester reconfig (CC_06):** 🟢 DONE (PR-A through PR-E.1.7 landed)
**v3.0 protocol redesign:** 🟢 DONE (`L_PROTOCOL.md` finalised; locked)
**Repo cleanup + consolidation:** 🟢 DONE (branches cleaned; inventory + arc history complete)

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
| Land `L_PROTOCOL.md` on main | 🔴 |
| Draft `docs/sub_protocols/heavy_ml_probe.md` | 🔴 |
| Draft `docs/sub_protocols/signal_discovery_probe.md` | 🔴 |
| Draft `ARC_TRACKER.md` skeleton (empty schema) | 🟢 |
| Lock Wave 1 / Wave 2 arc composition | 🔴 |
| Backup destination for HistData (separate folder off-repo) | 🟡 user pending |

### Round 3 — Backtester + data foundation

| Task | Status |
|---|---|
| HistData download completes + backup verified | 🟢 |
| Backtester reconfiguration for HistData as primary | 🟢 PR-A |
| Backtester support for new feature classes (cross-pair, currency strength, session, multi-TF) | 🟢 PR-C |
| Backtester WFO logic for 11-fold 2010-2020 + 1-shot 2021-2025 holdout | 🟢 PR-C |
| Backtester multi-pair simultaneous simulation (required for cross-pair features + portfolio architecture) | 🟢 PR-B |
| Backtester determinism (sha256 manifests, lineterminator='\n') | 🟢 PR-D |
| KH-24 strategy + EA-correction rounds 1-3 (anchor reproduction) | 🟢 PR-E.1/E.1.5/E.1.6 |
| `BACKTESTER_ARCHITECTURE.md` audit / update | 🟢 PR-E.1.7 |
| `DATA_FOUNDATION.md` finalised | 🟢 PR-E.1.7 |
| KH-24 anchor reproduction (Mode A) — Path B verdict | 🟢 PR-E.1.7 |

### Round 4 — Claude project folder cleanup

| Task | Status |
|---|---|
| Review `/mnt/project/` against new repo state | 🔴 |
| Remove pre-reset docs (v2.x amendments, dispatch handovers, etc.) | 🔴 |
| Sync currently-active docs (L_PROTOCOL, ARC_HISTORY, etc.) into project | 🔴 |

### Round 5 — Closure infrastructure

| Task | Status |
|---|---|
| Lock `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.0 | 🟢 |
| Extend `ARC_TRACKER.md` schema (cluster registry, cost-decomp registry, tag registry) | 🟢 |
| Land L_PROTOCOL §6 update referencing template | 🟢 |
| Build tracker parser `scripts/update_tracker_from_closure.py` | ⚪ deferred until 2-3 Wave 1 closures land |

---

## Phase 0 — Framework validation (PARALLEL)

Round 3 complete (2026-05-22). Phase 0 UNBLOCKED.

| Task | Status | Notes |
|---|---|---|
| Phase 0a: KH-24 full system anchor re-run on HistData | 🟢 | Completed PR-E.1.6 Mode A; chat Path B verdict (PR-E.1.7). F7 in-band; F2 sign recoverable; F1/F4/F5/F6 attributable to documented residuals (Sections G+H deferred + data-source drift). v3 certified Phase 0 ready. |
| Phase 0b: KH-24 base signal through L_PROTOCOL v3.0 | 🔴 | Tests whether the protocol rediscovers KH-24's design (filters, exposure, SL) from the base signal alone. Ready to dispatch. |

Phase 0a closed under "partial-by-attribution" reading; if a future
arc requires tighter reproduction, the path is Sections G (news
filter) + H (post-fill SL anchor) per `docs/dispatches/kh24_ea_full_diff.md`.

---

## Phase 1 — Arc 1-11 re-runs (WAVES OF 5 + 6)

Triggered after: Phase 0 closes clean.

**Wave 1 (5 arcs) — LOCKED. Strongest in-cohort edge:**
- Arc 5 — `mtf_alignment.2_down_mixed.kijun` (admit-only +0.14-0.21R; V-shape c1)
- Arc 8 — PR-HHHL long (c1 +2.59R/trade; V-shape FG-weak)
- Arc 10 — DLR long (oracle Sharpe 4.61; V-shape c1 near-miss)
- Arc 11 — SHB long (oracle 101% worst-fold; V-shape c1)
- Arc 7 — liquidity sweep + reclaim (first capturable-not-extractable; 3 V-shape units survive §2)

**Wave 2 (6 arcs) — LOCKED. Standard + low priority:**
- Arc 4 RERUN — `bar_range_top_decile.neg.h_001` (full-pool D1 failure under v2.x; v3.0 architecture search may recover)
- Arc 4 original — same signal as RERUN, but treated as separate due to pre-spread-audit numbers
- Arc 6 — failed-breakout reversal long (Stepwise cohort; max-F1 fallback case under v2.2)
- Arc 3 — `volatility_regime.d1_atr_top_decile` (closest miss was §2/§11 row-7 bimodal incompat)
- Arc 1 — LCHAR rank 1 (earliest, lowest expected value)
- Arc 2 — LCHAR rank 2 (SHELVED under v2.1.2)

**Deferred:**
- Arc 9 (IB-trend) — runs only if Wave 2 surfaces nothing. Known lookahead-corrupted under v2.x; signal class itself was clean in oracle. Reason for deferral: low expected value given the cohort wasn't extractable on causally-clean features.

| Task | Status | Notes |
|---|---|---|
| Lock Wave 1 composition | 🟢 | Locked: Arc 5, 8, 10, 11, 7 |
| Lock Wave 2 composition | 🟢 | Locked: Arc 4 RERUN, 4 original, 6, 3, 1, 2 (+ Arc 9 deferred) |
| Dispatch Wave 1 (5 arcs in parallel) | 🔴 | |
| Wave 1 closes; review tracker for patterns | 🔴 | |
| Dispatch Wave 2 (6 arcs in parallel) | 🔴 | |
| Wave 2 closes; cross-arc synthesis | 🔴 | |
| Decision: any deployable found? If yes → ship. If no → Phase 2 | 🔴 | |

---

## Phase 2 — Sub-protocols + new signals

Triggered after: Phase 1 closes (all 11 arcs).

### Sub-protocol probes (run on best Phase 1 candidates)

| Task | Status | Notes |
|---|---|---|
| `heavy_ml_probe` on best Phase 1 arc with cohort edge | 🔴 | AutoML + meta-labeling + survival models on Pipeline D |
| `signal_discovery_probe` on universal feature space | 🔴 | Genetic algorithm rule search with Bonferroni correction |
| t=x contamination probe (was drafted pre-reset) | 🔴 | Diagnostic only; useful for Pipeline D / DE validation |

### New signal classes (each becomes a Phase 2 arc)

| Task | Status | Notes |
|---|---|---|
| Currency strength signal (cross-sectional momentum) | 🔴 | Likely needs currency_strength sub-protocol |
| Asia session range breakout (Arc 13 from old queue) | 🔴 | Session-anchored |
| Failed-breakdown reversal in uptrend (Arc 15 from old queue) | 🔴 | Symmetric to Arc 6 |
| Persistent-momentum continuation (Arc 16 from old queue) | 🔴 | Bar-statistics feature class |
| Three-bar bullish reversal (Arc 12 from old queue) | 🔴 | Multi-bar sequence |
| News-window-anchored signals | 🔴 | Requires economic calendar feed |

---

## Standing items / open questions

- **Cross-asset data (DXY / US10Y / SPX as features):** on hold per user. Revisit if Phase 1 doesn't produce deployable.
- **OpenWebUI / OpenClaw:** evaluated and skipped — no immediate value.
- **Obsidian:** skipped — user doesn't search docs themselves.
- **KH-24 live VPS health:** confirmed running normally on 5ers MT5 broker feed. Independent of local data loss.
- **`L_ARC_PROTOCOL.md:717` prescriptive path:** deferred per UNCERTAIN flag resolution. Will be cleaned in protocol redesign land (next CC dispatch).
- **`STATUS.md` and `CHANGELOG.md` content:** untouched by reset. User decides if reset/delete or retain as historical record.

---

## Doc set state

### Locked / ready
- `L_PROTOCOL.md` (v3.0, finalised, ready to land)
- `ARC_HISTORY.md` (Arcs 1-11 + KH-24 anchor + pre-v2.0 summary)
- `REPO_INVENTORY.md` (classification of all 181 .md files)
- `README.md`, `CLAUDE.md`, `project_brief.md` (active orientation)

### To create
- `docs/sub_protocols/heavy_ml_probe.md`
- `docs/sub_protocols/signal_discovery_probe.md`
- `ARC_TRACKER.md` (empty skeleton)
- `DATA_FOUNDATION.md` (HistData reference — touched at HistData dispatch)
- Updated `BACKTESTER_ARCHITECTURE.md`

### Per-arc folders (created when arc opens)
- `results/<arc_name>/ARC_OPEN.md`
- `results/<arc_name>/ARC_CLOSURE.md`
- `results/<arc_name>/step_<N>/`

---

## Ideas parked (not yet planned, capture so they're not lost)

- **Heavy-ML "long-shot" exploratory budget:** include RL, tabular NN, LSTM/Transformer at very small compute budget for completeness, separate from AutoML core. Triggers only if standard ML doesn't break through. Compute cap to be defined.
- **Cross-asset features (DXY, US10Y, SPX):** Phase 2 candidate if Phase 1 fails. Data source: HistData if it has them, Stooq / Yahoo as alternative.
- **News-windowed sub-protocol:** requires economic calendar feed (NFP, FOMC, CPI dates). Not currently integrated.
- **D1 SHB diagnostic (was drafted pre-reset):** the original purpose was to test if 4H entry-feature ceiling is TF-specific. Under v3.0 with TF-flexible arcs, this is naturally tested when any 4H signal arc runs with `tf_mode: flexible`. May not need standalone probe.
- **Old dispatches (system-level reeval, currency-strength, t=x contamination, free-reign):** preserved as references. Each maps to Phase 2 work in this TODO; no separate re-instantiation needed.

---

## Reminder — what gets updated when

| Doc | Update cadence |
|---|---|
| `L_PROTOCOL.md` | Only at major redesign events |
| `docs/sub_protocols/*` | When sub-protocol is amended |
| `docs/templates/ARC_CLOSURE_TEMPLATE.md` | Only at major redesign events (template version bump) |
| `ARC_TRACKER.md` | Auto on arc open/close |
| `ARC_HISTORY.md` | Never (frozen at v3.0 start) |
| `TODO.md` (this file) | Manually as work progresses |
| `README.md`, `CLAUDE.md`, `project_brief.md` | Rarely |
| `BACKTESTER_ARCHITECTURE.md`, `DATA_FOUNDATION.md` | When backtester or data source changes |
| Per-arc `ARC_OPEN.md` / `ARC_CLOSURE.md` | Write-once |

This file (TODO.md) is the only doc with continuous manual updates. ARC_TRACKER is the only doc with continuous auto-updates. Everything else is rare or write-once.
