# TODO — Live Project Tracker

> The operational todo list. Append, check off, delete as work completes.
> Distinct from `ARC_TRACKER.md` (which is auto-updated arc state) and `ARC_HISTORY.md` (frozen pre-v3.0 record).
> Last updated: 2026-05-29 (post-Arc-10-deployment refresh — Wave 1 closed, Arc 10 LIVE; project now in live-ops + Arc 10 improvements mode)

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
- **Phase 1 — Arc 1-11 re-runs:** 🟡 PARTIAL — **Wave 1 CLOSED** (all 5 arcs run under v3.0.2: Arc 10 PASS-DEPLOYABLE → **LIVE**; Arcs 5 / 7 / 8 / 11 FAIL). **Wave 2 ⚪ PARKED** — deprioritized; project pivoted to deployment of the Arc 10 survivor.
- **Arc 10 — Live operations:** 🟡 ACTIVE — deployed on FundedNext (EET) + 5ers (UTC) demo; demo soak in progress; FundedNext $100k Challenge purchase pending soak verification. Operational source of truth = `arc_10/`.
- **Arc 10 — Improvements (research):** 🟡 the only open research thread — small exploratory probes on the deployed signal; nothing structural.
- **Phase 2 — Sub-protocols + new signals:** ⚪ PARKED — deprioritized along with Wave 2; project focus is deployment + Arc 10 improvements, not new signal classes.
- **Engine state:** **post-deployment-week** (post-PR-#236). Steps 1-5 + signal parity + Step 6 framework (Amendment 4) + Amendment 3 / 3.1 risk-normalised gates + Amendment 5 architecture selection + Amendment 6 EET daily-DD boundary + canonical exit-policy registry. **Live deployment stack wired & operationally validated:** convention-aware Python sidecar (UTC + EET, byte-identical parity proven on both) + thin MQL5 EA + multi-broker plumbing (`--mt5-path`) + `--quick-test`.
- **Architectures wired:** A1, A2, A3, A4, A6 (A5 not built — deferred; no VIABLE candidate triggered it)
- **Protocol state:** L_PROTOCOL v3.0 + Amendments 1, 2, 3, 3.1, 4, 5, 6 inline. Closure template v1.3.1. Parser v1.3 with Amendment-5 + Amendment-3 cutoff validation.

### Active parallel chat work

- **Arc 10 live operations** — chat-side mastermind work (demo soak, Challenge decision). Operational, NOT a research-pipeline item; detail lives in `arc_10/04_runbook/`.
- **heavy_ml_probe** sub-protocol build — 🟢 COMPLETE (PR-A → PR-F merged; latest #206 / #203). Not currently invoked (Phase 2 parked).
- **signal_discovery_probe** discovery runs — 🟢 infra built; `arc_discovery_01` ARCHIVED_PARTIAL (step-1 compute blowup), `arc_discovery_02` PARKED (#233).

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
| Backfill `AMENDMENT_3_1_CUTOFF_ISO` in `scripts/tracker_parser/schema.py` with PR Amendment-3.1 merge timestamp | 🔴 | Placeholder pinned at `2026-05-25T00:00:00Z`. Mirror existing cutoff backfill pattern. |
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
| CC_22 docs comprehensive refresh — post-Phase-1-engine-build state alignment | 🟢 | #207-#209 era | Tier 1-4 audit; full TODO + engine_capability_audit refresh; Amendment 6 prefatory block in L_PROTOCOL |

### Round 8 — Arc 10 deployment week (2026-05-25 → 2026-05-29)

Closes Wave 1 (all 5 arcs run under v3.0.2) and takes the Arc 10 survivor from PASS-DEPLOYABLE to LIVE.

| Task | Status | PR / Tag |
|---|---|---|
| Amendment 3.1 (r_max as deployment cap, not gate threshold) | 🟢 | #211 |
| Step 6 ultimate audit — wire spread P&L + harden every category | 🟢 | #212 |
| Driver-divergence investigation + risk-decoupling regression test | 🟢 | #213 |
| Arc 10 v3.0.2 PASS-DEPLOYABLE-PROVISIONAL under canonical 5ers_eet | 🟢 | #214 |
| Arc 10 v3.0.2 Amendment 3 + Step 6 addendum (bespoke methodology preserved) | 🟢 | #216 |
| Arc 10 v3.0.2 exhaustive contamination audit — AUDIT-CLEAN | 🟢 | #217 |
| Arc 10 v3.0.2 UTC rerun — PASS-DEPLOYABLE (convention-sensitivity characterised) | 🟢 | #218 |
| Arc 8 v3.0.2 retry — FAIL (chained DD blowout) | 🟢 | #231 |
| Arc 11 v3.0.2 retry — FAIL (step5_not_scalable) | 🟢 | #215 |
| Arc 7 v3.0.2 retry — FAIL (A2/A6 follow-up + r=2% analysis) | 🟢 | #202 + #210 |
| Arc 5 v3.0.2 retry — FAIL (step5_not_scalable) | 🟢 | #235 |
| Phase 1 sidecar + thin MQL5 EA (UTC-native deployment build) | 🟢 | #219 (+ #220 / #221 fixes) |
| Arc 10 cost-realism sweep (swap dominates; r_safe required) | 🟢 | #222 + #232 (FundedNext EET swap-free) |
| EA: FILE_COMMON + canonical R (r_atr fix) + strategic exit reasons | 🟢 | #223 |
| EA topology fix — single-chart-multi-pair (per-position H4 bar gate) | 🟢 | #225 + tag `arc-10-topology-validated` |
| Phase 2 parity — sidecar↔lab byte-identical (UTC) | 🟢 | #226 |
| Phase 2 EET parity — broker-convention-aware sidecar + FundedNext parity | 🟢 | #227 + tag `arc-10-eet-parity-validated` |
| Multi-broker plumbing (`--mt5-path` + portable-mode docs) | 🟢 | #228 |
| Sidecar `--quick-test` flag (bypass H4 boundary wait) | 🟢 | #229 |
| Broker server-clock offset check removal (#227 regression fix) | 🟢 | #230 |
| Park arc_discovery_02 (manifest fix + resumable artefact) | 🟢 | #233 |
| `arc_10/` consolidated holy grail folder (30+ docs across 5 subfolders) | 🟢 | #234 |
| Arc 10 deployment retrospective + generic `docs/SIDECAR_EA_BUILD_GUIDE.md` | 🟢 | #236 |
| VPS deployment (Python, MT5×2, services + watchdogs, EAs×2) | 🟢 | runbook in `arc_10/03_deployment/04_vps_setup_guide.md` |

Rounds 1-7 above are historical record — unchanged.

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

### Wave 1 status — CLOSED (all 5 arcs run under v3.0.2)

| Arc | v3.0 verdict | v3.0.2 retry verdict | Closure PR | Live status |
|---|---|---|---|---|
| Arc 10 | PASS-VIABLE → PASS-DEPLOYABLE | **PASS-DEPLOYABLE** (ratio 6.43 EET) | #174 / re-eval #177 / v3.0.2 #214/#216/#217/#218 | **LIVE on demo (FN EET + 5ers UTC); Challenge pending** |
| Arc 8 | FAIL (step 5) | FAIL (chained DD blowout) | #173 / v3.0.2 #231 | n/a |
| Arc 11 | FAIL (step 5) | FAIL (step5_not_scalable) | #171 / v3.0.2 #215 | n/a |
| Arc 7 | FAIL (step 5) | FAIL (step5_not_scalable) | v3.0.2 #202 (+ r=2% analysis #210) | n/a |
| Arc 5 | FAIL (step 5) | FAIL (step5_not_scalable) | v3.0.2 #235 | n/a |

→ **One deployable found: Arc 10.** Project's strategy-level goal (a PASS-DEPLOYABLE survivor) is met.

### Phase 1 task list

| Task | Status | Notes |
|---|---|---|
| Wave 1 composition locked | 🟢 | Arc 5, 8, 10, 11, 7 |
| Wave 2 composition locked | 🟢 | Arc 4 RERUN, 4 original, 6, 3, 1, 2 (+ Arc 9 deferred) |
| Wave 1 closures: all 5 arcs (v3.0.2) | 🟢 | Arc 10 PASS-DEPLOYABLE; Arcs 5/7/8/11 FAIL |
| Arcs 8/10/11 retroactive Amendment-3 re-evaluation | 🟢 | PR #177 |
| Arc 5 v3.0.2 retry under signal-parity engine + Amendment 3 gates | 🟢 | FAIL (step5_not_scalable) — PR #235 |
| Arc 7 v3.0.2 retry | 🟢 | FAIL (step5_not_scalable) — PR #202 + r=2% analysis #210 |
| ~~Arc 10 signal-parity rerun on UTC convention~~ | 🟢 | **Superseded** — UTC rerun (#218) + Phase 2 parity proven byte-identical on both UTC (#226) and EET (#227). |
| Wave 1 closes (cross-arc tracker review for patterns) | 🟢 | Wave 1 complete; only Arc 10 viable |
| **Decision: any deployable found?** | 🟢 | **YES — Arc 10. Goal achieved at the strategy level.** |
| **Arc 10 deployment to live** | 🟢 | **Deployed 2026-05-29 on FundedNext (EET) + 5ers (UTC) demo. See `arc_10/`.** |
| Dispatch Wave 2 (6 arcs in parallel) | ⚪ | **PARKED** — deprioritized. Project pivoted to deploying + improving the Arc 10 survivor rather than running more research arcs. Reopenable if Arc 10 fails its live kill criteria. **Any future Wave 2 dispatch must apply Amendment 5 four-gate architecture selection from first dispatch.** |
| Wave 2 closes; cross-arc synthesis | ⚪ | PARKED (gated on Wave 2 reopen) |
| Retroactive audit Arc 8 v3.0 + Arc 11 v3.0 under L_PROTOCOL Amendment 5 | ⚪ | Deferred. Criterion per Amendment 5 §7: "FAIL closed with at least one Step 3 surviving cluster whose Step 4 AUC ≥ 0.65 was not evaluated under A2 AND A6 under prior protocol." Low priority given deployment pivot. |

---

## Arc 10 — Live operations

> **Live system status as of 2026-05-29.** Operational detail lives in `arc_10/04_runbook/`. This TODO captures research-pipeline / decision-gate status only; it references `arc_10/`, it does not duplicate it.

Deployed on a single Contabo VPS (Frankfurt): FundedNext ($100k Challenge target, EET convention, 0.50% risk) + 5ers ($10k demo, UTC convention, 0.40% risk). Convention-aware Python sidecar + thin MQL5 EA per broker; watchdog auto-restart on stale heartbeat.

| Item | Status | Notes |
|---|---|---|
| Demo soak (both brokers) | 🟡 | In progress; verify before market reopen Sunday 2026-06-01 |
| FundedNext $100k Challenge purchase | ⚪ | Pending soak verification + operator decision |
| Risk ramp Week 1 (0.20%) | 🔴 | Triggers on Challenge purchase |
| Risk ramp Week 2 (0.30%) | 🔴 | Triggers Week 1 + 1 if clean |
| Risk ramp Week 3+ (0.50% target) | 🔴 | Triggers Week 2 + 1 if clean |
| KH-24 retirement | 🔴 | Triggers on Arc 10 reaching funded status |
| Weekly reconciliation script build | ⚪ | Deferred until 4+ weeks of live data exist; spec at `arc_10/04_runbook/02_weekly_check.md` |
| First 50-trade comparison vs live-tracking framework | 🔴 | Triggers automatically when ~50 trades closed; framework at `arc_10/04_runbook/06_live_tracking_framework.md` |
| FundedNext rules quarterly verification | 🔴 | Recurring — first one Q3 2026 |
| 5ers copy-trading policy clarification | ⚪ | Pending written 5ers confirmation before opening parallel 5ers accounts |

---

## Arc 10 — Improvements (research)

> The only open research thread. Small, exploratory probes on the deployed signal — nothing structural, nothing that gates deployment. Lower priority than live operations.

| Task | Status | Notes |
|---|---|---|
| Arc 10 improvement probes | 🟡 | Exploratory; scope set chat-side as interesting ideas surface. Must not touch the locked deployed config — any candidate improvement goes through a fresh validation arc, not an in-place edit. |
| Cross-arc V-shape clusterifier (carry-over candidate) | 🔴 | EXP-05 pool: AUC 0.6348 with `L1_minus_L0_atr` mandatory; Arc 7 c3 + Arc 10 c1 pooled. Leading research candidate to fold into Arc 10 improvements if pursued. |

---

## Phase 2 — Sub-protocols + new signals

> ⚪ **PARKED.** Deprioritized along with Wave 2. Project focus is deploying + improving the Arc 10 survivor, not opening new signal classes. Reopenable if Arc 10 fails live and the research pipeline is restarted.

Originally triggered after: Phase 1 closes (all 11 arcs). Wave 1 closed, but Wave 2 was parked rather than run, so the Phase 1 trigger condition is intentionally not met.

### Sub-protocol probes

| Task | Status | Notes |
|---|---|---|
| `heavy_ml_probe` build | 🟢 | COMPLETE — PR-A → PR-F merged (latest #206 / #203). Spec + engine shipped. |
| `signal_discovery_probe` infrastructure | 🟢 | Built via PR #175 |
| `signal_discovery_probe` discovery runs | 🟢 | `arc_discovery_01` ARCHIVED_PARTIAL (step-1 compute blowup, 511/10000 rules); `arc_discovery_02` PARKED (#233) |
| Discovery top-3 follow-up arcs | ⚪ | PARKED with Phase 2 |
| `heavy_ml_probe` invocation on best Phase 1 cohort-edge arc | ⚪ | PARKED with Phase 2 (build is ready; not invoked) |
| t=x contamination probe (drafted pre-reset) | ⚪ | PARKED; low priority diagnostic |

### New signal classes (each becomes a Phase 2 arc)

All ⚪ PARKED — defer to Phase 2 reopen (which is itself parked behind the deployment pivot).

| Task | Notes |
|---|---|
| Currency strength signal (cross-sectional momentum) | Likely needs currency_strength sub-protocol |
| Asia session range breakout (Arc 13 from old queue) | Session-anchored |
| Failed-breakdown reversal in uptrend (Arc 15 from old queue) | Symmetric to Arc 6 |
| Persistent-momentum continuation (Arc 16 from old queue) | Bar-statistics feature class |
| Three-bar bullish reversal (Arc 12 from old queue) | Multi-bar sequence |
| News-window-anchored signals | Requires economic calendar feed |

---

## Engine consolidation — deployment stack (WIRED this deployment week)

| Item | Status | PR | Notes |
|---|---|---|---|
| Phase 1 sidecar + thin MQL5 EA (UTC-native) | 🟢 | #219 | The live deployment build (+ #220/#221 fixes) |
| Sidecar convention-aware boundary (UTC + EET) | 🟢 | #227 | `deployment/sidecar/boundary.py` |
| Phase 2 parity (UTC + EET, byte-identical) | 🟢 | #226 / #227 | Both conventions proven against lab |
| Multi-broker `--mt5-path` | 🟢 | #228 | Deterministic MT5 attach + portable mode |
| Sidecar `--quick-test` | 🟢 | #229 | Bypass H4 boundary wait for smoke tests |
| Broker server-clock offset check removal | 🟢 | #230 | #227 regression fix |
| EA `r_atr` bug fix (canonical R = sl_distance) | 🟢 | #223 | + FILE_COMMON + strategic exit reasons |
| EA topology fix (per-position H4 tracking) | 🟢 | #225 | single-chart-multi-pair |
| Arc 10 deployment-readiness | 🟢 | (deployment week) | Was OPEN; now CLOSED — system live on FundedNext + 5ers demo |

## Engine consolidation — items the engine still needs

| Item | Status | Notes |
|---|---|---|
| A5 portfolio composition spec + build | 🔴 | Deferred — no VIABLE candidate triggered it (Wave 2 parked). |
| Full-window sim for chained DD (`chained_dd_method: full_window_sim`) | ⚪ | v3.0.2 follow-up; replaces equity-stitching default from PR #186. Was "required post-Wave-2"; parked with Wave 2. |
| Audit doc footer (items MISSING → WIRED as they ship) | 🟢 | engine_capability_audit_2026_05.md refreshed in CC_22 with post-PR-#197 footer |
| Arc 8 closure §10 correction | 🔴 | `primary_failure_mode` → `step5_not_scalable`; tracker per-failure-mode adjustment |
| Cross-platform CI | 🔴 | Linux-only currently; user dev is Windows. Non-blocking housekeeping. |
| Closure-writer YAML auto-emission | 🔴 | Currently hand-written; defer until orchestrator gains closure-emit capability |
| Legacy engine retirement (migrate `live/run_daily.py` + 32 importers off `core/backtester.py` + `core/signal_logic.py`) | 🔴 | Gate: KH-24 anchor preservation (±0.5pp ROI / ±1pp DD). Out of scope for current PRs; separate dispatch needed. |
| EA mid-trail update (live MT5 EA still uses bid-side `CopyClose`) | 🔴 | To restore backtest↔EA parity post-PR-#189; separate deployment PR |
| KH-24 anchor F2/F3 fold-2 anchor-drift bisect investigation | 🔴 | Pre-existing on main; F2/F3 divergence between legacy `KH24FoldRunner` (warmup_days=30) and A1 path (full-history warmup). Documented; queued investigation. |
| Triple-spread sensitivity diagnostic (Step 6 §6.3 enhancement) | 🔴 | Optional Step 6 framework enhancement |
| HistData ↔ 5ers MT5 spread comparison (calibration task) | 🔴 | User-side; procedure at [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md §5](docs/calibration/histdata_mt5_aggregation_parity_2026_05.md) |
| Cross-arc V-shape clusterifier | 🔴 | EXP-05 pool: AUC 0.6348 with `L1_minus_L0_atr` mandatory. Arc 7 c3 + Arc 10 c1 pooled. Moved to the **Arc 10 — Improvements** section as the leading candidate to pursue. |
| Closure-writer YAML auto-emission | 🔴 | Quality-of-life; defer |

---

## Standing items / open questions

- **Signal parity gap:** 🟢 RESOLVED via PR #189 (mid-price feature refactor + 5ers EET bar boundaries). All Wave 1 arcs were subsequently re-run under v3.0.2 on the parity engine; Arc 10's live deployment additionally proves sidecar↔lab byte-identity on both UTC (#226) and EET (#227).
- **Signal-module timezone gap:** 🟢 RESOLVED via PR #193 (canonical `core/signals/htf_alignment.py` utility + 8 modules fixed). Audit at `docs/audits/signal_module_eet_audit_2026_05.md`.
- **EET session semantics gap:** 🟢 RESOLVED via PR #197 (Amendment 6; daily-DD boundary aligned to EET broker trading day; `core/time_utils/session_boundary.py` canonical). Three convention-aware consumers wired: `distance.py`, `reset_floor.py`, `compute_per_day_max_dd`.
- **Broker convention verification:** 🟢 RESOLVED at deployment. 5ers deployed on **UTC** boundaries; FundedNext on **EET** boundaries. Sidecar is convention-aware and parity-proven on both (#226 / #227). The earlier "is 5ers EET or NY-close?" open question is moot — 5ers is deployed UTC.
- **HistData ↔ 5ers MT5 4H comparison (5 majors):** User pulls 5ers MT5 H4 closes from VPS for EURUSD/GBPUSD/USDJPY/AUDUSD/USDCAD over a 30-day post-2020 window; comparison procedure in [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md §5](docs/calibration/histdata_mt5_aggregation_parity_2026_05.md). Acceptance: per-pair mean abs diff <5 pips on majors.
- **5ers_eet cache build for 28 pairs × 7 TFs:** One-time workstation operation per PROTOCOL_RUNTIME.md §15.3.
- **Cross-asset data (DXY / US10Y / SPX as features):** on hold per user. Revisit if Phase 1 doesn't produce deployable.
- **KH-24 live VPS health:** confirmed running normally on 5ers MT5 broker feed. Independent of local data state. Untouched by Phase-1-engine-build sprint (legacy engine path preserved).
- **KMeans cluster-label assumption in `core/steps/classifier_persistence.py`:** 🔴 OPEN (low priority). The A2 orchestrator path assumes Step-2 cluster IDs start at 0, but KMeans can return labels `[1,2,3,4]` in different environments → `test_a2_end_to_end.py` cross-environment CI flake (`ValueError: cluster_id 0 not present in Step4Result`). Fix: cluster-label → candidate lookup must iterate over the actual returned labels, not assume a 0-based range. Affects strategy research only (L_PROTOCOL Step 2 clustering); does NOT affect deployment, sidecar, EA, or live trading. Fix post-FundedNext deploy.
- **GitHub PAT exposure (Arc 10 deployment):** A fine-grained PAT was pasted into chat during VPS setup; redacted from `arc_10/` docs via a follow-up commit. Operator declined to revoke; token expires Aug 2026. Tracked in `arc_10/05_history/04_open_items.md`.
- **`tests/ea/scenarios.json` path:** actual location is `tests/ea/scenarios/scenarios.json`; corrected in `arc_10/02_validation/04_st_scenarios.md` during PR #234 follow-up.
- **Weekly reconciliation script:** deferred until 4+ weeks of live data, then build per `arc_10/04_runbook/02_weekly_check.md`. (Also tracked in the Arc 10 live-operations table above.)
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
- `docs/sub_protocols/heavy_ml_probe.md` (spec + engine COMPLETE via PR-A → PR-F, latest #206 / #203) 🟢
- `docs/sub_protocols/signal_discovery_probe.md` (spec + engine wired via PR #175) 🟢
- `docs/audits/engine_capability_audit_2026_05.md` (REFRESHED in CC_22 PR with post-PR-#197 footer summarising items resolved across PRs #185-#197) 🟢
- `docs/audits/signal_module_eet_audit_2026_05.md` (landed PR #193) 🟢
- `docs/calibration/histdata_mt5_aggregation_parity_2026_05.md` (from PR #189; PR-# corrections in CC_22) 🟢
- `docs/calibration/arc_10_signal_parity_rerun_2026_05.md` (from PR #189; PR-# corrections + PR #193 / Amendment 6 impact notes in CC_22) 🟢
- `docs/BACKTESTER_ARCHITECTURE.md`, `docs/PROTOCOL_RUNTIME.md` (updated through PR #197 + CC_22 refresh; Amendment 5 architecture-selection note added) 🟢
- `REPO_INVENTORY.md` (classification of pre-reset .md files) 🟢
- `README.md`, `CLAUDE.md`, `project_brief.md`, `WORKFLOW.md` (CC_22 comprehensive refresh) 🟢
- **`arc_10/` consolidated holy grail folder (30+ docs across 5 subfolders: strategy / validation / deployment / runbook / history)** — single source of truth for the deployed Arc 10 system (PR #234; retrospective added #236) 🟢
- **`docs/SIDECAR_EA_BUILD_GUIDE.md`** — generic, system-agnostic guide for building a Python sidecar + thin MQL5 EA from a validated lab signal (PR #236) 🟢

### To create
- Weekly reconciliation script + corresponding markdown (deferred until 4+ weeks live data)
- Step 6 framework auto-dispatch verification on first PASS arc (⚪ parked with Wave 2)
- A5 portfolio composition spec (⚪ parked — no VIABLE candidate)
- Full-window-sim implementation for `chained_dd_method: full_window_sim` (⚪ parked with Wave 2)

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
