# CLAUDE.md — Forex Ignition Rebuild
> Last updated: 2026-05-31 | Phase: **Arc 10 DEPLOYED — live operations; forward research parked.** L_PROTOCOL v3.0 + Amendments 1-7 locked. v3.0 engine wired end-to-end across `core/arc/`, `core/steps/`, `core/architectures/` (A1, A2, A3, A4, A6 — A5 deferred), `core/runners/`, `core/step_6/`, `core/sim/exit_policies/`, `core/time_utils/`. KH-24 anchor preserved (A1 path byte-identical to legacy `KH24FoldRunner` under `boundary_convention="utc"`). KH-24 live deployment on Contabo VPS / 5ers MT5 unchanged.
> First file any AI assistant reads. Reflects where the project ACTUALLY is.
>
> **Arc 10 is deployed and analysis-complete; for live state read `arc_10/START_HERE.md`. Do not re-open the risk analysis — it is canonical in `arc_10/02_validation/07_canonical_wfo.md`.**

---

## Active protocol

**L_PROTOCOL v3.0** (`L_PROTOCOL.md`) is the overseer methodology for all forward research. Five steps as rankings + lazy Step 6 (causal audit). WFO at Step 5 is the only deployment gate. Seven amendments are inline / inline-archived:

- **Amendment 1** — Step 5 search policy (informed by Steps 3-4, not exhaustive).
- **Amendment 2** — ML architecture mechanics for A2, A3, A4, A6.
- **Amendment 3** — risk-normalised gates (`r_safe` / `r_hard` scaling; priority-ordered failure modes).
- **Amendment 4** — Step 6 causal-audit framework (six categories; auto-dispatch on Top-1 PASS candidate).
- **Amendment 5** — AUC-gated A2/A6 architecture selection (four-gate dispatch-time rule).
- **Amendment 6** — 5ers EET broker trading day as the daily-DD measurement boundary.
- **Amendment 7** — portfolio-level DD gating + EA-faithful floating-equity sizing (`ACCOUNT_EQUITY × r_base`, floating P&L included, re-read per entry) + both DD references reported (from-initial = broker MLL basis; trailing = conservative planning anchor). Applies to all future arcs.

Archive files: `archive/L_PROTOCOL_v3_0_AMENDMENT_3.md`, `..._AMENDMENT_4.md`, `..._AMENDMENT_5.md`. Amendments 1, 2, 6 are documented inline in `L_PROTOCOL.md` only.

Historical: Arcs 1, 2, ..., 11 (under v1.x and v2.x protocols) live in `ARC_HISTORY.md` (frozen). v3.0 Wave 1 is CLOSED (all 5 arcs run under v3.0.2): Arc 10 PASS-DEPLOYABLE → **LIVE**; Arcs 5 / 7 / 8 / 11 FAIL.

---

## Read These First, In Order

1. **`arc_10/START_HERE.md`** — entry point for the deployed live system. Arc 10 is the deployable survivor; read this for live state.
2. **`arc_10/02_validation/07_canonical_wfo.md`** — canonical results reference (EA-faithful, floating-equity). Source of truth for every Arc 10 number.
3. **`L_PROTOCOL.md`** — the methodology of record. Self-contained for gates, deliverables, and architectures.
4. **`TODO.md`** — operational tracker: current phase, in-flight arcs, engine work, standing items.
5. **`ARC_TRACKER.md`** — auto-managed arc state. Read for closed-arc status; do not hand-edit.
6. **`WORKFLOW.md`** — operational conventions, dispatch artefact pattern, branch strategy.

Then, depending on scope:
- Touching the live KH-24 system → `docs/KH24_SYSTEM_LOCK.md`
- Engine work or runtime questions → `docs/PROTOCOL_RUNTIME.md` + `docs/BACKTESTER_ARCHITECTURE.md`
- Opening or closing an arc → `docs/templates/ARC_CLOSURE_TEMPLATE.md` + `scripts/tracker_parser/README.md`
- Capability inventory → `docs/audits/engine_capability_audit_2026_05.md`
- Project history / narrative → `project_brief.md`

Sub-protocols at `docs/sub_protocols/` (`heavy_ml_probe.md`, `signal_discovery_probe.md`) define the canonical Step-1 / Step-4 overrides.

---

## CRITICAL: Current State

**Live system KH-24 is locked, passing, deployed.** Out of scope for any forward research work without an explicit modification phase.

**Arc 10 is LIVE-READY / DEPLOYED — analysis COMPLETE.** Arc 10 v3.0.2 (DLR signal) is the deployable survivor: deployed on FundedNext (EET, $100k Challenge) at **0.40% operating risk**, EA-faithful floating-equity sizing. The canonical gate is the **EA-faithful** run (`arc_10/02_validation/07_canonical_wfo.md`): worst-fold ROI 14.42%, mean-fold 32.48%, worst-fold trailing DD 8.21% / from-init 5.49% / daily 4.11%, 0 kills, PASS-DEPLOYABLE on FundedNext's from-initial basis. **0.50% is a marginal / evidence-gated future upgrade, not the operating tier** (FAILS trailing 10.89% / daily 5.16%). The risk-surface analysis (0.40% vs 0.50%, sizing basis, governors, gap/slippage) is canonical and closed — do not re-open it; if a number is questioned it traces to the EA-faithful CSVs in `results/l_arc_10_v3.0.2_ea_faithful/`.

The earlier "PASS-VIABLE → Amendment-3 PASS-DEPLOYABLE under UTC / parity rerun pending" framing is **superseded** by the EA-faithful run. The UTC / signal-parity-rerun narrative is legacy.

**Live-only residuals** (cannot be closed by backtest, tracked in `TODO.md`): tick-gap behaviour on first live spike day; 0.50% as a future evidence-gated upgrade; two pending FundedNext written confirms (cycle 4%-measurement basis; exact scale-event profit mechanic). OPEN-001 (EA floor re-baseline) is RESOLVED.

**Forward research is PARKED.** Wave 2 (Arcs 4-RERUN, 4-original, 6, 3, 1, 2) and Phase 2 (sub-protocols + new signal classes) are deprioritized behind the deployment pivot. Reopenable if Arc 10 fails its live kill criteria.

---

## What the Project Is

A long-only 4H trend-pullback system (KH-24) is in production. Parallel research arcs run through `L_PROTOCOL.md` v3.0 — five gates-as-rankings steps plus a lazy Step 6 causal audit framework. The goal is one or more PASS-DEPLOYABLE survivor systems that complement or supersede KH-24.

Any arc (standard, discovery, or diagnostic probe) plugs into the same overseer protocol, with sub-protocols at `docs/sub_protocols/` overriding specific steps when needed. The closure format and tracker integration are universal.

---

## Locked Philosophy

| Rule | Detail |
|------|--------|
| Structure-first | Signal is price structure, not indicator |
| WFO worst-fold is the only judge at Step 5 | Average fold, best fold — irrelevant |
| Dual-tier disposition at Step 5 | PASS-DEPLOYABLE / PASS-VIABLE / FAIL; DD ≤ 8% at `r_safe` (DEPLOYABLE), ≤ 10% at `r_hard` (VIABLE) |
| Step 6 causal audit is mandatory for PASS | Auto-dispatches on Top-1; critical failure downgrades verdict to FAIL with `primary_failure_mode = step6_causal_audit_fail` |
| Risk-normalised gates (Amendment 3) | Engine emits at `r_base`; gates evaluate at scaled `r_safe` / `r_hard` per scalability bounds (0.15% ≤ r ≤ 2.0%) |
| Ex-ante population always | `build_ex_ante_bounded_population` (or equivalent) — no outcome-aware filtering anywhere |
| No lookahead / no repainting | Hard invariant; lookahead-invariant tests required at every step |
| Config-driven (YAML only) | No hardcoded parameters |
| Real bid/ask spreads | HistData M1 bid+ask is canonical; zero-spread bars are a data-quality flag, not silently backfilled |
| Determinism | `random_state=42`, `n_jobs=1`, `lineterminator='\n'`; sha256 manifests; CI-enforced two-run identity |
| Anchor preservation | KH-24 worst-fold numbers within ±0.5pp ROI / ±1pp DD; A1 path byte-identical to legacy `KH24FoldRunner` under matched warmup |

---

## KH-24 System Parameters (Locked, Out of Scope for Forward Research)

```
Signal:     kb_exhaustion_bar (c1–c6, c8, c9)
            c7 DISABLED — volume gate removed
Direction:  Long only
Timeframe:  4H with D1 regime filter (one-day lag)
Pairs:      28 FX currency pairs
Broker:     5ers
Data:       HistData M1 bid+ask (cache at data/cache/<TF>/<PAIR>.parquet)
Entry:      Bar N+1 open after signal on bar N close
Stop:       Entry price - 2.0 × ATR(14) [entry price anchor]
Trail:      Activates at close ≥ entry + 2.0 ATR (close-based)
            1.5 ATR behind highest close, bar-close updates only
Exits:      trailing_stop | kijun_d1 | stoploss
Risk:       1.0% of current reset floor balance (KH-24 era; v3 arcs use 0.5% default)
Filters:    exposure cap=2; 1H CIR T=0.28
Spread:     Per-bar HistData M1 bid+ask
D1 align:   One-day lag — each 4H bar sees prior calendar day's D1 close
Convention: boundary_convention="utc" (KH-24 anchor; A1 path byte-identical)
```

KH-24 WFO gate: PASS. Worst-fold ROI +1.92% (F7); worst-fold DD 6.37% (F1); 214 trades across Oct 2020–Jan 2026; all 7 folds positive. Live on Contabo VPS / 5ers MT5.

---

## v3.0 Engine Status

- **Step 1 (Plumbing):** `core/arc/arc_pool_builder.py` + `core/features/pipeline.py` + `core/arc/integrity.py`. Signal-module Protocol at `core/arc/signal_protocol.py`. 27 features across 7 classes.
- **Step 2 (Clustering):** `core/steps/step_2_clustering.py`. KMeans over K ∈ {2..6}, silhouette selection, shape-tag assignment.
- **Step 3 (Capturability):** `core/steps/step_3_capturability.py`. Composite + candidate-cluster flag.
- **Step 4 (Extraction):** `core/steps/step_4_extraction.py`. RF + LGBM + LR, 5-fold TimeSeriesSplit, classifier persistence (PR #185), holdout-window training filter.
- **Step 5 (WFO):** `core/wfo/` + `core/architectures/` (A1, A2, A3, A4, A6 wired; A5 deferred). Amendment 3 risk-normalised gates wired (PR #186). Architecture-selection enforcement is dispatch-time per Amendment 5 (PR #194). Exit-policy registry at `core/sim/exit_policies/` (PR #195) includes `sl_partial_close_1r_runner_trail`.
- **Step 6 (Causal audit):** `core/step_6/` six-category framework (PR #188). Auto-dispatches on Top-1 PASS candidate per Amendment 4; manual CLI at `scripts/run_step_6.py`.
- **Signal parity (PR #189):** mid-price features + 5ers EET bar boundaries + worst-case fills. EET aggregation opt-in via `boundary_convention="5ers_eet"`; UTC default preserves KH-24 anchor.
- **Signal-module timezone alignment (PR #193):** canonical `core/signals/htf_alignment.py` utility replaces UTC-anchored HTF-lookup idioms. Audit at `docs/audits/signal_module_eet_audit_2026_05.md`.
- **EET session semantics (PR #197 / Amendment 6):** `core/time_utils/session_boundary.utc_to_eet_trading_day`; `Panel.boundary_convention` carries convention through orchestrator slicing; `compute_per_day_max_dd(boundary_convention="5ers_eet")` is the load-bearing daily-DD bucketing.
- **Tracker parser:** `scripts/update_tracker_from_closure.py` (template v1.0 / v1.1 / v1.2 / v1.2.1 / v1.3 / v1.3.1).
- **Determinism:** `core/determinism.py` — `seed_everything()`, `RANDOM_STATE=42`, `LINE_TERMINATOR="\n"`. CI-gated.
- **Legacy paths:** `core/backtester.py` + `core/signal_logic.py` retained for KH-24 anchor reproduction; 33 importers. Migration deferred until KH-24 anchor preservation is no longer load-bearing.

Key scripts:
- `scripts/anchor/run_anchor.py` — KH-24 anchor harness (7-fold rolling Oct 2020 → Jan 2026)
- `scripts/anchor/check_a1_equivalence.py` — A1-vs-legacy byte-equivalence harness
- `scripts/run_step_6.py` — manual Step 6 CLI on any closure
- `scripts/update_tracker_from_closure.py` — closure → tracker parser

Full engine capability map: [docs/audits/engine_capability_audit_2026_05.md](docs/audits/engine_capability_audit_2026_05.md).

---

## What Has Been Permanently Eliminated

- `jd_rf_evt_02_bounded_operational` — forward bias, fabricated results
- Clean labels in population selection
- Forward-conditioned dataset construction
- Indicator-driven C1 sweeps as primary research
- Exit indicator sweeps (57 tested, zero passed)
- Full NNFX stack as strategy
- GPT-4 / Aider for any implementation task
- Short signals (negative lift confirmed Phase KC; short-mirror exploratory run also failed)
- `signal_flip` exit (cuts winners); `kijun_4h` exit (fires on normal pullbacks)
- D1b slope filter (net negative across folds)
- Choppiness gate (redistributes damage)
- TP1 half-off structure as standalone gate (replaced by canonical `sl_partial_close_1r_runner_trail` primitive in `core/sim/exit_policies/`, which IS deployable per Arc 10)
- Currency exposure cap (KH era; superseded by exposure cap=2 in KH-22)
- `agree_count` gate (too rare)
- FOMC proximity filter (p=0.889)
- C7 volume gate on 5ers data (broker-specific, no lift validated)
- Range/ATR ceiling at 1.25× (fold 7 negative)
- 2% risk on 5ers data (daily cap breach, reset sim breach)
- Same-day D1 alignment (lookahead; permanently replaced by one-day lag)
- KH-25 re-entry exposure cap (KH-27 KILL — re-entries fire post-original-exit)
- 1H timeframe port of KH-24 (KI arc: mean R 0.004, t=0.095)
- L6.0 verbatim-as-gate framing (replaced by `L_ARC_PROTOCOL.md` v1.0; v1.x → v2.x → v3.0 `L_PROTOCOL.md`)
- Amendment 1's uniform archetype-driven architecture gating for A2/A6 (superseded by Amendment 5's four-gate AUC-driven rule; A2/A6 are archetype-agnostic by construction)

### Not eliminated, but flagged

- **Arc 2 signal** (`mtf_alignment.2_down_mixed.kijun`, h=120) — SHELVED 2026-05-16 under v2.x; not permanently eliminated. Cluster 2 has strong magnitude (fwd_mfe_p50 5.83R) on unextractable paths. Reopenable under v3.0 if signal-module re-fits cleanly. See `results/l_arc_2_redo/ARC_2_REDO_RESULT.md`.
- **Arc 4 signal** (`bar_range_top_decile__neg__h_001`, 1H) — SHELVED under v2.x; queued for Wave 2 v3.0 retry (Arc 4-RERUN + Arc 4-original).
- **Arc 6 signal** (failed-breakout reversal long) — Stepwise cohort with clean path quality but entry-time predictability below v2.x deployability bar; queued for Wave 2 v3.0 retry. May respond differently under Amendment 5's AUC-driven A2/A6 admit.
- **Arc 10 DLR signal** (D1 swing-low rejection long) — **DEPLOYED.** PASS-DEPLOYABLE under the EA-faithful canonical run (`arc_10/02_validation/07_canonical_wfo.md`); live on FundedNext at 0.40%. Cross-arc clusterifier build with Arc 7 c3 is a parked research candidate (see `TODO.md` Arc 10 — Improvements).

---

## Conventions

- **Queue ownership.** `results/ARC_QUEUE.md` (if/when restored under v3.0) is owned by exactly one CC session at any time. Parallel sessions on the same project must coordinate via the queue owner.
- **Determinism baseline.** `random_state=42`, `n_jobs=1`, `lineterminator="\n"` throughout. Audited via two-run sha256 comparison; CI-enforced at the mini-pipeline level via `tests/test_determinism.py`.
- **Closure docs.** v3.0 arcs land closures at `results/<arc>/ARC_CLOSURE.md` per `docs/templates/ARC_CLOSURE_TEMPLATE.md` (current v1.3.1). Tracker parser invoked pre-PR per `scripts/tracker_parser/README.md`. Historical v1.x/v2.x closures preserved at `docs/archive/arc_results/ARC_<N>_RESULT.md`.
- **Branch hygiene.** One worktree per active CC session; arc branches under `arc/<arc_name>`; infra branches under `infra/<work>`; cut from `origin/main`.

---

## Tool Assignments

| Task | Tool |
|------|------|
| Strategy, research planning, decisions, verdicts | This chat (Opus 4.7) |
| Multi-file features, atlas computation, WFO runs, cluster fits, predictor scans | Claude Code (Opus 4.7) |
| Single-file patches, YAML edits, doc updates | Cursor (Sonnet 4.6) |
| MT5 / MQL5 implementation | Claude Code |

GPT-4 and Aider are permanently excluded from all implementation work.

---

## Folder Convention

All v3.0 result documents are co-located with their artefacts under the arc folder.
- `results/<arc_name>/ARC_OPEN.md` + `ARC_CLOSURE.md`
- `results/<arc_name>/step_<N>/` step artefacts
- `results/<arc_name>/step_6/` (auto-dispatched) or `step_6_manual_<timestamp>/` (manual CLI)
- `docs/` retained for non-arc system specs (`KH24_SYSTEM_LOCK.md`, `PROTOCOL_RUNTIME.md`, `BACKTESTER_ARCHITECTURE.md`, `DATA_FOUNDATION.md`, sub-protocols, audits, calibration).

See `WORKFLOW.md` for branch + dispatch conventions.

---

## Risk Parameters (Prop Firm Constraints)

- Prop firm: 5ers
- Account constraints: max DD 10%, daily DD 5% — breach closes account permanently
- Per-trade risk: KH-24 uses 1% (live); v3.0 arcs use 0.5% as `r_base` (Amendment 3 scales to `r_safe` / `r_hard` at gate evaluation)
- Daily DD measurement boundary: 5ers EET broker trading day (Amendment 6, PR #197)
- Step 5 DD gates: ≤ 8% at `r_safe` (DEPLOYABLE), ≤ 10% at `r_hard` (VIABLE); both reflect safety margin against the 5ers hard limits

---

## Methodology in One Line

L_PROTOCOL v3.0 overseer. Five steps as rankings; WFO at Step 5 is the only deployment gate; Step 6 causal audit is the lazy verdict-downgrade. Six architectures tested per arc under Amendment 5's four-gate dispatch rule. Risk-normalised gates (Amendment 3); EET daily-DD boundary (Amendment 6); EA-faithful floating-equity sizing + portfolio-level DD gating (Amendment 7). First system clearing PASS-DEPLOYABLE ships and the project's 20%-annualised-on-$100k goal is met — **met by Arc 10, now deployed.**
