# CLAUDE.md — Forex Ignition Rebuild
> Last updated: 2026-06-04 | Phase: **DISCOVERY PIPELINE BUILT — continuous-authorized; deployable-system count = 0.** The repo was reset to a verified-green base (`reset/clean-base`); the self-running discovery pipeline is built in-repo and trial-validated (Arc 0 ran end-to-end on the honest engine and FAILED its signal, correctly), and continuous multi-chat operation is now AUTHORIZED — the overseer handover ([discovery/CONTINUOUS_OVERSEER_HANDOVER.md](discovery/CONTINUOUS_OVERSEER_HANDOVER.md)) and per-chat run dispatch ([discovery/CONTINUOUS_RUN_DISPATCH.md](discovery/CONTINUOUS_RUN_DISPATCH.md)) are landed and protocol §10 staging is continuous. The fast path-replay scorer was retired; the SL-honest `MultiPairBacktester` is the **sole** engine that scores any trade. All pre-2026-06-02 gate numbers were produced by the retired replay and are NOT trustworthy. **Deployable-system count = 0.**
> First file any AI assistant reads. Reflects where the project ACTUALLY is.
>
> **Before opening any new arc, read [docs/ARC_10_GATE_FIDELITY_DEFECT.md](docs/ARC_10_GATE_FIDELITY_DEFECT.md) — the lesson that reset this repo (internal consistency ≠ correctness).**

---

## CRITICAL: Current State

- **No live system.** No strategy is deployed. The live broker accounts are OWNED but DORMANT — the operator may place occasional keep-alive trades to avoid inactivity closure; that is the only live activity. No account runs a system. **Deployable-system count = 0.**
- **Arc 10 (DLR) is KILLED** — gate-fidelity defect. Its deployment verdict rested on a fast replay that skipped pre-partial stops; the SL-honest engine fails it. Archived; the lesson is kept live at [docs/ARC_10_GATE_FIDELITY_DEFECT.md](docs/ARC_10_GATE_FIDELITY_DEFECT.md).
- **KH-24 is retired/closed** — not a live system, not deployable. Its strategy code (`core/strategies/kh24/`) and anchor harness (`scripts/anchor/`) are retained ONLY as the A1 byte-identity engine anchor (a determinism fixture), never as a deployable or live system.
- **The fast replay is retired (2026-06-02).** No live-tree code path scores P&L from a precomputed shortcut. `MultiPairBacktester` (`core/sim/`) is the sole gate engine.
- **Forward research: the discovery pipeline is BUILT, trial-validated, and CLEARED for continuous operation.** The self-running discovery programme now lives in-repo under [discovery/](discovery/) — protocol [discovery/DISCOVERY_PROTOCOL.md](discovery/DISCOVERY_PROTOCOL.md) (authoritative), one-screen orientation [discovery/README.md](discovery/README.md), operator dispatch layer [docs/DISCOVERY_DISPATCH_TEMPLATE.md](docs/DISCOVERY_DISPATCH_TEMPLATE.md), per-chat run dispatch [discovery/CONTINUOUS_RUN_DISPATCH.md](discovery/CONTINUOUS_RUN_DISPATCH.md), overseer handover [discovery/CONTINUOUS_OVERSEER_HANDOVER.md](discovery/CONTINUOUS_OVERSEER_HANDOVER.md), measurement glue `core/wfo/discovery_measure.py`. Arc 0 (supervised trial) ran end-to-end on the honest engine and FAILED its signal — correctly — so the machinery is validated. Continuous multi-chat operation is now **AUTHORIZED** (protocol §10 staging updated: trial DONE → chats run continuously from arc 1); deployable-system count remains 0.
- **Short-side capability is ENABLED (2026-06-05, PR #273).** The v3 short path is merged and the honest-engine sweep's short-side re-read addendum reads SAFE (Parts C/D symmetric on short legs; longs byte-identical). The pre-reset "short signals eliminated" verdict is superseded. No short discovery arc has opened yet — the capability is available (gated canonical core) and the programme's immediate lane stays the long-only PORTFOLIO route until the operator flips it to short / relative-value work. **Deployable-system count = 0.**

---

## Read These First, In Order

1. **`README.md`** / **`NAVIGATION.md`** — what exists in the clean base and how to start.
2. **[docs/ARC_10_GATE_FIDELITY_DEFECT.md](docs/ARC_10_GATE_FIDELITY_DEFECT.md)** — **required reading.** Why the repo was reset; the only engine that may score a trade.
3. **`L_PROTOCOL.md`** — the methodology of record. Self-contained for gates, deliverables, architectures. Step 5 mandates `MultiPairBacktester` as the sole gate engine.
4. **`ARC_HISTORY.md`** — the elimination ledger (what was tried → not deployable; numbers untrusted).
5. **`WORKFLOW.md`** — operational conventions, branch strategy.
6. **`RESET_MANIFEST.md`** — what was kept / archived / retired in the reset, and the recovery tag.

Then, depending on scope:
- Engine work or runtime questions → `docs/PROTOCOL_RUNTIME.md` + `docs/BACKTESTER_ARCHITECTURE.md`
- Opening or closing an arc → `docs/templates/` + `scripts/tracker_parser/README.md`
- Capability inventory → `docs/audits/engine_capability_audit_2026_05.md`

Sub-protocols at `docs/sub_protocols/` (`heavy_ml_probe.md`, `signal_discovery_probe.md`) define the canonical Step-1 / Step-4 overrides.

---

## Active protocol

**L_PROTOCOL v3.0** (`L_PROTOCOL.md`) is the overseer methodology for all forward research. Five steps as rankings + lazy Step 6 (causal audit). WFO at Step 5 is the only deployment gate, scored **solely** by `MultiPairBacktester` (truth-engine triage on representative folds → full WFO on survivors). Methodology amendments (ex-ante population, cost cell, EET daily-DD boundary, causal audit, AUC-gated architecture selection) remain in force; the Arc-10-specific / risk-tier amendment is marked SUPERSEDED. See `L_PROTOCOL.md` for the authoritative, internally-consistent statement.

The take-the-loss invariant is locked: any stop breach at or before the +1R partial bar = −1R; same-bar = SL-first; ambiguity never resolves to a win. Pinned by `tests/sim/test_take_the_loss_invariant.py`.

---

## What the Project Is

A research programme that tests structurally-distinct, long-only price-structure signals through `L_PROTOCOL.md` v3.0 — five gates-as-rankings steps plus a lazy Step 6 causal audit. The goal is one or more PASS-DEPLOYABLE survivor systems. As of the clean-base reset there are none.

Any arc (standard, discovery, or diagnostic probe) plugs into the same overseer protocol, with sub-protocols at `docs/sub_protocols/` overriding specific steps when needed.

---

## Locked Philosophy

| Rule | Detail |
|------|--------|
| Single truth engine | `MultiPairBacktester` (`core/sim/`) is the ONLY engine that scores a trade for a gate. No fast-replay / precomputed-P&L path. |
| Take-the-loss | Stop breach at/before the +1R partial = −1R; same-bar = SL-first; ambiguity never wins. CI-pinned. |
| Structure-first | Signal is price structure, not indicator |
| WFO worst-fold is the only judge at Step 5 | Average fold, best fold — irrelevant |
| Dual-tier disposition at Step 5 | PASS-DEPLOYABLE / PASS-VIABLE / FAIL; DD ≤ 8% at `r_safe` (DEPLOYABLE), ≤ 10% at `r_hard` (VIABLE) |
| Step 6 causal audit is mandatory for PASS | Auto-dispatches on Top-1; critical failure downgrades verdict to FAIL |
| Risk-normalised gates | Engine emits at `r_base`; gates evaluate at scaled `r_safe` / `r_hard` (0.15% ≤ r ≤ 2.0%) |
| Ex-ante population always | `build_arc_pool` (`core/arc/arc_pool_builder.py`; the spec name `build_ex_ante_bounded_population` is an alias for it) — no outcome-aware filtering anywhere |
| No lookahead / no repainting | Hard invariant; lookahead-invariant tests required at every step |
| Config-driven (YAML only) | No hardcoded parameters |
| Real bid/ask spreads | HistData M1 bid+ask is canonical; zero-spread bars are a data-quality flag, not silently backfilled |
| Determinism | `random_state=42`, `n_jobs=1`, `lineterminator='\n'`; sha256 manifests; CI-enforced two-run identity |
| Anchor preservation | A1 path byte-identical to legacy `KH24FoldRunner` under matched warmup (engine determinism fixture only) |

---

## v3.0 Engine Status

- **Step 1 (Plumbing):** `core/arc/arc_pool_builder.py` + `core/features/pipeline.py` + `core/arc/integrity.py`. Signal-module Protocol at `core/arc/signal_protocol.py`.
- **Step 2 (Clustering):** `core/steps/step_2_clustering.py`. KMeans over K ∈ {2..6}, silhouette selection, shape-tag assignment.
- **Step 3 (Capturability):** `core/steps/step_3_capturability.py`. Composite + candidate-cluster flag.
- **Step 4 (Extraction):** `core/steps/step_4_extraction.py`. RF + LGBM + LR, 5-fold TimeSeriesSplit, classifier persistence, holdout-window training filter.
- **Step 5 (WFO):** `core/wfo/` + `core/architectures/` (A1, A2, A3, A4, A6 wired; A5 deferred). Risk-normalised gates wired. **Scored solely by `MultiPairBacktester`** (`core/sim/`) over the SL-honest exit-policy registry at `core/sim/exit_policies/`.
- **Step 6 (Causal audit):** `core/step_6/` six-category framework. Manual CLI at `scripts/run_step_6.py`.
- **EET session semantics:** `core/time_utils/session_boundary.utc_to_eet_trading_day`; `compute_per_day_max_dd(boundary_convention="5ers_eet")` is the load-bearing daily-DD bucketing.
- **Determinism:** `core/determinism.py` — `seed_everything()`, `RANDOM_STATE=42`, `LINE_TERMINATOR="\n"`. CI-gated.
- **Retired:** the fast path-replay scorer (`simulate_path` / precomputed realised-R) — archived 2026-06-02. `MultiPairBacktester` is the sole engine.

Key scripts:
- `scripts/run_step_6.py` — manual Step 6 CLI on any closure
- `scripts/update_tracker_from_closure.py` — closure → tracker parser
- `scripts/anchor/check_a1_equivalence.py` — A1-vs-legacy byte-equivalence harness (engine anchor fixture)
- `scripts/heavy_ml_probe/` — ML toolkit (kept; a separate honest-engine sweep will confirm it does not consume replay-derived labels)

Full engine capability map: [docs/audits/engine_capability_audit_2026_05.md](docs/audits/engine_capability_audit_2026_05.md).

---

## What Has Been Permanently Eliminated

Each entry below was tried and is **not deployable** — recorded so it is not repeated. No "promising / viable / revisit / close-to-gate" framing survives; every pre-2026-06-02 number behind these came from the retired replay.

- **DLR / Arc 10** — KILLED, gate-fidelity defect (see [docs/ARC_10_GATE_FIDELITY_DEFECT.md](docs/ARC_10_GATE_FIDELITY_DEFECT.md)).
- **KH-24** — retired, closed, not deployable (strategy code retained only as the A1 byte-identity engine anchor).
- The **fast path-replay / precomputed-realised-R gate path** — retired (gate-fidelity defect, 2026-06-02); never reintroduce a precomputed P&L shortcut.
- `jd_rf_evt_02_bounded_operational` — forward bias, fabricated results.
- Clean labels in population selection.
- Forward-conditioned dataset construction.
- Indicator-driven C1 sweeps as primary research.
- Exit indicator sweeps (57 tested, none deployable).
- Full NNFX stack as strategy.
- GPT-4 / Aider for any implementation task.
- ~~Short signals~~ — **RE-ENABLED 2026-06-05 (PR #273):** the v3 short path is merged and the honest-engine sweep short-side addendum reads SAFE (Parts C/D symmetric). The pre-reset elimination is superseded; the short capability is available but gated canonical core, and no short discovery arc has opened yet.
- `signal_flip` exit; `kijun_4h` exit.
- D1b slope filter.
- Choppiness gate.
- TP1 half-off as a standalone gate (the `sl_partial_close_1r_runner_trail` primitive in `core/sim/exit_policies/` remains, scored SL-honestly).
- Currency exposure cap (KH era).
- `agree_count` gate.
- FOMC proximity filter.
- C7 volume gate on 5ers data.
- Range/ATR ceiling at 1.25×.
- 2% risk on 5ers data (daily-cap breach).
- Same-day D1 alignment (lookahead; replaced by one-day lag).
- KH-25 re-entry exposure cap.
- 1H timeframe port of KH-24.
- L6.0 verbatim-as-gate framing.
- The arcs in `ARC_HISTORY.md` (Arcs 1–11) — all tried → not deployable; see the ledger.

---

## Conventions

- **Determinism baseline.** `random_state=42`, `n_jobs=1`, `lineterminator="\n"` throughout. CI-enforced via `tests/test_determinism.py`.
- **Closure docs.** New arcs write fresh result dirs and land closures per `docs/templates/`. Tracker parser per `scripts/tracker_parser/README.md`.
- **Branch hygiene.** One worktree per active CC session; cut from `origin/main`. Major structural / governance change → PR, CI green is the definition of done, no force-push.

---

## Folder Convention

- New research writes fresh `results/<arc_name>/` dirs (the prior `results/` tree is archived under `archive/results/`).
- `docs/` holds non-arc system specs (`PROTOCOL_RUNTIME.md`, `BACKTESTER_ARCHITECTURE.md`, `DATA_FOUNDATION.md`, the gate-fidelity lesson, sub-protocols, audits, templates).
- `archive/` holds everything removed from the working tree in the reset — fully recoverable, never trusted for numbers.

See `WORKFLOW.md` for branch + dispatch conventions.

---

## Methodology in One Line

L_PROTOCOL v3.0 overseer. Five steps as rankings; WFO at Step 5 is the only deployment gate, scored **solely** by the SL-honest `MultiPairBacktester`; Step 6 causal audit is the lazy verdict-downgrade. The first system clearing PASS-DEPLOYABLE on the truth engine ships. As of the clean-base reset, **none has** — deployable-system count = 0.
