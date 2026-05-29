# Forex Ignition Rebuild

A research-first FX trading system targeting prop-firm requirements (5ers) with WFO-validated edge.

---

## Current State

- **Live system:** KH-24 — running on Contabo VPS / 5ers MT5 broker feed, gate-passing, +1.92% worst-fold ROI / 6.37% worst-fold DD across 7 OOS folds. Unaffected by v3.0 engine work.
- **Active methodology:** `L_PROTOCOL.md` v3.0 + Amendments 1-6 (locked). Five steps as rankings; WFO at Step 5 is the only deployment gate; Step 6 causal-audit framework auto-dispatches on Top-1 PASS candidate.
- **v3.0 engine:** post-Phase-1-engine-build sprint (8 PRs: #185 / #186 / #188 / #189 / #193 / #194 / #195 / #197). Steps 1–5 + Step 6 framework wired end-to-end. Architectures A1, A2, A3, A4, A6 operational; A5 deferred until ≥ 1 VIABLE candidate. Signal parity (mid features + 5ers EET aggregation + worst-case fills) merged.
- **Data layer:** HistData M1 bid+ask, 28 pairs, 2010-2026 (52 GB tick + 18 GB M1 derived; verified 2026-05-21).
- **Active research:** Phase 1 Wave 1 v3.0 closures + retries. Arcs 8 / 10 / 11 closed; Arcs 5 / 7 closures in flight; Wave 1 retries under signal-parity engine pending. Parallel: `heavy_ml_probe` build (PR #187 PR-A); `signal_discovery_probe` 10k local run (user-side).

---

## Arc 10 Documentation

The complete Arc 10 system — strategy, validation, deployment, runbook, history — is consolidated in **[`arc_10/`](./arc_10/)**.

If you want to:
- Understand what Arc 10 does → [`arc_10/00_executive_summary.md`](./arc_10/00_executive_summary.md)
- See the validation work → [`arc_10/02_validation/`](./arc_10/02_validation/)
- Deploy or operate the system → [`arc_10/03_deployment/`](./arc_10/03_deployment/) + [`arc_10/04_runbook/`](./arc_10/04_runbook/)
- Understand the lineage → [`arc_10/05_history/`](./arc_10/05_history/)

---

## Start Here

For new sessions, read in this order:

1. **`L_PROTOCOL.md`** — methodology of record. Self-contained for gates, deliverables, architectures.
2. **`TODO.md`** — current phase, in-flight arcs, engine work, standing items.
3. **`ARC_TRACKER.md`** — auto-managed arc state.
4. **`CLAUDE.md`** — first-read context for AI assistants.
5. **`WORKFLOW.md`** — operational conventions, dispatch artefact pattern, branch strategy.
6. **`project_brief.md`** — long-form project history and locked decisions.

Specific scenarios:
- Touching the live KH-24 system → `docs/KH24_SYSTEM_LOCK.md`
- Engine / runtime questions → `docs/PROTOCOL_RUNTIME.md`, `docs/BACKTESTER_ARCHITECTURE.md`
- Opening or closing an arc → `docs/templates/ARC_CLOSURE_TEMPLATE.md`, `scripts/tracker_parser/README.md`
- Capability inventory → `docs/audits/engine_capability_audit_2026_05.md`
- Sub-protocols → `docs/sub_protocols/heavy_ml_probe.md`, `docs/sub_protocols/signal_discovery_probe.md`

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `core/` | v3.0 engine: data, spread/fill/sim, panel, WFO + features, parallelism, arc orchestrator, six architectures, Step 6 causal-audit framework, canonical exit-policy registry, EET session utilities |
| `scripts/` | Anchor harness (`scripts/anchor/`), Step 6 manual CLI (`scripts/run_step_6.py`), tracker parser (`scripts/update_tracker_from_closure.py`), per-arc dispatch scripts (`scripts/l_arc_*/`) |
| `configs/` | YAML configs (`configs/data_v3.yaml`, per-arc configs under `configs/l_arc_*/`) |
| `data/` | HistData M1 bid+ask under `data/histdata/`; v3 parquet cache under `data/cache/<TF>/<PAIR>.parquet` (UTC) and `data/cache/<TF>_5ers_eet/<PAIR>.parquet` (5ers EET); gitignored |
| `reference/` | Frozen reference artefacts — `kh24_ea/KH24_EA.mq5` is the deployed EA source |
| `results/` | Per-arc outputs (`results/<arc>/ARC_OPEN.md`, `ARC_CLOSURE.md`, `step_<N>/`, `step_6/`) |
| `tests/` | Unit + integration tests; protocol_runtime + step_6 + tracker_parser + sim/exit_policies + signals suites |
| `EA/` | MetaTrader 5 EA source (`KH24_EA.mq5`) — same file as `reference/kh24_ea/` |
| `archive/` | Frozen L_PROTOCOL amendment archives + pre-reset docs |
| `attic/` | MT5-era code/configs quarantined in PR-B (hard cut on MT5 per chat decision; excluded from pytest) |

---

## Tool Stack

- **Python v3.0 engine** — source of truth for all results
- **MetaTrader 5 EA** — used only for live KH-24 execution on the VPS
- **Claude (chat)** — planning, research interpretation, decisions, verdicts
- **Claude Code** — multi-file features, WFO runs, sub-protocol execution (Opus 4.7)
- **Cursor** — single-file patches, YAML edits, doc updates (Sonnet 4.6)
- **GitHub Actions** — CI: lint + pytest + smoke + determinism on every PR

Permanently excluded: GPT-4 (hallucinated indicator conversions), Aider (replaced by Claude Code).

---

## Documentation Hierarchy

Canonical reads — everything else is reachable from these six:

| File | When to read |
| --- | --- |
| `L_PROTOCOL.md` | Methodology of record. First read for any forward research work. Locked v3.0 + Amendments 1-6 inline. |
| `TODO.md` | Live operational tracker — current phase, in-flight PRs, standing items. Update at major state transitions. |
| `ARC_TRACKER.md` | Auto-managed arc state via `scripts/update_tracker_from_closure.py`. Read-only for chat. |
| `CLAUDE.md` | First-read context for AI assistants. Locked philosophy + KH-24 spec + permanently-eliminated list. |
| `WORKFLOW.md` | Operational conventions: tool boundaries, dispatch artefact pattern (intent → log → PR), branch strategy, halt discipline. |
| `project_brief.md` | Long-form project history and locked decisions. Strategic context. |

Engine-internal docs (`docs/PROTOCOL_RUNTIME.md`, `docs/BACKTESTER_ARCHITECTURE.md`, `docs/audits/*`, `docs/calibration/*`, `docs/sub_protocols/*`, `docs/templates/*`) are reachable from the six above. Per-arc closure docs live at `results/<arc>/ARC_CLOSURE.md` (v3.0) or `docs/archive/arc_results/ARC_<N>_RESULT.md` (v1.x / v2.x historical).

---

## Risk Parameters

- **Prop firm:** 5ers
- **Account constraints:** max DD 10%, daily DD 5% — breach closes account permanently
- **Per-trade risk:** KH-24 live uses 1%; v3.0 arcs use 0.5% as `r_base` (Amendment 3 scales to `r_safe` for DEPLOYABLE / `r_hard` for VIABLE at gate evaluation)
- **Daily DD measurement boundary:** 5ers EET broker trading day (Amendment 6, PR #197)
- **Step 5 DD gates:** ≤ 8% at `r_safe` (DEPLOYABLE), ≤ 10% at `r_hard` (VIABLE) — both safety-margin against 5ers hard limits

---

## Methodology in One Line

L_PROTOCOL v3.0 overseer. Five steps as rankings; WFO worst-fold at dual-tier disposition (PASS-DEPLOYABLE / PASS-VIABLE) is the deployment judge at Step 5; Step 6 causal audit is the lazy verdict-downgrade for PASS candidates. All six architectures tested per arc under Amendment 5's four-gate dispatch rule. Risk-normalised gates (Amendment 3); 5ers EET daily-DD boundary (Amendment 6). Pre-committed gates, accepted results, every phase a documented finding regardless of pass or fail.

---

## How to Run a Backtest (v3.0)

KH-24 anchor reproduction (7-fold rolling Oct 2020 → Jan 2026):

```python
from scripts.anchor.run_anchor import run

run(structure="kh24_anchor", output_dir="results/anchor_kh24_7fold_v3")
```

A1-path equivalence regression (full data):

```bash
python scripts/anchor/check_a1_equivalence.py
```

For an arc, dispatch through the orchestrator:

```python
from core.arc.arc_orchestrator import ArcConfig, ArcOrchestrator

cfg = ArcConfig(
    arc_name="my_arc",
    signal_class="...",
    pair_set=("EURUSD", "GBPUSD", ...),
    sl_atr_mult=2.0,
    architectures=(...,),
    architecture_configs=(...,),
)
result = ArcOrchestrator(cfg, signal_module, panels).run()
```

Per-arc dispatch scripts (e.g. `scripts/l_arc_10_v3/`, `scripts/l_arc_11/`) show end-to-end Step 1→5 wiring including Step 6 auto-dispatch via the orchestrator. Full API reference: [docs/PROTOCOL_RUNTIME.md](docs/PROTOCOL_RUNTIME.md).

Manual Step 6 on any closure:

```bash
python scripts/run_step_6.py results/<arc>/ARC_CLOSURE.md
```

Pre-PR-B MT5-era scripts (`scripts/phase_kgl_v2_4h_wfo.py` and the 21 L-arc YAML configs) were quarantined to `attic/` in PR-B and are not runnable on v3.0. Live KH-24 on the VPS uses `EA/KH24_EA.mq5` and is independent.

---

## Phase Workflow (Quick Reference)

Per `WORKFLOW.md` v2:

1. **Open arc:** write `results/<arc>/ARC_OPEN.md` declaring signal, sub-protocol, pair set, window, hypothesis. Cite Amendment 5's four-gate architecture-selection set per surviving cluster (dispatch-time).
2. **Run steps:** Step 1 → 5 via orchestrator; Step 6 auto-dispatches on Top-1 PASS candidate (Amendment 4).
3. **Close arc:** write `results/<arc>/ARC_CLOSURE.md` per `docs/templates/ARC_CLOSURE_TEMPLATE.md` (v1.3.1).
4. **Tracker update:** run `python scripts/update_tracker_from_closure.py results/<arc>/ARC_CLOSURE.md` on the arc branch; commit closure + tracker delta atomically.
5. **PR:** open closure PR; chat reviews + merges.

Dispatch artefact pattern for non-trivial CC work: intent doc → chat review → execute + log doc → PR. Detail in `WORKFLOW.md` §2.

---

## What This Project Is Not

- Not a single-shot strategy. It is a research framework that produces validated systems.
- Not indicator-first. It is structure-first — patterns and events come before indicators.
- Not optimised for headline numbers. It is optimised for worst-fold survival under realistic execution, with dual-tier disposition distinguishing measurable edge from deployment-ready economics.
- Not exploratory in an unbounded way. The protocol is gates-as-rankings; the WFO gate is the bound; Step 6 causal audit is the verdict-downgrade for any PASS candidate that doesn't withstand scrutiny.

---

*Last updated: 2026-05-25 — post-Phase-1-engine-build (PRs #185-#197); L_PROTOCOL v3.0 + Amendments 1-6 locked; Phase 1 Wave 1 in flight; KH-24 live deployment unaffected.*
