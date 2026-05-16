# Forex Ignition Rebuild

A research-first FX trading system targeting prop firm requirements (5ers) with WFO-validated edge.

---

## Current State

- **Live system:** KH-24 — running on Contabo VPS, gate-passing, +1.92% worst-fold ROI / 6.37% worst-fold DD across 7 OOS folds.
- **Active research:** L arc signal testing under `L_ARC_PROTOCOL.md` v2.0. Arc 3 opens under v2.0; Arcs 1, 2 are historical (ran under v1.x). Five L registry signals scheduled (Arcs 1–5).
- **Status as of 2026-05-16:** `L_ARC_PROTOCOL.md` v2.0 locked. Path-shape clustering + two-pipeline (E entry-filter / D1 deferred-identification) extractability gate. Calibration anchor: KH-24 K=4 archetype 3 (passes via D1 at t=3). v1.x protocol + ops spec archived under `archive/`.

---

## Start Here

For new sessions, read in this order:

1. **`L_ARC_PROTOCOL.md`** — methodology of record for all L arc signal-testing work. Locked v2.0; self-contained for methodology, deliverables, gates, and workflow. Read first.
2. `SESSION_ZERO.md` — 5-minute primer on current state.
3. `STATUS.md` — tight current-state snapshot.
4. `CLAUDE.md` — first-read context for AI assistants.
5. `WORKFLOW.md` v2 — phase management and folder convention.
6. `PROJECT_RULES.md` — tool roles and guardrails.
7. `docs/GOLDEN_STANDARD_LOGIC.md` — execution truth invariants.

For v1.x historical reference (Arcs 1, 2): `archive/L_ARC_PROTOCOL_v1_0.md`, `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md`, plus v1.1/v1.2 amendments.

Specific scenarios:
- Touching the live system → `docs/KH24_SYSTEM_LOCK.md`
- Opening a new arc → `L_ARC_PROTOCOL.md` §13 (live arc doc convention)
- Closing a phase → `WORKFLOW.md` v2 checklist

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `core/` | Backtester engine, signal logic, indicator implementations |
| `scripts/` | Phase scripts, analysis tools, WFO runners |
| `configs/` | YAML configs (locked: `wfo_kh24.yaml`, `wfo_baseline_clean.yaml`, `spread_floors_5ers.yaml`) |
| `data/` | OHLCV data (`4hr/`, `daily/`, `1hr/`) |
| `results/` | Phase outputs, organized by arc (`kh24/`, `kh27_preflight/`, `kh28/`, `kh29/`, `lchar/`, `l_arc_1/`, etc.) |
| `tests/` | Unit tests, smoke tests |
| `EA/` | MetaTrader 5 EA source (`KH24_EA.mq5`) |

**Folder convention (v2, locked 2026-05-13):** all phase docs and result artefacts are co-located under their arc folder in `results/`. The previous `docs/`-for-phase-docs convention is retired. See `WORKFLOW.md` v2.

---

## Tool Stack

- **Python backtester** — source of truth for all results
- **MetaTrader 5 EA** — used only for live execution on the VPS
- **Claude (chat)** — planning, research interpretation, decisions, step 3 verdicts, step 4 candidate selection
- **Claude Code** — multi-file features, atlas computation, WFO runs, cluster fits, predictor scans (Opus 4.7, xhigh effort)
- **Cursor** — single-file patches, YAML edits, doc updates (Sonnet 4.6, medium effort)
- **GitHub Actions** — CI: lint + pytest + smoke test on every PR

Permanently excluded: GPT-4 (hallucinated indicator conversions), Aider (replaced by Claude Code).

---

## Documentation Hierarchy

Top-level files in this repository serve specific roles. Read them in the right order:

| File | When to read |
| --- | --- |
| `L_ARC_PROTOCOL.md` | First read for any L arc work — methodology, gates, deliverables, exit-family map, workflow (v2.0, self-contained) |
| `archive/L_ARC_PROTOCOL_v1_0.md`, `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` | Historical v1.x protocol + ops spec (Arcs 1, 2 reference only) |
| `SESSION_ZERO.md` | Top of every session — current state primer |
| `STATUS.md` | Tight current-state snapshot |
| `CLAUDE.md` | First-read context for AI assistants |
| `docs/LCHAR_TOPN_REGISTRY.md` | When working on any L arc — the 5 candidate signals being tested |
| `docs/KH_Research_Roadmap.md` | When questioning what's been tried in KH (closed and deferred items) |
| `WORKFLOW.md` | When closing a phase or writing phase documentation |
| `PROJECT_RULES.md` | When deciding which tool to use or how to scope a change |
| `docs/GOLDEN_STANDARD_LOGIC.md` | When writing or reviewing backtester code; the formal invariants |
| `docs/BACKTESTER_AUDIT.md` | When debugging backtester behavior or extending the engine |
| `docs/BACKTESTER_SCHEMA.json` | When writing or modifying YAML configs |
| `docs/BACKTESTER_TEMPLATE.yaml` | Starting point for new WFO configs |
| `docs/SPREAD_SEMANTICS_LOCK.md` | When touching spread handling |
| `CHANGELOG.md` | When tracing a specific phase or change |
| `project_brief.md` | Long-form project history and locked decisions |
| `docs/KH24_SYSTEM_LOCK.md` | When working with or near the live KH-24 system |

Phase result documents live in `results/<arc_name>/PHASE_<NAME>.md` per WORKFLOW.md v2. Phase artifacts live in `results/<arc_name>/<step_subfolder>/`.

---

## Risk Parameters

- **Prop firm:** 5ers
- **Account constraints:** max DD 10%, daily DD 5% — breach closes account permanently
- **Per-trade risk:** KH-24 uses 1%; L arc work uses 0.5%
- **Step 6 gate:** DD < 8% applies to both PASS-DEPLOYABLE and PASS-VIABLE tiers — safety margin against the 10% prop limit

---

## Methodology in One Line

WFO worst-fold at dual-tier disposition (PASS-DEPLOYABLE / PASS-VIABLE) is the only judge of success. Pre-committed gates, accepted results, every phase a documented finding regardless of pass or fail.

---

## How to Run a Backtest

```
python scripts/phase_kgl_v2_4h_wfo.py -c configs/wfo_kh24.yaml
```

Replace the config with whichever scenario you're running. Outputs land in `results/<arc_name>/<step_subfolder>/` per the config's `outputs.dir` field. Locked baseline configs (`wfo_kh24.yaml`, `wfo_baseline_clean.yaml`, `spread_floors_5ers.yaml`) must never be modified — copy and rename for new scenarios.

---

## Phase Workflow (Quick Reference)

Per `WORKFLOW.md` v2:

1. Write phase document to `results/<arc_name>/<location_per_protocol>.md`
2. Update `SESSION_ZERO.md` "Current State" section only
3. Append entry to `CHANGELOG.md` (most recent first)
4. Update `STATUS.md` to reflect new current step
5. Write handover note for next chat (appended to phase doc)
6. Report which files changed; upload only those to the Claude project knowledge

Every phase produces a result document regardless of pass or fail.

---

## What This Project Is Not

- Not a single-shot strategy. It is a research framework that produces validated systems.
- Not indicator-first. It is structure-first — patterns and events come before indicators.
- Not optimized for headline numbers. It is optimized for worst-fold survival under realistic execution, with dual-tier disposition distinguishing measurable edge from deployment-ready economics.
- Not exploratory in an unbounded way. The L arc is bounded by the six-step extractability protocol; per-arc work is bounded by the dual-gate verdict logic in step 3 and the WFO gate in step 6.

---

*Last updated: 2026-05-16 — L_ARC_PROTOCOL v2.0 locked; Arc 3 opens under v2.0; v1.x archived under `archive/`.*
