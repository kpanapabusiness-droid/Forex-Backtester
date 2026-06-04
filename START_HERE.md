# START HERE — Forex Ignition Rebuild (clean base)

The repo was reset to a verified-green base on 2026-06-02. Read this, then route via
[`NAVIGATION.md`](./NAVIGATION.md).

## The one thing to know first

The fast path-replay scorer was **retired** for a gate-fidelity defect: it skipped pre-partial
stops and flattered every gate it scored. **`MultiPairBacktester` (`core/sim/`) is now the sole
engine that scores any trade.** Before you trust any backtest, read
**[`docs/ARC_10_GATE_FIDELITY_DEFECT.md`](./docs/ARC_10_GATE_FIDELITY_DEFECT.md)** — the lesson
that reset this repo (internal consistency ≠ correctness).

## State of the world

- **Deployable-system count = 0.** No strategy is deployed. Live broker accounts are owned but
  DORMANT (keep-alive trades only) — no account runs a system.
- **Arc 10 (DLR) is KILLED**, **KH-24 is retired/closed.** Their code is archived or retained only
  as engine fixtures; neither is a live system.
- **All pre-2026-06-02 gate numbers are NOT trustworthy** (retired replay). They survive only as a
  record of what was tried — see [`ARC_HISTORY.md`](./ARC_HISTORY.md).
- **Forward research: the discovery pipeline is BUILT and trial-validated, but pre-continuous.**
  The self-running discovery programme now lives in-repo under [`discovery/`](./discovery/) — read
  [`discovery/DISCOVERY_PROTOCOL.md`](./discovery/DISCOVERY_PROTOCOL.md) (authoritative) +
  [`discovery/README.md`](./discovery/README.md), with operator dispatch via
  [`docs/DISCOVERY_DISPATCH_TEMPLATE.md`](./docs/DISCOVERY_DISPATCH_TEMPLATE.md). Arc 0 (supervised
  trial) ran end-to-end on the honest engine and FAILED its signal — correctly — so the machinery is
  validated; **deployable-system count is still 0**. Continuous multi-chat operation is pending
  operator authorization.

## What exists (the base)

- The engine + data pipeline (`core/`, `core/sim/` = the sole gate engine, `signals/`, determinism harness).
- The methodology of record: [`L_PROTOCOL.md`](./L_PROTOCOL.md) (Step 5 = truth-engine-only, two-stage triage; take-the-loss invariant locked).
- The signal library, the ML toolkit (`scripts/heavy_ml_probe/`), the tracker parser, the data pipeline.
- A signal-agnostic MetaTrader 5 EA template + deployment/runbook machinery (`EA/`, `MQL5/`, `deployment/`) — not wired to any live system.
- The elimination ledger ([`ARC_HISTORY.md`](./ARC_HISTORY.md)) and the gate-fidelity lesson ([`docs/ARC_10_GATE_FIDELITY_DEFECT.md`](./docs/ARC_10_GATE_FIDELITY_DEFECT.md)).
- The self-running discovery pipeline ([`discovery/`](./discovery/)) — protocol, tool registry, measurement glue (`core/wfo/discovery_measure.py`), and the `/llm-council-discovery` skill. Built and trial-validated (Arc 0); pre-continuous, deployable-system count = 0.

## How to start a new arc

Copy [`ARC_RUN_TEMPLATE.md`](./ARC_RUN_TEMPLATE.md), supply the signal idea + any non-default
settings, and run through [`L_PROTOCOL.md`](./L_PROTOCOL.md). Every Step-5 P&L number must come
from `MultiPairBacktester` (two-stage triage: representative folds → full WFO on survivors).

To see the engine run end-to-end: `python scripts/smoke_test_multipair.py`.

## Recovery

Everything removed in the reset is under [`archive/`](./archive/) and recoverable from tag
`pre-reset-snapshot-2026-06-02`. See [`RESET_MANIFEST.md`](./RESET_MANIFEST.md) for the full
kept / archived / retired manifest.
