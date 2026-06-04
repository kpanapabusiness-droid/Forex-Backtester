# Forex Ignition Rebuild

A research-first FX backtesting programme. The SL-honest `MultiPairBacktester` is the sole
engine that scores any trade; `L_PROTOCOL` v3.0 runs research arcs through gated, walk-forward
validation.

> **New here? Start at [`NAVIGATION.md`](./NAVIGATION.md)** — the intent-router that maps what
> you're trying to do to the exact docs. For AI assistants, [`CLAUDE.md`](./CLAUDE.md) is the
> first read.

---

## Current State — clean base (2026-06-02)

The repo was reset to a verified-green base (`reset/clean-base`).

- **Deployable-system count = 0.** No strategy is deployed. The live broker accounts are owned
  but DORMANT — only occasional keep-alive trades to avoid inactivity closure. No account runs a
  system.
- **The fast path-replay scorer is RETIRED.** `MultiPairBacktester` (`core/sim/`) is the sole
  engine that scores a trade for a gate. There is no precomputed-P&L shortcut in the live tree.
- **Arc 10 (DLR) is KILLED** — gate-fidelity defect; its deployment verdict rested on the retired
  replay. **Required reading:** [`docs/ARC_10_GATE_FIDELITY_DEFECT.md`](./docs/ARC_10_GATE_FIDELITY_DEFECT.md).
- **KH-24 is retired/closed** — not a live system. Its strategy code is retained only as the A1
  byte-identity engine anchor.
- **Forward research: the discovery pipeline is BUILT, trial-validated, and CLEARED for continuous operation.**
  The self-running discovery programme now lives in-repo under [`discovery/`](./discovery/) — start
  at [`discovery/DISCOVERY_PROTOCOL.md`](./discovery/DISCOVERY_PROTOCOL.md) (authoritative) +
  [`discovery/README.md`](./discovery/README.md); operator dispatch via
  [`docs/DISCOVERY_DISPATCH_TEMPLATE.md`](./docs/DISCOVERY_DISPATCH_TEMPLATE.md), the per-chat run
  dispatch [`discovery/CONTINUOUS_RUN_DISPATCH.md`](./discovery/CONTINUOUS_RUN_DISPATCH.md), and the
  overseer handover [`discovery/CONTINUOUS_OVERSEER_HANDOVER.md`](./discovery/CONTINUOUS_OVERSEER_HANDOVER.md).
  Arc 0 (supervised trial) ran end-to-end on the honest engine and FAILED its signal — correctly — so
  the machinery is validated and **deployable-system count is still 0**. Continuous multi-chat
  operation is now AUTHORIZED (protocol §10 staging: trial DONE → chats run continuously from arc 1).

All pre-2026-06-02 gate numbers were produced by the retired replay and are NOT trustworthy.
See [`RESET_MANIFEST.md`](./RESET_MANIFEST.md) for what was kept / archived / retired and how to
recover anything (tag `pre-reset-snapshot-2026-06-02`).

---

## Start Here

1. **[`NAVIGATION.md`](./NAVIGATION.md)** — route by intent.
2. **[`docs/ARC_10_GATE_FIDELITY_DEFECT.md`](./docs/ARC_10_GATE_FIDELITY_DEFECT.md)** — required: why the repo was reset; the only engine that may score a trade.
3. **[`L_PROTOCOL.md`](./L_PROTOCOL.md)** — methodology of record (Step 5 = `MultiPairBacktester` sole gate engine).
4. **[`ARC_HISTORY.md`](./ARC_HISTORY.md)** — elimination ledger (what was tried → not deployable).
5. **[`CLAUDE.md`](./CLAUDE.md)** — first-read context for AI assistants; locked philosophy + eliminated list.
6. **[`WORKFLOW.md`](./WORKFLOW.md)** — operational conventions, branch strategy.

Engine internals → [`docs/PROTOCOL_RUNTIME.md`](docs/PROTOCOL_RUNTIME.md), [`docs/BACKTESTER_ARCHITECTURE.md`](docs/BACKTESTER_ARCHITECTURE.md). Sub-protocols → `docs/sub_protocols/`.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `core/` | The engine: data, spread/fill/sim, panel, WFO + features, arc orchestrator, six architectures, Step 6 causal-audit framework, SL-honest exit-policy registry, EET session utilities. `core/sim/` holds `MultiPairBacktester`, the sole gate engine. |
| `signals/` | Signal library (adapters + structure detectors). |
| `scripts/` | Engine-running infra: `scripts/anchor/` (A1 byte-identity anchor), `scripts/run_step_6.py`, `scripts/update_tracker_from_closure.py`, `scripts/tracker_parser/`, `scripts/heavy_ml_probe/`, data pipeline (`scripts/histdata_*`, `scripts/normalize_*`, `scripts/data/`), `scripts/smoke_test_multipair.py`. |
| `configs/` | YAML configs + schema/templates. |
| `data/` | HistData M1 bid+ask; parquet cache under `data/cache/<TF>/<PAIR>.parquet` (gitignored). |
| `tests/` | Unit + integration suites (engine, sim, signals, step_6, tracker_parser, …). CI: `pytest -m "not research"`. |
| `EA/`, `MQL5/`, `deployment/` | Signal-agnostic MetaTrader 5 EA template + deployment/runbook machinery. Not wired to any live system. |
| `docs/` | System specs + the gate-fidelity lesson + sub-protocols + templates. |
| `archive/` | Everything removed in the reset — prior `results/`, the `arc_10/` suite, arc/phase/replay research code + tests, superseded docs. Fully recoverable; numbers NOT trusted. |
| `attic/` | MT5-era code quarantined earlier; excluded from CI. |

---

## How to run the engine

Score a signal by walking bars through the sole gate engine:

```python
from core.sim.multipair_backtester import MultiPairBacktester
from core.sim.account import Account
# build a Panel of bid/ask bars + a strategy fn, then:
bt = MultiPairBacktester(panel=panel, account=Account(starting_balance=100_000.0),
                         strategy=my_strategy)
result = bt.run()
```

A runnable end-to-end smoke example is at [`scripts/smoke_test_multipair.py`](./scripts/smoke_test_multipair.py)
(`python scripts/smoke_test_multipair.py`). The take-the-loss invariant (stop-first; ambiguity
never wins) is pinned by [`tests/sim/test_take_the_loss_invariant.py`](./tests/sim/test_take_the_loss_invariant.py).

Manual Step 6 on any closure: `python scripts/run_step_6.py results/<arc>/ARC_CLOSURE.md`.

---

## Risk Parameters (planning constraints, not a live deployment)

- Prop-firm template constraints (5ers / FundedNext family): max DD 10%, daily DD 5%.
- v3.0 arcs use 0.5% as `r_base`; risk-normalised gates scale to `r_safe` (DEPLOYABLE) / `r_hard` (VIABLE).
- Step 5 DD gates: ≤ 8% at `r_safe`, ≤ 10% at `r_hard`. Daily-DD boundary: 5ers EET broker day.

No risk tier is live — deployable-system count = 0.

---

## CI

GitHub Actions on every PR: ruff + `pytest -m "not research"` + a smoke run. Determinism is
CI-enforced (`tests/test_determinism.py`). `archive/` and `attic/` are excluded from lint and
test collection.

---

*Last updated: 2026-06-04 — clean-base reset: fast replay retired, `MultiPairBacktester` is the sole gate engine, deployable-system count = 0. Discovery pipeline built + trial-validated + cleared for continuous operation; see [`discovery/`](./discovery/).*
