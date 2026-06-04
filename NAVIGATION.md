# NAVIGATION — Repo Front Door (route by intent)

> The single intent-router for this repo. Find what you're trying to do below, read the
> listed docs **in order**. Everything else is reachable from these.
> For full project context and locked philosophy, read **[`CLAUDE.md`](./CLAUDE.md)** first.

**What this project is:** a research-first FX backtesting programme. The SL-honest
`MultiPairBacktester` (`core/sim/`) is the **sole** engine that scores any trade; the
`L_PROTOCOL` v3.0 overseer runs research arcs through five gates-as-rankings steps plus a
lazy Step 6 causal audit. WFO worst-fold at Step 5 is the only deployment gate.

> **Clean base (2026-06-02):** the repo was reset to a verified-green base. **There is no
> deployable system — deployable-system count = 0.** The fast path-replay scorer was retired
> for a gate-fidelity defect; before starting any arc, read
> **[`docs/ARC_10_GATE_FIDELITY_DEFECT.md`](./docs/ARC_10_GATE_FIDELITY_DEFECT.md)**.
> Live broker accounts are owned but DORMANT (keep-alive trades only); no account runs a system.

---

## Intent map

| I want to… | Read, in order |
|---|---|
| **Understand the reset / why** | **[`docs/ARC_10_GATE_FIDELITY_DEFECT.md`](./docs/ARC_10_GATE_FIDELITY_DEFECT.md)** (required) → [`RESET_MANIFEST.md`](./RESET_MANIFEST.md) (what was kept/archived/retired + recovery tag). |
| **Start a new arc** | [`ARC_RUN_TEMPLATE.md`](./ARC_RUN_TEMPLATE.md) (self-run spec — copy, fill blanks) → [`L_PROTOCOL.md`](./L_PROTOCOL.md) (methodology; Step 5 = `MultiPairBacktester` sole gate engine). Supply the signal idea + any non-default settings. For autonomous self-running discovery, see **Run autonomous discovery** below. |
| **Run autonomous discovery** | [`discovery/DISCOVERY_PROTOCOL.md`](./discovery/DISCOVERY_PROTOCOL.md) (authoritative — start here) → [`discovery/README.md`](./discovery/README.md) (one-screen: arc loop, chat-range, STOP) → [`docs/DISCOVERY_COUNCIL_SKILL.md`](./docs/DISCOVERY_COUNCIL_SKILL.md) (the council, invoked via `/llm-council-discovery`). **§0 precondition:** the engine sweep must read clean ([`HONEST_ENGINE_SWEEP.md`](./HONEST_ENGINE_SWEEP.md)) — it does as of 2026-06-04. |
| **See what's been tried / failed (don't repeat it)** | [`ARC_HISTORY.md`](./ARC_HISTORY.md) (elimination ledger; pre-2026-06-02 numbers untrusted) + the eliminated list in [`CLAUDE.md`](./CLAUDE.md). |
| **Deep-dive the methodology** | [`L_PROTOCOL.md`](./L_PROTOCOL.md) (v3.0 + amendments inline; reset amendment at top). |
| **Understand the engine internals** | [`docs/BACKTESTER_ARCHITECTURE.md`](./docs/BACKTESTER_ARCHITECTURE.md) + [`docs/PROTOCOL_RUNTIME.md`](./docs/PROTOCOL_RUNTIME.md). The take-the-loss invariant is pinned by [`tests/sim/test_take_the_loss_invariant.py`](./tests/sim/test_take_the_loss_invariant.py). |
| **Run the engine on a signal** | `core/sim/` (`MultiPairBacktester`) → smoke example at [`scripts/smoke_test_multipair.py`](./scripts/smoke_test_multipair.py). |
| **Operational conventions / branches** | [`WORKFLOW.md`](./WORKFLOW.md). |
| **The deployment / EA template** | [`deployment/`](./deployment/) + `EA/`, `MQL5/` — a signal-agnostic template + runbook machinery. Not wired to any live system (deployable-system count = 0). |

---

## Active vs reference

- **Active / canonical:** the engine, data pipeline, signal library, `L_PROTOCOL` methodology,
  and the elimination ledger. No strategy is deployed.
- **Archived (reference only, numbers NOT trusted):** everything removed in the reset lives under
  [`archive/`](./archive/) — all prior `results/`, the `arc_10/` suite, arc/phase/replay research
  scripts and their tests, and superseded docs. It is fully recoverable (tag
  `pre-reset-snapshot-2026-06-02`) but no number in it should be cited as current — it was scored
  by the retired replay engine.

---

*Front door only — it routes, it does not duplicate. When a target doc moves, fix the link here.*
