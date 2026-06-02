# NAVIGATION — Repo Front Door (route by intent)

> The single intent-router for this repo. Find what you're trying to do below, read the
> listed docs **in order**. Everything else is reachable from these.
> For full project context and locked philosophy, read **[`CLAUDE.md`](./CLAUDE.md)** first.

**What this project is:** a research-first FX trading system. A long-only 4H trend system
(**KH-24**) and the **Arc 10 v3.0.2** DLR system are live; the L_PROTOCOL v3.0 overseer runs
research arcs through five gates-as-rankings steps plus a lazy Step 6 causal audit. WFO
worst-fold at Step 5 is the only deployment gate.

---

## Intent map

| I want to… | Read, in order |
|---|---|
| **Start a new arc** | [`ARC_RUN_TEMPLATE.md`](./ARC_RUN_TEMPLATE.md) (self-run spec — copy, fill blanks) → [`L_PROTOCOL.md`](./L_PROTOCOL.md) (methodology + Amendment 8 defaults) → [`BROKER_RULES.md`](./BROKER_RULES.md) (constraints). Supply only the signal idea + any non-default settings; everything else = Amendment 8 defaults. |
| **Understand / validate Arc 10** | [`arc_10/START_HERE.md`](./arc_10/START_HERE.md) → [`arc_10/02_validation/07_canonical_wfo.md`](./arc_10/02_validation/07_canonical_wfo.md) (canonical numbers; source of truth). |
| **Deploy / set up the EA** | [`EA_REFERENCE.md`](./EA_REFERENCE.md) (every EA input + per-broker overrides + attach/verify) → [`arc_10/03_deployment/`](./arc_10/03_deployment/) (per-broker setup, VPS, config artifacts). |
| **Run live health / monitoring** | [`arc_10/04_runbook/01_daily_health_check.md`](./arc_10/04_runbook/01_daily_health_check.md) → [`arc_10/04_runbook/`](./arc_10/04_runbook/) (weekly/Sunday checks, restart, incident response, emergency kill, kill criteria). |
| **Question a number / "where did X come from"** | [`arc_10/02_validation/07_canonical_wfo.md`](./arc_10/02_validation/07_canonical_wfo.md) + the `_final_canonical/` CSVs at [`results/l_arc_10_v3.0.2_final_canonical/`](./results/l_arc_10_v3.0.2_final_canonical/) (`matrix.csv`, `per_fold.csv`, `governor_log.csv`). |
| **Look up broker rules / per-account settings** | [`BROKER_RULES.md`](./BROKER_RULES.md). |
| **See what's been tried / failed (don't repeat it)** | [`ARC_HISTORY.md`](./ARC_HISTORY.md) (frozen pre-v3.0 record) + [`docs/SHELVED_ARCS.md`](./docs/SHELVED_ARCS.md) (parked threads + reopen criteria) + [`ARC_TRACKER.md`](./ARC_TRACKER.md) (closed-arc verdicts, auto-managed). |
| **Deep-dive the methodology** | [`L_PROTOCOL.md`](./L_PROTOCOL.md) (v3.0 + Amendments 1–8 inline). |
| **Understand the engine internals** | [`docs/BACKTESTER_ARCHITECTURE.md`](./docs/BACKTESTER_ARCHITECTURE.md) + [`docs/PROTOCOL_RUNTIME.md`](./docs/PROTOCOL_RUNTIME.md). |
| **Touch the live KH-24 system** | [`docs/KH24_SYSTEM_LOCK.md`](./docs/KH24_SYSTEM_LOCK.md) (locked — out of scope for forward research without an explicit modification phase). |
| **Know current phase / in-flight work** | [`TODO.md`](./TODO.md) + [`WORKFLOW.md`](./WORKFLOW.md) (conventions, dispatch pattern, branch strategy). |

---

## Active vs reference

- **Active / canonical:** Arc 10 v3.0.2 (DLR) is the deployed live system (FundedNext, 0.40%);
  its canonical gate is [`results/l_arc_10_v3.0.2_final_canonical/`](./results/l_arc_10_v3.0.2_final_canonical/)
  (fixed-initial sizing + `daily_ref=initial`-resetting — matches the live EA post-FIX-2b).
  KH-24 is the other live system. `arc_10/` is the complete, self-contained suite for it.
- **Reference only (not active):** the failed arcs (Arcs 5 / 7 / 8 / 11, both discovery folders)
  and the six superseded Arc 10 result folders (`l_arc_10`, `l_arc_10_v3.0.2`,
  `_canonical`, `_canonical_compound`, `_ea_faithful`, `_utc_rerun`) are retained for
  reference — each carries a SUPERSEDED banner pointing forward to `_final_canonical`. Do not
  cite their numbers as current.

---

*Front door only — it routes, it does not duplicate. When a target doc moves, fix the link here.*
