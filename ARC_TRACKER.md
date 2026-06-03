# ARC_TRACKER — Live State

> **Auto-updated** by `scripts/update_tracker_from_closure.py` on each arc closure. Do not edit manually except per `docs/templates/` mapping.
> Schema locked at v3.0.
>
> **Clean base (2026-06-02):** reset to a verified-green base. No arc has been closed under the
> sole gate engine (`MultiPairBacktester`). All prior arc state was produced by the retired
> replay and is archived at `archive/docs/ARC_TRACKER.md` (numbers NOT trusted). The full
> elimination ledger is in `ARC_HISTORY.md`. **Deployable-system count = 0.**

---

## Active arcs

| Arc | Signal | TF mode | Sub-protocol | Started | Branch | Step in progress |
|---|---|---|---|---|---|---|

(empty — no arcs running)

---

## Closed arcs (under the truth engine)

| Arc | Signal | Disposition | Worst-fold | Closed | Closure |
|---|---|---|---|---|---|

(empty — no arcs closed under `MultiPairBacktester` on the clean base. Prior arcs: see `ARC_HISTORY.md`.)

---

## Available sub-protocols

> Sub-protocols are reusable engine paths invokable by arcs via `sub_protocol: <name>` in `ARC_OPEN.md`.

| Sub-protocol | Status | Entry point | Spec |
|---|---|---|---|
| heavy_ml_probe | LANDED | scripts/heavy_ml_probe/run_probe.py | docs/sub_protocols/heavy_ml_probe.md |
| signal_discovery_probe | SPEC (entry point archived) | (was scripts/arc_discovery_01/ — archived) | docs/sub_protocols/signal_discovery_probe.md |

---
