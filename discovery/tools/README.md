# discovery/tools/ — BUILT experiment tools (committed, reusable)

This is the stable home for **experiment** tools CC builds during arcs — filters, clusterers,
transforms, exit-policy probes, signal modules, soundness controls (e.g. a random-entry null
baseline). They live here (committed), NOT in per-arc scratch, so they persist and compound across
arcs and chats.

**The rule (see [`../TOOL_REGISTRY.md`](../TOOL_REGISTRY.md)):**

- These are EXPERIMENT tools — what is *being tested*. A bug here fails the WFO and dies loudly, so
  CC builds them **freely, no human gate**.
- They must call the **CANONICAL** measurement apparatus (the engine / WFO runners / cost / scoring
  / pool build) — never re-roll it. The registry's CANONICAL section is the locked list.
- **Before building one, check `TOOL_REGISTRY.md` (BUILT).** If it exists, call it. If not, build it
  here, use it, and **append a row to the BUILT table at arc end** (name | what | path | how to call
  | arc that created it).

Measurement glue (per-year OOS folds, the all-folds-positive judge, the per-fold loop) is NOT here —
it is canonical and lives in `core/wfo/discovery_measure.py`. This folder is experiment-tier only.
