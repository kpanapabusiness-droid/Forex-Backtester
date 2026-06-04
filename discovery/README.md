# discovery/ — Autonomous Self-Running Signal Discovery

This folder is the home of the autonomous discovery programme: CC generates ideas, develops complete
trading systems, validates them honestly, documents the reasoning, and repeats — with minimal human
input. **[`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md) is authoritative**; this README is the
one-screen orientation.

> **Precondition (hard gate, protocol §0):** discovery is valid ONLY on a verified-honest engine.
> Confirm [`../HONEST_ENGINE_SWEEP.md`](../HONEST_ENGINE_SWEEP.md) reads clean before any arc. As of
> 2026-06-04 it does — `MultiPairBacktester` is honest end-to-end and safe as the sole gate engine.

## The arc loop (in brief)

read log + lessons → observe (data/charts) → form an idea → characterize (ex-ante population) →
cheap kills (pool floor, oracle-best-cluster ceiling, triage) → diagnose → address the diagnosis
(fail the BEST version) → validate (honest WFO, all-folds-positive on IS **and** OOS) → council
survivor stress-test → document. The sole judge is **all-folds-positive on IS and OOS**, honest engine,
FundedNext guardrails; ROI / DD / correlation are characterized, not gated.

## The Council of Five

At idea-forks (light), on the diagnosis (heavy), and on every survivor before promotion to `passed/`
(mandatory, heaviest), CC convenes the discovery council. Invoke it via the slash trigger
**`/llm-council-discovery`** (see [`../docs/DISCOVERY_COUNCIL_SKILL.md`](../docs/DISCOVERY_COUNCIL_SKILL.md)).
The council always RECOMMENDS; CC always COMMITS.

## Layout

| Path | What |
|---|---|
| `DISCOVERY_PROTOCOL.md` | The authoritative protocol. Read it first. |
| `DISCOVERY_LOG.md` | Two-tier append-only log: Tier-1 machine-scannable ledger + Tier-2 free-form reasoning. Chats APPEND only. |
| `LESSONS.md` | Operator-compressed distillation of the log. Chats READ only. |
| `arcs/` | Per-arc full records: `arc_<id>_<slug>.md`. |
| `passed/` | Deep records for survivors (config, per-fold IS+OOS, costs, council verdict, exact repro command + frame sha). The operator's deep-dive target. |
| `results/` | Raw run artifacts per arc. |

## Chat-range convention

Each concurrent discovery chat owns a static **1000-wide arc-id range** so ids never collide —
chat A: 1000–1999, chat B: 2000–2999, chat C: 3000–3999, … The operator assigns each chat its range at
launch. Per-arc files are `discovery/arcs/arc_<id>_<slug>.md` (unique per arc, never collide). Each chat
commits ONLY `discovery/` docs directly to main and PULLS main at step (a) to read other chats' latest.
**Code never auto-merges** — a chat that believes engine/tooling/gate code needs changing FLAGS it in its
arc doc; it does not merge it.

## STOP convention

Graceful, finish-then-halt. Create a **`discovery/STOP`** sentinel file to halt: it means "start NO new
arc" — a chat that sees STOP at step (a) starts nothing new; if STOP appears mid-arc, the chat FINISHES
the current arc (validation, documentation, log append, commit) THEN halts. Absent = keep running. STOP
is for deliberate full halts only (operator done for now, or about to change the
protocol/engine/guardrails) — routine check-ins are passive and never stop the run.
