# CC DISPATCH — RESUME CONTINUOUS DISCOVERY (per chat — assign a range)

You are RESUMING an autonomous discovery run. A prior chat on this range stopped at its context
budget; all its work is on `main`. You pick up from the shared log — nothing is lost, nothing to redo.
Authoritative spec is `discovery/DISCOVERY_PROTOCOL.md` — follow it; this dispatch only sets framing +
the range and does NOT restate the protocol.

## YOUR ASSIGNED RANGE
**Range: [ASSIGN — 1000-1999 / 2000-2999 / 3000-3999].** Use ids ONLY within your range. Per-arc files:
`discovery/arcs/arc_<id>_<slug>.md`.

## RESUME PROCEDURE
1. Pull `main`. Read `discovery/DISCOVERY_PROTOCOL.md` (follow it), `discovery/DISCOVERY_LOG.md` (both
   tiers, incl. the `disposition` column), `LESSONS.md`, `TOOL_REGISTRY.md`. The LOG is your memory;
   your context starts empty. Show the reading. FRESH EYES (honest-era log only).
2. Check `discovery/STOP`. Present -> write a halt note + stop. Absent -> continue.
3. Find the highest arc-id IN YOUR RANGE in the log; resume the loop at step (a) from that id + 1 (or
   the range floor if none). This is the bootstrap — it is contention-free; do NOT write any shared
   "resume point" file.

## MODE: CONTINUOUS until context budget, then GRACEFUL HANDOFF
Run arcs back-to-back per §5 until `discovery/STOP` OR your context runs low. A chat cannot clear its
own context, so it has a finite arc budget — that is expected.
- **Never block on the operator** (§8). Resolve via the rules + a conservative bias.
- **Run jobs foreground/synchronous** so control returns on completion and the arc continues
  automatically. Do NOT background-and-wait for a notification — it may never fire.
- **Handoff:** when context runs low, finish the CURRENT arc fully (arc doc + both-tier log + commit +
  **push**), then stop. A fresh chat resumes via the bootstrap above. Append-and-push BEFORE stopping —
  a stranded uncommitted arc is the only real failure.

## VERDICTS (three-way disposition — §11)
- **PASS** (all-folds-positive IS+OOS + council) -> `discovery/passed/<name>/`.
- **PORTFOLIO** (mean-positive net of costs but NOT all-folds-positive) ->
  `discovery/portfolio-candidates/<name>/`.
- **KILL** (everything else, incl. beats-null-but-net-negative).
Mandatory council on any PASS before `passed/`. A passer is a CANDIDATE, not a deployment.

## WHERE TO LOOK (search steer — see LESSONS)
Shallow single-condition directional on liquid FX is mapped (don't re-invent momentum/breakout/
reversion/trend, long or short — proven-dead). A genuinely novel directional mechanism with a
documented *because* still earns a test. **Pre-shorts (now):** hunt a 2nd net-positive long-only
component for the PORTFOLIO route, and novel structural mechanisms. **Post-shorts:** relative-value /
market-neutral and short-side asymmetries open.

## GUARDRAILS (anti-Arc-10, never cross)
- Call the canonical measurement core; NEVER reimplement it.
- Exit/transform tools = GEOMETRY ONLY, never realize P&L (engine does, take-the-loss). Before a FAIL
  on a non-coin-flip entry, sweep the canonical exit menu as a WFO-internal hyperparameter (select on
  IS fold, score that fold's OOS, freeze onto holdout) — never full-sample best-pick.
- Never auto-merge CODE to main (FLAG it). Only `discovery/` docs + `discovery/tools/` flow freely.
- Check `TOOL_REGISTRY` BUILT first before building an experiment tool.
