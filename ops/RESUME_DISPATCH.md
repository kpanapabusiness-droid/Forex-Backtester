# CC DISPATCH — CONTINUOUS DISCOVERY (per chat — assign a range)

Autonomous discovery run. You pick up from the shared log — nothing is lost, nothing to redo.
Authoritative spec: `discovery/DISCOVERY_PROTOCOL.md`. Range: **[ASSIGN — 1000-1999 / 2000-2999 /
3000-3999].** Use ids ONLY within your range. Per-arc files: `discovery/arcs/arc_<id>_<slug>.md`.

## RESUME PROCEDURE
1. Pull `main`. Read `discovery/DISCOVERY_PROTOCOL.md` (follow it), `discovery/DISCOVERY_LOG.md` (both
   tiers, incl. `disposition`), `LESSONS.md`, `TOOL_REGISTRY.md`. The LOG is your memory; context starts
   empty. Show the reading. FRESH EYES (honest-era log only).
2. Check `discovery/STOP`. Present -> write a halt note + stop. Absent -> continue.
3. Find the highest arc-id IN YOUR RANGE in the log; resume the loop at step (a) from that id + 1 (or the
   range floor if none). Contention-free; write no shared "resume point" file.

## MODE: CONTINUOUS until STOP, then GRACEFUL HANDOFF at context budget
Run arcs back-to-back per §5. The ONLY deliberate halt is the `discovery/STOP` sentinel (operator-set).
A chat cannot clear its own context, so it has a finite arc budget — when context runs low, finish the
CURRENT arc fully (arc doc + both-tier log + commit + **push**), then stop; a fresh chat resumes via the
bootstrap. Append-and-push BEFORE stopping. Never block on the operator (§8). Run jobs
foreground/synchronous so control returns automatically (never background-and-wait).

## WHERE TO LOOK (frontier — SHORTS ARE NOW OPEN)
**Shorts are merged and verified (PR #273, honest-engine sweep SAFE) — fully available. This supersedes
any "pre-shorts / long-only" framing still in LESSONS.** You are NOT restricted to shorts and NOT
required to use them — pursue the highest-expected-value frontier, which now includes both long and short:
- **Short-side asymmetries the corpus already flagged as STRONGER short** — the up-gap weekend SHORT
  (arcs 2001/2003: up-gap drift ~-0.57 ATR, ~0.64 acc, stronger than the down/long leg) and the
  failed-breakdown up-sweep SHORT (arc 1013 noted the up-sweep is arguably the stronger leg). These are
  where shorts add what a long cannot.
- **Relative-value / market-neutral** — now structurally possible (two simultaneous legs); the only lever
  that does not require beating 0.50 per trade.
- **The regime-orthogonal 3rd+ portfolio component (arc 2006 spec)** — positive when reversion bleeds
  (risk-off / strong-USD years 2015/16/18/20); most naturally a short / risk-off-positive construction.
- **Deep multi-factor directional, EITHER direction** — the arc-1013 template (structure × sequence ×
  magnitude conjunction) is proven productive; extend it (incl. its short mirror) and other structural
  mechanisms with a documented *because*.
- **Portfolio combination** — once ≥3 net-positive, regime-complementary components exist, the combined-
  book all-folds-positive WFO is the deployable gate.

To trade a short, set the signal/config `direction` (the merged short path; the configs' `direction:` key
is now load-bearing). Still MAPPED — do NOT re-run: **shallow single-trigger directional, long OR short**
(momentum / breakout / mean-reversion / trend, one condition, one pair). Shorts do not revive these — by
symmetry the shallow short base is the same coin-flip as the long; the value of shorts is in the
asymmetries and relative-value above, not symmetric shallow direction. A genuinely novel mechanism
(either side) with a documented *because* still earns a fresh test (§5a).

## VERDICTS (three-way disposition — §11)
- **PASS** (all-folds-positive IS+OOS + mandatory council) -> `discovery/passed/<name>/`.
- **PORTFOLIO** (mean-positive net of costs, NOT all-folds-positive) -> `discovery/portfolio-candidates/<name>/`.
- **KILL** (everything else, incl. beats-null-but-net-negative).
A passer is a CANDIDATE — independent re-verification before any deployment, never on the engine's word.

## MEASUREMENT INTEGRITY (non-negotiable)
Develop freely on IS; the gate stays honest: full-pool WFO; fair same-conditions null; pool floor (>=50)
at EVERY conditioning stage; **the OOS holdout is one-shot and FROZEN — never tune against it.** Before a
FAIL on a non-coin-flip entry, sweep the canonical exit menu as a WFO-internal hyperparameter (select on
IS fold, score that fold's OOS, freeze onto holdout) — never full-sample best-pick.

## GUARDRAILS (anti-Arc-10, never cross)
- Call the canonical measurement core; NEVER reimplement it.
- Exit/transform tools = GEOMETRY ONLY, never realize P&L (engine does, take-the-loss).
- Never auto-merge CODE to main (FLAG it). Only `discovery/` docs + `discovery/tools/` flow freely.
- Check `TOOL_REGISTRY` BUILT first before building an experiment tool.
