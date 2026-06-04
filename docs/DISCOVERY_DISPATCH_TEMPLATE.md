# DISCOVERY PIPELINE — OPERATOR PLAYBOOK + DISPATCH TEMPLATE

> For any Claude.ai chat helping the operator run the autonomous discovery pipeline. This doc tells
> you (the chat) how to generate run dispatches, launch/halt the process, and review results — so
> any chat is on the same page, not just the one that built it. The authoritative spec is
> `DISCOVERY_PROTOCOL.md`; this is the operating layer on top of it.

## THE PIPELINE IN ONE PARAGRAPH
Autonomous CC chats each run a continuous loop: read the shared log + lessons -> observe data ->
invent a complete trading system (entry/exit/filters/sizing, unlimited creativity) -> develop it on
IS (2010-2020) -> cheap-kill cheaply (pool floor 50, oracle-best-cluster ceiling, 3-fold triage) ->
diagnose + fail the best version -> validate all-folds-positive on IS then OOS (2021-current) on the
SL-honest engine with FundedNext costs -> council stress-test survivors -> document -> repeat. The
sole judge is all-folds-positive IS+OOS, honest engine. Survivors are CANDIDATES (not auto-deployed).
Each chat owns a static 1000-wide arc-id range. Chats coordinate via append-only docs to main; code
stays human-gated.

## PRECONDITIONS (verify before generating any run dispatch)
1. `HONEST_ENGINE_SWEEP.md` top verdict = SAFE (engine is the verified sole honest gate engine).
2. `discovery/` structure exists on main (DISCOVERY_PROTOCOL.md, DISCOVERY_LOG.md, LESSONS.md,
   arcs/, passed/, results/).
3. The `llm-council-discovery` skill is installed user-level; CC sessions must be started AFTER it
   was installed (skills load at startup).
4. Arc 0 (the supervised trial) has been run and reviewed clean (machinery works end-to-end).
If any fails, do NOT generate a continuous-run dispatch — resolve first.

## HOW TO GENERATE A RUN DISPATCH (what the operator asks for)
When the operator says "start an endless arc run" / "spin up discovery" / "launch N chats", produce
a dispatch per launched chat with these elements. Keep it short — the protocol carries the detail;
the dispatch only sets framing + the chat's range.

A run dispatch MUST contain:
- **Mode:** CONTINUOUS — run arcs back-to-back until the STOP sentinel appears (NOT one-and-halt;
  that was Arc 0 only).
- **Arc-id range:** assign this chat a unique 1000-wide block (chat A 1000-1999, B 2000-2999,
  C 3000-3999, ...). Per-arc files `discovery/arcs/arc_<id>_<slug>.md`. State the range explicitly.
- **Authoritative spec:** "Follow `discovery/DISCOVERY_PROTOCOL.md` exactly." Do not restate it.
- **The loop entry:** pull main -> read DISCOVERY_LOG (both tiers) + LESSONS -> proceed per §5.
- **Honors (one-liners, the load-bearing reminders):** the ONE rule (§1, FundedNext costs ON);
  question everything (§2); FRESH EYES — no pre-reset eliminated-list, no signal priors (§5a);
  unlimited creative range, IS-only development, never tune OOS (§3-4); cheap kills incl. oracle
  ceiling used asymmetrically (§5d); sole judge all-folds-positive IS+OOS (§5g); council via
  `/llm-council-discovery`, MANDATORY survivor stress-test (§7); never block on operator, conservative
  bias, safety-violation = skip+document (§8); append-only docs to main, CODE HUMAN-GATED (§9).
- **STOP check:** at the top of each arc, check for `discovery/STOP`; if present, finish the current
  arc, write a halt note, stop (graceful — never abort mid-arc).
- **Per-arc output:** full arc doc + Tier-1 row + Tier-2 entry, committed (docs only).

## HOW TO LAUNCH (operator actions, state them in the dispatch)
1. Merge any pending discovery PRs to main (folder, council doc).
2. Open N fresh CC sessions (fresh = loads the council skill). N = 2-3 on a 6-core/48GB machine
   (CPU-bound on WFO; oversubscribing past ~3 degrades throughput). Start with the smallest N that
   feels useful; scale only if arc-completion time doesn't degrade.
3. Give each session its dispatch with its assigned arc-id range.
4. Each runs continuously, pulling main each arc to read the others' appended lessons.

## HOW TO HALT
- **Graceful stop (the normal way):** create an empty file `discovery/STOP` on main. Each chat sees
  it at the top of its next arc, finishes the arc it's on (no mid-arc abort), writes a halt note,
  and stops. Use this whenever you want to stop, change the protocol/engine, or run compression.
- **Do NOT** force-kill sessions mid-arc except in emergency — it can leave half-written arc state.
- To resume: delete `discovery/STOP`, re-launch sessions (fresh, with ranges).

## HOW TO CHECK IN / REVIEW (passive, never stops the run)
- **Routine check-in:** drag the current `DISCOVERY_LOG.md` (+ `LESSONS.md`) into a Claude.ai chat
  and read. Scan the Tier-1 `passed` column for any `Y`. If none, nothing to do — the run continues
  untouched. Check-ins NEVER require stopping.
- **A passer (passed = Y):** open `discovery/passed/<name>/` — exact config, IS+OOS results, honest-
  engine verification, council verdict, repro command + frame sha. This is the deep-dive surface.
- **Before trusting/deploying ANY passer:** independent re-verification (protocol §11) — a second
  implementation OR a hand-audit of a trade sample vs raw price must agree before real capital. The
  engine is honest-as-tested, never proven 100% live-aligned; OOS + council + independent re-verify
  are the layered defenses. (This is the Arc-10 institutional lesson — never deploy on one engine's
  word.)

## COMPRESSION (operator-run, out-of-band — NOT by the running chats)
When stopped, the operator (or a dedicated chat) distills the Tier-2 free-form log into LESSONS.md,
value-weighted: dead/thin entries compress to a line; rich reasoning/threads stay verbose. This keeps
both the chats' step-(a) read-cost and the operator's draggable check-in artifact bounded. Running
chats only APPEND; they never compress.

## WHAT NOT TO DO (the guardrails that prevent another Arc-10)
- Never let a chat auto-merge CODE (engine/exit/tooling/gate) — code is human-gated; chats FLAG, the
  operator decides. Only `discovery/` docs auto-append to main.
- Never consult a pre-reset eliminated-strategies list — fresh eyes; no signal-level priors.
- Never treat a backtest number as proof of live edge — OOS + council + independent re-verify.
- Never deploy a passer on the gate engine's word alone — independent re-verification first.
- Never run continuous mode before Arc 0 was reviewed clean.
