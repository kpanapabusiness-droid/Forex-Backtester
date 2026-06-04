# CC DISPATCH: CONTINUOUS DISCOVERY RUN (per chat — assign a range)

> One of these per concurrent CC chat. The ONLY thing that differs between chats is the assigned
> arc-id range. Authoritative spec is `discovery/DISCOVERY_PROTOCOL.md` — this dispatch sets framing
> and the range; it does NOT restate the protocol.

## YOUR ASSIGNED RANGE
**Chat range: arc-ids [ASSIGN: e.g. 1000-1999].** Per-arc files: `discovery/arcs/arc_<id>_<slug>.md`.
Use ids only within your range — never collide with another chat's block. (Operator assigns: chat A
1000-1999, chat B 2000-2999, chat C 3000-3999.)

## PRECONDITIONS (verify before arc 1; STOP + report if any fail)
- `HONEST_ENGINE_SWEEP.md` top verdict = SAFE.
- `discovery/` exists on main (protocol, DISCOVERY_LOG.md, TOOL_REGISTRY.md, LESSONS.md, tools/,
  arcs/, passed/, results/).
- `/llm-council-discovery` is loadable (this session started AFTER the skill was installed).
- Canonical data: load via `histdata_root` = `C:\Users\panap\histdata_backup\` (one-time ~9min cache
  warm, then cached). Do NOT conclude "regenerate 12-24h" — the corpus is intact at that path.

## MODE: CONTINUOUS
Run arcs back-to-back per `discovery/DISCOVERY_PROTOCOL.md` §5 (a→k), continuously, until the STOP
sentinel appears. This is NOT one-and-halt (that was Arc 0). Each arc:
1. (a) Pull main → read DISCOVERY_LOG (both tiers) + LESSONS + TOOL_REGISTRY. Show the reading.
   FRESH EYES: no pre-reset eliminated-list, no signal-level priors. Honest-era log only.
2. (b–h) Observe → idea (documented *because*) → characterize → cheap kills (pool floor 50; oracle
   ceiling ASYMMETRIC: low→KILL, high→proceed-not-"works"; 3-fold triage) → diagnose → address +
   fail the best version → validate all-folds-positive IS then OOS (honest engine, FundedNext costs
   ON via the canonical measurement entry point) → MANDATORY council survivor stress-test on any
   IS+OOS passer (`/llm-council-discovery`) before `passed/`.
3. CALL the canonical measurement core (do NOT re-roll it): `build_arc_pool`, the fold runners,
   `core/wfo/discovery_measure.py` (per-year OOS + all-folds-positive judge). For EXPERIMENT tooling
   (filters/clusterers/transforms/exits/null-baseline): check `TOOL_REGISTRY` BUILT first → if it
   exists, call it → if not, build it under `discovery/tools/`, use it, and APPEND it to the registry
   at arc end. Build experiment tools FREELY (a bad one fails the WFO and dies). NEVER reimplement
   the CANONICAL/LOCKED measurement core.
4. (i) Document: full arc doc + Tier-1 row + Tier-2 reasoning. Commit DOCS ONLY direct to main
   (append-only). If CODE needs changing in the canonical core, FLAG it in the arc doc — do NOT
   merge code. (Experiment tools under discovery/tools/ are the exception — those are yours to add.)
5. (k) SELF-DEBLOAT: AFTER the log append (the save), shed this arc's transient detail (WFO output,
   council transcript, data dumps, scratch) from context; retain only loop-state (your identity,
   range, position). Then start the next arc fresh from the log. Append-BEFORE-shed always.
6. (j) STOP check: at the top of each arc, check for `discovery/STOP`. If present, finish the
   CURRENT arc fully (never abort mid-arc), write a halt note, and stop.

## FIRST-RUN VERIFICATION (this chat's arc 1 → arc 2 only)
Self-debloat (step k) has never executed live (Arc 0 ran one arc). On YOUR transition from arc 1 to
arc 2, explicitly confirm: the log append happened, context was shed, and arc 2 correctly re-read the
log fresh and continued. Report this once after arc 2 starts cleanly, then proceed continuously
without further per-arc reporting. If self-debloat does NOT free context in practice, fall back to a
hard context reset + re-read protocol/range/log, and note it.

## OUTPUT / RHYTHM
- Per arc: arc doc + both-tier log append, committed. No per-arc operator report (the log IS the
  report). Exception: the one-time arc-1→2 self-debloat confirmation above.
- A survivor (IS+OOS all-folds-positive + council-cleared) → full `discovery/passed/<name>/` deep
  record (config, IS+OOS results, honest-engine verification, council verdict, repro command + frame
  sha). This is the high-value event; everything else is routine.

## GUARDRAILS (the anti-Arc-10 lines — never cross)
- Never reimplement the canonical measurement core (call it). Bug there is invisible to the gate.
- EXIT/TRANSFORM TOOLS DEFINE GEOMETRY ONLY — NEVER REALIZE P&L. An experiment exit tool may
  specify WHERE stops/targets/trails sit (price geometry); it must NEVER compute realized R itself.
  ALL R realization routes through `MultiPairBacktester`'s bar-walk under the take-the-loss invariant
  (stop breach at or before the +1R partial = -1R; same-bar SL+TP = SL-first; ambiguity never wins).
  This is the LITERAL Arc-10 site: Arc 10 was a freely-built-looking exit scorer that realized its
  own P&L and skipped the pre-partial stop. A self-realizing exit tool is an Arc-10 bug the gate
  cannot see. If a tool needs to score a trade, it is the canonical core, not an experiment tool.
- Never auto-merge CODE to main (flag it). Only discovery/ docs + discovery/tools/ experiment tools.
- Never consult a pre-reset eliminated-strategies list. Fresh eyes.
- Never treat a backtest number as proof of live edge — all-folds-positive IS+OOS + council, and a
  passer is a CANDIDATE (independent re-verification before any deployment, per protocol §11).
- Never block on the operator. Resolve via the rules + conservative bias; safety-rule conflict =
  skip that arc, document, continue.

## DEFINITION OF DONE
N/A — this is continuous. It ends only on `discovery/STOP`. "Working correctly" = arcs flowing,
both-tier log appended each arc, canonical core called (not re-rolled), experiment tools registered +
reused, self-debloat confirmed on arc 1→2, survivors getting `passed/` records, no code auto-merged.
