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

## MODE: CONTINUOUS (within a finite arc budget — hand off, don't run forever)
Run arcs back-to-back per `discovery/DISCOVERY_PROTOCOL.md` §5 (a→k) until EITHER the STOP sentinel
appears OR your context budget runs low. This is NOT one-and-halt (that was Arc 0) — but it is also
NOT immortal: a chat runs ~4–8 arcs, then gracefully HANDS OFF (finish the current arc fully, commit +
**push**, stop) and a fresh chat bootstraps from the log and resumes at the next arc-id in your range
(protocol §10). The step-(k) re-orient drops working detail from active attention between arcs; it
does NOT free the context window. Each arc:
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
5. (k) RE-ORIENT: AFTER the log append (the save), drop this arc's transient detail (WFO output,
   council transcript, data dumps, scratch) from ACTIVE ATTENTION and re-anchor on loop-state (your
   identity, range, position); start the next arc fresh from the log. This does NOT free the context
   window (the transcript is append-only) — so when context runs low, HAND OFF (§10). Append-BEFORE-
   re-orient always.
6. (j) STOP check: at the top of each arc, check for `discovery/STOP`. If present, finish the
   CURRENT arc fully (never abort mid-arc), write a halt note, and stop.

## CONTINUITY (handoff + bootstrap, NOT in-chat self-clearing)
A chat cannot free its own context window (the transcript is append-only), so continuity is handoff +
bootstrap. When you approach a low-context handoff: FINISH the current arc fully (arc doc + both-tier
log + commit + **push**), then stop — never block on the operator, never background-and-wait. The NEXT
chat on your range bootstraps purely from the log: pull main, read protocol + log + LESSONS + registry,
find the highest arc-id in your range, resume at +1. If you ARE that bootstrapping chat, confirm once
that the log alone re-oriented you (no in-context memory of a prior chat needed), then proceed. The LOG
is the memory; context starts empty.

## OUTPUT / RHYTHM
- Per arc: arc doc + both-tier log append (Tier-1 row INCLUDING the `disposition` column), committed.
  No per-arc operator report (the log IS the report).
- Disposition (protocol §11): **PASS** (IS+OOS all-folds-positive + council-cleared) → full
  `discovery/passed/<name>/` deep record (config, IS+OOS results, honest-engine verification, council
  verdict, repro command + frame sha) — the high-value event. **PORTFOLIO** (mean-positive net of
  costs but not all-folds-positive) → `discovery/portfolio-candidates/<name>/` (config, per-fold
  IS+OOS series, correlation profile) — a decorrelated component for a later combination arc, not
  deployable solo. **KILL** (everything else, incl. beats-null-but-net-negative) → log row only.

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
N/A for a single chat — the RUN is continuous ACROSS chats; an individual chat ends on `discovery/STOP`
OR on a graceful low-context handoff (finish arc → commit + **push** → stop), after which a fresh chat
resumes from the log. "Working correctly" = arcs flowing, both-tier log appended each arc (with
`disposition`), canonical core called (not re-rolled), experiment tools registered + reused,
handoff-and-bootstrap continuity (NOT in-chat self-clearing), survivors getting `passed/` records and
PORTFOLIO edges getting `portfolio-candidates/` records, no code auto-merged.
