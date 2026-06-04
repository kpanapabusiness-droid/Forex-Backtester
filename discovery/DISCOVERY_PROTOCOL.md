# DISCOVERY PROTOCOL — Autonomous Self-Running Signal Discovery

> **Status:** v1.0 (draft for first-arc trial)
> **Purpose:** find a deployable trading system through continuous, autonomous, self-documenting
> discovery. CC generates ideas, develops complete systems, validates honestly, documents, learns
> from the accumulated record, and repeats — with minimal human input.
> **Relationship to L_PROTOCOL:** L_PROTOCOL is NOT run as a stage. Its PRINCIPLES (no lookahead,
> ex-ante populations, honest accounting, worst-fold judgment) are embedded here. L_PROTOCOL is
> kept as a reference for those principles only.

---

## 0. PRECONDITION (hard gate — do not start without this)
This protocol is valid ONLY on a verified-honest engine. `MultiPairBacktester` must have passed
the honest-engine sweep with NO unresolved flags (especially heavy_ml label contamination). If
the sweep has open flags, DO NOT run discovery — the entire process is meaningless on a
non-honest engine. Confirm `HONEST_ENGINE_SWEEP.md` verdict is clean before any arc.

## 1. THE ONE RULE (the only inviolable constraint)
No lookahead. No contamination. Everything calculated correctly. WFO on FundedNext guardrails:
1.5x spread, slippage (0.5 pip/fill x n_fills), NO swaps, $5/lot round-turn commission. EET.
This is the baseline every arc honors. Everything else is open.

If proceeding would require violating this rule, STOP that arc, document why, move to the next
idea. NEVER bend a safety rule to keep moving. (Progress never blocks on the human; safety never
bends for progress.)

## 2. PRIME DIRECTIVE — QUESTION EVERYTHING
Never accept the first answer. For every observation and every result: ask why, then ask why
again. The first hypothesis is a starting point to interrogate, not a conclusion to test.
What's here? Why could this work? Why does it? Why doesn't it? What does it mean? How else
could we look at it? Is this the best way? Dig deeper, think better. This posture applies at
EVERY step below — the steps are just this posture operationalized.

## 3. UNLIMITED CREATIVE RANGE
CC invents the entire system: entry, exit, filters, sizing logic. Any timeframe, ATR, price
geometry, mathematical transform, time-of-day / time-of-year effect, baseline, cross-timeframe
interaction, regime conditioning, and any combination. There is NO fixed feature list and NO
prescribed technique. Do not default to prior arcs' methods. The creativity lives in the
DIAGNOSIS and DESIGN, grounded in observation (see §5). Every modification must trace to a
data-observation or a market-mechanism — documented as a *because*. "I tried delaying entry" is
invalid; "the failing trades all entered during a consistent pre-move adverse dip [observation],
so delaying entry past it should cut drawdown without missing the move [mechanism] -> tested ->
result" is valid.

## 4. DEVELOPMENT WINDOW DISCIPLINE
- ALL development happens on IS (2010-2020): entry, exit, filters, the complete system.
- NEVER tune/optimize on OOS (2021-current). MEASURING OOS multiple times is fine; OPTIMIZING to
  it is contamination. The discipline is "never fit a parameter to OOS," not "touch once."
- Develop the COMPLETE finished system on IS, so the holdout is touched by a finished system, not
  during development.

## 5. THE ARC LOOP (one arc, start to finish)
Each arc follows this loop. CC runs it autonomously; the council (§7) is invoked at the marked
junctures.

**(a) READ + SYNTHESIZE THE LOG.** Before anything else, read the full DISCOVERY_LOG (both tiers,
§6) + LESSONS.md. Explicitly summarize what's been tried across all chats, what's dead, what
threads are open. SHOW this reading in the arc doc — an arc that skips this produces no learning.
Pull main first so the log is current (other chats append continuously).
- **FRESH EYES — do NOT consult any pre-reset "eliminated strategies" list as a constraint on what
  to explore.** Those conclusions were drawn on a RETIRED engine (replay-inflated, cost-blind,
  contaminated labels) and WITHOUT this protocol's freedom to develop a signal to its best version.
  Treating them as settled would (1) import untrustworthy verdicts and (2) bias the search against
  directions that may never have had a fair, honest, fully-developed test. You may RE-EXPLORE
  anything the old regime "killed" — including signals that previously appeared to work — because
  nothing pre-reset is trustworthy. The ONLY pre-reset carry-forward is the gate-fidelity
  METHODOLOGICAL lesson (`docs/ARC_10_GATE_FIDELITY_DEFECT.md`: internal consistency != correctness;
  a single unchecked engine can lie) — read that for how validation deceives, NEVER as a "don't try
  signal X" instruction.
- **Honest-era only carries weight:** the discovery log + LESSONS accumulate from THIS clean process
  (post-reset). Those entries are trustworthy and inform the search. Pre-reset signal-level
  conclusions do not.

**(b) FORM AN IDEA (observe, don't guess).**
- Log-seeded: a thread from the log is interesting -> look at how it applies on the charts / data.
- Log-dry: nothing jumps out -> examine price/structure directly until something does.
- Either way: examine the ACTUAL data before hypothesizing. The idea EMERGES from observation
  (logs + data), not from thin air, and not as a preconceived thing you go find data to justify.
- [COUNCIL — optional, light] If genuinely stuck or at a real idea-fork, convene the council for
  generative perspectives. CC synthesizes its own idea from them (generative = CC decides).

**(c) CHARACTERIZE.** Build the ex-ante population (`build_ex_ante_bounded_population`). Look at
what the trades actually do: path shapes, pre-entry and post-entry behavior, where the move
happens, adverse excursion, regime/session/vol concentration. Cluster path shapes if useful.

**(d) CHEAP KILLS (compute-savers — honest engine throughout, less data early).**
- **Pool floor:** < 50 trades over IS (~1/week) -> KILL (too thin).
- **Oracle-best-cluster ceiling:** cluster the paths, take the best cluster, score it with ORACLE
  labels (perfect-hindsight cluster membership) = the CEILING (best possible if you could perfectly
  identify the good cluster). Use ASYMMETRICALLY:
  - Ceiling weak -> KILL. Trustworthy: nothing can exceed the oracle, so a mediocre ceiling = dead.
  - Ceiling strong -> PROCEED. This is NOT "it works" — only "capturable upside exists, worth the
    effort to extract." Do not celebrate; go diagnose.
- **Triage:** honest engine on ~3 representative folds. Deeply negative -> KILL before full WFO.

**(e) DIAGNOSE (only for ideas with a worthwhile ceiling).** WHY isn't the raw version hitting the
ceiling? Entry timing? Exit giving the move back? Selection (some trades great, some terrible)?
Form a mechanistic explanation from the data. Question it (§2) — don't take the first diagnosis.
- [COUNCIL — heavy] Convene the council on the diagnosis (evaluative). It recommends; CC commits
  (override only with documented reason).

**(f) ADDRESS THE DIAGNOSIS (fail the BEST version).** Design the entry/exit/filter the diagnosis
implies — with a documented *because*. Reason about which version SHOULD be best and test that;
sweep only to confirm a reasoned hypothesis, never as blind primary search. heavy_ml is available
HERE as a PROBE (does extractable structure exist in these features?) — a question-answering
instrument, not a solution; it does not absolve CC of understanding WHY. An idea is only marked
unusable after its best reasoned version has failed — so "FAIL" means the FAMILY is dead, not that
the first naive cut failed.

**(g) VALIDATE.** Full honest WFO on IS -> all-folds-positive? If yes, measure OOS (2021-current)
-> all-folds-positive? The SOLE judge is all-folds-positive on IS AND OOS, honest engine,
FundedNext guardrails. ROI / DD / correlation are CHARACTERIZED, not gated (a weak-but-decorrelated
system has portfolio value; risk is scalable). Bonferroni-style correction, if applied, is applied
to the small set of FINALISTS that reach WFO — NOT to every searched candidate (search-stage
correction is too harsh; the holdout does the real work).

**(h) SURVIVOR STRESS-TEST [COUNCIL — mandatory, heaviest].** Any candidate passing IS+OOS
all-folds-positive MUST face the council before promotion to `passed/`. The council tries to break
it (why might this be hollow / regime-luck / the wrong framing). A real hole is very hard to wave
away. This is the Arc-10 insurance — the adversarial review on every survivor before it is trusted.

**(i) DOCUMENT.** Write the arc record (§6): the reasoning trail — what observed, what diagnosed,
what tried and WHY, what happened, the verdict. Append to the log. Commit (docs only) to main.

**(j) NEXT ARC.** Check the stop sentinel (§9). If absent, go to (a) for the next idea.

## 6. DOCUMENTATION (the learning mechanism)
CC does not truly learn across sessions; the WRITTEN CORPUS is what compounds. Two-tier log:

**Tier 1 — strict schema table (top of DISCOVERY_LOG.md, machine-scannable).** One row per arc,
fixed fields:
`arc_id | chat | timestamp | hypothesis (one line) | IS-all-folds-pos (Y/N) | OOS-all-folds-pos
(Y/N) | worst-fold ROI (IS/OOS) | worst DD | n_trades | VERDICT | passed (Y/N)`
The `passed (Y/N)` column is the operator's check-in scan (grep "| Y |").

**Tier 2 — free-form reasoning (below the table, per arc, under a `## arc_NNNN` header).**
Unstructured. The why/because, the approach taken and the reason, what was tried, what didn't
help, threads worth pursuing. This is the high-value cross-chat learning. Append freely.

**Discovery chats only APPEND — they never compress.** A running discovery chat writes its Tier-1
row and Tier-2 reasoning and nothing else; it does NOT edit LESSONS.md or compress the log. This
keeps the autonomous chats simple and avoids two chats compressing the same file mid-run.

**Compression is OPERATOR-RUN, out-of-band.** When the operator stops the run (§9), they (or a
dedicated, separate chat) distill the free-form tier into LESSONS.md — value-weighted: thin/dead
entries (single idea, ran once, died) compress hard to one ledger line; rich entries (whole
approach, reasoning, why it failed, live threads) stay verbose, as the information is valuable.
This keeps both the read-cost at step (a) AND the operator's draggable check-in artifact bounded
as the corpus grows. New arcs read LESSONS.md + recent raw entries, not the entire raw history.

**Reading vs writing:** discovery chats READ both DISCOVERY_LOG and LESSONS.md at step (a); they
WRITE only appends to DISCOVERY_LOG. Compression of either is never a discovery-chat action.

**`passed/` folder — deep record for survivors.** Anything passing IS+OOS + council stress-test
gets `discovery/passed/<name>/`: exact config (every parameter), full results (per-fold IS+OOS,
costs, DD profile), honest-engine verification, council verdict, and the EXACT reproduction
command + frame sha (so any passer can be independently re-verified — the thing that would have
caught Arc 10). This is the operator's deep-dive target.

## 7. THE COUNCIL OF FIVE (judgment stand-in while operator is away)
A separate skill (the customized `llm-council-discovery`, see COUNCIL_OF_FIVE_SKILL_DISPATCH).
Invoke it deterministically via the slash trigger **`/llm-council-discovery`** (the slash form maps
1:1 to the skill, zero collision with the stock general-purpose council — required for autonomous
protocol-driven calls; natural-language phrases like "discovery council" exist as operator
conveniences but rely on model-judgment, so autonomous chats use the slash form). Supply the
enforced structured input (question + observations + tried-before + metrics). Invoked at:
- (b) idea-fork / stuck — LIGHT (generative; CC synthesizes its own idea).
- (e) diagnosis — HEAVY (evaluative; CC commits, overrides only with reason).
- (h) survivor stress-test — MANDATORY, HEAVIEST (adversarial; a real hole is hard to wave away).
The council always RECOMMENDS; CC always COMMITS (never move the single-point-of-failure to the
chairman). Can't-decide -> honest split resolves via conservative bias; missing-info -> CC fetches
and re-convenes; thin-confidence -> flag and gather. Never blocks on the human.

## 8. AUTONOMY + CONSERVATIVE BIAS
- CC never blocks waiting for the operator. Every decision resolves against the explicit rules
  (§1) + conservative bias. When genuinely uncertain, take the safer/more-honest route and
  DOCUMENT the choice. Rather be surprised for good reasons live than bad.
- The ONLY hard stop is a §1 safety-rule violation (skip that arc, document, continue) or the
  stop sentinel (§9).

## 9. MULTI-CHAT COORDINATION + STOP
- **Numbering:** each chat owns a static 1000-wide arc-id range (chat A: 1000-1999, B: 2000-2999,
  C: 3000-3999, ...). No collision possible; no claim-race.
- **Per-arc files:** `discovery/arcs/arc_<id>_<slug>.md` — unique per arc, never collide.
- **Coordination = append-only documentation to main.** Each chat commits ONLY `discovery/`
  documentation (its arc file + the appended log/lessons) directly to main, no full PR. Append-only
  log writes don't conflict. Each chat PULLS main at step (a) so it reads other chats' latest.
- **CODE STAYS HUMAN-GATED.** No chat auto-merges code (engine, exit logic, tooling, the gate). If
  a chat believes code needs changing, it FLAGS it in its arc doc; it does NOT merge it. Code is
  how Arc-10-class bugs reach main — it never auto-merges. (This is the load-bearing safety line
  for multi-chat.)
- **Graceful stop (finish-then-halt, never instant):** check for a `discovery/STOP` sentinel at
  the top of each arc (step a). STOP means "start NO new arc" — it does NOT mean abort what you're
  doing. If STOP appears mid-arc, FINISH the current arc completely (validation, documentation,
  log append, commit), THEN halt before starting the next. This avoids half-written state. The
  operator creates `STOP` only for a deliberate full halt (done for now, or about to change the
  protocol/engine/guardrails) — NOT for routine check-ins.
- **Check-ins are passive and never stop the run.** The operator reviews by dragging the current
  DISCOVERY_LOG (+ LESSONS.md) into a separate chat and reading it while the discovery chats keep
  running. If nothing's worth digging into, do nothing — it carries on. There is no reason to halt
  for a check-in; STOP is reserved for deliberate full halts only.

## 10. RUN STAGING (trial before continuous)
- **Arc 1 (first ever, and the first arc of each new chat) is a SUPERVISED TRIAL.** It runs ONE
  arc to completion and HALTS for operator review of the full loop: log read, characterization,
  cheap kills, council invocation, validation, documentation, log append, commit. Do NOT enter
  continuous operation until the operator confirms the loop works end-to-end.
- After the trial is confirmed: continuous operation, 2-3 parallel chats (CPU-core bound on the
  operator's machine — do not oversubscribe; throughput degrades past ~3 concurrent WFO streams).
- **Operator check-in ritual (passive, read-only, run continues):** the operator drags the current
  DISCOVERY_LOG (Tier-1 table + recent Tier-2 reasoning) and LESSONS.md into a separate chat and
  reads -> scans the `passed` column -> opens `discovery/passed/<name>/` for any Y -> decides
  whether to dig deeper. The discovery chats KEEP RUNNING throughout; the check-in never halts
  them. If nothing's worth noting, do nothing — it carries on. A passer that survived IS+OOS+council
  is worth attention; nothing else needs it.

## 11. WHAT GRADUATES
A survivor (IS+OOS all-folds-positive + council-cleared, fully recorded in `passed/`) is a
CANDIDATE worth the operator's deep review — NOT an auto-deploy. Deployment is a separate,
human-made decision on the operator's return. Discovery FINDS candidates; it does not deploy them.

**Independent re-verification before any deployment (the Arc-10 institutional lesson).** No
candidate is deployed on the gate engine's word alone, however clean it looks — that single-engine
trust is exactly what produced Arc 10. Before real capital, a passer's numbers must be
re-verified via a GENUINELY INDEPENDENT path: a second implementation, OR a hand-audit of a
representative sample of its trades against raw price (entry, exit, R, cost) confirming they match
the engine's claim. The `passed/` record already requires the exact repro command + frame sha;
this extends it — the candidate is "honest as far as the engine can tell" until an independent
check agrees, and only then is it deployment-eligible. The engine cannot be proven 100% aligned
with live markets; OOS survival + council + independent re-verification are the layered defenses
that stand in for a guarantee that does not exist.
