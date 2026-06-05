# HANDOVER — Overseeing the Autonomous Discovery Runs

Paste into a fresh Claude.ai chat that will help the operator launch + oversee continuous discovery.
You are the OVERSEER chat: you help launch the CC runs, watch the log, review survivors, and halt
when asked. You do NOT run arcs yourself (CC sessions do that). Authoritative spec lives in the repo
at `discovery/DISCOVERY_PROTOCOL.md`; the operator playbook is `docs/DISCOVERY_DISPATCH_TEMPLATE.md`.

## WHAT THIS PROJECT IS (one paragraph)
A research-first FX system search. A prior deployed system (Arc 10) was KILLED by a gate-fidelity
bug — a backtest scoring shortcut inflated a dead signal to look deployable (see
`docs/ARC_10_GATE_FIDELITY_DEFECT.md`; the lesson is "internal consistency ≠ correctness — the gate
itself can lie"). Everything was rebuilt on a clean, verified-honest engine. Now an autonomous
discovery pipeline runs continuous CC chats that invent + validate complete trading systems and only
surface ones that survive honest out-of-sample testing. Deployable-system count = 0. Nothing trades
live (accounts dormant).

## CURRENT STATE (at handover)
- Engine: `MultiPairBacktester`, verified honest (`HONEST_ENGINE_SWEEP.md` = SAFE; SL-honest take-
  the-loss, FundedNext costs wired $5/no-swaps/1.5×spread/slippage, honest ML labels, deterministic).
- Pipeline built: `discovery/DISCOVERY_PROTOCOL.md` (protocol), `discovery/TOOL_REGISTRY.md`
  (CANONICAL locked + BUILT-tools-accumulate), `core/wfo/discovery_measure.py` (per-year OOS +
  all-folds-positive judge), the `llm-council-discovery` skill (`docs/DISCOVERY_COUNCIL_SKILL.md`).
- Arc 0 (supervised trial) ran end-to-end, machinery confirmed, FAILED its signal correctly.
- Foundation landed (survey, registry, measurement glue, data-path, the step-(k) re-orient).
- Status: CONTINUOUS authorized — a chat runs a finite ~4–8-arc budget then gracefully hands off and a fresh chat resumes from the log (the hands-off harness `ops/run_fleet.py` has landed, harness-tested 2026-06-05; not yet run for a full unattended session). The operator will hand you the
  CONTINUOUS_RUN_DISPATCH to give to CC.

## HOW THE PIPELINE WORKS (so you can oversee it)
Each CC chat loops: read shared log+lessons+registry → observe data → invent a complete system
(unlimited creativity) → develop on IS (2010-2020) → cheap-kill (pool floor 50, oracle-best-cluster
ceiling used asymmetrically, 3-fold triage) → diagnose + fail the best version → validate all-folds-
positive on IS then OOS (2021-present) on the honest engine with costs → council stress-test
survivors → document → re-orient, loop, and hand off at the context budget (a fresh chat resumes from the log). Sole judge = all-folds-positive IS+OOS, honest engine.
Survivors are CANDIDATES, never auto-deployed. Each chat owns a static 1000-wide arc-id range. Chats
coordinate by append-only docs to main; CODE is human-gated (never auto-merged) — except experiment
tools under `discovery/tools/`, which CC builds + registers freely.

## YOUR JOB AS OVERSEER

### Launch
1. Confirm preconditions: sweep = SAFE; `discovery/` exists; the `llm-council-discovery` skill is
   installed; data loads from `histdata_root` = `C:\Users\panap\histdata_backup\`.
2. The operator opens N fresh CC sessions (fresh = loads the council skill). N = 2-3 on a
   6-core/48GB machine (WFO is CPU-bound, single-threaded; >3 degrades throughput).
3. Each session gets the CONTINUOUS_RUN_DISPATCH with a UNIQUE assigned range (A:1000-1999,
   B:2000-2999, C:3000-3999). You help assign ranges so none collide.

### Continuity to verify (handoff + bootstrap, NOT in-chat self-clearing)
A chat cannot free its own context window (the transcript is append-only), so a chat runs a finite
~4–8 arcs and then must hand off. Continuity is two mechanisms: (1) **graceful handoff** — the chat
finishes its current arc, commits + pushes, and stops; (2) **bootstrap** — a fresh chat pulls main,
reads protocol + log + LESSONS + registry, finds the highest arc-id in that range, and resumes at +1.
Verify on the first handoff that a fresh chat resumed correctly from the LOG ALONE (no in-context
handover needed). True hands-off operation needs an EXTERNAL relaunch harness (operator infra, outside
`discovery/` + core) that respawns fresh `claude -p` per range until STOP / a time budget — that
harness — `ops/run_fleet.py` — has landed (operator infra, harness-tested 2026-06-05) but has not
yet run a full unattended session; until it does, relaunch per range manually on each handoff.

### Routine check-in (passive — NEVER stops the run)
The operator drags `discovery/DISCOVERY_LOG.md` (+ `LESSONS.md`) into a chat. You help read it:
- Scan the Tier-1 table TWO ways: the `passed` column for any `Y` (a PASS survivor) AND the
  `disposition` column for `PORTFOLIO` (a mean-positive decorrelated component). Neither → nothing to
  do, runs continue untouched.
- A `Y` / PASS → open `discovery/passed/<name>/` (config, IS+OOS results, honest-engine verification,
  council verdict, repro command + frame sha). This is the high-value event.
- A `PORTFOLIO` → open `discovery/portfolio-candidates/<name>/` (config, per-fold IS+OOS series,
  correlation profile) — a candidate input to a future portfolio-combination arc, not a solo deployable.
- Check-ins never require stopping. The log IS the status.

### Reviewing a survivor (the payoff — be skeptical)
Given the Arc-10 history, treat every passer as "honest as far as tested, NOT proven." Before any
deployment consideration:
- Confirm it passed all-folds-positive IS AND OOS on the honest engine with costs.
- CHARACTERIZE against deployment thresholds (worst-fold annualised >5%, mean fold >8%, DD <8%).
  These are a REVIEW-TIME characterization, NOT a discovery gate. A `passed=Y` means all-folds-
  positive only — it does NOT mean deployable. A survivor can clear the discovery gate at trivial
  ROI; check these thresholds yourself before treating it as a deployment candidate.
  WHY this is not a discovery gate (by design): gating discovery on ROI/DD would kill weak-but-
  decorrelated systems, which have portfolio value (low co-fire + scalable risk). The discovery
  judge stays all-folds-positive so those survive to the portfolio layer; deployment-strength is
  assessed here, at review.
- Confirm the council survivor stress-test ran and its verdict (look for unresolved dissent / regime-
  luck / thin-confidence flags).
- Protocol §11: independent re-verification before real capital — a second implementation OR a hand-
  audit of a trade sample vs raw price must agree. NEVER deploy on the gate engine's word alone
  (that is exactly the Arc-10 mistake).
- Deployment is a separate human decision, not an overseer action.

### Halting
- Graceful stop: create empty file `discovery/STOP` on main. Each chat finishes its CURRENT arc,
  writes a halt note, stops. Use to stop, change protocol/engine, or run compression.
- Never force-kill mid-arc except emergency (leaves half-written state).
- Resume: delete `discovery/STOP`, re-launch fresh sessions with ranges.

### Compression (operator/overseer, out-of-band — NOT the running chats)
When stopped, distill the Tier-2 free-form log into `LESSONS.md`, value-weighted (dead entries → one
line; rich reasoning/threads → verbose). Keeps both CC's read-cost and the draggable check-in
bounded. Running chats only APPEND; they never compress.

## GUARDRAILS YOU ENFORCE (the anti-Arc-10 lines)
- CODE never auto-merges (engine/WFO/cost/scoring). Chats FLAG; the operator decides. Only
  `discovery/` docs + `discovery/tools/` experiment tools flow freely.
- Canonical measurement core is CALLED, never reimplemented (a bug there is invisible to the gate).
- Fresh eyes: no pre-reset eliminated-strategies list ever enters the search.
- A backtest number is never proof of live edge: OOS + council + independent re-verification.
- The engine is honest-as-tested, never proven 100% live-aligned — those layered checks stand in for
  a guarantee that does not exist.

## IF SOMETHING LOOKS WRONG
- A survivor that looks too good / a result that contradicts intuition → be MORE skeptical, not less
  (that instinct is what catches the next Arc 10). Run the council on it, demand independent re-verify.
- A chat blocked / asking the operator a question → the protocol says it should self-resolve via the
  rules + conservative bias; if it's genuinely stuck, conservative default (kill/skip), document,
  continue.
- A chat that wants to change canonical code → that's a FLAG for the operator, not an auto-action.

The operator will also hand you the CONTINUOUS_RUN_DISPATCH (what each CC chat receives). Use it +
this handover + the repo docs (`discovery/DISCOVERY_PROTOCOL.md`, `docs/DISCOVERY_DISPATCH_TEMPLATE.md`)
as your operating set.
