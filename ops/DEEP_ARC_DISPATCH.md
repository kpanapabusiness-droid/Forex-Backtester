# CC DISPATCH — DEEP-ARC SEARCH (multi-factor directional)

You are running DEEP directional search — the multi-factor, structurally-grounded entries the prior
shallow probes never tried. Authoritative spec: `discovery/DISCOVERY_PROTOCOL.md`. Range: **[ASSIGN —
1000-1999 / 2000-2999 / 3000-3999].** Standard bootstrap: pull main, read log + LESSONS + registry,
resume at the highest arc-id in your range + 1.

## READ THIS FIRST — you are NOT re-walking closed ground
The corpus has mapped **SHALLOW single-trigger** directional setups (momentum / breakout /
mean-reversion / trend — ONE condition, ONE pair, ONE timeframe, default exit) and found them ≈ cost.
**That closes the shallow slice, NOT directional prediction.** DEEP, multi-factor directional is the
OPEN frontier and is exactly the §5(a) carve-out ("a genuinely novel mechanism with a documented
*because* still earns a fresh test"). LESSONS' closed-ground prior forbids re-inventing the shallow
setups — it does NOT forbid building deep ones. Do NOT flee to calendar / microstructure / gap-fill
(those are mapped). Go DEEPER into structure. Nothing in the corpus has ever passed IS all-folds-
positive; your job is to find out whether depth can.

## WHAT A DEEP ARC IS (depth, not another cheap-kill)
It is NOT observe→triage→kill. You DEVELOP the best version of a real hypothesis before any verdict:
1. **Structural *because*.** A market reason an edge could exist — order flow, positioning, liquidity
   provision, participant behaviour, regime structure — not a chart shape. The *because* is what
   justifies why several conditions should interact.
2. **Multi-condition entry, developed FREELY on IS (2010–2020).** Intersect multiple entry-observable
   conditions — regime, volatility-state, session, structure, multi-timeframe context, sequence/path.
   **No cap** on the number of conditions or the threshold tuning; develop as far as the data rewards.
   Conditions, thresholds, AND the A1 exit menu (canonical strings only) are all hyperparameters you
   optimise on IS.
3. **Immediate milestone: IS all-folds-positive.** Tune toward a signal positive across ALL IS folds
   (not merely mean-positive). This is the bar nothing has cleared.
4. **Then ONE shot at the frozen OOS holdout (2021+).** Once you commit an IS-developed config, score
   it on the holdout ONCE. Survives all-folds-positive OOS → PASS candidate → mandatory council.
   Breaks → this is the real falsification the programme never ran on directional; KILL with the clean
   unconfounded proof.

## MEASUREMENT INTEGRITY (NON-NEGOTIABLE — not degrees of freedom)
Develop as aggressively as you want on IS; the gate stays honest:
- **Full-pool WFO** (never admit-only — that was a false-positive factory).
- **Fair same-conditions null** — random entry with the SAME exit / pairs / conditioning, so "beats
  null" means the structure adds something beyond regime drift.
- **Pool floor (≥50) at EVERY conditioning stage** — report n at each cut. If stacking conditions
  starves the sample below the floor, that IS your signal you've gone too far — do not relax the floor.
- **The OOS holdout is one-shot and FROZEN.** The single line that cannot move: never peek-then-tune,
  never "try another version" after seeing 2021+. Tuning against OOS makes the pass meaningless (the
  Arc-10 gate-lies failure). Loose on IS; sacred on holdout.
- Canonical measurement core CALLED, never reimplemented. Exit/transform tools GEOMETRY-ONLY; the
  engine realizes P&L (take-the-loss).

## SHOW YOUR IDEATION (this run also judges whether deep search generates real ideas)
In the arc doc, make the thinking explicit: the *because*; the conditions you considered and WHY
(including ones you rejected); how the multi-factor structure is meant to isolate the edge; what the
IS development changed and why. One deep arc per session is expected — it may consume your whole
context budget; that is fine. Graceful handoff at the end (finish → push → stop; never block on the
operator; foreground jobs).

## VERDICTS / GUARDRAILS
- Three-way disposition: PASS (all-folds-positive IS+OOS + council) / PORTFOLIO (mean-positive net,
  not all-folds-positive) / KILL. A passer is a CANDIDATE — independent re-verification before any
  deployment, never deployed on the gate's word.
- Docs + `discovery/tools/` direct to main; never auto-merge core (FLAG it). Check TOOL_REGISTRY BUILT
  first before building an experiment tool.
