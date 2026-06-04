# Discovery Council Transcript — Arc 3004 (generative idea-fork)

> Invoked via `/llm-council-discovery`, LIGHT/generative weight. Juncture: genuine stuck-point after the
> 3000s chat closed the price/volume/structure/regime directional space (arcs 3000–3003). 5 isolated lenses →
> 3 anonymous peer reviews → chairman. CC committed to the chairman's measure-then-decide.

## Framed question (abridged)
After 13 arcs (3 chats) closed the directional FX long space under FundedNext costs, what genuinely-different
untested edge exists within the fixed apparatus (long-only, single-instrument, per-trade SL/TP, price+volume,
H4, 28 pairs; judge all-folds-positive), OR should we escalate the apparatus is incapable? Plus: pressure-test
the apparatus-is-binding meta-hypothesis; reframe "portfolio of sub-cost edges" without a net-positive component.

## Lens analyses (de-anonymized)
- **Mechanism:** "Directional FX has no edge" is mechanistically TRUE — the apparatus forbids the direction
  drivers (rates/CB-flow/positioning); price residual = coin-flip; spread prices in rebalancing latency. One
  untested in-apparatus idea: cross-rate triangulation lag (synthetic mid from the two USD legs vs quoted
  cross), low conviction / likely M1 / may be < cross spread. Meta: ESCALATE — min unlock = a second
  simultaneous leg. Portfolio: summing sub-cost directional legs fails (shared USD factor → correlated;
  N×neg=more neg); only honest version is COST-MUTUALISING (net flow internally, pay spread once per capital
  round-trip) — needs multi-leg.
- **Alternative framing:** The UNIT OF ANALYSIS is wrong — testing the ENTRY (coin-flip) when the
  EXIT/holding-period creates/destroys R; SL-first +0.019→−0.055 proves it. Reframe direction→PATH (asymmetric
  exit monetizes realized-vol convexity; payoff = range traversed). Volume "magnitude not direction" = the
  missing TARGET VARIABLE. Also "all-folds-positive PER-COMPONENT" is a self-imposed constraint forbidding the
  portfolio idea — it only lives if the judge becomes PORTFOLIO P&L.
- **Refinement:** Single highest-leverage UNTESTED change: REMOVE the per-trade hard SL, exit on TIME only.
  All 13 arcs used 2·ATR SL and paid the take-the-loss tax converting +gross drift → −honest expectancy. Test
  FIRST: re-run the arc-3001 post-up-spike pool with SL disabled/wide + time-exit, costs on. Flip toward +16.7%
  → stop was the wall; still bleeds → take-the-loss exonerated, space genuinely closed.
- **Steelman/Devil:** BULL — conditional path-asymmetry via exit (no regime forecast); the drift scan measured
  UNCONDITIONAL drift, never conditional-on-survival convexity. BEAR — that recycles convexity-harvest +
  exit-engineering (both closed); deeper kill is informational: long-only single-instrument = ONE bit; FX edges
  are relative; EDGE<COST structural. Verdict: BEAR decisively → escalate (specify min relaxation: second leg).
- **Soundness:** BOTH forks are comfortable lies. "New edge" decays to a coin-flip under long-only+SL.
  "Escalate" is the MORE seductive lie (ends the search). 13 arcs ≈ 2 mechanisms × few levers; two chats share
  priors (not independent). Unaudited escape hatch: the +0.019→−0.055 collapse is from the 2·ATR SL STOP
  GEOMETRY — EVERY arc used SL=2·ATR. SINGLE CHECK before escalating: re-score a real positive drift with NO
  tight SL / time-or-target exit — isolate EDGE vs STOP.

## Peer reviews (anonymous; key points)
- **R1:** Strongest = Soundness (names the meta-trap; demands cheap falsification first). Biggest blind spot =
  Steelman/Devil-BEAR (rules EDGE<COST structural on numbers measured *through* the SL confound). ALL missed:
  the COST is a wall INDEPENDENT of the stop — a 12-bar time-exit still pays full per-round-turn spread; lift
  the stop and cost confounds TOGETHER and read both numbers.
- **R2 (repo-read):** Strongest = Mechanism (correct + consistent with the record; second-leg escalation is the
  one unclosed path; cost-mutualising portfolio). Biggest blind spot = Refinement/Soundness: the time-exit is
  LARGELY already tested (arc 3001 ran time_exit_bars=120 with a live stop); only FULLY disabling the SL is
  unrun; arc 3003's +0.019→−0.055 shows slow drift doesn't clear the stop → "stop is the wall" likely already
  false. ALL missed: whether the same edges clear under a DIFFERENT (non-FundedNext) cost regime.
- **R3:** Strongest = Refinement/Soundness (isolate the single confound tainting all 13 arcs: SL=2·ATR; the
  experiment can RETIRE the escalation question). Biggest blind spot = Steelman/Devil (treats stop-confounded
  numbers as clean evidence). ALL missed: a SL-disabled long must STILL be scored by MultiPairBacktester and
  may relocate loss into tail DD; the real question is whether ANY exit both harvests drift AND is deployable.

## Chairman verdict
**Recommendation: investigate-X-first, do NOT escalate now.** Re-score the strongest +gross-drift pool under a
wide/disabled SL + time-exit (non-partial-runner policy, so the CI-locked invariant is untouched), costs ON;
read TWO numbers — (1) does worst-fold flip positive with the stop removed (STOP), (2) does net expectancy clear
zero after per-round-turn cost (COST). Pre-committed rule: stop-removed AND cost-cleared → stop was the wall,
pivot to wide-stop/time exits; stop-removed but still sub-cost → EDGE<COST structural → escalate with a CLEAN
proof + name the minimal unlock (second leg / non-FundedNext cost regime). Agreed: directional FX long has no
in-apparatus edge as tested; the drift scan is genuine and sub-cost; portfolio-of-directional-legs is blocked
in CC's power. Clashed: is the stop-free test already run (R2) vs genuinely untested (crux: is the HARD SL fully
disabled? — it was not in any prior run); is EDGE<COST structural or a stop artifact (crux: net expectancy
after removing the stop). Strongest dissent: R2/BEAR "escalate now, the edge is sub-cost before the stop." 
Confidence flag: shallow on the one decisive fact (no run ever fully disabled the SL) → measure then decide.

## CC commitment + outcome
CC committed to measure-then-decide. Ran the clean A/B (post-up-spike entry; baseline 2·ATR partial/runner vs
pure time-exit via the BUILT `make_time_exit_predicate`, wide 6·ATR sizing-only SL), full IS WFO, costs ON.
**Result:** removing the stop lifts mean fold ROI from −4.52% to ~0% (best N=48 +0.62%) — the take-the-loss tax
was a real ~4–5pp drag — but **no horizon is all-folds-positive** (best 5/10 neg, worst −20.4%; chop years stay
negative). → The chairman's "stop-removed but still sub-cost" branch. **Escalation EARNED with a clean,
unconfounded proof.** See `discovery/ESCALATION_apparatus_capability.md`.
