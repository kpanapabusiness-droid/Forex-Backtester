# arc 2009 — Climax-sweep SHORT (down-trend continuation) — the named 4th-component target

- **Chat / range:** 2000s (2000–2999)
- **Arc id:** 2009
- **Date:** 2026-06-05
- **Disposition:** **KILL** (cheap-kill at observation)
- **Council:** none (falsified at observation by three independent cheap discriminators; no
  worthwhile-ceiling fork — arc-1014/1010/3006 precedent)
- **Tooling:** reused BUILT `observe_long_capture(direction="short")` (the PR-#273 short lens);
  conditioning is a one-off scratch observer (arc-1014 convention). No canonical-core change, no new
  BUILT tool. Driver: `_disco2000_work/arc2009_observe_climax_short.py`.

---

## Idea + why (log-seeded — the route's named next target)

After 26 arcs the **portfolio route is the live deployable path and it is BLOCKED on one named fold.**
My own range's arcs 2006 (2-way) and 2008 (3-way) proved it by gridding the full convex simplex
(0/5151 weightings all-folds-positive). The binding fold is **2018** (strong-USD trend year): all
three current PORTFOLIO components (gap-fill 1006, month-end 1011, failed-breakdown 1013) are
*fade/reversion-flavored → implicitly short-trend*, and they tail-correlate in a strong-trend year.
Arc 2008's 4th-component spec: a **trend-continuation** edge positive when trends persist (2018, 2015),
and it named the concrete target — **"arc 2007's climax-sweep SHORT … the short side of the very 2018
breakdowns that beat the long failed-breakdown."**

Arc 2007 (a LONG arc) measured, as a side-observation, that the CLIMAX (big-range, violent) sweep
below a swing low is a FALLING KNIFE — forward drift −0.124 → −0.280 → −0.333 ATR, *worse the bigger
the climax* = the SHORT leg. Arc 1014 then tested the CLEAN confirmed-breakdown short (close-below, no
climax) → KILL (reverts; swing-low not load-bearing) but **explicitly left the climax / fast-violent-
drop variant engine-unvalidated:** "the −0.33 falling knife was inside a fast-3-bar-drop construction;
the clean confirmed-breakdown doesn't carry it."

This arc fills exactly that gap. **Hypothesis (§5a DEEP carve-out, the trend-continuation 4th leg):**
a violent fast-3-bar drop with a big-range bar that pierces a swing low (sweeps stops) and closes
below = a forced/climactic breakdown that **CONTINUES DOWN** → the down-trend-continuation short
positive in 2018. Distinct from arc 1014 (which had no climax/momentum requirement) and distinct from
the shallow momentum short (which is dead by symmetry, LESSONS) only by the **structural conjunction**
(the swing-low sweep) — so the swing-low must be load-bearing for the claim to be more than closed
ground. The decisive cheap test is arc 1014's structure control mirrored onto the climax.

## What happened — FALSIFIED at observation (three independent discriminators)

7 USD majors, H4, IS 2010–2020, short lens (`observe_long_capture(direction="short")`, honest
+1R-before-SL capture + 24-bar forward drift in ATR, short drift > 0 ⇔ price fell ⇔ good for short).
Base: n=121,432, cap **0.4849**, drift **−0.0151** (the long-base mirror reproduces).

**Q1 — the climax-sweep SHORT cell is coin-flip, drift near-zero-to-reverting.** Cell = swept &
close_below & fast-drop (drop3 ≤ −1·ATR) & big-range:
- range≥1.0: cap 0.492, drift **−0.057** (price REVERTS UP, bad for short)
- range≥1.5: cap 0.505, drift **−0.043**
- range≥2.0: cap 0.500, drift **+0.009** (only the thin big-range tail flips faintly positive, coin-flip cap)

The cap lift is ≤ +0.02 (coin-flip) and the short drift is **negative** (reverting) for every robust
cell. **Arc 2007's −0.33 falling knife does NOT reproduce in a clean SHORT construction.**

**Q2 — climax monotonicity FAILS / inverts** (within swept & close_below & drop3 ≤ −1, n=3,915):
small-range drift −0.155 → mid −0.026 → CLIMAX −0.038 — bigger range does **not** continue more. By
pierce depth: shallow<0.5 −0.136, 0.5–1.0 −0.099, **deep>1.0 +0.006** — deeper reverts *less* but only
reaches ~zero (coin-flip cap 0.51). This is arc 1014's "deeper reverts harder, only the thin extreme
tail faintly continues," reproduced on the climax variant. The arc-2007 monotone-toward-continuation
prediction is inverted.

**Q3 — STRUCTURE CONTROL is decisive and ANTI-load-bearing (the load-bearing test).** Same big-range
(≥1.5) fast-drop (≤ −1) CLIMAX bar:
- **AT a swept swing-low (close_below): n=2,069, cap 0.505, drift −0.043 (REVERTS UP)**
- **ELSEWHERE (not swept): n=3,249, cap 0.499, drift +0.128 (CONTINUES DOWN)**

The swing-low sweep is not merely inert for the short — it is the **wrong side**: a violent drop that
sweeps a swing low *reverts*, while a violent drop **not** at a structural low *continues*. Sweeping a
swing low is a **reversal** setup (exactly arc 1013's reclaim-LONG edge, AT-swept-low cap 0.55–0.61),
so the climax-sweep SHORT shorts *into the bounce*. The structure that was load-bearing for the long is
the *reason the short fails*.

**Q4 — per-pair 3/7 positive (the arc-1010/1014 noise signature), split by quote convention.**
EURUSD +0.314 / GBPUSD +0.294 / AUDUSD +0.283 (XXXUSD continue) vs USDCAD −0.507 / USDJPY −0.430 /
USDCHF −0.297 (USDXXX revert), NZDUSD −0.029. The split is **directional USD-beta by quote sign**
(short XXXUSD = ride USD strength in 2010–2020; short USDXXX = fight it), not the structural mechanism.

## Verdict: KILL (cheap-kill at observation)

The climax-sweep SHORT is a coin-flip-or-reverting short whose only "edge" is USD-beta quote-convention
noise. Three independent cheap discriminators agree: cell coin-flip + reverting (Q1), climax
monotonicity inverted (Q2), **structure control anti-load-bearing — the swept low REVERTS while
elsewhere continues (Q3)**, per-pair 3/7 noise (Q4). The named 4th-PORTFOLIO-component target (arc
2008's climax-sweep SHORT) is **dead**. Entry does not beat base and there is no robust positive gross
short drift → §5f exit-sweep does not bite; no pool/engine/council spent.

## Diagnosis — why the sweep has no tradeable short, from BOTH leans now

Arc 1013 proved the swing-low sweep is a **REVERSAL** structure: the deep pierce-and-reclaim LONG
captures 0.55–0.61 (AT-swept-low load-bearing, control-proven). A reversal structure has **no
symmetric short** — both the clean confirmed-breakdown (arc 1014, backward-confirming, enters at the
local low → reverts) and now the climax/violent variant (arc 2009, the swept low is *where the bounce
starts* → shorting it shorts into the reversal) fail at exactly the swept-low structure. Arc 2007's
−0.33 was a within-sweep-reclaim-long sub-bucket artifact (small-n, plausibly a pair/regime cell),
**not a robust short edge** — it does not survive a clean dedicated SHORT construction with a structure
control. The "elsewhere climax continues +0.128" is just **shallow momentum** (a big fast drop, not at
any structural level, continues) = closed ground (momentum/breakout shorts dead by symmetry, LESSONS);
stripping the (anti-load-bearing) structure leaves a dead shallow momentum short → not worth engine
compute, and not the 4th component.

## Threads / lessons

1. **The swing-low sweep is a REVERSAL structure with NO tradeable short mirror — confirmed from both
   leans.** arc 1013 (reclaim long = edge), arc 1014 (clean breakdown short = reverts, structure
   inert), arc 2009 (climax breakdown short = reverts, structure ANTI-load-bearing). The named
   4th-component "climax-sweep SHORT" (arc 2008) is closed. Arc 2007's −0.33 falling knife was an
   artifact of its long-construction sub-bucket, not a robust edge.
2. **Structure control is the one-line decisive discriminator for any structural short** (mirrors arc
   1014). "Climax AT swept-low (−0.043, reverts) vs elsewhere (+0.128, continues)" killed the claim in
   one read — and went further than 1014 (inert): for the climax it is *anti*-load-bearing, the swept
   low is precisely where the short loses. Re-usable: a structural short whose structure helps the
   OPPOSITE direction is dead.
3. **The 2018 binding fold's trend-continuation leg is NOT a structural-continuation short.** Both
   shorts the corpus could name from structure (clean + climax breakdown) inherit the reversal/coin-
   flip. The genuine trend-continuation edge positive in 2018 would have to be long-trend (dead, arcs
   0–3006) or a non-structural short — reinforcing the arc-3004 escalation: the portfolio route is
   blocked on a regime-orthogonal leg the apparatus (and now structural shorts) cannot express.
4. **Per-pair USD-quote-convention split is a short-side noise tell** — when a "short edge" is positive
   on XXXUSD and negative on USDXXX (or vice-versa), it is directional USD beta, not a structural
   mechanism; demand the structure control before believing any cross-section.
5. **Surviving short frontier (unchanged):** the one short candidate where the corpus measured
   directional accuracy >0.50 is the FLOW-event **up-gap weekend SHORT** (arcs 2001/2003: up-gaps drift
   −0.57 ATR, frac+ 0.64 — the gap-fill's stronger leg) — but it is a *fade/reversion* edge (would
   inherit the fade-book 2018 tail, not fix it) and is the 1000s chat's named arc-1015 lane. The
   regime-orthogonal trend-continuation 4th leg remains unfound; FLAG-1 (shorts/second-leg unlock) is
   already exercised, and the deeper unlock the route needs is a genuinely trend-positive construction
   the directional space has not yielded.

## FLAGS (code not merged)

None. No canonical-core change; no new BUILT tool (reused the direction-aware `observe_long_capture`;
conditioning is a one-off scratch observer). Carries the standing `A1Config.time_exit_bars`-unwired
flag only by inheritance (not exercised this arc). Driver `_disco2000_work/arc2009_observe_climax_short.py`
(reproducible: `PYTHONPATH=. py _disco2000_work/arc2009_observe_climax_short.py`).
