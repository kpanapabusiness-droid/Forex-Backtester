# arc 1014 — Confirmed-breakdown continuation SHORT (the short mirror of arc 1013)

**Chat:** 1000s | **Range:** 1000–1999 | **Date:** 2026-06-05 | **Disposition: KILL** (cheap-kill at
observation) | **The FIRST short-side arc in the programme.**

## Idea + why (ideation shown, per dispatch — shorts now open)
Shorts merged (PR #273, sweep SAFE; engine short-symmetric, `PerPairSignalState.direction` flows into
the pool bar-walk + Step-5 emission; `observe_long_capture(direction="short")` mirrors entry/SL/label/
drift-sign). The dispatch's lead priority is the short asymmetries the corpus flagged. My range's own
arc 1013 (the strongest directional long in the corpus — deep failed-breakdown RECLAIM, PORTFOLIO 3rd
component) is **negative in the strong-USD trend years 2018/2022/2025** *because* there "breakdowns are
real, not swept." The arc-2006 portfolio spec needs a **3rd/4th component positive in exactly those
risk-off years** (the lone mutually-negative fold blocking the 2-way book is 2015; the longs share 2018).

Hypothesis (the §5a DEEP multi-factor carve-out, *extend the arc-1013 template incl. its short mirror*):
the **short mirror** of arc 1013 — a deep pierce of a K-bar swing low that does **NOT** reclaim
(close_mid stays BELOW the swept level) with down-momentum = a **CONFIRMED breakdown** (sellers in
control, informational move, not a liquidity grab) → continuation DOWN. If real, it would be POSITIVE in
the strong-USD trend years where every long bleeds → the regime-orthogonal portfolio leg.

Corpus corroboration that made this worth a fresh test (not shallow-breakout-short, which is dead by
symmetry): **arc 2007** found the climax (big-range, non-reclaiming) sweep is a falling knife — drift
−0.124 → −0.280 → −0.333 ATR, *worse the bigger* (a real short edge in the long-drift lens, FLAG-1);
**arc 1013** flagged the deep non-reclaiming break as the short leg. This is the multi-factor cell
(depth × swing-low structure × close-below/no-reclaim × down-momentum), not the bare Donchian-low break.

## What happened — FALSIFIED at observation (the data inverted the corpus prior)
H4, 7 USD majors, IS 2010–2020, direction-aware `observe_long_capture(direction="short")` (honest
+1R-before-SL SHORT capture + 24-bar forward drift, drift sign flipped so a FALLING price reads
**positive** → a good short cell = capture >0.50 AND positive drift). GROSS, characterization only.
Swing lookback K=40 (arc-1013 best). Driver: `_disco_work/arc1014_observe_short.py`.

| cell | n | short cap | short drift (falling=+) |
|---|---|---|---|
| BASE (all bars) | 121,467 | 0.4850 | −0.0153 |
| pierced swing-low (any close) | 9,552 | 0.4825 | −0.0835 |
| pierced & RECLAIM (the 1013 long cell) | 5,025 | 0.4764 | −0.1102 |
| **pierced & NO-reclaim (confirmed breakdown)** | **4,500** | **0.4893** | **−0.0460** |

Confirmed-breakdown × **below-depth** (deeper close BELOW the swept level): (0,0.25] cap 0.489 drift
−0.012; **(0.25,0.5] cap 0.490 drift −0.212** (reverts HARD); (0.5,1.0] 0.481 −0.005; (1.0,∞) 0.504
+0.077 (thin n=623, coin-flip cap). **Deeper close-below REVERTS more, not continues** — the opposite of
the continuation hypothesis (only the extreme thin tail faintly continues).

Confirmed-breakdown × **down-momentum (climax proxy)**: non-monotone noise — (−∞,0.5] +0.149,
**(0.5,1.0] −0.243**, (1.0,1.5] −0.011, (1.5,2.0] +0.117, (2.0,∞) −0.026. No monotone climax→continuation
signal (the arc-3007 "non-monotone = not a lever" tell); does NOT reproduce arc 2007's −0.33 in this
clean confirmed-breakdown construction.

**STRUCTURE CONTROL (decisive, the arc-1013 discipline mirrored):** big bearish bar (down_range≥1·ATR)
**AT a swept swing-low** cap 0.4998 / drift +0.0175 vs the SAME big-red bar **ELSEWHERE** cap 0.4963 /
drift +0.0545 — ~identical coin-flips, and elsewhere is faintly *better*. **The swing-low structure is
NOT load-bearing for the short.** (Contrast arc 1013's long, where AT-swept-low 0.55–0.61 vs elsewhere
~0.49 — there the structure WAS the mechanism.)

**Per-pair deep cell (below_depth>0.25):** EURUSD +0.194 / GBPUSD +0.440 / AUDUSD +0.176 positive, but
USDJPY −0.414 / USDCHF −0.292 / USDCAD −0.667 / NZDUSD −0.001 negative — **3/7 positive = the arc-1010
NOISE signature**, not a robust edge.

## Verdict: KILL (cheap-kill at observation)
The confirmed-breakdown continuation short is a coin-flip-or-worse: cap 0.489 < 0.50, headline drift
negative (reverts), the swing-low structure control fails, momentum is non-monotone noise, and the
cross-section is 3/7 = noise. The entry does NOT beat the base and shows no positive gross short drift →
§5f exit-sweep does not bite (reserved for non-coin-flip entries); no pool/engine/council spent (the
arc-1001/1002/3006/1010 cheap-kill discipline). A sub-0.50/reverting gross signal is guaranteed sub-cost
on the honest engine (arc 3003); the lone faint tail cell (>1 ATR below, cap 0.504, drift +0.077) is the
arc-2007 thin-tail artifact, not a structure (the control proves it).

## Diagnosis — why the arc-1013 long has NO tradeable short mirror (the load-bearing asymmetry)
arc 1013's reclaim-LONG wins because **the reclaim is FORWARD-confirming**: the wick already swept the
stops and closed back above, so the i+1 long enters AFTER the adverse low — the move it bets on has not
yet happened (the fix to the capturability wall). The confirmed-breakdown SHORT is the mirror image but
the asymmetry does **NOT** flip: a close BELOW the swept level is **BACKWARD-confirming** — the down-move
it bets on has *already happened in the signal bar*, so the honest i+1 short enters at the local low, and
on liquid majors (where down-moves revert, arcs 3000/3001/3008) it reverts UP against the short. A
forward-confirming reversal structure (reclaim) is tradeable; a backward-confirming continuation
structure (confirmed breakdown) is not — it is shorting realized weakness, the arc-2001/2004/2007
buy/sell-into-the-move stop-tax, now on the short leg. The corpus's −0.33 "falling knife" (arc 2007) was
measured inside a fast-3-bar-drop sweep-RECLAIM construction (a multi-bar plunge = a momentum/breakout
short, closed ground by symmetry), never engine-validated; the clean confirmed-breakdown does not carry it.

## Threads / lessons
1. **The arc-1013 reclaim-long does NOT mirror to a tradeable short.** Forward-confirming reversal
   structure (reclaim) is capturable; backward-confirming continuation structure (confirmed breakdown) is
   a coin-flip that reverts on the honest i+1 short. Shorts do **not** revive directional STRUCTURE — by
   symmetry the structural short base is the same coin-flip as the long (cap 0.485 short ≈ 0.488 long), and
   the swing-low structure that was load-bearing for the long is inert for the short (control proves it).
2. **The regime-orthogonal short the portfolio needs is NOT this.** The hoped-for 2018/22/25 leg
   (positive when reversion bleeds) is not a confirmed-breakdown continuation short.
3. **The live short frontier is the FLOW-EVENT shorts where the corpus measured directional accuracy
   >0.50, not structural-continuation shorts** (which inherit the coin-flip by symmetry). The named
   candidate: the **up-gap weekend SHORT** (arcs 2001/2003: up-gaps drift −0.57 ATR, frac+ 0.64 — the
   gap-fill's STRONGER leg, direction NOT a coin-flip, previously blocked by long-only, now unblocked).
   That is the natural next arc (1015), distinct mechanism family (flow event, not structural continuation).
4. **Mirrored the arc-1013 structure control on the short side and it was decisive** — "big-red AT
   swept-low ≈ big-red elsewhere" killed the structural claim in one line; the control is the right first
   discriminator for any structural short, exactly as it was for the long.

## Tooling
No new BUILT tool. Reused the now-direction-aware `observe_long_capture(direction="short")` (CANONICAL
short plumbing verified live: `PerPairSignalState.direction` → pool bar-walk; the short lens reproduces
the base ≈ 0.485). Conditioning (swing-low pierce / reclaim / depth / momentum) is a one-off observer in
scratch, like arc 1013's. No engine compute spent (cheap-kill at observation).

## FLAGS (code not merged)
None. No canonical-core change. Driver scratch `_disco_work/arc1014_observe_short.py` (reproducible from
this doc). Carries the standing `A1Config.time_exit_bars`-unwired flag (arcs 1005/3004) by inheritance
(not exercised here). Confirms the short path (PR #273) works end-to-end in observation — the programme's
first live short measurement.
