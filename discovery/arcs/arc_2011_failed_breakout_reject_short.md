# arc 2011 — Failed-breakout REJECT SHORT at a swing HIGH (the true forward-confirming 1013 mirror) — KILL

- **Chat / range:** 2000s (2000–2999)
- **Arc id:** 2011
- **Date:** 2026-06-05
- **Disposition:** **KILL** (cheap-kill at observation, robustness-confirmed)
- **Council:** none (a +gross-drift entry whose drift is proven a 2-pair thin-tail artifact — robust
  population coin-flip-to-negative; arc-1010/3002/3003 cheap-kill discipline)
- **Tooling:** reused BUILT `observe_long_capture(direction="short")`; conditioning + LOO are one-off
  scratch observers. No canonical-core change, no new BUILT tool. Drivers:
  `_disco2000_work/arc2011_observe_failed_breakout_short.py`, `_disco2000_work/arc2011_loo.py`.

---

## Idea + why (the TRUE forward-confirming mirror nobody tested)

Arc 1013's failed-breakdown RECLAIM long is the strongest, cleanest edge in the corpus *because* the
reclaim is **forward-confirming**: a wick pierces a swing LOW (sweeps sell-stops) and CLOSES BACK ABOVE
it, so the honest i+1 long enters AFTER the adverse low — the up-move it bets on has not started.

The corpus tested two "short mirrors" and both KILLed for the SAME reason — both are
**backward-confirming** (enter into a move already made): arc 1014 (swing-LOW pierce + close BELOW =
confirmed-breakdown *continuation*, enters the local low → reverts) and arc 2009 (climax *continuation*
short, shorts a violent drop already made → reverts). **Neither is the actual mirror of a reclaim.** The
true forward-confirming short mirror is at the swing HIGH:

> a **failed-breakOUT REJECT** — price spikes ABOVE a prior K-bar swing high (sweeps resting BUY-stops /
> triggers breakout-longs), then **REJECTS back BELOW it** within the bar with a large UPPER rejection
> wick (≥ shadow·ATR) = a bull-trap → reversal DOWN. The reject is forward-confirming (the up-move is
> over AT the signal bar; the i+1 short enters AFTER the adverse high, not into it) — the exact property
> that made arc 1013 work, mirrored to the high.

*Why it could be the 2018-positive 4th PORTFOLIO leg:* on a USD major a failed rally that rejects at a
swing high → short → continues DOWN in a strong-USD trend year (2018), where the three fade longs bleed.
A structural reversal short, not a coin-flip continuation (1014/2009) nor a fade. Exact mirror of the
BUILT `FailedBreakdownReclaimLongSignal` (swing-low → swing-high; lower shadow → upper shadow; reclaim →
reject), tested at the same K=40, shadow≥1.25.

## What happened — promising pooled signal, FALSIFIED by robustness (a 2-pair thin-tail artifact)

7 USD majors, H4, IS 2010–2020, short lens (`observe_long_capture(direction="short")`; short drift > 0
⇔ price fell ⇔ good for short). Base cap 0.4849, drift −0.0151.

**The pooled observation looked like a genuine 1013 mirror — the first short to "pass":**
- **Q1** the reject cell (swept_high & reject & upper_shadow≥s) has POSITIVE, shadow-growing drift:
  s≥0.5 +0.171, ≥1.0 +0.242, ≥1.25 +0.261, ≥1.5 +0.463. **Monotone in the rejection wick** (the
  real-lever signature) — but **capture stays coin-flip (0.47–0.51)**, unlike 1013's 0.55–0.61.
- **Q2** shadow monotonicity holds on the drift (small +0.062 → deep>1.5 +0.463), capture flat ~0.49.
- **Q3 structure control APPEARED to pass:** big upper-wick AT a swept swing-high drift **+0.261
  (continues down)** vs the SAME wick ELSEWHERE **−0.213 (reverses up)** — the swing-high looked
  load-bearing in the right direction (the opposite of arc 2009's anti-load-bearing result).

**Robustness destroys it (the decisive cut — arc-1013 LOO + outlier + median discipline):**
- **Per-pair (Q4) is the arc-2009 USD-quote split, carried by 2 thin outliers:** EURUSD +0.41/GBPUSD
  +0.34/AUDUSD **+1.26** (n=25)/USDJPY **+1.49** (n=31) positive; NZDUSD −0.10/USDCAD −0.38/USDCHF −0.87
  negative — 4/7, dominated by two tiny-n pairs with enormous drifts.
- **Excluding BOTH outliers (AUDUSD+USDJPY) collapses the edge:** shadow≥1.0 +0.242 → **+0.085**
  (cap 0.489); shadow≥1.25 +0.261 → **−0.133** (cap 0.45, NEGATIVE). The deeper, "cleaner" cell goes
  *negative* on the robust population — the **inverse** of arc 1013's deepens-the-edge signature.
- **Mean ≫ median:** shadow≥1.0 mean +0.242 vs median +0.022; shadow≥1.25 mean +0.261 vs median
  **−0.069**. A handful of big down-moves (in the 2 thin pairs) drag the mean; the median trade is
  coin-flip-to-negative. **Textbook thin-tail artifact** (arc 1010/3003).
- The single-pair LOO "passed" (all drops > 0) only because dropping one outlier left the OTHER
  inflating the pool — a 2-outlier artifact hides from a 1-out LOO.

## Verdict: KILL (cheap-kill at observation, robustness-confirmed)

The failed-breakout reject SHORT is a **2-pair thin-tail artifact**, not a robust structural edge: the
robust (outlier-excluded) drift is coin-flip-to-negative, the median is ~0-to-negative, capture is
coin-flip, and the deeper-shadow cell INVERTS negative — the opposite of arc 1013's robust signature.
The Q3 structure-control "pass" was a pair-mix confound (the AT-swept-high cell is disproportionately
AUDUSD/USDJPY, the two thin-tail pairs). No engine compute: a coin-flip-capture, median-negative,
outlier-carried drift collapses SL-honest (arc 3003), and there is no robust edge for a §5f exit sweep
to rescue (arc 0/1004 — exits don't rescue a coin-flip entry). The §5f "robust gross drift" trigger is
not met; the robustness analysis (LOO + outlier-exclusion + median) is the airtight, more-rigorous
justification for the observation cheap-kill (arc-1010/3002 precedent).

## Diagnosis — why arc 1013's reclaim-long has NO tradeable short mirror, in ANY construction

Three short constructions of the swing-sweep idea are now tested and dead: confirmed-breakdown
*continuation* (1014, backward-confirming → reverts), climax *continuation* (2009, backward-confirming →
reverts, anti-load-bearing), and failed-breakout *reject reversal* (2011, the true forward-confirming
mirror → a 2-pair thin-tail artifact). The forward-confirming property that made 1013 robust does NOT
transfer to the high: arc 1013's swing-LOW reclaim is robustly load-bearing (LOO-all-positive, capture
0.55–0.61, deeper-grows-the-edge), whereas the swing-HIGH reject's apparent edge is pair-mix/thin-tail
and inverts on the robust population. Empirically, **the swing-low failed-breakdown reclaim is an
asymmetric long-only structural edge** — its short mirror does not exist on liquid USD majors (plausibly
because 2010–2020 USD-major bounces off swept lows are a broader, cleaner population than the few thin
failed-rally rejects, which only "work" where idiosyncratic USD strength continues — the quote-split
tell).

## Threads / lessons

1. **Arc 1013's reclaim-long has NO tradeable short mirror in ANY of the three constructions** —
   confirmed-breakdown continuation (1014), climax continuation (2009), failed-breakout reject reversal
   (2011). The strongest corpus edge is an **asymmetric long-only structural edge**; the swing-sweep
   short well is now exhausted. The forward-confirming property is necessary but not sufficient — it
   must also be robust (not a 2-pair thin-tail), which the high-side reject is not.
2. **A 2-outlier thin-tail artifact hides from a 1-out LOO** — drop-one all-positive can still be
   outlier-carried; the decisive cuts are **exclude-all-outliers, mean-vs-median, and deeper-cell-direction**.
   Re-usable: for a +drift/coin-flip-capture cell, check the median and the all-outlier-excluded pool,
   not just single-pair LOO. (Sharpens arc 1010's "positive drift under sub-0.50 capture = artifact.")
3. **A structure-control pass can be a pair-mix confound** — pooling AT-swept vs elsewhere across pairs
   conflates structure with pair composition when the AT-swept cell is dominated by a couple of
   high-drift pairs. Re-usable: re-run the structure control *within* the robust (outlier-excluded)
   universe, or per-pair, before believing it. (The one gap in the arc-1013/1014/2009 structure-control
   discipline, now patched.)
4. **The USD-quote-convention split (XXXUSD + / USDXXX −) is the short-side directional-USD-beta tell**
   — re-confirmed (arc 2009); a "short edge" that flips sign by quote convention is USD beta, not
   structure. Here it co-occurs with the thin-tail artifact (the + pairs are the thin-n outliers).
5. **The 2018-positive 4th component remains unfound; the structural-short route to it is now fully
   closed** (1014/2009/2011 across continuation AND reversal constructions). With relative-value
   cost-bound (arc 2010) and trend-shorts dead (arc 3010), the portfolio route's 2018 wall stands —
   reinforcing the arc-3004 escalation; the only un-mapped short with measured >0.50 directional
   accuracy is the FLOW-event up-gap weekend SHORT (1000s' arc-1015 lane, a fade that inherits the 2018
   tail). No in-apparatus structural short supplies the trend-continuation leg.

## FLAGS (code not merged)

None. No canonical-core change; no new BUILT tool (the swing-high reject conditioning + LOO are one-off
scratch observers; if a robust short ever appeared it would be registered as the high-side mirror of
`FailedBreakdownReclaimLongSignal`). The first-ever engine short-run was **not** needed (resolved at
observation by robustness), so the canonical short pool/engine path remains observation-verified only
(arc 1014) — a future short that clears observation will be the first to exercise it end-to-end. Drivers
scratch `_disco2000_work/arc2011_observe_failed_breakout_short.py`, `arc2011_loo.py` (reproducible from
the arc doc).
