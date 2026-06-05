# arc 2010 — Market-neutral relative-value (the escalation's #1 unlock), shorts-enabled — KILL

- **Chat / range:** 2000s (2000–2999)
- **Arc id:** 2010
- **Date:** 2026-06-05
- **Disposition:** **KILL** (cheap-kill at observation)
- **Council:** none (coin-flip + non-monotone-noise observation across 18 cells; no worthwhile-ceiling
  fork — arc-1014/2009/3007 precedent)
- **Tooling:** one-off scratch observer (spread/relative-strength drift; no SignalModule needed for a
  pre-pool screen). No canonical-core change, no new BUILT tool. Driver:
  `_disco2000_work/arc2010_observe_relmom.py`.

---

## Idea + why (the arc-3004 escalation's #1 named unlock, now testable)

The arc-3004 escalation ranked **market-neutral / relative-value** as unlock #1 — "a second simultaneous
leg; the only lever that does NOT require beating 0.50 per trade." Arc 2003 (chat 2000s) tried
relative-value long-only and could only **CONCEDE** — its Devil lens judged the long-only constraint the
binding blocker (couldn't short the leader of a divergent pair). Shorts are now open (PR #273), so the
concession is testable: build a genuinely market-neutral pair trade (long one leg, short the other of a
cointegrated pair) and ask whether it clears the gate.

Arc 2003's own finding gives the *because* for the **side**: relative performance of correlated majors
is NOT mean-reverting — "the laggard keeps lagging" = relative **MOMENTUM**. So the tradeable
market-neutral bet is long the outperformer / short the underperformer of a cointegrated pair, betting
the relative ordering PERSISTS — a documented cross-sectional FX momentum effect (Menkhoff et al.). It
would be **positive when orderings persist = strong-trend years (2018)** → a candidate for the
regime-orthogonal 2018-positive 4th component the portfolio route (arcs 2006/2008) needs.

**The decisive question this observation answers — THE COST REALITY.** A market-neutral pair trade pays
FundedNext cost on **BOTH legs** — ~2× the single-leg ~0.05–0.10R hurdle the entire corpus could not
clear directionally (EDGE<COST). The "doesn't need to beat 0.50" framing is only useful if the relative
drift is large enough to clear the *doubled* cost. So: measure, honestly and ex-ante, whether
relative-strength predicts forward relative-strength, and compare the per-trade relative move to the
2-leg cost — BEFORE building a two-leg book.

## What happened — FALSIFIED at observation (no edge clears the doubled cost)

3 cointegrated/correlated groups (EURUSD-GBPUSD, AUDUSD-NZDUSD, EURUSD-AUDUSD — all quoted XXX/USD so a
common USD factor cancels in the spread), H4 and D1, IS 2010–2020. Relative-strength = `logret_A(LB) −
logret_B(LB)`, z-scored on a trailing window (causal); forward relative move over the next FWD bars;
signed by the momentum bet (sign of z). 2-leg cost shown is **spread-only 1.5×** — the real FundedNext
2-leg cost is HIGHER (adds ~0.5pip slippage + $5/lot RT per leg).

**Relative strength has ≈ZERO predictive power.** `corr(relstr, fwd_rel)` across all 6 group×TF cells:
−0.029 / −0.034 / +0.002 (H4) and −0.048 / +0.006 / +0.003 (D1) — indistinguishable from 0; the
relative ordering predicts neither continuation nor reversion robustly. `frac+` of the momentum bet
hovers at coin-flip (0.43–0.55) at every threshold.

**No cell clears the doubled cost on an honest read:**
- **H4** — every group net-negative at every |z| threshold (−1.6 to −12.5 bp net, spread-only). EUR-GBP
  & AUD-NZD show faint *reversion* (negative momentum, corr −0.03), EUR-AUD faint momentum — both eaten
  by the 2-leg cost.
- **D1** — the only faintly-positive cells: EUR-GBP |z|≥2 net +4.53 bp, EUR-AUD |z|≥1 +1.23 / |z|≥2 +3.28
  — but **NON-MONOTONE in |z|** (EUR-GBP is −14.49 bp net at |z|≥1, +4.53 at |z|≥2; EUR-AUD +1.23 →
  −4.89 → +3.28) = the arc-3007 "**not a lever**" tell (a real lever is monotone; a bulging cell is
  selection noise), THIN (n=153–210, ~14–19/yr), `frac+` ≈ 0.51–0.55 (coin-flip), and net-positive ONLY
  on the spread-only cost that **understates** the true 2-leg FundedNext cost — adding ~4–8 bp of 2-leg
  slippage+commission flips them net-negative.

## Verdict: KILL (cheap-kill at observation)

Market-neutral relative-value on correlated FX majors does not clear the 2-leg cost hurdle at H4 or D1.
The relative move is a coin-flip (corr ≈ 0, frac+ ≈ 0.5) with no robust momentum or reversion; the
handful of marginally-positive cells are thin, non-monotone (noise, not a lever), and net-negative once
the second leg's full cost is counted. The entry does not beat coin-flip and shows no robust +gross
relative drift → §5f exit/engine test does not bite; no pool/engine/council spent (arc-1014/2009/3007
discipline — a non-monotone +gross bucket on understated cost collapses SL-honest, engine-confirmed in
3007).

## Diagnosis — the second leg doubles cost faster than it adds edge

The escalation's hope was that a second leg sidesteps the "beat 0.50 directionally" wall. It does — but
it replaces that wall with a **doubled cost** wall, and on liquid FX majors the relative drift (~2–10 bp
per trade) is *smaller* than the doubled FundedNext cost (~6–14 bp), not 2× larger. The deeper kill is
that relative predictability itself is ≈0 (corr ≈ 0) — there is no robust relative edge to scale,
hedge-ratio refinements aside. This **directly closes arc 2003's CONCEDE**: the binding constraint for
relative-value was NOT (only) the long-only block — with shorts enabled, relative-value STILL fails, now
on the doubled-cost EDGE<COST wall against a coin-flip relative move. Removing direction risk did not buy
an edge; it bought a second spread.

## Threads / lessons

1. **The escalation's #1 unlock (market-neutral/relative-value) is itself EDGE<COST on liquid FX
   majors** — tested honestly now that shorts are open. The second leg doubles cost without doubling
   edge; the relative move is a coin-flip (corr ≈ 0) sub-2×-cost at both H4 and D1. The "doesn't need to
   beat 0.50" framing is true but moot when 2× cost exceeds the relative drift.
2. **Closes arc 2003's CONCEDE with the real reason** — relative-value's blocker was not the long-only
   constraint (now lifted) but the doubled cost vs a coin-flip relative move. A clean correction: the
   FLAG-1 shorts unlock does NOT revive relative-value (it was never the binding constraint there).
3. **Non-monotone-in-|z| net-positive cells are the arc-3007 noise signature** — re-confirmed on a new
   (relative-value) construction; a real lever is monotone in its conditioning extremity. When the
   strongest |z| is the only positive cell while the milder ones are deeply negative, it is
   multiple-comparison selection noise, not an edge. Don't spend engine compute on it.
4. **Cost-favorable-TF discipline applied (arc 3008 lesson, in reverse):** because the 2-leg cost is
   spread-dominated (per round-trip, ~constant in bp across TF) while a longer hold captures a bigger
   relative move, D1 is the cost-favorable resolution — checked it explicitly before the KILL (the
   arc-2002/3008 "confirm at the cost-favorable resolution" rule). D1 did not rescue it.
5. **The 2018-positive 4th component remains unfound, and the relative-value route to it is now closed.**
   The portfolio route (arcs 2006/2008) still needs a regime-orthogonal trend-continuation leg positive
   in 2018; neither structural-continuation shorts (1014/2009) nor market-neutral relative-value (this
   arc) supplies it. Reinforces + narrows the arc-3004 escalation: of its three ranked unlocks,
   relative-value (#1) is now cost-bound on majors and short-side asymmetries (#2) are mapped (the
   up-gap weekend SHORT, 1000s' arc-1015 lane, is a fade that inherits the 2018 tail) — leaving the
   genuinely-tighter-cost execution regime (#3) as the operator lever the apparatus cannot self-supply.

## FLAGS (code not merged)

None. No canonical-core change; no new BUILT tool (a pre-pool spread/relative-drift screen is a one-off
scratch observer, like arc 3005's triangulation / arc 1008's gotobi observers). The two-leg book would
be scored — if ever an edge appeared — as two canonical single-pair legs combined via the BUILT
`combine_fold_roi` (additive; sidesteps any multi-leg apparatus gap), so no engine change is required
for relative-value scoring; the kill is the absence of edge, not an apparatus block. Driver scratch
`_disco2000_work/arc2010_observe_relmom.py` (reproducible from the arc doc).
