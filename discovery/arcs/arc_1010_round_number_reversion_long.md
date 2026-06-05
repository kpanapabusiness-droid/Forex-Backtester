# Arc 1010 — Round-Number / Order-Cluster Reversion Long (Osler 2003 microstructure)

> **Arc id:** 1010 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (cheap-kill at observation) → KILL.** The documented round-number
> order-clustering reversal (Osler 2003) confers **no robust capturable long edge** on liquid FX at H4.
> The strongest cut — a support-rejection (H4 bar dips into the big-figure "00" level from above, close
> holds) — lifts honest +1R-before-SL capture only **+0.5pp (0.4883 vs base 0.4831), drift ~0 (+0.005 ATR)**,
> with the per-pair lift scattered **3/9 pairs >0.50 = noise-consistent, NOT cross-pair robust.** A finer
> **H1** confirm (arc-2002/1008 discipline) **evaporates** the lift entirely (touched 0.4746 *below* base
> 0.4755; EURUSD 0.5155→0.4834, USDJPY 0.5003→0.4638). Below the ~0.05–0.10 ATR cost hurdle at every
> resolution → no engine compute warranted.
> **Lever tested:** MECHANISM CHANGE — execution/microstructure ORDER-FLOW at structural price LEVELS
> (frontier item 3, available pre-shorts); a genuinely novel structural mechanism with a documented
> *because*, decorrelated from the weekend gap-fill (fires intraweek on any pair).

Scored only descriptively (gross honest +1R-before-SL capture + forward-24-bar drift in ATR, via the BUILT
`observe_long_capture`) — falsified at observation before pool/engine, exactly as arcs 1001 / 2000 / 2003 /
3005 / 1008. No council (no worthwhile-ceiling fork). No null baseline (no edge to compare). The honest
engine (costs + SL-first) can only make a no-edge gross signal worse (arc 3003), so a sub-0.50 / ~0-drift /
non-robust signal is guaranteed sub-cost — no engine compute spent.

## (a) Log read — FRESH EYES (honest-era only)

Pulled main. Honest-era corpus = 20 arcs / 3 chats (0, 1000–1009, 2000–2004, 3000–3005). **Closed
comprehensively:** single-condition shallow directional prediction (momentum / breakout / mean-reversion /
trend, long OR short) across H1/H4/D1, majors + crosses (28 pairs), under both the +1R-capture and the
forward-drift lens, every exit/SL, and stop-removed (arcs 3004 momentum + 2004 reversion — the 2·ATR stop is
a real ~4–5pp drag / ~halves DD but is NOT the wall); regime conditioning failed 3 ways (dispersion / vol /
Kaufman-ER, strong trends INVERT); volume = magnitude not direction (3002); turn-of-month USD drift real but
sub-cost (1005); gotobi absent/arbitraged (1008); triangulation residual ≈ 0 (3005); relative-value catch-up
falsified (2003). The binding wall is **EDGE < COST** for directional bets; the arc-3004 escalation says
durable FX edges need a structural unlock (relative-value / shorts) the operator hasn't flipped to yet.

**The one positive lead:** weekend gap-DOWN-fill long on JPY crosses (arc 1006; audited + thinned in 1009) —
mean-positive IS, beats a fair same-exit null by ~+0.36pp (gap-specific excess, ~half the +0.69% headline),
low DD, fold-fragile, threshold-fragile (lives at 0.5 ATR) → **PORTFOLIO** disposition. It is an overshoot,
not a fill-to-target (1007); majors negative (2001); TF-invariant (2002). The mechanism that WORKED is a
**discrete-liquidity / flow EVENT reversion**, not continuous prediction.

**My lane (dispatch, pre-shorts):** hunt a **2nd net-positive decorrelated long-only component** for the
PORTFOLIO route OR a novel structural mechanism with a *because*. Shorts (PR #273) are GATED canonical-core,
NOT merged → I stay long-only, no short arc. Refining the gap-fill again yields a *correlated* component
(mapped 5 ways) → low value. I need a **decorrelated** mechanism. STOP absent.

## (b) Idea + observation — round-number order-cluster reversion (documented *because*)

The validated-positive template is an event-driven flow/liquidity reversion. A genuinely novel, untested,
decorrelated long-only mechanism with a strong documented *because* is **round-number / order-cluster
reversion** (Osler 2003, *"Currency Orders and Exchange-Rate Dynamics,"* J. Finance): take-profit orders
cluster at big-figure "00" levels → predictable price **reversals** at those levels (shorts taking profit =
buying pressure); stop-loss orders cluster just beyond → trends through. This is an **order-flow microstructure**
mechanism — structurally distinct from the indicator-reversion family already mapped (RSI/z/Bollinger, arcs
3000/3001), it is frontier item 3 (execution/microstructure), and it is decorrelated from the weekend gap
(fires intraweek on any pair) → a candidate 2nd PORTFOLIO component.

Long-tradeable side = price dips **into a big-figure support from above** and the take-profit cluster bounces
it up. Observe-first (§5b, gotobi/volume discipline): condition the per-bar honest long capture on
round-number geometry and check for a robust >0.50 lift before believing.

**Round-number grid.** Big-figure ("00") spacing = 100 pips: **0.0100** for 5-decimal pairs, **1.00** for JPY
pairs. Universe = 7 USD majors + 2 JPY crosses (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD,
EURJPY, GBPJPY). Window IS 2010–2020 (OOS never touched). Driver: `_disco_work/arc1010_observe_roundnumber.py`.

## (c)/(d) Observation = cheap-kill (no pool built — falsified at obs)

`observe_long_capture` (honest +1R-before-SL, gross, sl=2·ATR, hold=120, drift=24 bars), 156,130 IS bars.
**Base capture 0.4831, base drift −0.0399.**

**A. Proximity of close to nearest round (ATR buckets).** NO structure — bars nearest a round (prox 0–0.1)
capture **0.4765**, *below* base. No raw round-number effect.

| prox to nearest round (ATR) | n | capture | drift |
|---|---|---|---|
| 0.0–0.1 | 11,144 | 0.4765 | −0.0140 |
| 0.1–0.25 | 17,791 | 0.4861 | −0.0105 |
| 0.25–0.5 | 29,759 | 0.4781 | −0.0599 |
| 0.5–1.0 | 51,468 | 0.4852 | −0.0283 |
| >1.0 | 45,793 | 0.4843 | −0.0568 |

**B. Support-rejection** (low touches round-below from above within 0.1·ATR, close holds above): a tiny lift.

| touched_support | n | capture | drift |
|---|---|---|---|
| False | 121,951 | 0.4817 | −0.0525 |
| **True** | 34,179 | **0.4883** | **+0.0048** |

+0.5pp capture over its own base, drift only to ~zero — sub-0.50, far below the ~0.05–0.10 ATR cost hurdle.

**C. dist_low_to_round** (how deep the bar's low sits vs the support): best bucket (low just above support)
**0.4948** — still sub-0.50; the pierce-and-recover "spring" bucket (low < round, −inf,0] = **0.4873**, NOT
better → the sharper Osler stop-run-and-reverse version is *also* dry.

**D. Per-pair support-rejection (the tell — NOT cross-pair robust):** EURUSD 0.5155, USDJPY 0.5003 (drift
+0.248), GBPUSD 0.5013, NZDUSD 0.4961 lift; AUDUSD 0.4822 (drift −0.227), USDCAD **0.4464**, USDCHF 0.4884,
EURJPY 0.4826, GBPJPY 0.4819 do NOT. **3/9 pairs >0.50 = consistent with noise** across 9 independent draws.

### Finer-resolution confirm (H1) — arc-2002/1008 discipline

Osler's effect is INTRADAY; if a real bounce completes inside the H4 bar (honest next-bar entry misses it), a
finer H1 entry would reveal it. Checked the 3 best-looking H4 pairs (EURUSD/USDJPY/GBPUSD) at H1
(`_disco_work/arc1010_h1_confirm.py`), 204,807 IS bars, base capture 0.4754:

| H1 support-rejection | n | capture | drift |
|---|---|---|---|
| False | 183,629 | 0.4755 | −0.0772 |
| True | 21,178 | **0.4746** | +0.0400 |

The aggregate lift **vanishes** (touched 0.4746 *below* the no-touch 0.4755). Per pair: EURUSD **0.4834**
(H4's 0.5155 collapses), GBPUSD 0.4750, USDJPY **0.4638** (H4's 0.5003 collapses; +0.118 drift sits under
sub-0.47 capture = thin-tail artifact, not a +1R-before-SL edge). The H4 "3/9 >0.50" does NOT replicate →
confirmed noise.

## (g)–(i) Verdict + threads

**Verdict: FAIL (cheap-kill at observation) → KILL.** Round-number order-cluster reversion produces no robust
capturable long edge on liquid FX: best H4 cut sub-0.50 (support-rejection 0.4883, +0.5pp, drift ~0), the
per-pair lift is noise (3/9 >0.50), and the marginal H4 lift evaporates at H1. One of the most-published FX
microstructure effects is fully arbitraged at H4/H1 in 2010–2020 — the same dissolution as gotobi (arc 1008)
and triangulation (arc 3005). The honest engine (costs + SL-first) can only make a sub-0.50 / ~0-drift gross
signal worse (arc 3003) → no pool / triage / council / null warranted.

**Threads / lessons.**
1. **Round-number / big-figure order-cluster reversal (Osler) is absent/arbitraged at H4 & H1, 2010–2020.**
   A famous documented microstructure *because* (the orders genuinely cluster) does NOT imply a still-capturable
   H4/H1 edge — the second documented-microstructure mechanism (after gotobi 1008) to dissolve on honest data.
   The "structural price LEVEL as a setup" sub-lane of frontier item 3 is now tested and dry for round numbers.
2. **A scattered 3/9-pairs-above-0.50 cross-section is the noise signature, not an edge** — and a finer-resolution
   re-test is the cheap discriminator (arc 2002/1008): the H4 lift did not survive to H1, confirming it was noise.
   Re-usable discipline: when a marginal capture lift is carried by a minority of pairs, confirm at a finer TF
   before believing.
3. **Positive forward drift under sub-0.50 capture = thin-tail artifact, not edge** (USDJPY +0.118/+0.248 drift
   with 0.464–0.500 capture) — re-confirms arc 3003 (raw drift overstates SL-honest expectancy; take-the-loss
   kills slow/tail-driven drifts). The drift lens alone would mislead here; the capture lens + cross-pair
   robustness + finer-TF confirm together make the KILL airtight.
4. **21 arcs, still ONE (thin) net-positive long-only component** (weekend gap-fill, PORTFOLIO). The novel
   long-only structural-mechanism well continues to come up dry (gotobi absent, round-numbers absent,
   triangulation ≈0); the arc-3004 escalation + FLAG-1 shorts/second-leg unlock remains the highest-leverage
   operator move. **Surviving pre-shorts untested sub-lanes:** multi-timeframe STRUCTURE as a setup (not a regime
   filter), and other discrete liquidity/flow EVENTS (the template that produced the one win) beyond the
   weekend gap — e.g. month-end fixing-flow *dislocation-and-reversion* (distinct from arc 1005's turn-of-month
   *drift*).

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (round-number grid is a one-off
conditioning helper kept in scratch, like arc 3005's triangulation observer and arc 1008's gotobi tagger).
Reused BUILT `observe_long_capture`. Drivers scratch `_disco_work/arc1010_observe_roundnumber.py`,
`arc1010_h1_confirm.py` (reproducible from this doc).
