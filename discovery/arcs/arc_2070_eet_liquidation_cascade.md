# arc 2070 — Prop-firm synchronized-liquidation footprint at the EET daily-DD boundary (the OVERLOOKED run-2 menu item E1)

**Chat 2000s.** Fresh bootstrap; resumed the 2000s range at 2069+1. §5d obs cheap-kill (honest
take-the-loss capture/drift lens; §5f does not bite — sub-0.50, no above-null cell). OOS (2021+) NEVER
touched. BUILT-tool only (`observe_long_capture`); no canonical change; no FLAG; no council; no
engine/null. Components UNCHANGED (all 4 PORTFOLIO). Driver
`_disco2_work/arc_2070_eet_cascade_obs.py`.

## Step (a) — log read / fresh eyes

Pulled main. Read DISCOVERY_PROTOCOL (v1.1 continuous), the full 2000-series ledger + recent Tier-2
(2057-2069 + the sibling 1000s 1057-1064 handoffs), LESSONS ("Closed ground" + the open-frontier
ranking), DISCOVERY_DIRECTION (run-1 + run-2 menus), TOOL_REGISTRY (CANONICAL + all BUILT rows). No
STOP sentinel.

State synthesis (honest-era):
- **4 PORTFOLIO components** (gap 1006, me_long 1011, me_short 1019, fbr 1013); book NEVER
  all-folds-positive, dies at **2015 & 2018** (strong-USD/risk-off "wall", combination-invariant —
  co-sim item E, path-B PROVEN closed arc 3021). §5f exit-honesty collapsed the deploy object to
  me_long-solo / {me_long,fbr}, both vehicle-INFEASIBLE (2053/2055/2059/2060); §11 verification
  COMPLETE; **deployable-system count = 0.**
- The recent 2000s arcs (2066-2069) are all **terminal re-confirmations** declaring the explore-now
  **MENU exhausted** — but each lists only **`M1/O1/L1/Q1/G1/S1`**, i.e. the **run-1**
  `DISCOVERY_DIRECTION` menu. **The run-2 strategist re-run (2026-06-06) ADDED two genuinely-new
  in-charter candidates not in run-1 and not in closed ground — E1 and E2 (§3) — and NEITHER has an
  arc.** A grep of the whole log confirms E1's mechanism (synchronized-liquidation / final-EET-hour /
  auto-flatten / daily-DD-boundary as a *signal source*) was never tested (arc 2033 is the inverse —
  whether the book can *pass* a prop challenge).

Per §2/§5a + the arc-3004 warning that *"the apparatus is incapable" is a seductive, search-ending
conclusion*, the honest move at a "frontier exhausted" claim is to CLOSE the overlooked gap with a real
test, not assert it. So arc 2070 runs **E1** — zero new data, the strategist's "cheapest possible test
of the conservation law."

## Step (b) — the idea (E1, verbatim mechanism)

Tens of thousands of funded-prop accounts share near-identical daily-DD rules (~5%, EOD-EET) and
concentrate in the same instruments (EURUSD/GBPUSD/USDJPY). On a large adverse intraday move, broker
risk engines auto-flatten *en masse at the same EET boundary* → a synchronized forced-liquidation
cascade in the **final EET hour**, then a snap-back when forced flow exhausts → a **next-session
reversion**. Direction-agnostic: a big DOWN day flattens losing longs (overshoot down → revert UP / a
LONG); a big UP day flattens losing shorts (overshoot up → revert DOWN / a SHORT).

**Falsifiable prediction.** On big-|day-move| days the final EET hour shows abnormal range (the cascade
footprint) AND the next session reverts it (capture > base, drift > 0 in the reversion direction),
**stronger than the same big-move reversion entered at a NON-final hour** — the decisive discriminator:
does EET-boundary clustering add edge OVER the generic big-move / stop-sweep reversion the corpus
already killed (1013/3013, closed-ground daily reversion)? Placebo: small-move final-hour days show
nothing.

## Step (c)/(d) — construction + cheap kill (H1, EURUSD/GBPUSD/USDJPY = the MOST prop-concentrated, IS 2010-2020)

EET trading day per H1 bar via canonical `utc_to_eet_trading_day(convention="5ers_eet")`; final-hour
flag = next bar is a new EET day; day-move-so-far = `(close_mid − day_open_mid)/H1-ATR` (causal at the
final-hour close); "big day" = top-20% |day move| = **3.71 H1-ATR**. Reversion measured with the BUILT
`observe_long_capture` (honest +1R-before-SL take-the-loss capture + 24-bar drift), direction = the
reversion side, restricted to the relevant mask. Baselines: H1 long 0.4368 / short 0.4405.

**(1) CASCADE FOOTPRINT — FALSE. The final EET hour is the QUIETEST hour, not a liquidation climax.**
Final-hour range/ATR: EURUSD 0.663 (big-day) vs 0.579 (small-day); GBPUSD 0.647 vs 0.637; USDJPY 0.702
vs 0.682 — **no big-vs-small difference**, and the final hour (~0.66) is *far smaller* than the
non-final hours (**1.03**). The EET daily-DD boundary = EET midnight = 22:00 UTC (winter) / 21:00
(summer) = the NY-close / Asia-open handover, the thinnest hour of the 24h FX day. The hypothesized
synchronized-liquidation footprint does **not exist** in OHLC.

**(2)/(3) REVERSION — DEAD, and the discriminator FAILS.**
- LONG reversion (big DOWN days): final-hour capture **0.3888 ≪ base 0.4368**, drift **−0.2525**
  (big down days *continue* down — no reversion); WORSE than the generic non-final big-move cell
  (0.4187 / −0.1747).
- SHORT reversion (big UP days): final-hour 0.4323 (still < base 0.4405, < 0.50), drift +0.0873;
  only marginally above non-final (0.4134 / −0.1350) but never clears coin-flip.
- **Discriminator verdict:** EET-boundary clustering does NOT add capturable edge over the generic
  (already-dead) big-move reversion — on the long side it is strictly *worse*.

**Placebo** (small-move final-hour): long 0.4153 / short 0.3709 — sub-base, nothing.

**(4) Binding folds:** 2015 pooled cap 0.526 but drift −0.048 ≈ 0 (capture>0.50 with flat drift =
regime-luck within a coin-flip, the arc-3010/1016 tell); 2018 cap 0.412 / drift +0.011 ≈ 0. Not the
+2015/+2018 leg the book needs.

The prop-concentrated-vs-prop-avoided cross-sectional contrast was **not needed**: the footprint is
absent and the reversion dead in the MOST prop-concentrated majors (the strongest case for the
mechanism); it cannot be present in prop-avoided crosses, and the final-vs-non-final discriminator
already settles it.

## Verdict — KILL (obs cheap-kill; premise false + no edge)

E1 dies on two independent sufficient grounds: **(i) the cascade footprint does not exist** — the final
EET hour is the quietest hour, with no big-vs-small-day range difference; **(ii) no reversion edge** —
big-down days continue (capture 0.389, drift −0.25), big-up days barely revert (0.432, sub-0.50), and
EET-boundary timing does not beat the generic big-move reversion. §5f does not bite (sub-0.50, no
above-null cell). No engine/null/council spent.

**Diagnosis / NEW lesson.** Prop-firm daily-DD auto-flattening leaves **no OHLC footprint at the EET
boundary**: either it fires *intraday at the breach* (not clustered at one clock hour) or, more likely,
it is *too small relative to the 24h global FX market* to move price at the boundary hour — which is
itself the thinnest, lowest-participation hour (NY-close/Asia-open). This is the **same closure as arc
2061 (US-holiday "thinness-too-mild")**: a single homogeneous-actor forced flow does not dislocate
globally-traded FX; the only tradeable forced flow is a genuine market-CLOSURE window (weekend gap,
1006) or a mandate-sized fix flow (month-end, 1011). The "funded-prop ecosystem as a forced-flow
source" hypothesis (E1) joins gotobi/round-number/WM-fix/turn-of-month/holiday as a documented
discrete-flow effect that is sub-footprint on liquid FX. And big-move days at the boundary CONTINUE,
re-confirming the closed-ground daily-reversion death (the move IS the trend, 2018-class).

**ADDITIVE record correction (the genuine value of this arc).** The 2066-2069 "MENU exhausted" claim
was scoped to the **run-1** menu (M1/O1/L1/Q1/G1/S1). E1 (run-2) is now CLOSED here with a real test;
the in-charter explore-now menu is genuinely exhausted **except E2** — equity-return-spread-conditioned
month-end USD flow, which requires **external free daily equity-index closes (SPX/NKY/SX5E)** not in
the corpus, so it is **needs-enablement-adjacent (external data fetch), not a zero-data in-charter
candidate**; deferred (one arc = one idea; flag for a future data-enabled arc). With E1 dead and E2
data-gated, the run-2 Frontier-A menu is closed — consistent with the run-2 **conservation law**
(`frequency × per-trade-edge ≈ constant` in a price-only charter): E1 was, as predicted, a thin
rare-event leg that is *also* a coin-flip, never both fundable-frequency and edge-positive.

Components UNCHANGED (all 4 PORTFOLIO); deploy object UNCHANGED; lever = operator path-A / charter
unlock (M macro / O options for the binding +2015/+2018 leg; equity-data for E2). No canonical change,
no FLAG, no new BUILT tool (single-use EET masks in scratch; reused BUILT `observe_long_capture`). OOS
NEVER touched.
