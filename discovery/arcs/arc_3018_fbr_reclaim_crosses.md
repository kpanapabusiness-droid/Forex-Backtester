# arc 3018 — Failed-breakdown-reclaim LONG on CROSSES (2018-neutral decorrelated 5th-leg candidate)

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> The 4-way book (arc 1020) is ~0.11% from deployable, blocked only by MARGINAL **2015 (−0.08)** &
> **2016 (−0.12)**. Hypothesis: apply the corpus's strongest structural edge — the failed-breakdown
> RECLAIM (arc 1013, USD majors) — to a DIFFERENT universe (crosses) to get a decorrelated, plausibly
> 2018-NEUTRAL component that lifts 2015/2016. **Falsified at observation: the fbr structure does NOT
> transfer to crosses — the swept-low reclaim captures 0.42–0.46 (BELOW coin-flip) and the structure
> control INVERTS (reclaim WORSE than a generic deep down-wick), the opposite of arc 1013; and the
> target-year sign is wrong (2016/2018 negative; 2015 a thin-tail artifact). The fbr edge is
> USD-major-SPECIFIC.**

---

## Log reading (step a — FRESH EYES, honest-era only; pulled main, no STOP)

Fresh chat, resumed 3000s at the range floor +1 (highest 3000s id was 3017 → I open 3018). Read the full
Tier-1 ledger + recent Tier-2 + LESSONS + TOOL_REGISTRY. The corpus is deeply mapped (~55 arcs). State
synthesis that matters for the frontier:

- **Four net-positive PORTFOLIO components exist:** gap-fill 1006 (JPY-cross H4), month-end-long `me`
  1011 (USD-major D1), failed-breakdown-reclaim `fbr` 1013 (USD-major H4, the STRONGEST: IS +1.85%
  9/10), and **month-end-SHORT `me_short` 1019/3017 (USD-major D1) — the FIRST robustly 2018-positive
  component.**
- **The portfolio route moved decisively since arc 3016's "in-apparatus exhausted / 2018 unfound"
  pessimism:** `me_short` (found AFTER 3016) supplied the 2018 leg. arc 1020's **4-way combination WFO
  BREACHES the 2018 wall** (2018 was the 3-way binding fold that killed ~12 routes; now +0.08 at the
  book optimum), worst-fold improved **7× (−0.77% → −0.115%)** — the strongest book in the corpus — but
  **STILL not all-folds-positive**, now blocked by **2015 (−0.08) & 2016 (−0.12), both MARGINAL.**
- **The precise live frontier (arc 1020 spec):** a 5th component **positive in 2015 AND 2016 without
  dragging 2018**. arc 1020's reasoning: 2018 needs heavy `me_short` (which drags 2015/2016); 2015/2016
  are carried by gap+`fbr` but those get down-weighted because 2018 rejects them (`fbr` is −2018). So the
  optimizer can't satisfy 2015/2016 and 2018 at once.
- **Closed/dead for the 2018 leg (now solved) AND the search texture:** shorts (structure 1014/2009/
  2011/3011, trend 3010, up-gap flow 1016/2013, vol 3012, rel-value 2010, carry-unwind 1017), all TF
  incl W1 (3014), intraday session structure (3016) — directional/flow/calendar/structure space
  comprehensively mapped. The recurring lesson: edges are **universe-SPECIFIC** (gap-fill is
  JPY-cross-specific, arc 1018; `fbr` level is the structural swing pivot, arc 3013).

## Idea + why (a documented *because*, with a documented spec-override)

The only thing **tradeable** in 2015/2016 is washout-bounces — reversion off event spikes (SNB/China-
deval 2015, oil/Brexit 2016). That is **exactly why `fbr`-USD-majors is +2015/+2016** (it longs the
failed-breakdown reclaim — a bounce off a swept low). But `fbr`-USD-majors is **−2018** (in 2018's
sustained USD trend the failed breakdown becomes a REAL breakdown, the reclaim doesn't hold — arc 2014
proved this is mechanism-intrinsic and un-conditionable), so the optimizer can't weight it up enough to
fix 2015/2016 without re-breaking 2018.

**Hypothesis:** run the SAME `fbr` reclaim mechanism on a DIFFERENT universe (crosses). A cross-`fbr`
would be (a) **decorrelated** from the USD-major `fbr` (different pairs), (b) plausibly **+2015/+2016**
(crosses had violent event washouts that bounced), and crucially (c) potentially **NEUTRAL in 2018**
(crosses were OUT of the USD trend that makes the USD-major `fbr` −2018) — satisfying the REAL 5th-leg
constraint (positive 2015+2016 *without dragging 2018*).

**Documented spec-override (CC commits, §2/§8):** arc 1020's spec says "NON-reversion (the gap/month-end
family is 2016-saturated)." I tested a reversion-family idea anyway, with a reason: mechanistically the
ONLY edge in 2015/2016 IS reversion (washout-bounce), so a strict non-reversion 2015/2016 leg is likely
unachievable. The genuinely NEW lever here is not non-reversion but **2018-NEUTRALITY via a different
universe** (the existing `fbr` is 2018-NEGATIVE; a 2018-neutral cross-`fbr` is non-redundant). Falsifiable
acceptance test, observation-first.

**Acceptance test (pre-registered):** the `fbr` reclaim must (1) transfer to crosses — honest
+1R-before-SL capture > 0.50 AND the swept-low STRUCTURE control load-bearing (reclaim ≫ generic deep
down-wick, as on USD majors); AND (2) show the right target sign — **+2015 AND +2016, not −2018.** A
real falsification risk is flagged up front: arc 1018 found cross GAPS are efficient/random-walk; the
open question is whether the `fbr` STRUCTURE (a stop-run reversal, a different mechanism) transfers.

## Method (CALLED canonical; observation only — cheap-kill)

Reused the EXISTING BUILT tool `FailedBreakdownReclaimLongSignal` geometry (K=40 swing-low, lower
shadow ≥1.25 ATR) — no new tool. Driver `_disco3_work/arc3018_observe_fbr_crosses.py`:
`Panel.from_pairs` (H4, real bid/ask, EET, recovered 65 GB backup) → compute per-pair `fbr` fire mask +
a **structure-control mask** (same deep down-wick ≥1.25 ATR but NOT a swept-low reclaim) → canonical
`observe_long_capture` (honest `reached_1r_before_sl` take-the-loss capture + 24-bar forward drift)
restricted to each mask → group by pair and by year (IS 2010–2020). Two universes tested for a fair
family kill:

- **JPY crosses** (cached; high-vol, risk-driven): AUDJPY, CADJPY, CHFJPY, EURJPY, GBPJPY.
- **non-JPY crosses** (warmed from M1; cleaner decorrelation, less carry-confounded): GBPAUD, EURAUD,
  GBPNZD, EURCHF, AUDNZD.

## What happened — FALSIFIED at observation (both universes)

**STRUCTURE CONTROL INVERTS on BOTH universes — the opposite of arc 1013 (USD majors).** On USD majors
the swept-low reclaim captures 0.55–0.61 and the structure is load-bearing (same wick ELSEWHERE = coin-
flip). On crosses the reclaim is WORSE than a generic deep down-wick, and both are sub-0.50:

| universe (IS 2010–2020) | fbr reclaim AT swept low | generic deep down-wick |
|---|---|---|
| JPY crosses | n=177 **cap 0.4633** drift_med −0.232 | n=410 cap 0.5463 drift_med +0.343 |
| non-JPY crosses | n=153 **cap 0.4248** drift_med −0.540 | n=329 cap 0.4620 drift_med +0.016 |

The swing-low pierce/reclaim REMOVES edge on crosses (reclaim cap < generic-wick cap, both < 0.50) —
structure ANTI-load-bearing. (The generic-wick 0.55 on JPY crosses is just generic reversion, dead per
arcs 3000/3001, not structure-controlled.)

**The target-year sign is WRONG** (per-year IS `fbr` fires):

| year | JPY crosses (cap / drift_mean / drift_med) | non-JPY crosses (cap / mean / med) |
|---|---|---|
| **2015** | 0.333 / **−0.997** / −0.180 | 0.444 / +1.247 / **−0.295** (thin-tail, n=9) |
| **2016** | 0.391 / +0.067 / **−0.708** | 0.364 / **−0.952** / −0.622 |
| **2018** | 0.313 / **−0.455** / −0.297 | 0.333 / −0.154 / +0.150 |

Both target years are negative-or-thin: 2016 robustly negative on non-JPY (drift −0.95); 2015 is either
strongly negative (JPY −0.997) or a **thin-tail artifact** (non-JPY mean +1.25 but median −0.295, n=9 —
the arc-2011 tell). 2018 negative on both. Per-pair: mostly sub-0.50 capture (the lone >0.50, EURAUD
0.571, has median-negative drift). No robust edge anywhere.

## Diagnosis — the `fbr` edge is USD-major-SPECIFIC; cross swing-lows aren't defended pivots

On USD majors (the most liquid pairs, densest resting-stop clustering at visible swing lows) a swept
swing low is a genuine **reversal pivot** — institutional support, the reclaim confirms the stop-run is
over (arc 1013 control-proven; arc 3013 showed the STRUCTURAL swing pivot beats time-based liquidity
pools). On crosses (thinner liquidity, more momentum/carry-driven) swing lows are swept **routinely
without being defended supports** → the reclaim doesn't confirm a reversal, it enters mid-cascade and
catches a falling knife (capture sub-0.50, median-negative drift). On JPY crosses the carry/risk
mechanism makes this worst in risk-off (2015 China-deval, 2018-Q4) — JPY-cross washouts CONTINUE down
(carry unwind, arc 1017), so the reclaim-long is a knife exactly in the target years. This generalizes
arc 1018 (cross gaps efficient) from FLOW to STRUCTURE: **the corpus's structural edges, like its flow
edges, are universe-specific and do NOT transfer off USD majors.**

§5f does not bite: no non-coin-flip entry exists (capture sub-0.50, structure anti-load-bearing, wrong-
sign target years), so no exit/SL sweep can manufacture an edge. No pool/engine/null/council spent
(arc-3010/3012/3016 efficiency discipline).

## Verdict: KILL (cheap-kill at observation)

The `fbr` reclaim does NOT transfer to crosses (JPY or non-JPY): structure control inverts (reclaim cap
0.42–0.46 < generic-wick cap, both < 0.50), and the 2015/2016/2018 sign is wrong (2016/2018 negative;
2015 thin-tail). The 2018-neutral decorrelated-`fbr` 5th-leg route is closed. USD-major `fbr` (arc 1013)
UNCHANGED (still PORTFOLIO); the 4-way book's 2015/2016 blockers stand unfixed.

## Threads / lessons

1. **The `fbr` structural edge is USD-major-SPECIFIC — it does NOT transfer to crosses.** The swept-low
   reclaim's load-bearing structure (arc 1013: reclaim 0.55–0.61 ≫ generic-wick coin-flip on USD
   majors) **INVERTS** on crosses (reclaim 0.42–0.46 < generic-wick, both sub-0.50). Cross swing-lows
   are swept routinely without being defended pivots (thinner liquidity / momentum-driven), so the
   reclaim enters mid-cascade. This generalizes arc 1018 (cross GAPS efficient) from FLOW to STRUCTURE:
   **the corpus's edges are universe-specific; don't expect a USD-major structural edge to port to
   crosses.** Re-confirms arc 3013 (the structural swing pivot is load-bearing; weaker/looser levels
   dilute).
2. **The 2015/2016 leg is NOT supplied by a cross-universe reversion.** 2015/2016's only tradeable edge
   is washout-bounce reversion, but applying it to crosses gives the WRONG sign in the target years
   (JPY-cross knives in risk-off 2015; non-JPY 2016 drift −0.95). The arc-1020 "NON-reversion 2015/2016
   leg" spec is mechanistically hard precisely because 2015/2016's edge IS reversion — and the
   reversion that works (USD-major `fbr`) is the one that's −2018. This is a genuine structural bind:
   the 2015/2016 lift and the 2018 lift want opposite universes/directions.
3. **5th-leg search status (unchanged, narrowed):** the 4-way book is ~0.11% from deployable, blocked by
   marginal 2015/2016; the decorrelated-`fbr`-on-crosses lead is now dead. The remaining candidate
   spaces for a +2015/+2016/non-(−2018) component: (a) a `me_short` SIBLING that is robustly +2015/2016
   (1019 found me_short's 2015 fragile/GBPUSD-leaning — is there a more robust acute-event-driven short
   for the SNB/EUR-collapse 2015 + oil/Brexit 2016?); (b) operator-gated levers (tighter-cost execution
   regime — arc-3004 escalation #3 — or a non-OHLCV data source). The honest read (arc 3016 thread 4,
   reaffirmed): the in-apparatus FX-major OHLCV well is nearly dry for this specific bi-fold spec.

## Tooling

No new BUILT tool — reused `FailedBreakdownReclaimLongSignal` (arc 1013) geometry + canonical
`observe_long_capture` + a one-off structure-control observer (`_disco3_work/arc3018_observe_fbr_crosses.py`,
reproducible). Same observation pattern as arcs 3012/3013/3014/3015/3016. Not promoted (one-off
cross-universe conditioner).

## FLAGS (code not merged)

None. No canonical-core change. Carries the standing flags (the `risk_pct` PERCENT-vs-FRACTION unit
split, arc 3017 FLAG-1; `A1Config.time_exit_bars` unwired, arc 1005). Driver reproducible from this doc.
