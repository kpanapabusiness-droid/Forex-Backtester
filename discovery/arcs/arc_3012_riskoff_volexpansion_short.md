# arc 3012 — Risk-off vol-EXPANSION SHORT on high-risk-beta majors (AUD/NZD)

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> The candidate 2015 **&** 2018-positive 4th PORTFOLIO leg, attacked from the one un-tested
> conditioner: a realized-volatility-EXPANSION (risk-off) trigger on the highest risk-beta
> majors. Falsified at observation — and falsified on the binding fold (2018 robustly negative).

---

## Log reading (step a — FRESH EYES, honest-era only)

Resumed 3000s range at arc 3012 (highest in-range = 3011). Pulled main; no `discovery/STOP`.
State of the programme after 37 arcs:

- **Directional space comprehensively closed** (long OR short; momentum/breakout/reversion/trend;
  H1/H4/D1; majors+crosses): coin-flip ~0.49 base, EDGE<COST. Regime conditioning failed across
  dispersion / vol-LEVEL / Kaufman-ER; strong-trend regimes invert (3003). Carry OFF.
- **Three PORTFOLIO components** (net-positive long-only, not all-folds-positive): gap-fill JPY-cross
  H4 (1006), month-end USD-major D1 (1011), failed-breakdown-reclaim USD-major H4 (1013, the
  strongest+cleanest corpus edge).
- **Portfolio combination** (2008/3009/1015, triple-independent): the 3-way book is the strongest
  corpus result (risk-parity 2/10 neg, worst −0.77%, mean +0.55%) but **provably blocked,
  combination-invariant, by 2015 AND 2018** (0/5151 convex weightings pass; 2015 positive only in
  fbr, 2018 positive only in me — mutually exclusive). Needs a **4th component positive in BOTH 2015
  & 2018**.
- **2018-leg hunt exhausted across every directional/flow construction:** structural short
  (1014/2009/2011), trend short (3010), flow short up-gap weekend (1016), deep continuation long
  (2012), relative-value (2010) — ALL KILL. Refrain: "the 4th leg needs a genuinely non-price-
  direction construction, or the route is escalation-bound."

## Idea + why (the un-tested conditioner, with a *because*)

Both residual blocking folds — **2015** (SNB unpeg, Aug-2015 China deval) and **2018** (Feb VIX
spike, Q4 selloff) — are **risk-off / vol-spike years**, exactly where the three long-reversion fades
bleed (trends persist, dips keep falling). Every short tried for the 2018 leg conditioned on **price
structure** (sweep/reclaim/breakout) or **price trend** (3010) or a **flow event** (1016). The one
conditioner never tested: a **risk-STATE** proxy — a realized-**volatility-EXPANSION** trigger — used
to **SHORT the highest risk-beta majors (AUDUSD, NZDUSD; + the AUDJPY risk-cross)**.

*Because:* the dollar-smile / flight-to-quality mechanism — in risk-off deleveraging, risk-currencies
fall vs USD, a funding/positioning flow that **persists for weeks**, so the i+1 short should still have
move ahead (forward-confirming, unlike the backward-confirming flow shorts 1014/1016). This is
genuinely distinct from (a) the price-trend conditioner (3010, which inverted), and (b) the vol-LEVEL
momentum filter in closed ground (that conditioned a generic long across all pairs; this is
vol-EXPANSION × risk-beta-pair-SELECTION × SHORT, now expressible with shorts open, PR #273).

**Acceptance test (pre-registered):** robustly short-positive in **2015 AND 2018** (per arc-3010's
regime-luck screen — not just "the cells happen to be positive"), with >0.50 capture and a robust
(median, per-pair, outlier-excluded) drift. Observation-first cheap-kill (§5d) before any engine spend.

## Method (CALLED canonical; observation only)

`Panel.from_pairs` (H4, cached, 5ers_eet) on AUDUSD, NZDUSD (RISK), AUDJPY (RISKX), EURUSD, GBPUSD,
USDJPY (controls). Direction-aware honest short capture + 24-bar forward drift via BUILT
`observe_long_capture(direction="short")` (CHARACTERIZATION ONLY — gross, not a gate). Conditioner:
**vol-expansion** = causal `wilder_atr(14).shift(1)` ÷ its trailing 100-bar rolling median ≥ **1.3**
(ATR ≥ 1.3× trailing norm). Grouped base vs expansion, by pair, by year, per-pair 2015/2018.
Driver: `_disco3_work/arc3012_observe_riskoff_short.py` (reproducible).

## What happened — FALSIFIED at observation

**RISK basket (AUDUSD+NZDUSD), vol-expansion short, pooled:** cap **0.5025** (coin-flip; vs 1013 long
0.55–0.61), drift_mean **+0.264 but median −0.004** — the positive mean is a **thin-tail artifact**
(the exact arc-2011 mean≫median tell; the median short bar has ≈zero drift).

**Acceptance test FAILS on the binding fold.** RISK-basket vol-expansion short by year:

| year | n | cap | drift_mean | drift_med |
|---|---|---|---|---|
| 2015 | 246 | **0.654** | **+0.748** | +0.798 |
| 2016 | 187 | 0.578 | +0.789 | +0.778 |
| 2017 | 65 | 0.323 | −0.491 | −0.452 |
| **2018** | 111 | **0.369** | **−1.170** | −1.295 |
| 2020 | 338 | 0.382 | +0.626 | −0.517 |

2015 is strongly positive — **but 2018 is robustly NEGATIVE, and all three pairs agree**
(AUDUSD 2018 cap 0.328 / −1.24; NZDUSD 0.415 / −1.09; AUDJPY 0.545 / −0.30). The leg LOSES in the
exact fold it was built to cover. Controls (EURUSD/GBPUSD) show vol-expansion capture ~0.50 too —
vol-expansion adds no directional edge on either side (magnitude, not direction; closed-ground
re-confirmed).

## Diagnosis — the *because* inverts in 2018

Vol-expansions in **2018** were **capitulation / washout spikes at local lows** (Feb & Q4 risk-off
flushes that snapped back), so shorting the spike shorts **into the bounce** — the **backward-
confirming** failure mode of arc 2009 (climax-at-swept-low reverts) and arc 1016 (up-gap reversion
already spent). In **2015** the spikes came mid-decline (sustained EUR/AUD weakness → continuation →
the short wins). So the trigger is **regime-dependent**: it captures the easy fold (2015, already
covered by fbr) and fails the binding one (2018). A threshold sweep that "rescued" 2018 would be
fishing the target fold (arc-1012 trap) — the mechanism (vol-spike = local capitulation low →
bounce) is structural and all three pairs agree, so 2018-negative is the honest verdict.

## Verdict: KILL (cheap-kill at observation)

Coin-flip capture (0.50), thin-tail median≈0 drift, and **2018 robustly negative** → §5f exit-sweep
does not bite (reserved for beats-null / +gross-drift entries; a coin-flip-capture/median-≈0/
target-fold-negative cell collapses SL-honest, arc 3003). No pool/engine/null/council spent.

## Threads / lessons

1. **The risk-off vol-EXPANSION short does NOT supply the 2018 leg** — it is +2015 but −2018, because
   2018's vol spikes are capitulation lows (short bounces) while 2015's are mid-trend (short continues).
   Adds a sixth dead route to the 2018-leg hunt (structure 1014/2009/2011, trend 3010, flow 1016,
   continuation-long 2012, now risk-state vol-expansion short 3012).
2. **Vol-EXPANSION confers no directional edge** (cap ~0.50 both sides; mean drift is a thin-tail
   artifact, median ≈0) — the symmetric completion of arc 1001's vol-CONTRACTION-breakout finding and
   closed-ground's vol-LEVEL result: volatility predicts magnitude, never direction, at H4 on majors,
   in either vol direction.
3. **The 2018 wall is mechanism-deep, not construction-shallow.** Every short/trend route fails 2018 by
   the SAME structural reason — 2018's tradeable price events are **mean-reverting capitulations**, so
   any *fade* (the 3 PORTFOLIO longs) bleeds AND any *vol/structure-triggered short* shorts into the
   bounce. 2018 is positive only for a *slow trend-follower that is already short before the spike* —
   which is a coin-flip on the +1R-capture lens across all years (3010). The 2018 leg is not a
   price/vol-conditioned directional bet in either direction → **arc-3004 escalation reinforced** (the
   4th leg needs a genuinely non-price-direction construction or a tighter-cost execution regime the
   apparatus cannot self-supply).
4. **Surviving frontier (unchanged):** the in-apparatus directional/flow/vol short leads for the 2018
   leg are now exhausted; what remains is operator-gated (tighter-cost execution regime, #3 of the
   arc-3004 escalation) or a non-price-direction construction not yet conceived.

## Tooling

No new BUILT tool — reused BUILT direction-aware `observe_long_capture(direction="short")`; the
vol-expansion ratio is a one-off scratch conditioner (like arcs 3005/1008/2010 observers).

## FLAGS (code not merged)

None. No canonical-core change. Carries FLAG-1 (shorts open but the named structure/flow/trend/
vol-state short leads are now all dead for the 2018 leg) + the standing `A1Config.time_exit_bars`-
unwired flag. Driver `_disco3_work/arc3012_observe_riskoff_short.py` (reproducible from this doc).
