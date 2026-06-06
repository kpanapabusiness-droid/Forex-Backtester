# Arc 1055 — O1 spread-z inelasticity-state conditioning of fbr (the last untested O1 proxy)

**Chat:** 1000s **Date:** 2026-06-06 **Disposition:** KILL (obs cheap-kill)
**Components changed:** none **OOS:** never touched (IS 2010–2020 only; observation, no engine/gate)

---

## (a) Log synthesis (carried from arc 1054)

The portfolio route's 4-way book is mean-positive but never all-folds-positive; the failure is a
per-year noise floor (path-B closed, 3021); the honest deploy object collapses to me_long-solo (1046),
vehicle-infeasible (1053/2053). The strategist explore-now MENU is nearly fully closed: M1 (1027/2023),
L1 (1054, just closed), Q1 (1028), G1 (2052) all dead. **O1** (inelasticity-state conditioning) had two
of its three collinear proxies closed — calendar-density (1029) and trigger-shallowing/depth (1025) —
leaving exactly one named sub-thread untested: the **honestly-costed entry-bar spread-z** proxy (arc
1029: "spread-z (cost-trap-flagged) the only untested O1 sub-thread"). This arc closes it.

## (b) Idea — O1 spread-z, the council's flagged cost-trap proxy

*Because (the O1 thesis):* forced flows move price more per unit when the book is **inelastic**; the
subsequent reversion is then larger. A wide bid-ask spread at the dislocation bar is a lag-free proxy
for "how inelastic was the book." Concentrating the proven fbr edge onto its highest-spread-z entry bars
should raise edge-per-trade without adding coin-flip trades — a direct attack on the fold-resolution
problem (the book's actual blocker). **Council cost-trap warning:** wide-spread bars are exactly where
you pay the most — the test MUST charge the *realized* wide spread, or it is an Arc-10 cousin.

**O1 falsifiable prediction:** per-trade realized R rises **monotonically** with the spread-z state
variable, AND a book restricted to its top decile has a **higher worst-fold** than the unconditioned edge.

## (c)/(d) Method (obs cheap-kill, §5d)

Single-use driver `discovery/_disco1_work/arc1055_fbr_spreadz.py` (BUILT-tools only:
`FailedBreakdownReclaimLongSignal` 1013 + `observe_long_capture`). fbr fires (H4, 7 USD majors,
K40/shadow1.25, IS 2010–2020; n=235 ≡ canonical fbr) joined with each fire's entry-bar **spread-z**
(causal: `(spread − rolling-mean)/rolling-std`, 250-bar trailing window, shift1) and the **realized
round-trip spread cost in R** (`1.5×spread / (2·ATR)`). Bucketed by spread-z tercile (gross drift,
capture, cost-R, NET drift) and a top-decile-vs-full per-year fold-sign comparison. No engine/null/
council; IS-only so the holdout stays pristine.

## Result — O1 spread-z FALSIFIED (and INVERTED)

```
tier        n  spread_bp  gross_drift  capture  cost_R  net_drift  med_drift
LO_tight   79     0.7453       0.2999   0.5823  0.0363     0.2636     0.2671
MID        79     0.9862       0.0334   0.5823  0.0438    -0.0104     0.1037
HI_wide    77     2.8891       0.1015   0.5974  0.0725     0.0290     0.1209

FULL        n=235  pos_years=7/11  worst_year_drift=-1.527  2015=+0.799  2018=-1.282
TOP_DECILE  n= 24  pos_years=2/9   worst_year_drift=-6.044  2015=+0.016  2018=nan(no fires)

NET-drift monotone-rising in spread-z?   False (LO +0.264, MID -0.010, HI +0.029)
GROSS-drift monotone-rising in spread-z? False (LO +0.300, MID +0.033, HI +0.102)
```

**Decisive readings:**
1. **Non-monotone — INVERTED vs the O1 prediction.** fbr's edge is *highest in the tightest-spread*
   bucket (gross +0.300, net +0.264) and ~3× weaker in the widest/"inelastic" bucket (+0.102/+0.029).
   The inelasticity thesis is backwards for fbr: a wide spread at the fbr bar marks genuine stress (a
   real breakdown where the reclaim fails), not a fadeable thin-book overshoot.
2. **Cost trap confirmed.** HI_wide median spread 2.89 bp ≈ **4× the tight bucket's 0.75 bp**; realized
   cost_R doubles (0.036→0.073) with no gross-edge gain → net is *worse* than the tight bucket. Charging
   the realized wide spread (the council's mandated guard) is exactly what kills the wide-spread cell.
3. **Top-decile craters fold resolution.** The top spread-z decile has 2/9 positive years, worst-year
   drift −6.04 (vs full −1.53), and **zero 2018 fires** — the O1 "top decile → higher worst-fold"
   prediction is decisively FALSE; this is the same thinning death as arc 1029 (calendar-density).

## Diagnosis

The spread-z proxy fails on all three of the O1 prediction's clauses (monotone↑, cost-honest, better
worst-fold). For a stop-run-reclaim reversion (fbr), the wide-spread bars are not "more inelastic →
larger reversion"; they are genuine-stress bars where the failed-breakdown becomes a real breakdown and
the reclaim does NOT hold (the 2018 mechanism, arcs 2014/2052). So inelasticity (as proxied by spread-z)
*anti*-selects fbr's edge, and even where it didn't, charging the 4× realized spread plus thinning to a
top decile destroys both the net edge and the fold resolution. This CONFIRMS the council's collinearity
note (density/spread-z/depth are one latent variable in three costumes) AND the 1029 thinning death,
with a sharper mechanism: fbr is a **liquid-market** stop-run reclaim, not an illiquidity event.

## Verdict & meaning

**KILL (obs cheap-kill).** Closes the last O1 sub-thread (spread-z) → **O1 fully closed** across all
three proxies (density 1029, depth 1025, spread-z 1055). With M1 (1027/2023), L1 (1054), Q1 (1028), G1
(2052) already dead, **the strategist explore-now MENU is now exhaustively closed.** The OHLC-only EDGE
frontier is mined out on the directional (1052), convergence/stat-arb (1054), AND
inelasticity-conditioning (this) axes. Components UNCHANGED; operator path-A remains the sole lever
(deploy object me_long-solo, 1046/1053); deployable-system count = 0.

**NEW lesson:** the inelasticity-state thesis (wide entry-spread → larger reversion) is INVERTED for the
corpus's strongest reversion edge — fbr's edge is concentrated in TIGHT-spread (liquid) bars; a wide
spread at the fbr bar signals a genuine stress breakdown (reclaim fails), not a fadeable thin-book
overshoot. So the honest spread-z proxy both anti-selects the edge AND triggers the cost trap (4×
realized spread) AND thins fold resolution — three independent reasons it fails. Generalizes the O1
collinearity (density/depth/spread-z = one variable) with a directional correction: for stop-run
reversions, inelasticity is a NEGATIVE signal, not the positive one O1 hypothesized.

## (i) Tooling

Single-use driver `discovery/_disco1_work/arc1055_fbr_spreadz.py` (BUILT-tools only:
`FailedBreakdownReclaimLongSignal` + `observe_long_capture`; no new BUILT registry entry, per the
arc-1053 precedent for one-off diagnostics). No canonical change, no FLAG.
