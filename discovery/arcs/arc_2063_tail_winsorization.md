# arc 2063 — TAIL-DEPENDENCE of the {me_long, fbr} deploy vehicle (runner winsorization)

**Chat:** 2000s. **Type:** DIAGNOSTIC (decision-support for the path-A lever). **Disposition:**
KILL (no new component; all 4 UNCHANGED, PORTFOLIO). **Deployable count:** 0.
**Driver:** `_disco2000_work/arc2063_tail_winsorization.py` (analysis script; CALLS canonical
apparatus + BUILT `build_component`/`fit_weights`; experiment-side = winsorization arithmetic only,
never realizes P&L). OOS = measure-once CHARACTERIZATION off the already-spent frozen series
(§4; arc 2055/2056/2059 precedent) — nothing selected on OOS.

## Why this (resolves an explicitly-owed thread at MY chat's deploy object)

Fresh-eyes step-(a) read: the OHLC-only EDGE frontier is mined out on every mapped axis (both
chats converged; the last live arcs 2057–2062 are calendar/closure cheap-kills + vehicle
diagnostics). The single live lever is the operator's **path-A gate-governance decision** —
deploy (as an already-funded diversifier) vs not. Arcs 2059/2060 established the **corpus-best OOS
deployment vehicle is the 2-leg `{me_long, fbr}`** (global leg-count Calmar optimum, beating solo
2055 and the 4-way book 2056) and flagged it as **doubly fat-tail-fragile at the YEAR level**
(ex-2024 → +0.31%/yr; ex-both-tail-years ≈ +0.06%/yr flat; the +1.87 RP year = fbr's 2024 runner).

The 1000s chat's arc-1056 handoff left an explicitly-owed, decision-grade thread (noted
non-colliding with the 2000s vehicle lane): **quantify what fraction of the deploy mean comes from
fbr's convex runner tail, and re-score significance with the runner tail winsorized at +2R/+3R
(geometry-only, the engine still takes-the-loss).** That is the rigorous TRADE-level version of
2059's year-level fragility note, measured at exactly the object my own 2059/2060 arcs named the
deploy optimum. *because:* a mean carried by a handful of convex trailing-runners is not
broad-based — it is high-variance to realize over a short challenge window and is the classic
"high Calmar from a least-repeatable fat tail" pattern (2054/2055/2059); a mean that survives a
+2R cap is deployment-robust.

## Method (CALLS canonical; geometry-only upside cap)

1. Build `me_long` + `fbr` via the committed/frozen deploy configs (`build_component`, IDENTICAL to
   `subset_deploy_profile`/`validate_4way_book`): me_long = `sl_only` + 2-bar time-exit (D1 USD
   majors); **fbr = `sl_plus_trailing_atr`, `trail_enabled=True` double-trail (H4 USD majors) — the
   convex-runner source.** Canonical `ArcFoldRunner` → A1 → `MultiPairBacktester`, FundedNext costs.
2. Per fold (= per year) capture per-**position** NET P&L (`apply_cost_model` chokepoint; positions
   whose `final_exit_time` ∈ the fold's OOS window) + denom (OOS-start net equity) — the arc-1056 unit.
3. **R-unit** = `risk_pct·SB` = 0.005·100 000 = **$500** (linear sizing risk — the size at which a
   full SL = −1R). Winsorize UPSIDE only: `net_capped = min(net_pnl, K·$500)`, K ∈ {2,3,4}. **Losses
   are UNTOUCHED** (take-the-loss preserved; strictly conservative — capping can only LOWER the mean).
4. Recompute per-year leg + book ROI (equal + RP frozen-IS weights), deploy mean / neg-count / worst,
   fbr tail-fraction, and the year-level (cluster) bootstrap P(mean<0) + 95% CI — uncapped vs capped.

**Reproduction anchor (Arc-10 guard).** Uncapped per-year leg ROIs reconcile to the committed 2056
numbers **on the decision-relevant quantity (the mean) to ≤0.04pp**: IS-RP committed 0.534 vs 0.515
(0.019pp); OOS-RP committed 0.570 vs 0.606 (0.036pp). Individual YEARS deviate up to 0.69pp (fbr's
high-activity 2011/2012), because the per-position-realized unit attributes open-position
mark-to-market at year boundaries differently than the contiguous cosim equity curve (2056's source)
— this CANCELS in the mean and is attribution-invariant for the total-excess-above-cap that drives
the finding. (Soft note, not a FLAG: the canonical `fs.roi_pct`/cosim is correct; the per-position
Σnet/denom is the documented experiment-side approximation, faithful on the mean.)

## Result

**fbr runner tail is concentrated** (the edge lives in few positions):
- IS: 104/210 fbr positions positive; **23 exceed +2R and carry 20.5%** of fbr's gross-positive P&L
  (10 >+3R → 8.6%; 4 >+4R → 3.8%).
- OOS: 49/109 positive; **9 exceed +2R and carry 25.3%** of gross-positive P&L (5 >+3R → 15.7%).

**Deploy mean survival under the upside cap** (RP frozen-IS weights = me_long 0.814 / fbr 0.186, the
deploy weighting):

| window / weighting | uncapped | +2R | +3R | +4R |
|---|---|---|---|---|
| **OOS** RP | +0.606%/yr (P<0=**0.001**) | +0.310 (51%, P=**0.061**, CI[−0.07,+0.72]) | +0.442 (73%, P=0.022) | +0.513 (85%, P=0.006) |
| **OOS** equal | +0.766 (neg 3/6, P=0.117) | **+0.039 (5%!**, neg 5/6, P=0.437) | +0.326 (43%) | +0.517 (68%) |
| **IS** RP | +0.515 (P<0=0.003) | +0.255 (50%, P=**0.050**) | +0.406 (79%, P=0.009) | +0.468 (P=0.003) |
| **IS** equal | +1.025 (neg 1/10, P=0.008) | +0.326 (32%, P=0.132) | +0.731 (71%) | +0.897 |

## Diagnosis / finding (decision-grade)

The {me_long, fbr} deploy mean is **substantially fbr-runner-tail-carried, NOT broad-based.** A +2R
geometry cap — which touches ONLY fbr's convex trailing-runners; every loss is already taken
SL-first — removes **~49% of the RP-weighted deploy mean** (OOS 0.61→0.31; IS 0.52→0.26) and
**~95% of the equal-weighted OOS mean** (0.77→0.04, flipping 3/6→5/6 neg). It pushes the
cluster-bootstrap significance from clearly-positive (P(mean<0) ≈ 0.001–0.008) to **borderline /
non-significant at +2R** (RP P ≈ 0.05–0.06, CI straddling zero); significance only recovers at +3R/+4R,
i.e. only when the runners keep most of their natural convexity.

This is the trade-level confirmation of 2059's year-level fragility: the positivity leans on a
handful of large convex trailing-runners (9 OOS / 23 IS positions > +2R), so the edge is **convexity-
dependent**, not a broad win-rate edge. The RP deploy weighting (fbr down-weighted to 0.186)
deliberately tempers this — under RP the mean stays positive even at +2R (51% survives) — but it is
still half tail, and under equal weight the OOS mean is essentially ALL tail.

**Why it matters for path-A:** a tail-/convexity-dependent mean is exactly a mean that needs MANY
trades / a LONG horizon to realize reliably — the same property that produced 2059/2060's
vehicle-infeasibility (T_min ≥ 2yr ≫ a challenge's weeks–months: you may simply not catch a runner in
the window). The two findings are the same coin: the deploy object's edge is **real and honestly
take-the-loss-realized, but convex-tail-carried**, which (a) confirms it is not a challenge-account
strategy and (b) means even as an already-funded diversifier its realized short-run return is
high-variance. It does NOT change the deploy disposition (still PORTFOLIO/vehicle-infeasible) but it
sharpens *why*: not regime-luck per se, but **trade-count/convexity dependence**.

## NEW lesson

The corpus-best OOS deployment vehicle ({me_long, fbr}, 2059/2060) is **convex-tail-carried at the
trade level**: a geometry-only +2R upside cap (losses untouched) removes ~half the RP-weighted deploy
mean and essentially ALL the equal-weighted OOS mean, dropping cluster-bootstrap significance from
P(mean<0)≈0.003 to borderline (~0.05–0.06). The edge is fbr's handful of convex trailing-runners (9
OOS / 23 IS positions > +2R carry ~20–25% of fbr's gross-positive P&L), not a broad win-rate edge.
This is the trade-level identity of 2059's year-level "ex-tail-years ≈ flat" and the mechanistic root
of vehicle-infeasibility: a convex-tail-carried mean requires a long horizon / many trades to realize,
so it cannot clear a short prop-firm challenge window AND is high-variance even as a funded
diversifier. Generalizes the 2054/2055/2059 "high Calmar from a least-repeatable fat tail" pattern to
the trade level via the winsorization lens, and resolves arc-1056's explicitly-owed tail-dependence
thread at the {me_long, fbr} object. The RP deploy weighting (fbr → 0.186) is the right partial defense
(keeps the mean positive at +2R) but does not make it broad-based.

## Threads / handoff

Resolves the 1056 owed thread; no new thread opened. Components UNCHANGED (all 4 PORTFOLIO); honest
deploy object UNCHANGED (me_long-solo on mean / {me_long, fbr} on the OOS vehicle axis, both
vehicle-infeasible AND — now established — convex-tail-carried). Lever = operator path-A. No
engine/null/council (diagnostic KILL, not a survivor); no canonical change; no FLAG (the
per-position-vs-cosim boundary difference is a documented experiment-side approximation, faithful on
the mean). No new BUILT tool (one-off analysis driver; reuses canonical + `build_component`/`fit_weights`).
My read after 2061/2062/2063: the OHLC-only charter is mined out on edges AND the deploy object is now
fully characterized (vehicle matrix + tail-dependence); remaining value sits with the operator's
gate/deploy decision or an operator charter unlock (NEEDS_ENABLEMENT), not the autonomous edge-hunt.
