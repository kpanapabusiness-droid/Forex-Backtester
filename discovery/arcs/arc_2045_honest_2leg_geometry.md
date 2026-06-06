# arc 2045 — Deployment GEOMETRY of the honest 2-leg (me_long+fbr) book

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **DIAGNOSTIC → KILL** (no new
component; me_long & fbr stay PORTFOLIO, the 2-leg book is a SUBSET not a new edge, and it is
still not all-folds-positive) · **Council:** none (a measurement; composes 2044 + 1033/2033)

---

## (a) Log read / synthesis

Pulled main; STOP absent; highest 2000s arc = 2044 (mine), next = 2045. State (honest-era):
edge-hunt **structurally closed** (3021 path-B proof, MENU exhausted, every short construction dead,
~18 routes to the 2015/2018 leg dead); 4 PORTFOLIO components (gap 1006, me_long 1011, fbr 1013,
me_short 1019); 4-way book mean-positive but never all-folds-positive; **§11 independent verification
COMPLETE** end-to-end. The exit-honesty thread (2040→2044, +1042/1043/1044) showed the committed-exit
book characterization is systematically optimistic: under honest §5f exits gap & me_short flip
mean-NEGATIVE, fbr −40%, only me_long exit-robust; the book's significance (2043), diversification
(2044/1044), temporal stability (2044), and cost cushion (2044) pillars all weaken; **arc 2044
established the honest deploy OBJECT is the 2-leg me_long+fbr** (the 4-leg book is mean-DRAGGED below
its 2 honest legs; arc 1044 independently confirmed me_long+fbr beats me_long-solo on risk-adjusted
Sharpe via fbr's −0.5 decorrelation).

## (b) Idea (the *because*)

arc 2044 named the honest deploy object (2-leg me_long+fbr) but never PROFILED it. Arc 1033/2033
proved deployability is governed by GEOMETRY — **Calmar / time-underwater / prop-firm feasibility, NOT
depth or per-year sign** (a thin book can clear a vol-adjusted bar yet be undeployable on a DD-gated
vehicle) — but computed that geometry only for the COMMITTED 4-leg book (RP maxDD 1.59%, Calmar
0.24–0.36, ~4.7yr/98% underwater, prop-firm T_min 1.4–8.4yr = infeasible). **Does the honest deploy
object — drop the two exit-fragile drag legs (gap, me_short) AND use fbr's HONEST §5f exit — improve
that geometry?** A cleaner, fewer-leg, honest-exit book is the operator's actual candidate; if its
Calmar/underwater are materially better, the vehicle wall (2033) might soften. The deploy object uses
FIXED single configs per leg (me_long committed `sl_only`/2-bar = exit-robust, committed≈honest per
2042; fbr's FROZEN honest §5f exit = `freeze_best_over_folds`, the deployable single config, NOT the
committed full-sample pick 2040 flagged ~40% optimistic) → a genuinely deployable contiguous account.

## (c)/(f) Method

- 100% canonical scoring (A1→MultiPairBacktester, FundedNext, risk 0.005). fbr's frozen honest exit
  via BUILT `nested_exit_selection.freeze_best_over_folds` (3 metrics). Contiguous co-sim via the
  canonical `cosim_book_fold` (reusing `equity_risk_profile`'s `_build_4way_contiguous` pattern,
  specialized to 2 legs + per-year-reset re-id). Geometry via BUILT `compute_risk_profile`; vehicle
  feasibility via BUILT `propfirm_feasibility` (`compute_sharpe`, `feasibility_horizon_years`).
  Committed 4-leg numbers cited from arcs 1033/2033 (reproduced many times) as the baseline; me_long
  per-year ROIs cross-checked vs the committed record as the in-script anchor. No new BUILT tool
  (composition of existing). OOS NEVER touched. Driver `_disco_work/arc2045_honest_2leg_geometry.py`.

## Results

### Anchors reproduce
- me_long per-year ROI == committed record, **max |dev| 0.004pp** (anchor).
- fbr frozen honest §5f exit: **afp/worst metrics → `sl_partial_close_1r_runner_trail`/SL1.5, IS mean
  +1.477%** (vs committed trail-off +2.084% / double-trail +1.854% — ~25–30% honest haircut, consistent
  with arc 2040). (mean_roi metric freezes the high-variance `trailing_swing`/SL1.5 +4.263% = the §5f
  trap arc 2040 flagged dies OOS → correctly NOT the deploy choice.)

### The honest 2-leg book — per-fold (per-year-reset), n=10
| weighting | mean | sd | t | pos | worst |
|---|---|---|---|---|---|
| risk-parity (me_long .745 / fbr .255) | **+0.549%** | 0.634% | +2.74 | 8/10 | **−0.251% (2015)** |
| equal | +0.854% | 0.997% | +2.71 | 8/10 | −0.815% (2018) |

- **Worst-fold −0.251% (RP) is the SHALLOWEST book worst-fold in the corpus** (vs the committed 4-leg's
  best −0.77%) — but **STILL not all-folds-positive** (2015 −0.15%, 2016 −0.25% marginally negative).
- **The 2-way RECOVERS borderline significance the honest 4-way had LOST** (corrected post-hoc by the
  concurrent independent arc 1045 — see CONVERGENCE; my initial draft over-cautiously guessed it would
  "collapse to ~1," which was WRONG). My frozen-config 2-way is t=+2.74 (n=10); arc 1045's genuinely
  no-lookahead NESTED 2-way is **t=+2.12, P(mean<0)=0.008, CI [+0.07,+0.77] excludes zero** (afp metric,
  8 eval folds) — both say the 2-way is borderline-significant, **vs the honest 4-way's t≈0.4–1.0 with
  CI spanning zero** (2043). Trimming the two drag legs concentrates the book on the me_long↔fbr
  anti-correlation (−0.5…−0.6) and thereby recovers significance. **Caveat (no over-claim): borderline,
  RP-weighting- and metric-dependent** (1045: equal-weight 2-way afp t=+1.76, CI spans zero) — a real
  improvement, not a robust >2σ. So the honest-§5f *significance* picture is: 4-way NOT sig (2043), but
  its 2-leg honest CORE is borderline sig — the drag legs were destroying the significance, not the edge.

### DEPLOYMENT GEOMETRY (the headline) — honest 2-leg vs committed 4-leg
| book / weighting / cap | ann ret | maxDD | Calmar | underwater | Sharpe | prop-firm T_min (10/10) |
|---|---|---|---|---|---|---|
| honest 2-leg RP cap-ON (faithful) | +0.382% | **1.559%** | **0.245** | 1999d (99%) | 0.305 | **4.08yr** |
| honest 2-leg RP cap-OFF (bound) | +0.635% | 1.543% | 0.412 | 890d (99%) | 0.459 | 2.43yr |
| honest 2-leg EQUAL cap-ON | +0.579% | 2.180% | 0.266 | 1459d | 0.303 | 3.76yr |
| **committed 4-leg RP (arc 1033/2033)** | ~+0.4–0.6% | **1.59%** | **0.24–0.36** | ~4.7yr/98% | 0.4–0.7 | **1.4–8.4yr** |

- **The geometry is essentially UNCHANGED.** maxDD ~1.55% (≈ 4-leg 1.59%), Calmar 0.245–0.46
  (≈ 4-leg 0.24–0.36), ~99% time underwater, Sharpe ~0.3–0.46, prop-firm T_min 2.4–4.1yr (inside the
  4-leg's 1.4–8.4yr). Daily 5% cap never breached (worst-day 0.29–0.99% × leverage). **Dropping the two
  drag legs + honest exits does NOT break the vehicle wall** (arc 2033's Calmar-bound prop-firm
  infeasibility) — T_min stays YEARS vs the weeks-months a challenge expects.
- **Modest, cap-convention-dependent improvements only:** cap-OFF the 2-leg RP underwater shortens to
  ~890d (~2.4yr) vs the 4-leg ~4.7yr and Calmar 0.412 edges the 4-leg's top — but cap-ON (faithful) the
  2-leg underwater is 1999d (~5.5yr, LONGER than the 4-leg) and Calmar 0.245 (same) → the improvement
  is not robust to the exposure-cap convention; the binding (faithful) geometry is unchanged.

## (h)/(g) Verdict — DIAGNOSTIC → KILL (no new component; components UNCHANGED)

**Findings.**
1. **The honest deploy object (2-leg me_long+fbr) is CLEANER but NOT more deployable.** As a fixed-config
   book it has a decent point mean (+0.55% RP / +0.85% equal), the shallowest book worst-fold in the
   corpus (RP −0.25%, 8/10), and a fixed-config t≈2.7 — but it is **still not all-folds-positive** (2015
   & 2016 marginally negative) and its **deployment geometry is essentially identical to the committed
   4-leg** (faithful Calmar ~0.245, ~99% underwater, prop-firm T_min ~4yr = infeasible). Simplifying
   4→2 legs + honest exits removes the exit-optimism of the drag legs but does **not** soften the
   vehicle wall (arc 2033).
2. **The 2-way improves STATISTICAL QUALITY but not the DEPLOYMENT geometry — the arc-2033 split
   persists.** Trimming to the 2 robust anti-correlated legs ~DOUBLES the Sharpe (RP cap-OFF 0.459 /
   cap-ON 0.305 vs the honest 4-way's ~0.13–0.20 per arc 1045) AND recovers borderline significance
   (my frozen t=2.74 / arc-1045 nested t=2.12, CI excludes 0) — the drag legs (gap, me_short) were
   net-SUBTRACTING (mean-negativity outweighed their decorrelation), so dropping them lifts both. BUT the
   DD-shaped vehicle metrics — faithful Calmar ~0.245, ~99% time-underwater, prop-firm T_min ~4yr — are
   UNCHANGED. This is exactly arc-2033's "Sharpe OK but Calmar much less than Sharpe → DD-gated vehicle
   infeasible," now shown to PERSIST for the cleaner 2-way: the drag legs added mean-noise (hurting
   Sharpe/significance), not drawdown duration (which sets Calmar) — so trimming them is a real
   statistical-quality win but not a deployability fix. (My initial draft wrongly guessed the 2-way
   significance "collapses to ~1"; the concurrent arc 1045 computed the nested 2-way directly and got
   t=2.12 — corrected here, see CONVERGENCE.)
3. **fbr's honest deployable exit = `sl_partial_close_1r_runner_trail`/SL1.5 (+1.477% IS)** — a ~25–30%
   haircut on the committed trailing-atr headline, and a partial-runner (mechanism-defensible), not the
   committed trail config. (The §11 outcome audit 2036 verified the committed `trail_enabled=True`
   variant — the deployable honest fbr exit differs, re-confirming arc-2040's F2 config note: the
   deployable trail-off / honest-exit fbr's outcome layer is a small owed §11 re-derivation.)

**FAIR caveat (§8).** The 2-leg book's borderline significance (frozen t=2.74 / nested t=2.12, 1045) +
~2× Sharpe + decent point mean are REAL improvements over the honest 4-way — but borderline and
RP-weighting/metric-dependent (1045: equal-weight 2-way CI spans zero), not a robust >2σ. The binding
weakness is the per-year-gate noise floor (not AFP) + the vehicle-Calmar wall (arc 2033), both now shown
INVARIANT to the 4→2-leg honest simplification.

**FLAG F1 (docs only).** The honest deploy object (2-leg me_long+fbr) does not improve deployability
over the committed 4-leg: faithful Calmar ~0.245, ~99% time underwater, prop-firm T_min ~4yr
(infeasible), still not all-folds-positive. Path-A's vehicle wall (arc 2033) is invariant to dropping
the drag legs + honest exits; the only viable deployment remains already-funded low-risk capital as a
slow ~0.4–0.6%/yr diversifier (operator risk-appetite call), not a prop-firm challenge.

**Components UNCHANGED** (me_long & fbr stay PORTFOLIO; the 2-leg book is a subset, still not AFP → no
survivor). Lever unchanged = operator path-A gate-governance call (+ the arc-2033 vehicle reality). No
canonical change, no new BUILT tool, no council, **no OOS**.

## (i) NEW lesson

For a thin multi-leg book whose deployability is GEOMETRY-bound (Calmar / time-underwater /
DD-gated-vehicle, arc 2033), pruning to the honest exit-robust subset (here 4→2 legs, me_long+fbr) and
using each leg's honest §5f exit **cleans the object and can shrink the per-year worst-fold to the
corpus minimum (−0.25%, 8/10) — but it does NOT move the binding deployment geometry** (faithful Calmar
~0.245, ~99% underwater, prop-firm infeasible all stay put). The drag legs were adding mean-noise, not
the drawdown duration; the duration/Calmar wall is a property of the SURVIVING reversion legs
(me_long+fbr earn in lumpy flow bursts and grind flat-to-bleed between, regardless of how many legs are
combined). Pruning the drag legs DOES lift the vol-adjusted quality — ~2× Sharpe and a recovered
borderline significance (nested t=2.12, 1045) the full book had lost (2043), because gap+me_short were
net-SUBTRACTING (their mean-negativity outweighed their decorrelation) — but it does NOT lift the
DD-adjusted Calmar (the metric the deployment vehicle gates on). So a thin geometry-bound book has two
nearly-independent quality axes: trimming to the honest exit-robust core fixes the Sharpe/significance
axis but not the Calmar/vehicle-feasibility axis. Simplifying to the honest core is the right deploy
FRAMING and a real statistical-quality win, but it is not a deployability FIX — the vehicle wall is
invariant to leg-count.

## CONVERGENCE — independent concurrent reproduction (arc 1045, 1000s chat)

The 1000s chat ran the SAME 2-leg me_long+fbr honest-book idea concurrently (arc 1045, landed on main
at my push). **Same headline, complementary measurements, and it CORRECTED my over-caution (Arc-10
defense in action):**
- **Agreement (the shared headline):** the honest deploy object is the 2-leg me_long+fbr;
  **gap & me_short are NET DRAGS under honest exits, not diversifiers** (their mean-negativity outweighs
  their decorrelation); the 2-way is still **NOT all-folds-positive** (1045: no convex weighting AFP,
  best worst −0.21%, binding 2014 + the 2015↔2018 mutual-exclusion; mine: RP 8/10 worst −0.25% at 2015)
  → no survivor, components unchanged. Two chats, different drivers, same verdict.
- **Complementary measurements:** arc 1045 computed the 2-way's Sharpe + significance + convex-AFP scan
  on the nested 8-eval-fold series; **I** computed the 2-way's contiguous **DEPLOYMENT GEOMETRY**
  (Calmar / time-underwater / prop-firm feasibility) — the axis 1045 didn't touch and arc 1033/2033
  only did for the 4-leg. Together: the 2-way ~doubles Sharpe (1045: +0.66–0.75 RP vs 4-way +0.13–0.20;
  mine: +0.31–0.46) and recovers borderline significance (1045 nested t=2.12 / mine frozen t=2.74) — a
  real **statistical-quality** win — **but its DD-shaped Calmar (~0.245)/underwater(~99%)/prop-firm
  T_min(~4yr) are unchanged** → still vehicle-infeasible. The synthesis is arc-2033's Sharpe-vs-Calmar
  split, now shown to persist for the cleaner 2-way.
- **The correction (why independent reproduction matters):** my draft over-cautiously claimed the 2-way
  significance "collapses to ~1 under the nested reading" (extrapolating the 4-way mechanism). arc 1045
  actually computed it — nested 2-way t=+2.12, CI [+0.07,+0.77] EXCLUDES zero — so the 2-way RECOVERS
  the significance the 4-way lost; dropping the drag legs HELPS the significance, it doesn't kill it. I
  corrected the doc accordingly. Both numbers (my frozen 2.74, 1045's nested 2.12) agree: borderline-sig,
  RP/metric-dependent, ~2× the honest 4-way.
- **Combined operator statement:** the honest deploy object is the 2-leg {me_long+fbr} at risk-parity
  (drop gap+me_short — they net-subtract) — cleaner, ~2× Sharpe, borderline-significant, shallowest
  worst-fold (−0.25%) — but STILL not all-folds-positive AND its deployment geometry (Calmar/underwater/
  prop-firm) is unchanged → still vehicle-infeasible as a challenge; viable only as a low-risk
  already-funded slow diversifier. Path-A gate-call + arc-2033 vehicle reality remain the levers.

## (k) Re-orient

Detail persisted (this doc + DISCOVERY_LOG append, committed + pushed). Tools: REUSED
`nested_exit_selection` (2040) + `combine_fold_roi`/`fit_weights` (2006) + `cosim_book_fold`
(canonical) + `compute_risk_profile` (1033) + `propfirm_feasibility` (2033); no new BUILT tool. Driver
`_disco_work/arc2045_honest_2leg_geometry.py`, output `_disco_work/arc2045_out.txt`. No canonical
change, no council, no OOS. The honest-exit thread is now complete on every axis (mean 1042, sig
2043/1043, diversification 2044/1044, temporal 2044, cost 2044, and now the honest deploy object's
geometry 2045) — the edge-hunt is structurally closed and the deployability lever is the operator's
path-A call + the invariant arc-2033 vehicle wall. Next: resume at arc 2046 (or graceful handoff if
context low).
