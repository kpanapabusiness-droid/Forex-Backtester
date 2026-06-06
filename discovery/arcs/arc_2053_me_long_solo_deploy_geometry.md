# arc 2053 — me_long-SOLO standalone deployment-vehicle geometry (the arc-1046 deploy object, never profiled standalone)

**chat:** 2000s · **date:** 2026-06-06 · **disposition:** (pending) · DIAGNOSTIC, OOS-preserving (§5g)

---

## (a) LOG READING + SYNTHESIS (fresh eyes, honest-era only)

Pulled main (up to date, HEAD e9a05c8). Read DISCOVERY_PROTOCOL, full Tier-1 ledger (arcs 0–3022 across
all three chats), LESSONS, DISCOVERY_DIRECTION, NEEDS_ENABLEMENT, TOOL_REGISTRY. No `discovery/STOP`.
Highest 2000-range id = 2052 → resume at **2053**.

**State of the programme (what's settled):**
- **Closed ground:** single-condition shallow directional prediction (momentum/breakout/mean-reversion/
  trend), H1/H4/D1/W1, majors+crosses, BOTH directions, capture AND drift lenses, stop-removed — forward
  drift ≈ cost everywhere. The 2018/2014 strong-USD regime INVERTS every reversion leg.
- **4 PORTFOLIO components** (mean-positive, not all-folds-positive): gap-fill (1006, JPY-cross H4),
  me_long (1011, USD-major D1), fbr (1013, USD-major H4), me_short (1019, USD-major D1).
- **The book is NOT all-folds-positive** — blocked combination-invariantly by **2015 & 2018**
  (1015/2008/3009: 0/5151 convex weightings pass; 2015 wants fbr, 2018 wants me_short, mutually exclusive).
- **The 2018-positive 4th/regime-orthogonal leg does NOT exist in OHLC FX** — ~22 routes dead (structure
  shorts 1014/2009/2011/3011, trend short 3010, vol-state 3012, flow shorts 1016/1017, rel-value 2010,
  deep-continuation 2012, weekly 3014, session 1047/2050, fix daily 1051, intrabar 1052, fbr universe/TF/
  USD-factor 2031/2049/2051/2052, …). Directional/trend bets are coin-flips both directions; the leg the
  book needs is structurally a trend/risk-off bet → not in the OHLC charter.
- **The strategist menu (DISCOVERY_DIRECTION) is fully closed:** M1 driver-shock cross-TF residual
  (1027/2023), O1 inelasticity-state all 3 proxies (calendar-density 1029 / spread-z 1030 / trigger-depth
  1025), L1 triangulation 2nd-moment/OU (1031/2028), Q1 peg-defense (1028), S1 take-the-loss geometry
  (subsumed by fbr), G1 dollar-factor residual (2018/2031). Remaining unlocks are operator-gated
  (NEEDS_ENABLEMENT: co-sim item E [BUILT/in-review], passive-limit-fill item D, options/macro data).
- **Honest-exit + OOS collapse (arc 1046, the load-bearing recent finding):** under §5f-frozen exits
  scored on the 2021+ holdout, the whole 4-component portfolio's deploy value **collapses to its single
  robust leg, me_long.** fbr's IS −0.57 decorrelation (which lifts the 2-way IS Sharpe, 1045/2045) is
  **mean-NEGATIVE OOS** → adding it introduces neg OOS folds me_long-solo lacks. me_long-solo OOS:
  Sharpe ~0.51–0.53, **5/6 OOS years positive**, +0.33–0.45%/yr, t≈1.3 (borderline), NOT AFP (2021 neg).
  Deploy object went 4-way (1033/2033) → 2-way {me_long+fbr} (1045/2045) → **1-leg me_long-solo (1046)**.
- **§11 verification COMPLETE** for the committed book (signal 2034/2035, outcome 2036–2039/1038/1039,
  cost 2039) and for the 2-way deploy object at its DEPLOY configs (2046).
- **Deployability is GEOMETRY/VEHICLE-bound, not sign/depth-bound** (1033/2033/2045): the book is
  shallow-DD (maxDD ~1.6%) and daily-cap-SAFE but **Calmar ~0.24–0.36, ~99% time-underwater (~4.7–5.5yr
  deepest DD), prop-firm T_min ~1.4–8.4yr** → vehicle-infeasible as a challenge account; viable only as a
  low-risk already-funded slow diversifier. Calmar is risk-invariant (1024) → leverage can't rescue it.
- The operative lever is **operator path-A** (gate-governance / deploy call), repeated across the last ~15
  arcs. The leg-hunt is closed.

**The gap I picked (why this arc):** arc 1046 NAMED me_long-solo as the honest OOS deploy object but
profiled deployment GEOMETRY only for the **4-way** (1033/2033) and **2-way** (2045) books — **never the
solo object the operator would actually deploy.** The deploy object shrank 4→2→1 leg; the geometry/vehicle
axis was measured at 4 (1033) and 2 (2045, ≈ unchanged vs 4) but NOT at 1. This is genuinely owed and
decision-grade for the path-A call, and the geometry axis (Calmar / time-underwater / prop-firm T_min)
is NOT predictable from the mean/Sharpe axis: dropping fbr's decorrelation could LENGTHEN underwater /
deepen DD (the diversification that died OOS on *mean* may still have bought *DD-duration* value). This
arc quantifies what collapsing to the one robust leg costs/saves on the vehicle axis.

Fresh-eyes check for a genuinely-novel MECHANISM first (§5a): the forced-flow taxonomy (settlement gap /
WMR fix / stop-run + ~14 tested named flows) is exhausted; every remaining candidate I could name is
either covered, data-gated (options/equities/commodities — corpus is pure-FX, 1047), or closed-by-
generalization (info events priced efficiently, 1048). Forcing a weak-*because* mechanism would violate
§3. The decision-grade diagnostic is the higher-EV honest move.

## (b) IDEA
Profile me_long-solo (committed/robust exit `sl_only` + 2-bar time-exit, D1 USD majors — the 1042/1046
exit) as a STANDALONE deployment vehicle: its contiguous 2011–2020 IS equity curve's max-DD, Calmar,
time-underwater, daily-cap headroom, Sharpe/Sortino, and prop-firm-challenge T_min — the exact inputs
1033/2033 produced for the 4-way and 2045 for the 2-way, now for the 1-leg object. Compare against those
published book numbers to quantify the vehicle-axis cost/benefit of the honest collapse to one leg.
OOS-preserving: uses the already-measured 1046 OOS as characterization; touches no new OOS.

## (c) CHARACTERIZE / (g) MEASURE — me_long-solo contiguous 2011–2020 IS curve

Built via `discovery/tools/solo_deploy_profile.py` (BUILT this arc; canonical A1 + MultiPairBacktester
per-year scoring → re-id → `cosim_book_fold` single component weight=1.0 → BUILT `compute_risk_profile` +
`compute_sharpe` + `feasibility_horizon_years`). me_long committed/robust exit (`sl_only` + 2-bar
time-exit, D1 USD majors). **Anchor: per-year ROI reproduces the committed me_long EXACTLY** (matches the
`equity_risk_profile` REC me_long column to the pp): 2011 +0.40 / 2012 +0.29 / 2013 +0.96 / 2014 −0.23 /
2015 −1.14 / 2016 −0.51 / 2017 +0.34 / 2018 +0.90 / 2019 +1.16 / 2020 +0.15 → **mean +0.232% / sd 0.673% /
3-of-10 neg (2014/15/16) / worst −1.141% (2015).**

**Deployment-vehicle geometry (the never-before-computed solo numbers):**

| metric | me_long-SOLO (this arc) | 2-way {me_long+fbr} (2045) | 4-way book (1033/2033) |
|---|---|---|---|
| contiguous max-DD | **3.002%** (cap-off ≡ cap-on) | 1.559% (RP cap-on) | 1.59% (RP cap-off) / 1.53% |
| Calmar | **0.081 (cap-off) / 0.064 (cap-on)** | 0.245 | 0.36 (cap-off) / 0.24 (faithful) |
| Sharpe (daily, ann) | **0.213 (cap-off) / 0.175 (cap-on)** | 0.459 / 0.305 | ~0.13–0.20 |
| time-underwater | ~99% (1859–1951d) | 99% (1999d) | 98% (1858d) |
| prop-firm T_min | **9.9–26.2 yr** | 4.08 yr | 1.4–8.4 yr |
| daily 5% cap | never binds (worst-day ~0%) | never binds | never binds |

Deepest solo DD: **peak 2014-10-01 → trough 2017-03-06** = the 2014/15/16 strong-USD block chaining
contiguously (which the per-year table's −1.14% worst-fold structurally HIDES — the arc-1033 lesson, now
sharper for one leg). cap-on n_dropped=2 (2-per-ccy cap trivially binds on a 7-USD-pair leg), ret
+0.191% vs cap-off +0.242%, maxDD identical 3.00%.

## (e) DIAGNOSE — the two deploy axes pull in OPPOSITE directions; the 2→1 collapse breaks the vehicle axis

The mean/OOS axis (1046) and the deployment-vehicle/geometry axis pull OPPOSITE ways:
- **Mean/OOS axis (1046):** solo > 2-way > 4-way — gap/me_short/fbr net-SUBTRACT on mean and die OOS, so
  the honest OOS-survivor is me_long-solo.
- **Vehicle/geometry axis (this arc):** 4-way ≈ 2-way ≫ solo — the book's max-DD is ~1.6%, the solo leg's
  is **3.0% (~2× deeper)**; Calmar 0.24–0.36 (book) vs **0.06–0.08 (solo, ~3× worse)**; T_min ~4yr (book)
  vs **~10–26yr (solo)**.

**Mechanism (clean + specific):** arc 2045 found the **4→2** collapse left geometry ~unchanged (gap+me_short
didn't cover me_long's hole — gap is JPY-cross, me_short fires opposite months). This arc finds the **2→1**
collapse is where the vehicle geometry BREAKS, because **fbr's IS value for the book is concentrated
exactly on me_long's deepest hole — fbr is +3.17% in 2015 (me_long's −1.14% worst fold) and net-positive
across the 2014–16 block** → in the book, fbr fills me_long's 2014–16 contiguous trough, holding book
max-DD to 1.6%; dropping fbr (because it's mean-NEGATIVE OOS, 1046) re-exposes the full 2014–16 chain →
solo max-DD 3.0%. **So fbr's diversification value for me_long was DRAWDOWN-DEPTH cover, NOT mean — and
that is precisely the value that does not survive the holdout (1046).** The 1046 logic (drop fbr, it's
mean-neg OOS) and the vehicle logic (keep fbr, it halves the IS drawdown) are in direct tension; both,
however, still land outside the prop-firm-vehicle envelope (solo Calmar 0.06–0.08 / T_min 10–26yr is the
WORST in the corpus; the book's 0.24 / ~4yr was already infeasible, 2033/2045).

## (h)/(i) VERDICT — DIAGNOSTIC → KILL (no new component; me_long & all components UNCHANGED, PORTFOLIO)

No council (diagnostic, no survivor, no idea-fork). No canonical change. No OOS spent (IS geometry only;
1046 OOS cited as characterization). Components UNCHANGED. New BUILT tool `solo_deploy_profile.py`
registered (step i). Deployable-system count = 0.

**NEW LESSON.** For a thin geometry-bound multi-component book, the *honest-OOS-survivor* selection (drop
the legs that net-subtract on mean / die OOS — arc 1046) and the *deployment-vehicle geometry* selection
(keep the legs that reduce contiguous drawdown) **pull in opposite directions**, because a decorrelating
leg's IS contribution can be drawdown-DEPTH cover rather than mean — and depth-cover is exactly the kind
of IS-diversification benefit that need not survive the holdout (here fbr covers me_long's deepest 2015
hole on IS but is mean-NEGATIVE OOS). Collapsing a book to its single OOS-robust leg therefore makes the
MEAN axis more honest (1046) while making the VEHICLE axis strictly WORSE: a single lumpy single-flow
reversion leg has the corpus's worst vehicle geometry (max-DD ~2× the book, Calmar ~3× worse, T_min
~10–26yr) precisely because it has no decorrelating partner to fill its multi-year underwater stretches.
Deploy-object selection cannot optimize one axis without costing the other; **both axes still land
outside the prop-firm-vehicle envelope at every leg-count → the lever stays operator path-A**, and the
honest deploy framing is: me_long-solo is a mean-honest but vehicle-worst ~0.23%/yr (IS) / ~0.33–0.45%/yr
(OOS, 1046) slow already-funded diversifier, NOT a challenge-account strategy. This sharpens arc 2045's
"4→2 geometry unchanged" with "**2→1 geometry breaks** (the dropped fbr was drawdown-depth cover, not
mean)" and completes the deploy-object geometry analysis at all three leg-counts (4 / 2 / 1).

**Datum banked:** me_long-solo IS contiguous maxDD 3.00% / Calmar 0.06–0.08 / underwater ~99% / prop-firm
T_min 10–26yr; deepest DD = 2014-10→2017-03 (the 2014–16 strong-USD block chained).

