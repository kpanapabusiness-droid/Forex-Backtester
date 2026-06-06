# arc 2059 — 2-leg (me_long + one partner) OOS deployment-vehicle geometry

**Chat:** 2000s | **Date:** 2026-06-06 | **Disposition:** DIAGNOSTIC → KILL (no new component;
updates the deploy recommendation; lever unchanged = operator path-A)

## Step (a) — log read (fresh eyes, honest-era only)

Pulled main. Read DISCOVERY_PROTOCOL, LESSONS, TOOL_REGISTRY, and the Tier-1 ledger + recent Tier-2.
Synthesis of the honest-era corpus as it stands at arc 2058:

- **4 PORTFOLIO components** (mean-positive net of costs, none all-folds-positive): `gap` (1006,
  weekend down-gap fill, JPY crosses H4), `me_long` (1011, month-end WMR reversion long, USD majors
  D1), `fbr` (1013, failed-breakdown reclaim long, USD majors H4), `me_short` (1019, month-end
  reversion short).
- **Deploy object = `me_long`-SOLO** (1046): the only OOS-mean-robust leg; the others die on OOS
  mean. Vehicle-infeasible (~0.3–0.5%/yr, T_min ≥ 2.4yr).
- **The 4-way book fails all-folds-positive at the 2018 wall** — a combination-invariant noise floor
  (1015/2008/2016/2022); strong-USD risk-off year where every reversion leg trends instead of
  reverts. No OHLC ex-ante regime separator flips it (regime conditioning failed on dispersion /
  vol-level / Kaufman-ER; fbr-2018 unconditionable per-pair [2014] AND USD-factor [2052]).
- **Edge frontier verified mined out (every axis):** relative-value (2010, doubled-cost-vs-coin-flip),
  all three short asymmetries (climax 2009 / reject 2011 / up-gap 2013), the calendar series
  (gotobi / fiscal-YE / daily-fix / NFP / IMM-roll / turn-of-quarter / day-of-week 3015), fbr
  conditioning (per-pair / level-significance / universe / finer-TF / USD-factor), option-B
  thickening (2018 / 2051). The 1000s chat already handed off on this basis; the 2000s lane's
  remaining value is **decision-support diagnostics**.

**The one genuinely-unmeasured, decision-relevant cell:** the deploy-vehicle matrix had **solo**
(1-leg me_long: 2053 IS / 2055 OOS) and the **full 4-way book** (2054 IS / 2056 OOS), but NOT the
interpolating **2-leg** cell on OOS. The operator's rational path-A deploy choice is `me_long` (the
sole OOS-mean-robust anchor) PLUS *at most one* decorrelating partner for drawdown-depth cover. The
question: **does any 2-leg book DOMINATE — better OOS Calmar than me_long-solo (0.31, 2055) WITHOUT
collapsing to the 4-way book's halved OOS mean (~+0.25%/yr, 2056, dragged by three weak legs)?**

## Step (b/c) — idea + apparatus

Not a new edge — a deploy-geometry completion. Built `discovery/tools/subset_deploy_profile.py`, the
parameterized generalization of arc-2056's `book_deploy_profile.py` (the ONLY change: `NAMES` is any
subset; every measurement primitive — `build_component`, `cosim_book_fold`, `fit_weights`,
`compute_risk_profile`, `compute_sharpe`, `feasibility_horizon_years` — reused UNCHANGED). GEOMETRY
ONLY on an already-computed contiguous co-sim equity curve; never realizes P&L / scores / touches the
gate / *selects* on OOS.

**§4 compliance:** RP weights fit on the subset's IS per-year ROIs and **frozen**, applied unchanged
to OOS; exits are the committed/frozen per-component configs; the per-component frozen OOS series were
already spent in 1042/2055/2056, so reading 2-leg geometry off them adds NO selection (2055/2056
precedent). OOS = measure-once characterization.

Ran the three `me_long`-anchored 2-leg books: `me_long+fbr`, `me_long+gap`, `me_long+me_short`.

## Step (g) — result

**OOS (2021–2026), RP frozen-IS weights, cap-ON faithful (the actual deploy config):**

| Vehicle | ret %/yr | maxDD % | Calmar | T_min (FN 2-step) | neg/6 | worst % |
|---|---|---|---|---|---|---|
| me_long-solo (2055) | +0.487 | 1.75 | 0.310 | 2.4 yr | 1 | −0.63 |
| **me_long + fbr** | **+0.697** | 1.72 | **0.405** | **2.0 yr** | 2 | −0.20 |
| me_long + gap | +0.459 | 1.84 | 0.250 | 3.2 yr | 2 | −1.08 |
| me_long + me_short | +0.411 | 1.45 | 0.284 | 2.8 yr | 4 | −1.29 |
| 4-way book (2056) | +0.47 | 1.43 | 0.328 | 2.4 yr | 2 | −1.16 |

**IS (2011–2020), RP cap-ON, for context (bracket):** me_long+fbr Calmar 0.217 / me_long+me_short
0.232 (best IS partner) / me_long+gap 0.069.

### Three findings

**(1) `me_long+fbr` is the BEST OOS vehicle in the entire deploy matrix** — Calmar **0.405**, ret
**+0.697%/yr**, T_min **2.0 yr** — strictly beating me_long-solo (2055: 0.31 / 0.49% / 2.4yr) AND the
4-way book (2056: 0.328 / 0.47% / 2.4yr) AND the other two 2-leg books on Calmar, return, and T_min.
This **updates the deploy recommendation**: *if* deploying, the best vehicle is the 2-leg
`{me_long, fbr}` (RP cap-ON: me_long 0.812 / fbr 0.188), NOT me_long-solo (the prior named object,
1046) and NOT the full 4-way book.

**(2) `fbr` is the UNIQUE value-adding partner — gap and me_short DEGRADE me_long-solo.** Adding gap
(Calmar 0.310→0.250) or me_short (0.310→0.284) makes the OOS vehicle *worse* than solo; only fbr
improves it. Why: fbr is the only leg with a large **decorrelated positive tail (2024 raw leg
+7.61%) landing in a year me_long is flat** (2024 me_long +0.54%) → it lifts RETURN (0.49→0.70%/yr)
more than it adds DD, at a low RP weight (0.188) that dampens its OOS-negative years. gap and me_short
are OOS-negative-heavy (me_short 4/6 neg, gap deep 2024-07→2025-12 DD) with no compensating
decorrelated tail, so they drag Calmar below solo. This **refines arc 2053**: there, fbr's value for
me_long was *IS-trough depth-cover* (filling me_long's 2014–16 hole) — a property 2053 said "dies
OOS." On OOS the partner is the SAME (fbr) but the mechanism SHIFTS to a 2024 **return-tail** (the
2014–16 trough doesn't recur, 2055). Same leg, different *why*, IS→OOS.

**(3) IS-best partner ≠ OOS-best partner** — re-confirms 2054 at the 2-leg level. By IS Calmar the
best partner is me_short (0.232) ≈ fbr (0.217); by OOS Calmar it is decisively fbr (0.405) ≫ me_short
(0.284). The partner whose value is a fat tail (fbr) ranks by which window's tail happened to be
large; **IS standalone/2-leg vehicle quality is anti-predictive of OOS durability** (2054's lesson,
now on the partner-selection axis).

### The honest KILL (why this is still not a challenge strategy)

`me_long+fbr` is a better VEHICLE but does NOT change the verdict:

- **STILL vehicle-infeasible.** T_min 2.0 yr ≫ a prop-firm challenge's weeks-to-months; the daily-5%
  cap never binds (maxDD/Calmar is the wall, not the daily limit).
- **NOT all-folds-positive OOS** — 2/6 neg (2021, 2026).
- **Doubly fat-tail-fragile.** The OOS book rests on TWO single-year tails in DIFFERENT years —
  fbr's 2024 (+7.61% raw leg) and me_long's 2025 (+2.26%). Book per-year (RP): −0.20 / +0.18 / +0.35
  / **+1.87** / **+1.32** / −0.09. Ex-2024 → +0.31%/yr; **ex-both-tail-years → ≈+0.06%/yr (flat).**
  The two tails landing in different years is a mild tail-timing diversification (the 2-leg has
  positive years more reliably than either leg alone) but the LEVEL is tail-carried, not broad-based
  — the exact arc-2054/2055 "high Calmar from a least-repeatable fat tail" pattern, now compounded
  across two legs.

So: a genuinely better deploy vehicle on every summary axis, still outside the prop-firm-challenge
envelope and tail-fragile → **path-A verdict UNCHANGED**; an honest read is a low-risk ~0.5–0.7%/yr
already-funded **diversifier** (deploy as `{me_long, fbr}` RP cap-ON, not solo), not a challenge
strategy.

## NEW lesson

**The deploy matrix's missing 2-leg cell is its SWEET SPOT, but only with the right partner — and the
right partner is the one with a decorrelated TAIL, not the best IS vehicle.** One decorrelating
partner (fbr) lifts me_long's OOS return via a fat tail without the full 3-weak-leg mean drag of the
4-way book → `me_long+fbr` beats BOTH solo (2055) and the 4-way book (2056) on OOS Calmar / return /
T_min, making it the corpus-best deploy vehicle. But (i) only fbr helps — gap/me_short DEGRADE solo
(no compensating tail); (ii) the lift is doubly fat-tail-fragile (2024 fbr + 2025 me_long; ex-both ≈
flat); (iii) IS-best partner (me_short) ≠ OOS-best partner (fbr), re-confirming 2054's "IS vehicle
quality anti-predictive of OOS durability" on the partner axis. The deploy *object* should be named
the 2-leg `{me_long, fbr}` on the VEHICLE axis even though `me_long`-solo is the mean-honest object
(1046) — the two axes name different objects, completing the 2053 "honest-OOS-survivor vs
vehicle-geometry pull opposite" thread at the partner-selection level. Still vehicle-infeasible at
every leg-count (best T_min 2.0yr); lever = operator path-A.

## Bookkeeping

- **Disposition: KILL** (DIAGNOSTIC — no new component, no PASS; the 4 components UNCHANGED, all
  PORTFOLIO). Updates the deploy *recommendation* (best vehicle = `{me_long, fbr}` 2-leg, not solo).
- Completes the deploy-vehicle matrix on its last interpolation axis (solo / **2-leg** / 4-leg × IS /
  OOS).
- **BUILT tool registered:** `subset_deploy_profile.py` (geometry-only; parameterized N-leg subset
  generalization of `book_deploy_profile.py`).
- No council (cheap diagnostic, no survivor, §5d/§5g). No canonical change, no FLAG. OOS = measure-once
  characterization (§4; frozen IS weights + committed exits, already-spent series). No engine/gate
  touch.
- Data: canonical loader (histdata_backup) + cache; reproduces committed per-year component ROIs.
