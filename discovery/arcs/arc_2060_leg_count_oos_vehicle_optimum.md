# arc 2060 — leg-count optimum of the OOS deployment vehicle (is 2 the sweet spot?)

**Chat:** 2000s | **Date:** 2026-06-06 | **Disposition:** DIAGNOSTIC → KILL (no new component; closes
the leg-count optimum question; lever unchanged = operator path-A)

## Why (direct follow-up to arc 2059)

Arc 2059 found `{me_long, fbr}` is the best **2-leg** OOS vehicle, beating me_long-solo (2055) and the
full 4-way book (2056). It left one question open: **is 2 the optimal leg count, or does adding a 3rd
partner to the winning pair improve it further?** This closes that — a cheap completion reusing the
BUILT `subset_deploy_profile.py` (no new tool), running the two 3-leg books that CONTAIN the winning
pair: `{me_long, fbr, gap}` and `{me_long, fbr, me_short}`. §4 identical to 2059 (frozen-IS RP
weights, committed exits, OOS measure-once characterization off already-spent series).

## Result — OOS (2021–2026), RP frozen-IS weights, cap-ON faithful

| Vehicle | legs | ret %/yr | maxDD % | Calmar | T_min (FN 2-step) | neg/6 |
|---|---|---|---|---|---|---|
| me_long-solo (2055) | 1 | +0.487 | 1.75 | 0.310 | 2.4 yr | 1 |
| **me_long + fbr (2059)** | **2** | **+0.697** | 1.72 | **0.405** | **2.0 yr** | 2 |
| me_long + fbr + me_short | 3 | +0.526 | 1.34 | 0.393 | 2.0 yr | 3 |
| me_long + fbr + gap | 3 | +0.602 | 1.64 | 0.367 | 2.2 yr | 2 |
| 4-way book (2056) | 4 | +0.47 | 1.43 | 0.328 | 2.4 yr | 2 |

## Finding — Calmar is HUMP-SHAPED in leg count, peaking at 2

Both OOS Calmar (0.310 → **0.405** → 0.367–0.393 → 0.328) and OOS return (+0.487 → **+0.697** →
+0.526–0.602 → +0.47) **peak at the 2-leg `{me_long, fbr}` and decline monotonically as the 3rd and
4th legs are added.** So `{me_long, fbr}` is not just the best 2-leg — it is the **global OOS-vehicle
optimum across all leg counts (1/2/3/4)**.

**Mechanism (decomposes the hump):** the two forces that move with leg count pull opposite —

- **maxDD keeps falling with each added leg** (1.75 → 1.72 → 1.34/1.64 → 1.43%) — the staggered-trough
  depth-diversification of arc 2054 (more legs, troughs land in different years, shallower contiguous
  DD; the 3-leg `+me_short` 1.34% is the shallowest fbr-containing book). This force *wants* more legs.
- **return falls faster** because every partner beyond the first is a weak-OOS-mean leg (gap 2/6,
  me_short 4/6 neg) that drags the book mean more than its incremental DD-cut helps. This force *wants*
  fewer legs.

The FIRST partner (fbr) is special: it lifts return (its decorrelated 2024 +7.61% tail lands when
me_long is flat) MORE than it adds DD → Calmar rises 0.31 → 0.405. Every SUBSEQUENT partner only
drags mean → Calmar falls back toward the 4-way 0.328. Hence the peak at exactly 2. This is the
quantitative resolution of arc 2059's "fbr is the unique value-adding partner": not only does no other
*single* partner help — no *additional* partner helps either; the vehicle optimum is the single best
partner and stop.

## Still KILL (caveats unchanged from 2059)

The optimum is better but the verdict does not move: T_min 2.0yr (vehicle-infeasible — ≫ a
prop-firm challenge's weeks-to-months; daily-5% cap never binds at any leg count); NOT all-folds-
positive OOS (2/6 neg); the `{me_long, fbr}` lift is doubly fat-tail-fragile (fbr-2024 + me_long-2025;
ex-both ≈ +0.06%/yr flat, 2059). Path-A verdict UNCHANGED.

## NEW lesson

The OOS deployment-vehicle Calmar is **hump-shaped in leg count, peaking at 2** — adding the single
best decorrelated-tail partner (fbr) to the mean-honest anchor (me_long) maximizes the vehicle, and
every leg beyond that *lowers* Calmar (depth-diversification keeps cutting maxDD but the weak legs'
mean-drag dominates). The "more decorrelated legs is always a better book" intuition is FALSE on the
vehicle (Calmar) axis on this corpus: the 4-way book is the WORST vehicle of the lot (0.328), the
solo is second-worst (0.31), and the optimum is the 2-leg in between. Generalizes 2054
(depth-diversification) + 2059 (unique value-adding partner) into a leg-count optimum: deploy the
anchor + its single tail-decorrelating partner, not the full book. Still vehicle-infeasible at the
optimum; lever = operator path-A.

## Bookkeeping

Disposition KILL (DIAGNOSTIC — no new component; all 4 UNCHANGED, all PORTFOLIO). Confirms + sharpens
2059's deploy recommendation: the vehicle-axis deploy object is the 2-leg `{me_long, fbr}` (the global
leg-count optimum), the mean-axis object is me_long-solo (1046). Reused BUILT `subset_deploy_profile.py`
(no new tool). No council, no canonical change, no FLAG. OOS = measure-once characterization (§4).
