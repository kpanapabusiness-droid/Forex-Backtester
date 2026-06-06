# arc 2054 — per-leg standalone vehicle map: vehicle-quality is the REVERSE of OOS-deploy-priority; book DD is staggered-trough diversification that fails at 2018

**chat:** 2000s · **date:** 2026-06-06 · **disposition:** DIAGNOSTIC → KILL (no new component) · OOS-preserving (§5g)

## (a) LOG READING
Continues directly from arc 2053 (this chat, same loop, log re-read at 2053). Pulled main. No `discovery/STOP`.
State unchanged: 4 PORTFOLIO legs (gap/me_long/fbr/me_short), book not AFP (2015/2018 wall), 2018-leg
unfound (~22 routes), strategist menu closed, deploy object collapses to me_long-solo under honest OOS
(1046), deployability vehicle-bound (1033/2033/2045), lever = operator path-A. Arc 2053 just profiled
me_long-SOLO vehicle geometry (maxDD 3.00% / Calmar 0.06-0.08, ~2× deeper / ~3× worse than the book) and
found the honest-OOS-survivor axis and the vehicle-geometry axis pull OPPOSITE, with the **2→1 collapse**
breaking the geometry because the dropped fbr was DRAWDOWN-DEPTH cover of me_long's 2015 hole, not mean.

## (b) IDEA
Complete the per-leg vehicle map: run the BUILT `solo_deploy_profile.py` (arc 2053) on the other three
committed legs (fbr / gap / me_short) at their committed exits, to (1) quantify 2053's "decorrelation buys
depth-cover" mechanism across all legs, (2) give the operator the full standalone vehicle map, and (3) test
whether the per-leg drawdown windows are time-staggered (the mechanism that would make the book's maxDD
shallower than any single leg). Reuses the BUILT tool + canonical machinery; IS-only; zero new OOS.

## (c)/(g) RESULT — the four solo legs' contiguous 2011–2020 IS vehicle geometry

| leg | solo mean / neg / worst-fold | solo maxDD (off/on) | Calmar (off/on) | Sharpe-daily (off/on) | deepest-DD window | prop-firm T_min | OOS fate (1046) |
|---|---|---|---|---|---|---|---|
| **me_long** | +0.232% / 3 / −1.14 (2015) | 3.00 / 3.00% | 0.081 / 0.064 | 0.213 / 0.175 | 2014-10 → 2017-03 | 9.9–26.2 y | **SURVIVES** |
| **fbr** | +1.854% / 1 / −4.20 (2018) | 7.84 / 6.10% | **0.223 / 0.306** | 0.403 / 0.451 | 2017-10 → 2019-10 | 2.6–7.5 y | dies |
| **gap** | +0.685% / 5 / −6.79 (2018) | 12.73 / 11.38% | 0.060 / 0.032 | 0.153 / 0.085 | 2016-12 → 2019-08 | 13.3–51.8 y | dies |
| **me_short** | +0.683% / 3 / −0.91 (2016) | 3.96 / 4.07% | 0.149 / 0.143 | 0.261 / 0.252 | 2014-09 → 2017-09 | 5.4–11.7 y | dies |
| **4-way book (1033/2033)** | +0.58%/yr | **1.59 / 1.53%** | 0.36 / 0.24 | ~0.13–0.20 | 2014-10 → 2019-06 | 1.4–8.4 y | (collapses to me_long, 1046) |

All per-year ROIs reproduce the committed components EXACTLY (the REC table in `equity_risk_profile.py`).
cap-ON notes: fbr IMPROVES (maxDD 7.84→6.10, Calmar 0.22→0.31 — the 2-per-ccy cap, n_dropped=11, trims
its worst clustered fires); gap return WORSENS (0.77→0.37 — cap hurts gap); me_long/me_short ~neutral.
No leg breaches the daily 5% cap. RP weights (1033) heavily favor the low-vol legs (me_long 0.531 /
me_short 0.284) and down-weight the high-maxDD legs (gap 0.078 / fbr 0.107).

## (e) DIAGNOSE — three sharp results

**1. The book's maxDD (1.59%) is SHALLOWER THAN EVERY SOLO LEG (3.0–12.7%).** Diversification cuts
contiguous max-DD below even the best single leg (me_long 3.00%) by a further ~47%. The book's shallow DD
is a genuine diversification PRODUCT, not any leg's property — the strongest quantitative confirmation of
arc 2053's depth-cover mechanism. (Two drivers: RP down-weights the deep-DD legs gap/fbr, AND the troughs
are time-staggered — next.)

**2. The deepest-DD windows are TIME-STAGGERED → that is WHY book DD is shallow, but they all OVERLAP
2018 → that is the binding-fold tail-correlation.** me_long (2014-10→2017-03) & me_short (2014-09→
2017-09) trough in the **2014–17 strong-USD** window; fbr (2017-10→2019-10) & gap (2016-12→2019-08) trough
in the **2017–19** window. The two reversion-flow families bottom in DIFFERENT epochs → no common deep
trough → book maxDD 1.59% ≪ any solo. **But 2018 sits inside BOTH the fbr and gap deep-DD windows** — so
the staggering that protects DEPTH everywhere else FAILS at 2018, which is exactly the combination-invariant
AFP-blocking fold (1015/2008/3009: "avg-corr hid the shared 2018 tail"). Depth-diversification (staggered
troughs) and sign-diversification (the 2018 fold) are DIFFERENT properties — the book has the former, lacks
the latter — mirroring 2053's "depth-cover ≠ mean/sign-cover" at the book level.

**3. The standalone-vehicle-QUALITY ranking is the REVERSE of OOS-deploy-PRIORITY.** Ranked by solo IS
vehicle quality: **fbr (Calmar 0.22–0.31, T_min 2.6–7.5y) ≫ me_short (0.14) > me_long (0.06–0.08) ≈ gap
(0.03–0.06).** But 1046's honest-OOS ranking is the OPPOSITE: **me_long SURVIVES OOS; fbr/gap/me_short all
die** (mean-negative on the holdout). The leg with the BEST IS solo vehicle (fbr — highest mean, fewest neg
folds, best Calmar) is exactly the one whose edge does NOT survive the holdout (1013 thin-OOS, 2040
fished-exit-dies-OOS); the leg with the WORST IS solo vehicle (me_long — lowest mean, poor Calmar) is the
sole OOS survivor (the WMR month-end rebalancing mandate is the most structurally durable mechanism). So a
strong IS standalone-vehicle profile is, if anything, ANTI-predictive of OOS durability here — high IS
Calmar came from the high-variance fat-tail legs (fbr/gap), whose tails are the least repeatable
out-of-sample.

## (h)/(i) VERDICT — DIAGNOSTIC → KILL (no new component; all 4 components UNCHANGED, PORTFOLIO)
No council, no canonical change, no FLAG, no OOS spent (IS geometry; 1046 OOS cited as characterization).
No new BUILT tool (reused `solo_deploy_profile.py`, arc 2053). Deployable count = 0.

**NEW LESSON.** Across a thin reversion book the per-leg standalone vehicle-quality ranking can be the
REVERSE of the honest OOS-deploy-priority ranking — a strong IS standalone profile (high Calmar/Sharpe,
few neg folds) is driven by the high-variance fat-tail legs whose tails are the least OOS-repeatable, so
it is anti-predictive of holdout durability; never rank deploy candidates by IS standalone vehicle quality.
And the book's shallow contiguous max-DD (below ANY single leg) is a genuine diversification product of
TIME-STAGGERED per-leg drawdown troughs (reversion-flow families bottom in different epochs) — but that
depth-staggering is a DIFFERENT property from sign-diversification: it holds everywhere except the one
epoch (2018) whose trough is shared across legs, which is precisely the combination-invariant AFP wall.
Completes the deploy-object geometry analysis: solo-per-leg (this arc) + 1-leg me_long (2053) + 2-leg
(2045) + 4-leg (1033/2033) — the whole vehicle picture is now mapped, and every leg-count lands outside the
prop-firm-vehicle envelope (best solo Calmar 0.31 / book 0.36; T_min ≥ 2.6yr). Lever stays operator path-A.

**Datum banked:** solo vehicle geometry — me_long maxDD 3.00%/Calmar 0.06-0.08; fbr 6.10-7.84%/0.22-0.31;
gap 11.38-12.73%/0.03-0.06; me_short 3.96-4.07%/0.14-0.15. Book maxDD 1.59% < all solo. Deepest-DD windows
staggered (me_long/me_short 2014-17, fbr/gap 2017-19), all overlap 2018.
