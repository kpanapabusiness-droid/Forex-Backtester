# Arc 1030 — O1: inelasticity-STATE conditioning via the entry-bar SPREAD-Z proxy

> **Arc id:** 1030 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-06
> **Final verdict:** **KILL (cheap-kill at observation)** — the O1 "wide spread = inelastic book =
> bigger capturable reversion" thesis is FALSIFIED, and **INVERTS** on the corpus's only fold-resolving
> edge. Within `fbr` (failed-breakdown reclaim), per-trade capture & drift are HIGHEST in the
> **tightest**-spread entries (T1 capture 0.61 / drift-median +0.74 ATR) and lowest/NEGATIVE in the
> **widest**-spread entries (T3 capture 0.56 / drift-median −0.28 ATR); corr(spread, drift) **−0.18**
> (spread-z) / −0.01 (spread/ATR) — monotone DOWN, the opposite of O1's prediction. Within the
> month-end legs it is NON-monotone (T2 mid-tier peak — the arc-3007 "not-a-lever" tell), corr ≈ 0.
> Wide-spread fbr losses CLUSTER in 2016/2018/2020 (risk-off / strong-USD) → conditioning on
> inelasticity would CONCENTRATE the binding-fold losses, not separate them. And the **cost trap is
> real and doubly fatal**: the wide-spread tier carries ~7× the spread/ATR (0.13 vs 0.019) for a WORSE
> gross edge. **Closes O1 entirely** — all three collinear proxies now dead: calendar-density (arc
> 1029), trigger-depth/shallowing (arc 1025), inelasticity-state/spread-z (this arc).
> **Idea source:** `DISCOVERY_DIRECTION.md` MENU item **O1** (strategist council's #2-EV `explore-now`,
> the one item that attacks the book's actual blocker — per-year fold resolution), via its last untested
> proxy. Arc 1029 (same chat) closed the calendar-density proxy and flagged spread-z as "the only
> untested O1 sub-thread"; this arc closes it.

Scored by the direction-aware honest capture/drift harness (`observe_long_capture`, take-the-loss
`reached_1r_before_sl`) over the canonical H4/D1 USD-major panels (real bid/ask, 5ers_eet). No
pool/engine — the observation is decisive (§5d): the conditioning is wrong-signed (fbr) / non-monotone
(me), so there is no above-baseline best-version to carry to the engine (§5f does not bite). OOS never
touched.

## (a) Log read + synthesis

Pulled main; read DISCOVERY_PROTOCOL, the full Tier-1 ledger (arcs 0–3022), LESSONS, DISCOVERY_DIRECTION,
TOOL_REGISTRY. State of the corpus (honest-era, ~95 arcs):

- **The route has converged on a 4-component PORTFOLIO** — gap-fill (1006, JPY-cross H4 long), me_long
  (1011, USD-major D1 long), fbr (1013, USD-major H4 long, the STRONGEST/only fold-resolving edge), and
  me_short (1019, USD-major D1 short, the first robustly +2018 leg). The book is mean-positive
  (t=2.66, arc 1023), cost-robust (break-even κ=3.32, arc 3022), temporally stable (arc 2021), ~3
  independent bets (ENB 3.32/4, arc 2019) — but **FAILS the per-year all-folds-positive gate**, blocked
  by MARGINAL 2015/2016 folds that sit BELOW the components' noise floor (arcs 2016/2017).
- **Path-B (add decorrelated legs) is quantitatively CLOSED** (arc 3021: at the empirical residual
  correlation ρ≈0.115 the book Sharpe ceilings below the gate at ANY N; arc 2022: even a perfect
  maximally-decorrelated 5th leg fails honest weighting). **Edge-hunting for the book is closed**; the
  sole deployability lever is the operator's **path-A gate-governance call**.
- **fbr's −2018 is mechanism-intrinsic / entry-unconditionable** across FOUR prior axes — daily regime
  (2014), per-fold CI (2017), M1 reclaim-quality (2020), trigger-depth (1025), cross-pair breadth (3020).
- **O1 (inelasticity-state concentration)** is the MENU's one item attacking the actual deficit (fold
  resolution). Its three council-named proxies are collinear ("the same variable in three costumes —
  build it once"): calendar-density (arc 1029 → KILL, concentration WORSENS fold resolution),
  trigger-shallowing/depth (arc 1025 → fbr doesn't thicken), and **entry-bar spread-z** — the
  lag-free, calendar-ORTHOGONAL proxy, **still untested** and flagged a cost-trap. This arc tests it.

FRESH EYES: pre-reset conclusions ignored; only honest-era log carries weight.

## (b) Idea + because

**O1 spread-z hypothesis.** Forced flows move price more per unit when the book is **inelastic** (thin
depth); the subsequent reversion is then larger. A wide bid-ask spread at the dislocation bar is a
**lag-free proxy** for how inelastic the book was. **Because** the surviving edges (fbr, me) are
forced-flow reversions that infer inelasticity only from price/calendar, the spread (orthogonal, book-
state) should add information: per-trade realized edge should rise **monotonically** with signal-bar
spread-z, and concentrating risk onto high-spread-z entries should raise the worst-fold ROI — a direct
attack on the fold-resolution deficit that blocks the book.

**Falsifiable prediction (O1).** Within fbr / month-end, per-trade capture & drift rise T1 < T2 < T3 in
spread-z, AND the top tier separates fbr's binding 2018 fold (wide-spread 2018 reclaims hold better).
**Falsifier:** non-monotone (arc-3007 not-a-lever tell) OR wrong-signed OR 2018 not separated. **Cost
discipline (the trap):** wide-spread bars cost the most — any positive obs must clear the realized wide
spread; the gross lens here is only a screen, and a gross failure is already decisive.

Causal/no-lookahead: spread = `close_ask − close_bid` at signal bar t (known at t close; entry t+1);
spread/ATR uses ATR shift1; spread-z is a trailing 250-bar rolling z-score (shift1, std-floored to kill
divide-by-≈0 flat-spread HistData windows). Conditioner read at the signal bar, never the future.

## (c)/(d) Observation → verdict (cheap-kill)

Conditioner built two ways (clean): **spread/ATR** (spread in price-vol units — the direct inelasticity
measure) and **spread-z** (trailing rolling z-score). Terciles of each, on the corpus's fold-resolving
edge (`fbr`, K40/shadow1.25, H4 USD majors, n=237) and the pooled month-end legs (me_long fades a DOWN
move, me_short a UP move into ME; D1 USD majors, thr 1.0/into2, n=240). IS 2010–2020.

**fbr — spread-z terciles (n=50 each):**

| tier | capture | drift (mean) | drift (median) | spread/ATR |
|---|---|---|---|---|
| T1 (tightest) | **0.60** | **+0.361** | **+0.533** | 0.026 |
| T2 (mid) | 0.60 | +0.027 | +0.208 | 0.032 |
| T3 (widest) | **0.54** | **−0.767** | **−0.310** | 0.070 |

corr(spread-z, capture) **−0.040**, corr(spread-z, drift) **−0.176**. **Wrong-signed — monotone DOWN.**

**fbr — spread/ATR terciles (n=79 each):** T1 cap **0.608** / drift-median **+0.739** → T3 cap 0.557 /
drift-median **−0.276**; corr ≈ 0/−0.025. Same inversion.

**fbr 2018 separation (the binding fold):** the wide-spread (T3) 2018 entries are drift −2.36 (spread-z)
— still a wipeout; the negative wide-spread drift CLUSTERS in **2016 (−3.84), 2018 (−2.36), 2020 (−3.71)**
— the risk-off / strong-USD years. Conditioning on wide spread would CONCENTRATE the losses.

**Month-end pooled — spread/ATR terciles (n=80 each):** cap 0.488 → **0.563** (T2) → 0.538 — **NON-
monotone** (mid-tier peak = the arc-3007 "not-a-lever" tell); corr(spread/ATR, capture) +0.018 ≈ 0. The
slight drift rise (T1 −0.03 → T2 +0.27 → T3 +0.07) is within-noise and the take-the-loss-relevant
capture metric is flat/non-monotone.

**Cost trap (decisive even had it worked):** the wide-spread tier carries spread/ATR ~0.13 vs ~0.019 in
the tight tier = **~7× the cost** — so concentrating onto inelastic bars pays ~7× more cost for a WORSE
(fbr) or non-monotone (me) gross edge. Doubly dead.

## (e) Diagnosis (the why — §2)

**Why does fbr's edge INVERT vs the O1 thesis?** fbr is a stop-run RECLAIM — a sweep of resting stops
below a swing low that *fails* (reclaims) and reverts. A **wide spread at the reclaim bar marks genuine
stress / informational, disorderly trading** (illiquid session, news, real selling pressure into a thin
book) → the "failed breakdown" is then more likely a REAL breakdown that resumes down → the reclaim
**fails** (exactly the 2016/2018/2020 risk-off clustering). A **tight spread marks an orderly, liquid
book** → the sweep was mechanical/liquidity-driven (the clean stop-run the edge is built on) → the
reclaim holds → the real fbr edge. **For a reclaim-type edge, book ELASTICITY (tight spread / orderly)
is the favorable state, NOT inelasticity** — O1's premise inverts. This generalizes arc 1018's "efficient
cross gaps" theme: the surviving edges key off ORDERLY mechanical dislocations; stressed/inelastic
conditions degrade them. (No council: the obs is wrong-signed/non-monotone, decisive §5d like arcs
1029/2014/2020/3020/1025.)

## (f)/(g) Best version / validation

§5f does NOT bite: the conditioning is wrong-signed (fbr) / non-monotone (me) → there is no above-baseline
best-version filter to carry to the honest engine (and the base edges are already engine-validated
PORTFOLIO components; this arc only asked whether spread-z IMPROVES them, and it does not). No engine, no
null, no council spent — a §5d cheap-kill of a refinement. OOS never touched.

## (h) Council

Not invoked (cheap-kill at observation; nothing reached WFO/survivor stage).

## (i) Disposition + NEW lesson

**KILL (cheap-kill at observation).** Components UNCHANGED (all 4 still PORTFOLIO). The O1 inelasticity-
state route is now **closed on all three collinear proxies** (calendar-density 1029 / trigger-depth 1025 /
spread-z 1030) — concentration cannot buy fold resolution by any of them.

**NEW lesson.** The O1 "wide spread = inelastic book = bigger CAPTURABLE reversion" thesis **inverts for
stop-run-RECLAIM edges**: a wide spread at the reclaim bar marks genuine *stress* (disorderly/
informational), under which the failed breakdown becomes a real breakdown and the reclaim FAILS — so
fbr's edge lives in the ORDERLY / elastic (tight-spread) state, and its wide-spread losses cluster in
exactly the risk-off years (2016/2018/2020). For the month-end legs the spread axis is a non-monotone
"not-a-lever." Combined with the structural cost trap (wide-spread bars cost ~7×), **inelasticity-state is
not a fold-resolution lever in any proxy.** The corpus's surviving edges key off ORDERLY mechanical
dislocations conditioned on a LARGE *price* displacement, not on book-state inelasticity. The route's
deployability lever remains the operator's path-A gate-governance call (arcs 2019/3021/2016/2017/1023/
2021/3022).

**Frontier note for the next arc:** the explore-now MENU's last named-live thread is **L1** (triangulation
residual SECOND moment / OU amplitude at finer resolution) — cost-walled (H1) but the variance dimension
is genuinely virgin; a cheap multi-resolution observation would close the triangulation door fully (level
3005/1027/2023 + variance). Everything else on the MENU is now tested/dead. No autonomous edge-hunt
remains above the noise floor; the lever is the operator's call.

## Reproduction

`discovery/results/arc_1030_o1_spread_z_inelasticity/observe_spread_z.py` (committed) — H4+D1 USD-major
panels from `C:\Users\panap\histdata_backup`, fbr K40/sh1.25 + me_long/me_short thr1.0/into2, IS
2010–2020. Output saved alongside as `output.txt`. Base conda env (pyarrow for parquet cache).
