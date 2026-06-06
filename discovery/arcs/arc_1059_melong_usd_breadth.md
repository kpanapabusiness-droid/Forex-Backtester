# arc 1059 — USD-trend-BREADTH conditioning of me_long (the deploy object)

**Chat:** 1000s · **Type:** obs cheap-kill (§5d), OOS-preserving · **Disposition:** KILL ·
**Deployable count:** 0 · Driver: `discovery/_disco1_work/arc1059_melong_usd_breadth.py` (BUILT tools only;
single-use diagnostic, no new BUILT registry entry — arc-1029/1053/1055 precedent). No engine / null /
council; no canonical change; no FLAG; OOS untouched.

## Fresh-eyes read (step a)

Pulled main; read DISCOVERY_PROTOCOL, the Tier-1 table + recent Tier-2 (1050-1058 + sibling 2053-2058),
LESSONS, TOOL_REGISTRY. State at resume: the corpus is at a **terminus** — 4 PORTFOLIO components
(gap/me_long/fbr/me_short), the 4-way book mean-positive but never all-folds-positive (the 2015/2018
wall); the §5f exit-honesty thread (2040-2046/1042-1046) collapsed the honest deploy object to
**me_long-solo**; the deploy-vehicle matrix is complete at every leg-count × IS/OOS (vehicle-infeasible,
Calmar 0.06-0.34, T_min ≥ 2.4yr); §11 verification is complete end-to-end; and the strategist explore-now
EDGE MENU is **exhaustively closed** on every axis — directional (1052), calendar/forced-flow (a 4-mode
failure taxonomy: sub-cost / priced-in / instrument-neutral / fat-tail-mirage, 2026/2057/2058),
convergence/stat-arb (1054), inelasticity (1055), relative-value/market-neutral (2003/2010/2018 =
doubled-cost-vs-coin-flip, NOT the long-only block). The lever is the operator's **path-A** gate-governance
call. Both chats converged on this. The honest remaining within-charter value is decision-support on the
characterized object — but per §5a fresh-eyes + the arc-3004 warning against prematurely concluding "the
apparatus is incapable," a genuinely-novel, mechanistically-grounded test of the SURVIVING edge still
earns a fresh look.

## Idea + why (step b) — genuinely untested, pointed-at by arc 2018

me_long buys a pair that fell ≥1 ATR into month-end, betting the post-fix WMR rebalancing reverts it UP.
Its known weakness is the contiguous **2014/2015/2016 strong-USD-bull IS hole** (−1.14/−0.51/−0.23); the
signal docstring's hypothesized *because* is "in a strong-USD trend a big down move into ME IS the trend →
continues, doesn't revert." Arc **2018** found me_long carries a REAL positive 2018 via **directional
WMR**, and that stripping USD beta (cross-sectional/USD-neutral) REMOVES the +2018 help → me_long's edge
has a directional-USD component.

So the untested refinement: condition me_long on the **BREADTH / strength of the common USD move into
month-end**. This conditioner is genuinely new — distinct from every me-conditioner already mapped:
calendar-density (1029), trigger-depth (1025), spread-z inelasticity (1055), trend-SMA (1012),
quarter-end (1012/1029). Hypothesis (falsifiable): (A) capture DECREASES with |common-USD-move| (a strong
broad USD move = a TREND that continues → reversion fails); (B) per-fire IDIOSYNCRATIC fires (the pair fell
against the common USD direction → pair-specific over-extension) REVERT, SYSTEMATIC fires (with the herd)
CONTINUE; (C) the negatives concentrate in the high-breadth bucket so an idiosyncratic-only filter lifts
the worst fold. Failure modes: flat/non-monotone (arc-3007 not-a-lever) OR the favorable bucket thins
folds (1029/1055 concentration-death).

## Method

BUILT-tools-only obs cheap-kill: `MonthEndReversionLongSignal` (1011) fires + honest take-the-loss
capture/fwd-drift via `observe_long_capture` (1007), IS 2010-2020, D1, 7 USD majors. USD-signed 2-day
move into each month-end (XXXUSD up = USD down; USDXXX up = USD up); **common_usd_move** = the mean
USD-signed move across the 7 majors at that month-end; **breadth = |common_usd_move|** (trend strength).
Each of the n=126 me_long fires tagged with its month's breadth + its own USD-signed move (systematic =
sign agrees with common). Grouped by breadth tercile, by systematic/idiosyncratic, and per-year fold-sign.

## Result

**(A) Capture declines MONOTONICALLY with USD-breadth — the mechanism is real:**

| breadth tercile | n | capture | drift_mean | drift_med |
|---|---|---|---|---|
| LO (|common| 0.01-0.52) | 42 | **0.595** | +0.176 | +0.292 |
| MID (0.53-0.88) | 42 | 0.548 | +0.146 | +0.202 |
| HI (0.91-2.44) | 42 | **0.405** | +0.113 | +0.070 |

The HI bucket (strong broad USD move) is **below coin-flip** (0.405) — me_long fails precisely when the
move is a strong broad USD trend. Unconditional capture 0.516.

**(B) Per-fire systematic/idiosyncratic is a WEAK split:** idiosyncratic (n=22) capture 0.500 / drift
+0.311; systematic (n=104) capture 0.519 / drift +0.110. Idiosyncratic carries higher drift but coin-flip
capture and is thin — the month-level breadth (A) is the cleaner separator. The **2014-16 block** confirms
the *because*: there, SYSTEMATIC fires (n=23) drift **−0.117** (trend continues = the hole), IDIOSYNCRATIC
fires (n=5) drift **+0.335** (revert) — right sign, but 5 fires over 3 years.

**(C)/(D) No fold-resolution lever in either filter direction:**
- **Concentrate** onto idiosyncratic (n=22 total, median 2/yr; 2017 has ZERO) or LO-breadth (n=42,
  neg-years **4/11** worse than ALL 3/11) → the 1029/1055 thinning death; 2014-16 stays mixed-negative on
  1-4 trades/yr.
- **Exclude only the worst HI-breadth tercile** (the §5f "best version", keeps n=84, ~2/3, no thinning):
  pooled capture IMPROVES 0.516→**0.571**, but per-year fold resolution gets **WORSE** — neg-years
  3→**4/11**, worst-year −0.237→**−0.317**. The reason is decisive: dropping HI-breadth fires removes
  me_long's **directional-WMR +2018 help** (2018 drift +0.060 → **−0.317**; 2011 +0.155 → −0.037) —
  exactly arc 2018's "stripping the USD-beta component removes the +2018."

## Diagnosis + meaning

USD-breadth is a **real, monotone, mechanistically-correct per-trade separator** (capture falls cleanly
with breadth; the HI tercile is sub-coin-flip) that data-confirms both the signal docstring's hypothesized
*because* and arc 2018's directional-WMR finding — **but it is NOT a deployable worst-fold lever.**
me_long's 2014-16 failure (systematic trend-continuation) and its +2018 help (directional WMR reversion)
live on the **SAME high-USD-breadth exposure**: concentrating onto low-breadth thins folds to noise, and
excluding high-breadth improves per-trade capture only by **removing the +2018 fires that the worst fold
needs**. Conditioning can trade 2014-16 risk for 2018 risk; it cannot lift the worst fold. me_long is
UNCHANGED (PORTFOLIO; the honest deploy object, 1046/1053). The §5f exit-sweep is moot — the conditioning
fails at the fold-SIGN level (which years are pos/neg), which no exit/SL choice can move, and me_long's own
exit is already settled-honest (committed sl_only/2-bar, 1012/2042).

## NEW lesson

The "edge & tail are the same exposure" closure (arc 1018 universe / 2018 cross-sectional) extends to
**state-conditioning the deploy object**: me_long's worst-fold failure (2014-16 strong-USD systematic
trend) and its best directional help (+2018) are the **same USD-trend-breadth exposure** measured at
opposite outcomes, so a breadth filter that removes the failure necessarily removes the help — it trades
one bad fold for another and cannot lift the worst fold. A conditioner can be a genuine, monotone per-trade
separator (capture 0.60→0.40 across breadth) and STILL be a non-lever, because per-trade capture ≠
per-year fold sign when the edge and its regime-tail share an exposure. Adds USD-trend-breadth to the
mapped-dead me-conditioner list (density 1029 / depth 1025 / spread-z 1055 / trend-SMA 1012 / quarter-end
1012-1029): every state-conditioning of the surviving reversion edge either thins folds or removes the help
it needs — the deploy object's worst fold is structurally unfilterable within OHLC.

## Threads / handoff

Components UNCHANGED (all 4 PORTFOLIO; me_long-solo the honest deploy object, vehicle-infeasible). Lever =
operator path-A. This arc adds a data-confirmation of arc 2018's directional-WMR mechanism and a closure on
the last natural me-conditioner (breadth). Within the OHLC-only charter the EDGE frontier stays mined out;
the next genuinely-new direction needs a charter unlock (operator-gated macro/options data,
`NEEDS_ENABLEMENT.md`) or the operator's path-A decision. **Datum:** me_long capture by USD-breadth tercile
0.595/0.548/0.405 (monotone, HI sub-coin-flip); excluding HI improves pooled capture 0.52→0.57 but worsens
worst-year −0.24→−0.32 by removing the +2018 directional help.
