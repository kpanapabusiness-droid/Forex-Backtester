# arc 2014 — Failed-breakdown reclaim long × downtrend-regime gate (improve arc 1013 toward solo-PASS)

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **KILL** (cheap-kill at observation) · **Council:** none (falsified at obs; no worthwhile-ceiling fork)

> Attacks the corpus's single most valuable open thread: arc 1013 (the strongest edge — IS 9/10 folds
> positive, the ONLY negative fold is 2018) is ONE fold from a solo PASS. arc 1013 thread #5 explicitly
> left open: "condition OUT the strong-USD regime with a PRE-REGISTERED causal measure (NOT fished to flip
> 2018, arc-1012 trap)." This arc tests the one regime conditioner obs#3/arc-3013 never tried.

---

## 1. READ + SYNTHESIZE THE LOG (step a)

Resumed 2000s range at arc 2014 (highest in-range id = 2013). Pulled main; no `discovery/STOP`. Read
DISCOVERY_PROTOCOL, the full Tier-1 ledger (0–3014), LESSONS, TOOL_REGISTRY, and the arc 1013 record.

State at entry (honest-era only, fresh eyes):
- **Closed ground:** shallow single-condition directional prediction (momentum/breakout/reversion/trend),
  longs OR shorts, H1/H4/D1/W1, majors+crosses — coin-flip base (~0.487), drift ≈ cost, timeframe-invariant
  through W1 (arc 3014). Regime conditioning failed 3 ways (dispersion/vol/Kaufman); strong trends invert.
- **3 PORTFOLIO edges (only net-positives):** gap-fill 1006 (JPY crosses H4, +0.685%), month-end 1011 (USD
  majors D1, +0.232%, **+2018**), failed-breakdown reclaim **1013** (USD majors H4, +1.854% **9/10**, the
  strongest; +2015/16/20, **−2018** the sole negative fold; OOS +0.94%, negs 2022/2025 share the strong-USD
  signature).
- **Portfolio provably blocked** (2006/2008/3009/1015): 0/5151 convex weightings all-folds-positive; binding
  folds **2015 & 2018** (2015 wants 1013, 2018 wants 1011 — mutually exclusive). Needs a 4th leg **+both**.
- **The 2018 leg is unfound across EVERY *separate-component* route** — 16 arcs: structure shorts
  (1014/2009/2011/3011), trend short (3010), flow shorts (1016/2013), vol short (3012), relative-value
  (2010), deep-continuation long (2012), carry-unwind (1017), weekly trend (3014). Log's own verdict.

**My read of the highest-EV untested lever:** every 2018 attempt hunted a *separate* 4th component, all
dead. The one thing NOT tried is *improving 1013 itself* past its single 2018 fold — its own thread #5's
open refinement. If a causal regime gate flips 2018 while keeping the 9 good folds, **1013 becomes the
first solo PASS survivor**; if it can't, the last 1013-improvement lane closes cleanly. Either way is
high-value, and it's a LONG refinement (no short-mirror death).

## 2. THE IDEA + WHY (step b — pre-registered before measuring the 2018 effect)

**Mechanism / *because*.** The reclaim is a LIQUIDITY-grab reversal (arc 1013's controlled mechanism). In
an EXTREME / established downtrend (2018-style strong-USD/risk-off), a swept swing low is an
**informational breakdown** (real sellers), and the "reclaim" is a dead-cat pause before continuation, not
a grab. So the edge should LIVE in balanced/mild-down context and DIE in extreme-persistent-downtrend
context → a **TAIL gate on downtrend strength** (consistent with arc-1013 obs#3, which found *mild*-down is
FINE; it only tested a BINARY D1≤SMA50 split, never downtrend STRENGTH/PERSISTENCE as a continuous gate).

**Anti-arc-1012-trap discipline (the design's whole point).** The measure is pre-registered with a because
BEFORE looking at its 2018 effect. The real falsification is OOS: a *causal* regime gate must also flip the
unseen strong-USD OOS years 2022/2025 — if it only patches 2018-IS, it's fished → KILL. (Did not reach that
stage; see below.)

## 3. OBSERVATION (step b/c) — the data led AWAY from the hypothesis

Reproduced the 1013 pool (K=40, shadow≥1.25, 7 USD majors, H4, IS 2010–2020) via canonical `build_arc_pool`
(n=237, mean final_r **+0.137**, win 0.198 — entry edge reproduces). Computed three causal downtrend-strength
features at each fire (SMA200 slope/50 in ATR; distance vs SMA200 in ATR; consecutive bars below SMA200) and
the pool's SL-honest `final_r`. Driver: `_disco2000_work/arc2014_observe_downtrend_gate.py`.

**Edge by downtrend strength (gross pool final_r):**

| cut | bucket | n | mean_r |
|---|---|---|---|
| SMA200 slope | <−0.5 (strong down) | 103 | **+0.218** |
| | −0.5..−0.1 (mild down) | 27 | −0.245 |
| | >0.1 (up) | 99 | +0.194 |
| dist vs SMA200 | −1..1 (near/balanced) | 38 | **+0.397** |
| | >1 (above/uptrend) | 49 | +0.041 (win 0.102) |
| consec below SMA200 | 1–30 | 71 | +0.329 |
| | 31–100 | 48 | +0.339 |
| | >100 (persistent) | 52 | **−0.299** |

1. **The hypothesis INVERTED.** Strong-down-slope is the *best* bucket (+0.218, n=103), not the worst —
   reclaim bounces are *sharpest* when stretched down. "Extreme downtrend = informational = bad" is wrong.
2. The only genuinely negative pockets (`consec_below200>100`: −0.30; the small mild-down slice: −0.24) do
   **not form a coherent regime**, and gating them out **destroys the good folds** (per-year: 2012
   +1.14→−0.09, 2013 +0.36→−0.31, 2019 −0.08→−0.47, 2020 +0.40→+0.12). No free lunch.

**The decisive datum — 2018 is NOT a regime-separable subset, it's a near-total wipeout:**

| 2018 per-pair (final_r) | AUDUSD | EURUSD | GBPUSD | NZDUSD | USDCAD | USDCHF | USDJPY |
|---|---|---|---|---|---|---|---|
| n | 3 | 2 | 4 | 1 | 2 | 5 | 2 |
| mean | −1.00 | −1.00 | −1.00 | −1.00 | −1.00 | −1.00 | **+0.24** |

- **18 of 19 trades hit the full −1R stop; 6 of 7 pairs went 100% to −1R.** Both 2018 trades in the "best
  context" (near-SMA200) are −1.00. The gate drops 58% of 2018 fires yet the *kept* trades still average
  −0.69. There is no entry-time-separable good subset to keep.
- The lone non-loser is **USDJPY** (+0.24, 2 trades) — the one pair trending *up with* USD in 2018. But at
  ~2 trades/yr a "trade only pairs aligned with the USD trend" filter is un-scalable (pool floor; the exact
  arc-1017 un-scalability failure), and arc 3010 already killed the USD-trend-alignment angle.

## 4. VERDICT — KILL (cheap-kill at observation; 1013 UNCHANGED)

The downtrend-regime-gate refinement is **falsified at observation** (no engine/null/council spent — like
arcs 3013/3010/1014/1016): (1) the hypothesis inverted (strong-down is the best bucket); (2) 2018 is a
near-uniform −1R wipeout across all 7 pairs and ALL trend-contexts (incl. the "best"), so it is not a
regime-separable subset; (3) the only negative pockets don't cohere and gating them destroys 4 good folds.

This **independently re-confirms** arc 1013's and arc 3013's finding — 1013's 2018 weakness is
**mechanism-intrinsic** — via a NEW conditioner (downtrend STRENGTH/PERSISTENCE) that neither obs#3 (binary
D1≤SMA50) nor arc 3013 (liquidity-level variants) had tested. arc 1013 thread #5's open
"condition-out-the-regime" refinement is now **CLOSED**: the strong-USD-regime failure of the reclaim-long
is **not entry-time-conditionable** — in risk-off the failed breakdown simply becomes a real breakdown and
the reclaim doesn't hold. The last lane to lift 1013 to a *solo* PASS is shut. **1013 unchanged (still
PORTFOLIO).**

**Threads / lessons.**
1. **NEW lesson:** 1013's 2018 drag is not a tail of bad-context trades but a near-total wipeout (18/19 −1R,
   6/7 pairs 100% −1R, every trend-context incl. "best") → the strong-USD-regime failure is mechanism-
   intrinsic and **entry-time-unconditionable**. The only non-loser is USDJPY (the lone USD-aligned pair),
   but at ~2 trades/yr it is un-scalable (arc-1017 mode). Closes the "improve 1013 toward solo-PASS" lane.
2. **Reclaim edge texture (decorrelation note, not a gate):** the entry edge is strongest in *balanced*
   context (near SMA200 +0.40) and *strong-down* slope (+0.22), weak in established *uptrend* (dist>1 +0.04,
   win 0.10). This is a mean-reversion-works-when-not-trending-up texture — but it does not separate 2018
   and any all-folds gate built on it is the arc-1012 fishing trap.
3. The 2018 portfolio leg is now unfound across ALL routes INCLUDING improving the best component itself —
   the route's 2018 wall stands (arc-3004 escalation reinforced).

**FLAGS (code not merged):** none requiring the canonical core. No new BUILT tool (observation reused
`build_arc_pool` + `FailedBreakdownReclaimLongSignal`; downtrend features are one-off diagnostic, not a
reusable signal). Driver in scratch `_disco2000_work/arc2014_observe_downtrend_gate.py` (reproducible from
this doc).
