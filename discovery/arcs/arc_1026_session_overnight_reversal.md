# arc 1026 — Session / overnight inventory-REVERSAL (the equity overnight-intraday-reversal analog in FX)

**Chat:** 1000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> The corpus's one genuinely-open productive route is **arc-2017 "option B": a THICK, fold-resolving
> standalone edge** (a 5th decorrelated *reversion* leg cannot make the book all-folds-positive — arc
> 2019). Every prior edge is an intrinsically-thin forced-flow reversion (the "unified theory"). I
> attacked that theory with a **thick-by-design** mechanism: does the illiquid Asian-session net move
> **REVERT** when London/NY liquidity arrives — the FX analog of the equity overnight-intraday reversal
> (dealer inventory warehoused in thin hours, mean-reverted on the liquidity transition)? It fires
> **daily (~580/yr)**, so if real it could resolve per-year folds. **Falsified at observation: there is
> no autocorrelation across the session boundary (corr(asian, day) = +0.0006); the only faint reversal
> is a thin deep-tail (~17/yr) and is ~5× sub-cost.**

---

## Log reading (step a — FRESH EYES, honest-era only; pulled main, no STOP)

Resumed 1000s as a fresh chat. Pulled main (already current); read DISCOVERY_PROTOCOL, the full
DISCOVERY_LOG (both tiers), LESSONS.md, TOOL_REGISTRY.md. No `discovery/STOP`. Highest arc-id in my
range (1000-1999) is **1025** → resume at **1026**.

State after 27 honest-era arcs (both chats), the decisive recent picture:

- **A 4-component PORTFOLIO book exists** — gap-fill (1006, JPY-cross H4 weekend), `me_long` (1011,
  USD-major D1 month-end), `fbr` (1013, USD-major H4 failed-breakdown-reclaim, the strongest edge
  +1.854%/9-of-10), `me_short` (1019, the first robustly **+2018** leg).
- **Diagnostics 2016/2017/1023/2019 converged hard:** the book is a genuine **~3-independent-bet**
  (ENB 3.32/4), **mean-positive** (P(mean<0)=0.004; risk-parity +0.589%, t=2.66), negative-tail
  PORTFOLIO. It fails all-folds-positive **only** because the per-calendar-year gate sits *below the
  components' noise floor* — every worst fold is a single thin leg dipping within-noise-negative,
  never a co-drawdown.
- **arc 2019 (latest 2000s, council-driven):** *edge-hunting for the book is closed.* A 5th
  decorrelated *reversion* leg cannot make it AFP (no diversification deficit, no tail to hedge). Only
  an **arc-2017 "option B"** — a *thick, fold-resolving standalone* — or the **operator
  gate-governance call** remains.
- **Option B attempts so far:** arc 1025 (mine) closed it for `fbr` (depth↔edge coupled — can't
  thicken without trading edge for whipsaw variance); arc 2018 closed it for `me` (cross-sectional,
  doubled multi-leg cost vs a coin-flip).
- **Closed ground:** shallow directional (long OR short, H1/H4/D1/W1, majors+crosses, capture AND
  drift, stop-removed) ≈ cost everywhere; ~15 routes to a +2018/+2015/+2016 leg dead; the **H1 cost
  wall** (gotobi 1008, month-end-fix 3008, round-number-H1 1010, **session-breakout 3016**) kills
  every intraday/discrete-flow construction tried so far.

**Where that leaves the frontier.** The instrument universe is **pure FX** (28 currency pairs, no
metals/indices in `histdata_backup`), so the "new universe" lever is unavailable — closed-ground
applies to all of it. The genuinely-open route is option B, and the corpus's "unified theory" (real FX
edges are intrinsically-thin forced-flow reversions → per-year gate structurally unsatisfiable) is
exactly the *seductive search-ending conclusion* the arc-3004 council warned against (§2). The way to
either break it or earn the right to believe it is to test a **thick-by-design** mechanism with a
documented *because* that the corpus has NOT cleanly tested.

## Idea + why (a thick-by-design mechanism with a *because*)

**The equity overnight-intraday reversal, ported to FX sessions.** In equities the overnight return
reverses during the next regular session (Lou–Polk–Skouras 2019; liquidity providers absorb
closed-market order flow and mean-revert it when the liquid session opens). The FX analog: the
**illiquid Asian/Tokyo session** warehouses dealer inventory imbalance; when the **London open** — the
single highest-liquidity FX event — arrives, that inventory is offloaded, **reversing the Asian net
move**. *because* = inventory mean-reversion at the liquidity transition.

Why this is the right shape for option B and **distinct from arc 3016:**
- **Thick by design.** It fires every trading day (~580/yr at a 0.25-ATR overnight-move filter) — the
  one property a fold-resolver needs and that every existing (thin, forced-flow) edge lacks.
- **Genuinely different construction from arc 3016.** 3016 tested the London-open **BREAKOUT of the
  Asian range** — a whipsaw entry at a volatility spike (it died because a 2·ATR SL is hit BOTH ways,
  cap ~0.38 each side). This arc tests the **net-session-return REVERSAL**: enter at London open in the
  direction *opposite* the overnight net move — a calm directional bet on inventory reversion, NOT an
  entry at a break point.
- **Tradeable at ~H4 cost, not the H1 break point.** The mechanism lives at the day grain (overnight
  net move → day reversal), so a one-round-trip-per-day hold pays cost once.

**Acceptance test (pre-registered).** The mechanism predicts **negative autocorrelation** between the
Asian net move and the subsequent London+NY day move. SL-agnostic first (a tight 2·ATR_H1 SL over a
day-long hold would merely reproduce 3016's whipsaw, telling us nothing about the mechanism). Require:
(1) a clearly negative corr; (2) a conditional reversal drift, in the deep-move tail, that clears the
~0.06R (H4) – ~0.125R (H1) cost-in-R hurdle; (3) the 2015/2018 sign checked. Observation-first; only if
the mechanism survives does an engine/exit sweep follow.

## Method (CALLED canonical; observation only)

`Panel.from_pairs` (H1, base-conda env — the working-tree env lacks pyarrow; H1 generated from M1,
parquet-cached) on 7 USD majors (EURUSD/GBPUSD/USDJPY/AUDUSD/USDCHF/USDCAD/NZDUSD). UTC index, all 24
hours present, 2010-2026. Per UTC trading day:
- **Asian net move** = mid open[00:00] → mid close[06:00] (Tokyo 09:00–15:00 JST = 00:00–06:00 UTC).
- **Day move** = mid open[07:00] → mid close[20:00] (London + NY).
- **Early-London move** = mid open[07:00] → mid close[09:00] (the liquidity-arrival point).
- All normalized by a **daily ATR** (Wilder-14 on resampled D1 mid OHLC, `shift(1)` ex-ante).
- Reversal drift = `−sign(asian_ret) · day_ret` (positive ⇒ the day reverted the Asian move).

Drivers (reproducible): `_arc1026_work/observe_reversal.py`, `_arc1026_work/observe_early_london.py`.
CHARACTERIZATION ONLY (gross, no cost/SL/portfolio) — same observation-first discipline as arcs
3010/3012/3014/3015/3016. IS = 2010–2020 (OOS never touched).

## What happened — FALSIFIED at observation

**1) No autocorrelation across the session boundary.** corr(asian_ret, day_ret) pooled = **+0.0006**
(IS n=19,839). Per-pair: −0.025 (EURUSD) to +0.021 (NZDUSD), **mixed signs** — the overnight move
neither reverts nor continues; it is a **random walk across the Tokyo→London transition.**

**2) The conditional "reversal" drift is NEGATIVE / coin-flip — the opposite of the hypothesis.**

| overnight filter | n (/yr) | reversal drift mean | median | frac+ |
|---|---|---|---|---|
| \|asian\|≥0.25 ATR | 6386 (581) | **−0.0124** | −0.0187 | 0.486 |
| \|asian\|≥0.50 ATR | 1729 (157) | **−0.0381** | −0.0529 | 0.463 |
| \|asian\|≥1.00 ATR | 192 (17) | +0.1008 | **−0.0215** | 0.490 |

A *negative* reversal drift = a slight **continuation** (shallow-directional momentum, closed ground,
frac+ < 0.51 = sub-cost). The lone positive mean (thr≥1.0, +0.10) is a **thin-tail artifact** — median
−0.02, frac+ 0.49 (the arc-2011/3012 mean≫median tell).

**3) Year test fails too — and in the wrong direction.** Full-day reversal drift (thr 0.5) is negative
in 9/11 years, including **2015 (−0.121)** and **2018 (−0.045)**.

**4) The faint early-London tail reversal is thin AND ~5× sub-cost.** The 07:00–09:00 window (the
liquidity-arrival point) is the most favorable slice: corr(asian, early) = **−0.031** (noise);
|asian|≥1.0 reversal drift **+0.054 ATR** (median +0.040, frac+ 0.552) — the only cell that looks like
reversion. But it is **17/yr (thin — couldn't resolve folds even if real)** and ~0.054 ATR ⇒ ~**0.027 R**
(1R = 2·ATR) against a ~**0.125 R** H1 round-trip cost = **~5× sub-cost.** |asian|≥0.25/0.5 early-London
drift is +0.006 / −0.008 ATR (coin-flip).

## Diagnosis — FX is too efficient across the session boundary for the equity reversal to exist

The equity overnight-reversal relies on **closed-market / retail frictions**: order flow accumulates
while the market is shut and is absorbed at a price concession when it reopens. **24-hour FX never
closes** and is the deepest, most-arbitraged market — the Tokyo→London "transition" is a liquidity
*gradient*, not a reopening, so there is no warehoused-inventory concession to mean-revert. The
near-zero autocorrelation (+0.0006) is the signature of an efficient continuous market. The faint
deep-tail early-London reversion (+0.054 ATR) is consistent with a tiny stop-cascade unwind on the
largest overnight moves, but it is thin and sub-cost — the same verdict every intraday construction
reaches (the H1 cost wall).

**For option B specifically:** the one *thick-by-design* candidate has **zero edge** at the thick
filters (~580/yr, drift coin-flip-to-negative); edge appears *only* in the thin deep tail (17/yr,
sub-cost). So the mechanism cannot deliver a thick fold-resolver — and the result **reinforces** the
unified theory rather than breaking it: where the corpus has real edge it is intrinsically rare
(forced-flow reversions); where a mechanism is thick it is efficient/coin-flip.

## Verdict: KILL (cheap-kill at observation)

The session/overnight inventory-reversal does not exist in FX: no cross-boundary autocorrelation
(+0.0006), a slight *continuation* (not reversion) at tradeable thickness, a thin deep-tail reversion
that is ~5× sub-cost, and negative 2015/2018 signs. §5d cheap-kill — capture proxy is coin-flip and
forward drift is sub-cost in every cell, so no pool/engine/null/council is warranted (arc-3010/3012/
3016 efficiency discipline). OOS never touched.

## Threads / lessons

1. **The equity overnight-intraday reversal has NO FX analog at the session boundary** — corr ≈ 0
   (+0.0006 full-day, −0.031 early-London). Mechanistically: 24h FX never closes, so there is no
   warehoused-inventory price concession to mean-revert (the friction the equity effect needs). A
   *thick-by-design* candidate that turns out efficient/coin-flip — the cleanest demonstration yet that
   FX's depth is what makes the corpus's edges intrinsically thin.
2. **Complements arc 3016 (decisively closes intraday session structure).** 3016 killed the session
   **BREAKOUT** (whipsaw, cap 0.38 both ways). This arc kills the session **NET-RETURN REVERSAL** (no
   autocorrelation; faint deep-tail reversion sub-cost). Intraday session structure is now dead in
   BOTH its break and its reversal constructions — there is no third obvious construction.
3. **Reusable caution: `1 − continuation ≠ reversal`, and a thick filter can have ZERO edge while a
   thin tail shows a (sub-cost) signal** — the reversal only appears at |asian|≥1.0 (17/yr), and even
   there median<mean (thin-tail). Don't read a thick mechanism's deep-tail signal as a thick edge.
4. **Option-B status:** the *thick fold-resolver* route now has three closed attempts —
   `fbr`-thicken (1025), `me`-cross-sectional (2018), and a thick-by-design new mechanism (1026, this
   arc). The unified theory holds: the corpus's real edges are rare forced-flow reversions, and the
   one demonstrably-thick construction is efficient. The lever remains the **operator gate-governance
   call** (arcs 2016/2017/2019/1023 flag) — which I cannot make and must not pre-empt; OOS stays frozen.
5. **Surviving frontier:** with intraday session structure fully mapped (break + reversal) and option B
   closed on three constructions, the in-apparatus (H1/H4/D1/W1 FX-major OHLCV) idea well is, on the
   honest read, dry. A genuinely novel thick mechanism with a *because* still earns a fresh test (§5a),
   but none is currently visible; the route to deployable is one regime-orthogonal fold-resolver or the
   operator's gate-resolution decision.

## Tooling

No new BUILT tool — focused scratch observers (`observe_reversal.py`, `observe_early_london.py`) using
canonical `Panel.from_pairs` (H1) + a daily-ATR-normalized session-return construction, same pattern as
arcs 3012/3014/3015/3016. Not promoted (one-off session conditioner; reuse the pattern, not a tool).

## FLAGS (code not merged)

None. No canonical-core change. **Environment note (not a code FLAG):** the working-tree conda env
`forex_backtester` lacks `pyarrow` (cannot read the parquet cache); run discovery scripts via
`conda run -n base python` (base has numpy 2.4.2 / pandas 3.0.0 / pyarrow 18.1.0). Carries the standing
`A1Config.time_exit_bars`-unwired flag (arc 1005) and the operator gate-governance question (arcs
2016/2017/2019/1023). Drivers reproducible from this doc.
