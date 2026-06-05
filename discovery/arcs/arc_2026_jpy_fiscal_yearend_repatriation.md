# arc_2026 — Japanese fiscal-year-end (March 31) repatriation

**Chat:** 2000s | **Date:** 2026-06-05 | **Disposition:** KILL (obs cheap-kill, §5d — hypothesis inverted / priced-in) | **OOS:** untouched

## (a) Log read

(Carried from arcs 2024-2025, same chat.) Corpus near-terminal: 4-component PORTFOLIO book is the deepest
result (mean-positive, cost-robust to ~3.3× FN [arc 3022], temporally stable [2021], NOT all-folds-positive);
leg-hunt/portfolio route proven closed (2019/2022/3021); option-B thick-standalone closed on 4 constructions
(incl. my arc 2024 me-window); the lever is the operator path-A gate call. My arcs 2024/2025 produced a
unifying read: **surviving edges (gap/me/fbr) clear cost only because they condition on a LARGE (≥1 ATR)
price displacement + a forced flow; pure calendar-fix flows (month-end WMR 3008, intra-month gotobi 2025)
are real-but-sub-cost.** Open test of that read: is there a LARGE-displacement (multi-week) calendar forced
flow that DOES clear cost?

## (b) Idea + because

The **Japanese fiscal-year-end (31 March) repatriation** is the canonical candidate: Japanese institutions
(insurers, pensions, corporates) close books on 31 March and repatriate foreign assets / settle FX hedges
into the fiscal year-end → persistent JPY-BUYING through late March → JPY-pairs (USDJPY, EUR/GBP/AUD-JPY)
should FALL (JPY strengthens) into 31 March. Unlike the sub-cost intraday fixes, this is a MULTI-WEEK
directional flow → the displacement is multi-ATR → clears cost easily IF directionally reliable. A clean
test of the arc-2024/2025 hypothesis on a genuinely new instance; candidate risk-off-ish decorrelated
component. Distinct from `me` (price-move-triggered reversion at every month-end) — this is a CALENDAR
directional bias SPECIFIC to March, not conditioned on a price move.

## (c)/(d) Observe + cheap-kill (§5d — gross; NOT a gate)

Driver: `discovery/_disco2_work/arc_2026_jpy_fye_repat_obs.py`. 4 JPY pairs, D1, IS 2010-2020. For each
month-end, the JPY-strength drift over the trailing 10 D1 bars (the repatriation window) = −(pair return)/ATR
(positive = JPY strengthened). March vs other-month-ends → March EXCESS + per-year sign.

```
pair      MARCH mean   MARCH med   frac+    other mean   MARCH EXCESS   per-yr March>0   2015     2018
USDJPY     -0.4629     +0.0898     0.545     -0.0381       -0.4248         6/11        +0.468   -0.231
EURJPY     -0.0994     -0.0866     0.455     +0.0032       -0.1026         5/11        +1.067   -0.087
GBPJPY     -0.5869     -0.1754     0.364     -0.0708       -0.5161         4/11        +1.449   -0.175
AUDJPY     -0.4528     -0.4346     0.455     +0.1665       -0.6193         5/11        +2.361   +0.276
```

## (e) Diagnose — the hypothesis INVERTS

1. **JPY WEAKENS into March-end, the OPPOSITE of repatriation-driven strength.** March jpy_strength is
   NEGATIVE on all 4 pairs, and the March EXCESS is NEGATIVE on all 4 (−0.10 to −0.62 ATR) — robust across
   the JPY leg (USD-vs-JPY and the 3 crosses alike). The predicted positive (JPY-strength) drift does not
   appear; if anything the late-March move is JPY weakness.

2. **And it's coin-flip per-year** (March>0 only 4-6/11; frac+ 0.36-0.55), n=11/pair (thin). Even the
   inverted (JPY-weakness / long-the-pair) direction is not per-year-reliable.

3. **Mechanism — anticipated flow is priced-in / front-run.** The fiscal-year-end repatriation is
   well-known and scheduled; the hedging/repatriation is front-loaded into Feb/early-March, so by the actual
   year-end the JPY-buying is largely done and the late-March approach sees position-squaring / reversal →
   JPY weakness. A textbook "the flow everyone knows about is in the price." (2015 is the lone +JPY-strength
   March on all 4 pairs — an SNB/EUR-collapse-era risk-off, an idiosyncratic event, not a March-specific
   tell; 2018 is mixed-to-negative.)

## (f)/(g) Verdict — KILL (obs cheap-kill, §5d)

The directional bet is falsified twice over: wrong sign (negative March excess on all 4 pairs) and coin-flip
per-year on a thin n=11/pair. No reliable above-cost entry → §5f does not bite. OOS untouched.

## (i) What this closes + lesson

- **Closes the Japanese fiscal-year-end repatriation as a tradeable late-March directional flow** — real
  flow, but anticipated → priced-in / front-run → the realized late-window move is coin-flip-to-reversed.
- **NEW lesson — the TWO failure modes of known-calendar forced flows, and why the surviving edges avoid
  both.** A calendar flow fails to be tradeable in one of two distinct ways: (i) **sub-cost** — the
  displacement is too small (the fix-flow family: WMR 3008, gotobi 2025), or (ii) **priced-in** — the
  displacement is large enough but the flow is well-known and anticipated, so it is front-run and the
  realized move at the calendar date is coin-flip-to-reversed (this arc). The corpus's surviving edges
  (gap/me/fbr) avoid BOTH because they key off a **SURPRISE price displacement** (a weekend gap, a ≥1-ATR
  move into the fix, a deep stop-sweep) — an unscheduled, large, not-fully-anticipated event — NOT a known
  calendar date. This sharpens the arc-2024/2025 unifying read: tradeability needs **large displacement AND
  not-fully-anticipated** — a known calendar date supplies at most one of the two. Operative frontier
  unchanged (operator path-A). Components UNCHANGED.

**Tooling:** no new tool (self-contained obs). No TOOL_REGISTRY append. **FLAGS:** none.
Driver: `_disco2_work/arc_2026_jpy_fye_repat_obs.py`.
