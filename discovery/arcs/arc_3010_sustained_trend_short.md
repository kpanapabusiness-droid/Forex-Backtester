# arc 3010 — sustained-USD-strength SHORT: can a trend-following short provide the 2018-positive 4th portfolio leg?

**Chat:** 3000s · **Date:** 2026-06-05 · **Disposition:** KILL (cheap-kill at observation) · **Council:**
none (falsified at observation, regime-inversion is a falsification not a tuning fork — arc-1014/3003
discipline; no worthwhile-ceiling fork).

## (a) Log read / synthesis (fresh eyes, honest-era only)
Pulled main; read the log through arcs 3009 (mine), 2008, 1014. State after 31 arcs:
- The 3-way PORTFOLIO combination (arc 3009 + the 2000s' independent arc 2008) is the route's frontier: 3
  net-positive long-only components combine to a book that is NOT all-folds-positive, blocked on **2 folds —
  2015 and especially 2018** (a strong-USD *trend* year where all three long/structural-reclaim edges bleed).
  Both arcs' spec: the 4th leg must be **net-positive on 2018/2015 → a SHORT / trend / risk-off-positive
  construction** (the long/trend menu for those years is dead).
- **arc 1014 (1000s) ran the FIRST short arc** — confirmed-breakdown continuation short → KILL: *shorts do
  not revive directional STRUCTURE* (coin-flip by symmetry; the swing-low structure is inert for the short).
  It claimed the **up-gap weekend short as its next arc (1015)** and named the live short frontier as
  FLOW-EVENT shorts (acc>0.50), not structural/trend shorts.
- The two named short leads are thus CLAIMED: up-gap weekend short → 1000s (1015); climax-sweep short →
  2000s (arc-2008/2007 lead). The unclaimed, highest-EV open question for my range is the **deployment
  blocker itself**: can ANY trend-following short — held through a trend year — provide the 2018-positive
  leg? arc 1014 tested *structure* shorts; the *sustained-trend-regime* short is untested post-shorts.

## (b) Idea + why (the portfolio's exact need, observation-grounded)
The portfolio needs a component positive in the strong-USD *trend* year 2018 (XXXUSD majors trending down =
USD strengthening). The only thing structurally positive in a sustained directional move is a position
*aligned* with it — here a SHORT on the falling majors. The LESSONS close *shallow* trend shorts (coin-flip
by symmetry), so the test is the regime-conditioned version: does conditioning a short on an **established
downtrend** (close<SMA200 & SMA50<SMA200 — the regime the portfolio needs) isolate a cost-clearing,
2018-positive short drift? This is the symmetric mirror of arc 3003 (which asked the same of momentum
*longs* via the Kaufman efficiency-ratio and found a regime INVERSION). The decisive, observation-cheap
question: **is the USD-majors short positive in 2015 & 2018, and does the downtrend filter lift it there?**

## (c)/(d) Method (observation-first; CHARACTERIZATION ONLY)
Reused the BUILT direction-aware `observe_long_capture(direction="short", sl_mult=2.0, hold=120,
drift_bars=24)` (validated by arc 1014 — short base ≈0.485 reproduces the long-base mirror) on the 7 USD
majors H4, IS 2010–2020. Joined a per-symbol established-downtrend regime (close<SMA200 & SMA50<SMA200) and
the calendar year; grouped short capture + gross forward-24-bar drift (ATR units, sign flipped so a falling
price = positive for the short) by regime, by year (esp 2015/2018), and per pair. Gross observation — the
engine is the gate, but a sub-+0.05R coin-flip drift is guaranteed SL-honest-negative (arc 3003 lesson #2),
so a coin-flip/reverting observation cheap-kills before the engine (arc-1001/1014/3006 discipline).

## (g) Results — FALSIFIED at observation
```
SHORT base (IS, USD majors H4): n=121,432  cap 0.4849  drift -0.0151 ATR   (coin-flip; mirrors long base)
  downtrend bars   : n=51,267  cap 0.4777  drift -0.1271   (WORSE — the regime filter INVERTS)
  non-downtrend    : n=70,165  cap 0.4902  drift +0.0667
```
Per-year short drift (the portfolio-leg question):

| year | ALL cap | ALL drift | DOWNTREND cap | DN drift |
|---|---|---|---|---|
| 2011 | 0.4860 | +0.028 | 0.4265 | −0.407 |
| 2012 | 0.4593 | −0.205 | 0.4524 | −0.126 |
| 2013 | 0.4662 | −0.206 | 0.4365 | −0.491 |
| 2014 | 0.4772 | −0.012 | 0.4895 | +0.380 |
| **2015** | 0.5047 | **+0.125** | 0.5095 | **−0.107** |
| 2016 | 0.4775 | +0.059 | 0.4801 | −0.426 |
| 2017 | 0.4809 | −0.208 | 0.4589 | −0.250 |
| **2018** | 0.5022 | **+0.152** | 0.5022 | **+0.103** |
| 2019 | 0.5119 | +0.090 | 0.4896 | −0.312 |
| 2020 | 0.4798 | −0.028 | 0.5040 | +0.129 |

Per-pair downtrend-short drift: EURUSD −0.023, GBPUSD −0.030, AUDUSD −0.022, NZDUSD −0.132, USDCAD −0.232,
USDCHF −0.139, USDJPY −0.319 — **all 7 negative.**

**Two findings, both fatal to the trend-short-as-2018-leg hypothesis:**
1. **The downtrend regime filter INVERTS the short** (the symmetric completion of arc 3003). Conditioning on
   an established downtrend makes the short drift *more* negative (−0.127 vs +0.067 non-downtrend), negative
   on every one of the 7 pairs, and negative in 7 of 10 years — established trends REVERT against the
   trend-aligned position, on the short exactly as arc 3003 found on the momentum long. The regime filter is
   anti-predictive: it is the wrong lever, not a weak one.
2. **The 2018-positivity is uncapturable regime-luck, not a conditionable edge.** The *unconditional* short
   is faintly positive in 2018 (+0.152) and 2015 (+0.125) — but only as part of a yearly coin-flip (positive
   in 5/10 years, negative mean drift −0.015 over IS; sub-cost and below the 0.50 capture line). The regime
   filter that should *isolate* the 2018-favorable trend instead KILLS 2015 (downtrend −0.107) and is
   negative on all pairs — so the 2018 positivity cannot be conditioned on, predicted, or separated from the
   coin-flip. A portfolio leg built on it would add a net-negative coin-flip short (drift −0.015 gross →
   solidly negative SL-honest + cost), negative in 5 other years, and would NOT reliably rescue 2018.

## Verdict: KILL (cheap-kill at observation)
A sustained-trend-following short on USD majors does NOT provide the 2018-positive (or 2015-positive)
portfolio leg. The short is a coin-flip (base cap 0.485, drift −0.015), the established-downtrend regime
filter is anti-predictive (inverts, all pairs negative — arc-3003 mirror), and the raw short's 2018/2015
positivity is uncapturable regime-luck within a yearly coin-flip. No engine/council compute spent (a
reverting/coin-flip gross short is guaranteed worse SL-honest — arc 3003 lesson #2).

## What this closes (and what it does NOT)
- **CLOSES:** the most direct candidate for the portfolio's 2018 leg — a trend-following / regime-conditioned
  short. It is not trend-buildable. The **3-way book's 2018 wall (arc 3009/2008) is not bridgeable by a
  price-trend short**, reinforcing the arc-3004 escalation: the 2018-positive leg needs something beyond
  price-trend *direction*, which liquid FX prices to a coin-flip in BOTH directions (long: arcs 0–3006;
  short: arcs 1014 structure, 3010 trend) and whose strong-trend *regimes revert* (3003 long, 3010 short).
- **Does NOT close:** the FLOW-EVENT short asymmetries (up-gap weekend short → 1000s arc 1015; climax-sweep
  short → 2000s) — those are not trend-direction bets and remain the live short frontier (arc 1014's steer).
  They are reversions/bursts, though, so whether either is *2018-trend-positive* (the portfolio's specific
  need) is itself open and doubtful — most reversion edges bleed in trend years (the whole reason 2018 is
  the wall).

## Threads / lessons
1. **Regime-conditioning on trend strength is anti-predictive in BOTH directions — the symmetric completion
   of arc 3003.** Strong uptrends revert (3003, momentum long); strong downtrends revert (3010, trend short).
   Conditioning a directional bet on "the trend is established" makes it WORSE, not better, on liquid FX H4 —
   a now-two-sided, robust closure of regime-detection as a directional lever.
2. **The portfolio's 2018 wall is not a price-trend problem.** A short is positive in 2018 only as
   uncapturable regime-luck; no conditionable trend-short isolates it. The 4th leg — if it exists — is NOT a
   trend-following short. This sharpens the arc-3009/2008 spec into a near-impossibility within the
   price-direction apparatus and points back at the arc-3004 escalation (relative-value / second-leg /
   genuinely non-directional structure; carry is OFF on FundedNext).
3. **Confirms arc 1014 from the trend angle.** Shorts revive neither directional STRUCTURE (1014) nor
   directional TREND (3010); the short base ≈ the long base by symmetry, and both fail the same way. The
   value of shorts is confined to genuine ASYMMETRIES (flow events where measured accuracy >0.50), not
   symmetric direction/trend.
4. **Observation-first cheap-kill was correct and cheap** — the reverting, all-pairs-negative, regime-
   inverting signature is decisive without engine compute (arc 3003 lesson #2: a sub-+0.05R / reverting
   gross drift is guaranteed SL-honest-negative).

## Tooling
No new BUILT tool — reused the BUILT direction-aware `observe_long_capture(direction="short")`
(`discovery/tools/observe_long_capture.py`). The downtrend-regime conditioning is a one-off scratch observer
(SMA50/SMA200 mask), like prior observation cheap-kills.

## FLAGS (code not merged)
None. No canonical-core change. Carries FLAG-1 (the regime-orthogonal 4th leg is a non-price-direction /
short-asymmetry / relative-value construction — shorts unblocked PR #273, but trend-direction shorts are now
shown dead, so FLAG-1's *value* narrows to the flow-event asymmetries + the standing escalation) and the
standing `A1Config.time_exit_bars`-unwired flag. Driver scratch `_disco3_work/arc3010_trend_short_obs.py`
(reproducible: `PYTHONPATH=. py _disco3_work/arc3010_trend_short_obs.py`,
`histdata_root=C:\Users\panap\histdata_backup`).
