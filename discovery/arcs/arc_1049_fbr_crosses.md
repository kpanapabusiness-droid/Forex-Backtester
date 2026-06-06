# arc 1049 — failed-breakdown RECLAIM (arc 1013, best edge) on CROSSES: the 2018 universe test

**Chat:** 1000s | **Range:** 1000-1999 | **Date:** 2026-06-06
**Disposition: KILL** (obs cheap-kill — hypothesis directionally confirmed but the 2018 cross-signal is thin-tail; cross-fbr is coin-flip/mean-negative, no usable leg)

---

## (a) Log read / synthesis

(Carried from arcs 1047/1048, same chat.) Honest-exit deploy thread closed → me_long-solo, not AFP. The
unfound **2018/2014 regime-orthogonal component** is the only path to an AFP book (~20 dead routes incl.
1047 session-flow, 1048 NFP). Survivor DNA = MECHANICAL forced dislocation → reversion (1048 sharpening:
information events priced efficiently, don't revert; mechanical flows do). The corpus's BEST edge is arc
1013 failed-breakdown reclaim (USD majors H4, IS 9/10) — its only failing fold is 2018, diagnosed
(1014/2014/3013) as a USD-TREND artifact ("in risk-off the failed breakdown becomes a real breakdown").

## (b) Idea (observe, don't guess)

arc 1013 was only ever run on **USD MAJORS**. 2018 was a USD-SPECIFIC one-way trend; on **USD-NEUTRAL
crosses** 2018 was not a clean trend → the reclaim may HOLD there, supplying the regime-orthogonal
(2018-positive) complement. *Because*: remove the USD-trend confound (the named cause of the 2018 failure)
by changing the universe, not the mechanism. None of 1013's refinements (1014/2014/3013/1040/2030) tested
crosses — genuinely untested, and it attacks the exact wall via the best-understood edge. This is a
universe lever on a DEEP structural edge, not a shallow-directional re-test (closed ground covers the
latter).

## (c)/(d) Characterization + cheap kill (OBSERVATION ONLY — no engine/P&L)

Driver `_disco_work/arc1049_fbr_crosses.py`. Reused BUILT `FailedBreakdownReclaimLongSignal`
(K=40, shadow≥1.25) on the 4 H4-cached USD-neutral crosses (AUDJPY, EURGBP, EURJPY, GBPJPY) vs the 7 USD
majors (1013 reference). Honest +1R-before-SL capture + forward 24-bar drift (ATR) via BUILT
`observe_long_capture`, restricted to fbr fires, grouped by year.

**Finding 1 — the 2018 hypothesis is DIRECTIONALLY CONFIRMED (pooled).** 2018 head-to-head:
**MAJORS 2018 drift −1.282** (the known failing fold) vs **CROSSES 2018 drift +1.486** (n13) — the reclaim
flips sign positive on USD-neutral crosses in 2018. The USD-trend confound diagnosis (1014/2014/3013) is
validated: the 2018 weakness is USD-specific and relaxes on crosses.

**Finding 2 — but the cross edge is WEAK overall (not a usable component).** Cross fbr pooled: **capture
0.508** (≈ coin-flip; majors 0.582), **mean drift −0.128 ATR (negative)**, 9/17 drift-positive years, and
**2015 −1.808 (negative)**. Per §11 a mean-negative component is KILL (cannot diversify net-negative
positive, 3000/3001). The structural reclaim is materially weaker on crosses — its USD-majors strength was
partly USD-specific structure, not pure mechanism.

**Finding 3 — the 2018 cross-positivity is THIN-TAIL + pair-inconsistent (the decisive kill).** Per-pair
2018: EURGBP **+3.462 (n4)**, GBPJPY **+2.374 (n2)** carry the entire pooled +1.486; EURJPY +0.173 (n6,
weak); **AUDJPY −0.318 (n1)** — and AUDJPY is the ONLY individually-decent cross overall (cap 0.566, drift
+0.197). So the encouraging pooled 2018 number is dominated by 6 trades across 2 thin-tail pairs — exactly
the thin-tail/pair-mix confound the corpus repeatedly flags (arcs 2011/3011/3012). Not robust; not a
deployable 2018 leg.

## (e)–(h) Diagnose / validate / council

Not reached — §5d cheap-kill. Cross-fbr is a coin-flip with negative mean gross drift (capture 0.508,
drift −0.128); the §5f nested-exit step does not bite (no gross edge to exit-optimize, doesn't beat a
fair null by construction), and the 2018 cross-positivity is thin-tail (n13) not robust. No engine, null,
or council spent (3003 lesson #2; 2011 thin-tail lesson). OOS never touched.

## (i) Verdict

**KILL (obs cheap-kill).** The cross-universe route to the 2018 leg via the corpus's best edge is closed:
the hypothesis is directionally confirmed (fbr's 2018 weakness IS USD-trend-specific — crosses 2018
+1.486 vs majors −1.282, a genuine mechanistic insight) but does NOT yield a usable component — cross-fbr
is coin-flip/mean-negative overall, and the 2018 cross-signal is a thin-tail artifact (n13, carried by
EURGBP n4 + GBPJPY n2; AUDJPY, the best cross, is 2018-negative). No new component; 1013 UNCHANGED (still
PORTFOLIO, USD majors). No canonical change.

**NEW lesson.** The corpus's best structural edge (failed-breakdown reclaim, 1013) is materially WEAKER on
USD-neutral crosses (capture 0.582→0.508, mean drift +0.191→−0.128) — its USD-majors strength was partly
USD-SPECIFIC structure, not pure mechanism, so the universe lever does not transplant it. The 2018 weakness
DOES relax on crosses (confirming the USD-trend diagnosis), but the relaxation is thin-tail and
pair-inconsistent, so removing the named confound (USD trend) by changing universe still does not produce
a robust 2018 leg. Confirms once more: the 2018 leg is not reachable by re-pointing an existing reversion
edge — it needs a genuinely different mechanism. And the thin-tail/pair-mix confound (a few big-drift
trades on 1-2 pairs masquerading as a 2018 edge) recurs whenever a near-coin-flip base is sliced to a
single fold — always check per-pair n before believing a pooled fold sign (2011/3011/3012/now 1049).

## Tooling

No new BUILT tool (reused `FailedBreakdownReclaimLongSignal` 1013 + `observe_long_capture`). Driver
`_disco_work/arc1049_fbr_crosses.py`. No canonical change, no council, OOS untouched. Cross universe
limited to the 4 H4-cached crosses (others need the ~75s/pair backup load) — a fuller cross universe could
be swept, but the coin-flip overall capture + thin-tail 2018 make a broader run low-EV.

## Threads / handoff

The 2018/2014 component remains unfound (~21 dead routes). Reinforced prior: re-pointing/refining existing
reversion edges (universe, level, gate, exit, confluence — 1012/1014/2014/3013/1040/2030/1049) cannot reach
2018; only a genuinely novel MECHANICAL/structural mechanism can. Intraday (sub-cost), scheduled-news
(efficient), all shallow directional, and pure-FX universe are closed. **This is my (chat-1000s, this
session) graceful handoff after arcs 1047-1049.**
