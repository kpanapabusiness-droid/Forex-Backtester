# arc_2064 — Weekend gap-fill on the NON-JPY cross universe (gap survivor's last untested universe cell)

**Chat:** 2000s | **Date:** 2026-06-06 | **Verdict:** KILL (obs cheap-kill) | **Disposition:** KILL

## (a) Log read + synthesis (FRESH EYES, honest-era only)
Resumed at the highest 2000-range arc (2063) + 1. No `discovery/STOP`. Pulled main; read protocol
§0–§11, TOOL_REGISTRY (CALL vs BUILT, full BUILT table), LESSONS + DISCOVERY_DIRECTION + NEEDS_ENABLEMENT,
the full Tier-1 ledger (0→2063) + recent Tier-2 (2055–2063, 1055–1058).

State of the corpus: 4 PORTFOLIO components (`gap` 1006 weekend down-gap-fill **JPY-cross** H4; `me_long`
1011 month-end reversion long USD-major D1; `fbr` 1013 failed-breakdown-reclaim long USD-major H4;
`me_short` 1019 month-end reversion short USD-major D1). The 4-way book is mean-positive (t≈2.66, 1023)
but never all-folds-positive — 2015/2018 strong-USD wall, combination-invariant (0/N convex weightings
at every leg-count 2006/2008/3009/1015/2015), and the leg-hunt is STRUCTURALLY closed: even a perfect
+2015/+2018 maximally-decorrelated 5th leg fails honest weighting (the weighting dilemma, 2022), and the
AFP failure is a noise-floor artifact not a diversification deficit (2016/2017/2019/1023). §5f exit-honesty
audits collapsed the honest deploy object to me_long-solo (mean axis, 1046) / {me_long,fbr} (OOS vehicle
axis, 2059/2060) — both vehicle-infeasible (T_min≥2yr) AND convex-tail-carried (2063/1057). §11 verification
complete end-to-end. Edge frontier mined out on every mapped axis; explore-now MENU exhausted (1054/1055).
Lever = operator path-A gate-governance call or a NEEDS_ENABLEMENT charter unlock (E built/in-review; C/D/J/M).

**Survivor DNA (the *because* filter):** a capturable FX edge needs an identifiable institutional forced
flow + a LARGE displacement of the ACTUAL TRADED SPOT instrument (2057) + information-free (information
events are priced efficiently and don't revert — NFP 1048, generalized). Calendar-flow failure catalogue:
(i) sub-cost (2025/3008), (ii) priced-in (2026), (iii) instrument-neutral (2057), (iv) fat-tail mirage
(2058), (v) thinness-too-mild / closure-not-duration-driven (2061/2062).

## (b) Idea (the gap survivor's one specifically-named untested cell)
Per §2/§5a + the arc-3004 warning against the seductive "apparatus incapable" closure, I did NOT merely
declare the frontier closed — I picked the single mechanism whose death is *least* pre-written and tested
it cheaply. The gap survivor (`gap` 1006) lives on **5 JPY crosses** (mean-POS); arc 2001 found it
mean-NEG on USD majors (uncapturable, adverse i+1 continuation). The **15 non-JPY / non-USD-major crosses**
(EURGBP, EURAUD, EURCAD, EURCHF, EURNZD, GBPAUD, GBPCAD, GBPCHF, GBPNZD, AUDCAD, AUDCHF, AUDNZD, CADCHF,
NZDCAD, NZDCHF) were **NEVER tested** — arc 1006's own thread #112 flagged "larger universe... untested."

**Why it isn't pre-killed + the *because*:** arc 1009 found the JPY-cross gap edge carries a **JPY-basket
forward-drift confound** (its fair same-exit null was itself +0.327% from JPY drift; the edge over the null
was only ~+0.36pp, half the headline). A clean non-JPY, non-USD cross universe **isolates the gap-fill
MECHANISM** (closed-market repricing overshoot → reversion) from both the JPY-basket drift AND the USD-beta
pair-mix (2001/2009). Two outcomes are decision-grade: (1) a mean-positive decorrelated cross gap-fill = a
genuinely new edge in a fresh universe (still only PORTFOLIO — a 5th reversion leg can't make the book AFP,
2019/2022 — but real); (2) dead/JPY-specific = a clean, documented closure of the gap survivor's universe
and a sharpening of the survivor DNA. In the survivor *family* (the gap mechanism that actually works) so
it has the best non-coin-flip prior of any fresh test still available.

## (c)/(d) Characterize + cheap obs (§5d) — OBSERVATION ONLY, IS 2010-2020, OOS untouched
Driver `_disco2000_work/arc2064_nonjpy_cross_gapfill.py`: reuses BUILT `WeekendGapFillLongSignal` (fire
mask, thr∈{0.5,1.0}, gap_hours=20) + BUILT `observe_long_capture` (honest +1R-before-SL capture, take-the-loss;
24-bar fwd drift in ATR; gross, NOT a gate). Restricted to gap-fire bars, IS years only. Includes the
arc-1006 JPY-cross benchmark (anchor) + a structure control (all week-opens, no gap filter).

| Universe | thr | n (IS) | capture | fwd24 drift mean | median | per-pair drift>0 |
|---|---|---|---|---|---|---|
| **JPY crosses (1006 benchmark)** | 0.5 | 399 | **0.5163** | **+0.376** | **+0.392** | (EURJPY +0.76, CHFJPY +0.52, GBPJPY +0.35) |
| NON-JPY crosses | 0.5 | 1227 | **0.4523** | **−0.090** | **−0.079** | 5/15 |
| NON-JPY crosses | 1.0 | 337 | **0.4748** | +0.062 | **−0.026** | 7/15 |
| structure control (all week-opens) | — | 8668 | 0.4730 | −0.157 | −0.092 | — |

**Decisive reads.**
- **The harness reproduces the survivor** (JPY cap 0.516 >0.50, drift +0.376/median +0.392 broad — the
  real, capturable gap-fill; EURJPY +0.76) → the negative non-JPY result is a genuine universe contrast,
  not a tooling artifact.
- **Non-JPY cross gap-fill is dead — coin-flip-to-ADVERSE.** thr0.5: capture **0.4523 <0.50** + drift
  **−0.090 mean / −0.079 median** (both negative, median≈mean = broad, not a tail) → the down-gaps
  **CONTINUE down, they don't fill.** Only 5/15 pairs positive-drift; binding folds NOT positive (2015
  +0.07≈0, 2018 −0.256).
- **The thr1.0 "positive mean" (+0.062) is a thin-tail MIRAGE** — median **−0.026** (mean≫median, the
  arc-2011/2058 fat-tail tell); carried by EURCHF +1.38/n17 (the SNB-floor-break pair-epoch, arc-1028
  artifact). 2018 +0.293 sits on a 0.50 coin-flip capture / median-neg pool = regime-luck, not an edge.
- **Structure control: the gap is NOT load-bearing.** All non-JPY-cross week-opens drift −0.157 (mild
  continuation); the big-gap subset (−0.090) is no *more* reversionary than a random week-open — adding
  the ≥0.5-ATR gap filter does not create reversion. (Contrast JPY, where the gap filter lifts drift to
  +0.376.)

§5f does NOT bite (sub-0.50 capture + no gross reversion drift to rescue with an exit — §5f is for
non-coin-flip entries; this is coin-flip-to-adverse). §5d cheap-kill. No engine/null/council spent; OOS
NEVER touched.

## Verdict — KILL (obs cheap-kill). The gap-fill is JPY-cross-SPECIFIC, not universal.
The weekend gap-fill reversion does **not generalize beyond JPY crosses.** On 15 non-JPY/non-USD crosses
(n=1227 ≥0.5-ATR weekend gaps) the honest i+1 capture is **adverse (0.452)** and the forward drift is
**negative (−0.090)** — these gaps continue, they don't fill; the structure control shows the gap is not
load-bearing.

**NEW lesson (sharpens the survivor DNA + confirms arc-1009 from the other side).** "Weekend gap-fill" is
a misnomer for a universal closure-repricing-overshoot reversion: it is specifically a **JPY-cross
risk-off gap snap-back.** Mechanism — a JPY-cross weekend down-gap is a risk-off/carry-unwind Tokyo-session
spike that **mean-reverts** (the snap-back of a forced de-risking move); a non-JPY cross weekend down-gap
is a genuine relative-rate / EUR-bloc repricing that **persists** (mild continuation). The reverting force
lives in the JPY-basket carry/risk-off structure (the +0.327% JPY-basket drift arc 1009 isolated as a
confound is, from this angle, the *mechanism itself*), not in the calendar-closure window. This bounds the
gap survivor's universe **definitively** (JPY crosses POS 1006 / USD majors NEG 2001 / non-JPY crosses
DEAD here) and adds a **sixth** calendar/structural failure mode to the catalogue: **(vi) mechanism is
universe-specific — a forced-flow reversion sourced in one currency bloc's risk-off carry structure does
not transfer to instruments lacking that structure.** No 5th component; even a success would have been only
PORTFOLIO (2019/2022). Closes arc-1006 thread #112.

## (i)/(k) Bookkeeping
Components UNCHANGED (all 4 PORTFOLIO); deploy object UNCHANGED (me_long-solo / {me_long,fbr}, both
vehicle-infeasible). Lever = operator path-A. No new BUILT tool (driver reuses BUILT `WeekendGapFillLongSignal`
+ `observe_long_capture`; scratch obs driver under `_disco2000_work/`). No canonical change, no FLAG, no
council (coin-flip base, §5f doesn't bite), no engine/null. OOS NEVER touched (IS-only obs). Data:
canonical `Panel.from_pairs` (histdata_backup) + cache. My read after this arc: the OHLC-only edge frontier
is exhausted — this was the gap survivor's last specifically-named untested universe cell, and it confirms
the survivor does not generalize. Remaining value sits with the operator's gate/deploy decision or a
NEEDS_ENABLEMENT charter unlock, not the autonomous edge-hunt.
