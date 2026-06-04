# Arc 1006 — Weekend Gap-Down-Fill Long (JPY crosses) — generative-council idea

> **Arc id:** 1006 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL** (not all-folds-positive: IS 5/10 neg worst −6.79%; OOS 4/6 neg worst −4.13%)
> — **BUT the most promising result of the run:** the FIRST mean-positive IS edge (+0.69%), it BEATS the
> random null, and DDs are the lowest of any arc (2–9%). A REAL but fold-fragile edge.
> **Idea:** synthesized from a LIGHT generative council (§5b) at the idea-fork arc 3001 flagged.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Canonical engine called; signal + the
time-exit predicate are experiment tools.

## (a) Log read — FRESH EYES (incl. other chats)

Pulled main. Corpus = 10 arcs (0; 1000–1005; 2000; 3000–3001), ALL FAIL. The price-structure directional-long
space is closed fleet-wide under BOTH the +1R-capture and the mean-forward-drift metrics (arc 3001's 28-pair ×
6-condition scan, incl. up- AND down-spikes), across H4/D1, majors+crosses, exit engineering (1004), and
turn-of-month calendar (1005). **EDGE<COST is mechanism-general.** Arc 3001 explicitly flagged: the next arc
needs a *generatively-different* idea → a §5b LIGHT generative council. Key constraint surfaced: the data is
PRICE-ONLY (no exogenous), long-only. No STOP.

## (b) Idea — light generative council → synthesis

Convened the §5b light generative council (5 isolated lenses; transcript:
[`../results/arc_1006_weekend_gap_fill/council_transcript.md`](../results/arc_1006_weekend_gap_fill/council_transcript.md)).
Lenses proposed: weekend-gap effects, triangular-divergence reversion, vol/cost conditioning, spread-tier
gating, SL-geometry sweep, and a Devil's null-confirmation. **CC synthesis:** rejected the SL-geometry lever
(first-order cost-in-R and gross-in-R both scale 1/SL-width → cannot flip the net SIGN); rejected triangular
divergence (arb closes sub-second → dead at H4); deferred the Devil's null/passive baseline as premature while
novel ideas remain. **Chose the WEEKEND GAP-FILL** — the most genuinely-novel, untested (untouched by arcs
0–3001), price-only, long-only mechanism: documented FX gap-fill tendency; mechanism = weekend positioning/
liquidity overhang reverting as flow resumes Monday. Long-only → buy weekend **gap-DOWNS**, bet on the upward fill.

**Observation** (10 pairs, week-open bars = timestamp gap >36h, IS 2010–2020, 5,720 week-opens; gap std 0.66
ATR, 24% of gaps >0.5 ATR — meaningful). Honest +1R-before-SL capture + forward 24-bar drift in ATR:

| week-open gap (ATR) | n | capture | mean fwd drift (ATR) |
|---|---|---|---|
| < −0.5 (big gap-down) | 735 | 0.4653 | **+0.2028** |
| −0.5..−0.25 | 821 | 0.4397 | −0.1336 |
| flat / small | ~1750 | ~0.42 | ~−0.07 |
| > +0.25 (gap-up) | 1343 | ~0.46 | +0.08..0.16 |

**Only the LARGEST gap-downs (<−0.5 ATR) fill** (+0.20 ATR forward); moderate gap-downs CONTINUE down. Per-pair
gap-downs are heterogeneous — **JPY crosses fill strongly** (EURJPY capture 0.5417, mean fwd **+0.4765 ATR** —
the biggest gross edge in the run; USDJPY +0.21; GBPJPY +0.12) while GBPUSD/USDCAD/USDCHF continue down. →
best-reasoned version: **long a JPY cross on a >0.5-ATR weekend gap-down**, 24-bar time exit, SL=2·ATR.

## (c)+(d) Characterize + cheap kill (triage)

`build_arc_pool`, 5 JPY crosses (EURJPY GBPJPY AUDJPY CADJPY CHFJPY), H4 5ers_eet, IS 2010–2020. **396 IS
trades**, capture 0.5000, **mean final_r +0.1127 (positive gross — the fill works)**. Time exit via the BUILT
`make_time_exit_predicate` (24 bars). Pool floor PASS.

3-fold triage (A1, SL=2·ATR + 24-bar time exit, `exit_policy=None`, FundedNext costs ON): **2013 −2.06% / 2016
+3.20% / 2019 +7.45% — worst −2.06%, mean +2.86%, 2/3 POSITIVE, tiny DDs (2.5–5%).** Best triage of the run;
the positives are in the CHOPPY years (2016/2019), not a single trending fluke → **did NOT cheap-kill →
PROCEED to full WFO.**

## (g) Full WFO verdict-of-record + null

One config (the IS-defined system; no OOS tuning), all IS folds + per-year OOS:

**IS (10 folds):** +(-0.07, +8.23, −2.06, +2.94, −4.19, +3.20, +0.53, −6.79, +7.45, −2.39)%. **all-folds-positive
NO** — worst −6.79%, **5/10 negative**, but **mean +0.69% (the only mean-POSITIVE IS of any arc)**, DDs 2–9%.

**OOS (2021–26):** −4.13 / +6.84 / −3.26 / −2.25 / −3.17 / +5.98 %. **all-folds-positive NO** — worst −4.13%,
4/6 negative.

**Null baseline:** REAL IS mean **+0.69%** (worst −6.79%, 5/10 neg) **BEATS** random entry (−0.60%, 5–7/10 neg,
worst ~−14%) → a **genuine edge**, not luck — but not consistent enough for all-folds-positive.

## (f) Fail the best version — one reasoned IS refinement

Hypothesis: gap-downs FILL when they're dips in an uptrend but CONTINUE when extending a downtrend → add
`close > SMA50`. **Result (IS):** negatives cut 5/10→4/10, folds mostly small-positive, but **still NOT
all-folds-positive** (worst −4.40%, mean +0.34%) AND it thinned folds dangerously (min 5 trades/fold —
statistically meaningless). The refinement does not rescue fold-consistency and over-thins. No further tuning
(more refinements on a thin, observation-selected signal = overfitting / "ritual").

## (h) Survivor stress-test — NOT REACHED

Not all-folds-positive → not a survivor; the mandatory survivor council + `passed/` record were correctly NOT
triggered.

## Final verdict — FAIL (but the run's most promising lead)

The JPY-cross weekend gap-down-fill **FAILs the sole judge** (all-folds-positive on neither IS nor OOS): it is
**fold-fragile** because JPY-cross weekend gaps are tail-event-timing-dependent (a few big risk-event weekends
drive the fill; quiet years are flat-to-negative). BUT it is **categorically different from the 9 prior FAILs**:
the FIRST mean-positive IS edge (+0.69%), it BEATS random, and it has the lowest DDs of the run. A real,
mechanistically-grounded edge that is simply not consistent enough to deploy as-is.

## Lessons (candidate for LESSONS.md)

1. **Weekend gap-FILL is REAL on JPY crosses (the first net-positive long edge of the run)** — big weekend
   gap-downs (<−0.5 ATR) partially fill (+0.20 ATR pooled; EURJPY +0.48 ATR). It is mean-positive (IS +0.69%),
   beats random, low-DD — but FOLD-FRAGILE (5/10 IS, 4/6 OOS negative; tail-event-timing-dependent), so it
   FAILs all-folds-positive. Moderate gap-downs (−0.5..−0.25) CONTINUE down; gap-FILL is a large-gap-only effect.
2. **This RE-OPENS the portfolio/selection thread.** Arc 3001 correctly noted you cannot diversify net-NEGATIVE
   components into a positive system — but arc 1006 is the FIRST net-POSITIVE (if fold-fragile) component. A
   decorrelated combination of several net-positive-but-fold-fragile edges could plausibly reach all-folds-
   positive (diversification cuts fold-variance while preserving positive mean). The thread activates the
   moment a SECOND net-positive edge is found. **High-value steer for the fleet.**
3. **Generative council earns its keep at a real fork** — it steered off the exhausted directional/calendar
   rut to the first promising lead. (It also queued the SL-geometry, spread-tier-gating, vol/cost-conditioning,
   and Devil's null-confirmation ideas for future arcs.)

## Threads

- **Open / HIGH VALUE:** (a) weekend gap-fill as a portfolio COMPONENT (net-positive, decorrelated from
  trend/calendar) — pursue once a 2nd net-positive edge exists; (b) is the fold-fragility reducible by a
  larger universe (all JPY-cross + commodity gaps) to add trades without thinning? (untested — but beware
  cherry-pick); (c) other large-gap events (intraday session gaps, post-holiday gaps). 
- **Queued from the council (untested):** spread-tier gating (majors-only cost cut), vol/cost-ratio
  conditioning, Asia-compression→London-open timing, and the Devil's random-entry/passive-buy-and-hold
  null-confirmation (the decisive "is the space dry" closure test, if novel ideas dry up).
- **Closed:** the gap-down-fill as a STANDALONE all-folds-positive system (fold-fragile); the uptrend-filter refinement.

## Flags / Reproduction

No canonical-core change. Reused BUILT `make_time_exit_predicate`. Signal + drivers scratch
(`_disco_work/arc1006_observe.py`, `arc1006_kill.py`, `arc1006_wfo.py`, `arc1006_refine.py`);
`PYTHONPATH=. py _disco_work/<script>.py`. Data: 5 JPY crosses, H4 5ers_eet, `histdata_root=C:\Users\panap\histdata_backup`.
Signal: long on week-open (timestamp gap >36h) gap-down < −0.5 ATR, 24-bar time exit, SL=2·ATR. Engine:
`A1Architecture`+`ArcFoldRunner` (`exit_policy=None, trail_enabled=False`), FundedNext costs at
`build_fold_stats_from_run`, IS `build_v3_folds`, OOS `build_oos_year_folds(2021)`, judge `judge_all_folds_positive`,
null `discovery/tools/null_entry_baseline`.
