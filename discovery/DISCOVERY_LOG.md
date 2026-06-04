# DISCOVERY_LOG

This is the machine-scannable index of every discovery arc. The **Tier-1** table below is one row per
arc with fixed fields; the operator's check-in is a single scan of the `passed` column for `Y`. Discovery
chats **APPEND only** — they never edit or compress this file (compression is operator-run, out-of-band;
see [`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md) §6). The log is READ at arc step (a) and appended
at arc step (i).

## Tier 1 — Arc Ledger (strict schema)

| arc_id | chat | timestamp | hypothesis | IS_all_folds_pos | OOS_all_folds_pos | worst_fold_ROI_IS | worst_fold_ROI_OOS | worst_DD | n_trades | VERDICT | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | trial | 2026-06-04 | Pullback-in-uptrend long (close>SMA50 & pierce prior-5-bar-low); best ver = excursion-banking exit | N | N | -14.2% | -14.3% | 17.1% | 4985 | FAIL | N |
| 1000 | 1000s | 2026-06-04 | Cross-sectional momentum long (top-quintile 24-bar ret, rising; partial/runner exit) | N | N | -21.20% | -15.36% | 24.50% | 7757 | FAIL | N |
| 1001 | 1000s | 2026-06-04 | Volatility-contraction breakout long (low-ATR coil + 10-bar-high break) — cheap-kill at triage | N | n/e | -9.41% | n/e | 12.30% | 3186 | FAIL (triage) | N |
| 1002 | 1000s | 2026-06-04 | D1 daily trend-following long (Donchian-20 breakout in uptrend) — TF lever; cheap-kill at triage | N | n/e | -5.72% | n/e | 6.61% | 831 | FAIL (triage) | N |
| 1003 | 1000s | 2026-06-04 | Cross trend-momentum long (Donchian-20 breakout in uptrend, 8 trending crosses) — universe lever; cheap-kill | N | n/e | -9.75% | n/e | 12.39% | 4078 | FAIL (triage) | N |
| 1004 | 1000s | 2026-06-04 | Cross-trend exit/cost engineering (let-it-run / 3R vs partial-runner) — EXIT lever; cheap-kill | N | n/e | -9.75% | n/e | n/e | 4078 | FAIL (triage) | N |
| 3000 | 3000s | 2026-06-04 | Mean-reversion long on coupled crosses (RSI<25 oversold) — instrument-universe + reversion lever; cheap-kill | N | n/e | -20.22% | n/e | 22.54% | 1009 | FAIL (triage) | N |

---

## Per-Arc Reasoning (free-form)

Each arc appends under its own `### arc_<id>` header — the why/because, the approach taken and its rationale, what was tried, what didn't help, and threads worth pursuing. Append freely; empty until the first arc lands.

### arc_0

**SUPERVISED TRIAL** (first run of the protocol; arc id 0, real arcs start at 1000). Full record:
[`arcs/arc_0_pullback_in_uptrend_long.md`](arcs/arc_0_pullback_in_uptrend_long.md). Council
transcript: [`results/arc_0_pullback_in_uptrend_long/council_transcript.md`](results/arc_0_pullback_in_uptrend_long/council_transcript.md).

**Idea + why.** Log was empty (cold start, fresh eyes — no pre-reset priors). Observed EURUSD H4
(IS 2010–2020): naive long breakouts FADE (p_win 0.477/0.471 < 0.492 unconditional) — momentum-
long rejected by the data; the best simple long trigger is buying a short-term dip in an uptrend
(p_win 0.505). Hypothesis: trend resumes after shallow pullbacks; breakouts revert at extremes.
Signal v0: long when close>SMA50 & bar pierces prior-5-bar-low, refractory 6, SL=2·ATR. 8 liquid
pairs, real HistData bid/ask (recovered from the 65 GB backup), H4 5ers_eet, honest engine, costs ON.

**What happened.** Pool 4,985 IS trades (floor PASS). Raw exit slightly negative (mean −0.045R)
but huge favorable excursion given back (≥1R MFE 0.50, ≥2R 0.33; honest +1R-before-SL 0.498).
Clustering (k=4) cleanly split ~50% "good" (clusters 0/1, median MFE 5.9R/2.0R) vs ~50% "bad"
(clusters 2/3, wrong-way-first 96–99.7%). Oracle-best-cluster ceiling worst-fold **+12.7%** (strong
→ PROCEED); raw triage worst −14%.

**The crux (why it died).** Diagnosis: the gap is SELECTION, but entry-time observables do NOT
separate good from bad clusters — every univariate AUC ≈0.5 (max atr_pct 0.527), no structural
filter lifts the 0.497 good-fraction. The oracle ceiling is **answer-key hindsight** (clusters
defined from realised path). Diagnosis council (heavy) recommended: don't run an ML entry-filter
(it would chase hindsight clusters + overfit at 5k samples); test an excursion-banking EXIT instead,
gated by a random-entry null baseline. CC committed. Full honest WFO: `partial_1r_runner_trail`,
`tp_2r`, `sl_only` ALL fail all-folds-positive (best worst-fold IS −14.2%, 7/10 folds negative; OOS
likewise). Null baseline: real signal (mean fold ROI −4.7%, 7/10 neg) beats random entry (−8.7%,
9–10/10 neg) → **a real but sub-cost edge**; not enough to clear FundedNext costs + SL-first.

**Verdict: FAIL.** Family dead for deployment.

**Threads / what didn't help.** Entry-selection on this signal is closed (the discriminator is
post-entry, not on the chart). Excursion-banking exits (partial/TP/trail) do not rescue a near-
coin-flip entry once costs + take-the-loss apply. WEAK open thread: the signal is decorrelated-ish
and sub-cost — possible portfolio value only if a *structurally different* entry refinement raises
the bankable +1R fraction; low priority, not standalone.

**Carry-forward lessons (candidate for LESSONS.md compression):** (1) a high oracle-best-cluster
ceiling proves nothing when clusters are realised-path-defined — always ask "can entry-time
observables reach it?" before believing it (Arc-10 lesson applied); (2) gross MFE-touch rates are
path-optimistic — the bankable fraction after costs + SL-first is materially lower; (3) for buy-
the-dip-in-uptrend H4 majors, the resume/fail split is not in entry-time price structure.

**FLAGS (code not merged):** council's preferred BE-after-+1R exit policy absent from
`core/sim/exit_policies/` (needs human-gated code to test); `build_ex_ante_bounded_population` named
in protocol/CLAUDE.md is `build_arc_pool` in-tree (doc nit); arc drivers kept in scratch `_arc0_work/`.

### arc_1000

**Cross-sectional momentum long** (chat 1000–1999, first continuous arc). Full record:
[`arcs/arc_1000_xsect_momentum_long.md`](arcs/arc_1000_xsect_momentum_long.md). Council transcript:
[`results/arc_1000_xsect_momentum_long/council_transcript.md`](results/arc_1000_xsect_momentum_long/council_transcript.md).

**Idea + why.** Arc 0 closed single-pair entry-time price structure for a long. So I probed the axes
ORTHOGONAL to it: WHEN (time-of-day/session, day-of-week, vol regime, prior move) and WHICH
(cross-sectional relative strength across the 8 majors). Observation (138,816 hypothetical honest longs,
IS 2010–2020, unconditional capture 0.4877): the WHEN axes are DRY (max lift +1.3pp, noise, DST-confounded
seasonality artifact). The WHICH axis showed a faint, monotone, mechanistic momentum tilt (strongest
cross-sectional quintile capture 0.495 vs weakest 0.483) — so the best reasoned version = long the
top-quintile 24-bar-return pair that is also rising (refractory 6, SL=2·ATR, partial/runner exit). Because:
cross-sectional FX momentum is a continuation effect; the absolute-positive filter keeps us long a real uptrend.

**What happened.** Pool 7,757 IS trades, honest capture 0.4939 (matches obs). Cluster structure MIRRORS
arc 0 (~50% resume / ~50% fail; cluster 0 mfe_p50 5.79R; clusters 2/3 wrong-way 0.97/1.0). Cheap kills:
oracle ceiling STRONG (worst-fold +22.4%), raw 3-fold triage NOT deeply negative (2013 +11.98%, 2016 −0.92%,
2019 −3.03%) → proceeded to diagnose (did not cheap-kill).

**The crux (why it died).** Diagnosis: the selection lever is CLOSED — every entry-time observable,
INCLUDING the cross-sectional ones that are the signal's whole novelty (rank/dispersion/board-drift/relative-
strength), separates good {0,1} from bad {2,3} at AUC ≈ 0.50–0.51; good-fraction by tercile flat 0.48–0.51.
The regime hypothesis is FALSIFIED: the highest cross-sectional-dispersion (clearest-trend) quintile is the
WORST (mean final_r −0.24R). The 2013 +11.98% is regime/luck, not separable edge. HEAVY council (5 lenses +
peer review + chairman): KILL — dispersion inversion = falsification not tuning; capture 0.4939 over 7,757
is indistinguishable from 0.50 (CI ±0.011); oracle ceiling is hindsight exit-variance; the lone "keep
testing" dissent (extension-ceiling sweep) was settled against by all 5 reviewers as a non-orthogonal
re-cut of the already-falsified dispersion axis ("ritual not rigor"). CC committed (no override); did NOT
run the sweep. Full WFO verdict-of-record: IS 7/10 folds neg (worst −21.20%) → NOT all-folds-positive; OOS
5/6 neg (worst −15.36%) → NOT all-folds-positive. Null baseline: REAL mean fold ROI −4.36% (7/10 neg) BEATS
random −9.19% (9–10/10 neg) → a real but SUB-COST edge, the SAME signature as arc 0.

**Verdict: FAIL.** Family dead for deployment.

**Threads / what didn't help.** Cross-sectional momentum as a TRADE-LEVEL directional long is closed (the
discriminator is post-entry, not in the cross-sectional structure). The extension-ceiling refinement is
closed (non-orthogonal to the falsified dispersion axis). Temporal/regime conditioning of a naive long is
closed. OPEN (future candidate arc, NOT a rescue of this one — council's reframe): use cross-sectional rank
to select WHICH pairs/universe to run a *different* entry on, or harvest portfolio diversification from a
decorrelated sub-cost edge — a DIFFERENT claim (selection/portfolio, not trade-level direction) needing its
own ex-ante population. Extends arc 0's portfolio thread.

**Carry-forward lessons (candidate for LESSONS.md):** (1) cross-sectional relative strength adds NO
separable entry-time edge on H4 majors (AUC≈0.50, same as arc 0's single-pair features) — the dry zone is
wider than single-pair structure; (2) a dispersion INVERSION (worst performance in the clearest-trend
regime) is a falsification, not a regime to filter — don't sweep to rescue it; (3) WHEN-context
(session/DoW/vol) does not condition a naive long-capture edge on H4 majors; (4) two independent long
families (arc 0 pullback, arc 1000 XS-momentum) now share the SAME real-but-sub-cost signature (beat
random, fail costs) — evidence the binding constraint on H4-major longs is the cost/SL-first hurdle against
a ~coin-flip directional base, not the specific entry construction.

**Tooling:** built + registered the random-entry NULL baseline (`discovery/tools/null_entry_baseline.py`,
TOOL_REGISTRY BUILT) — the protocol's first expected BUILT tool; mask randomization only, scoring stays
canonical (`ArcFoldRunner`). Reusable by all future arcs.

**FLAGS (code not merged):** none requiring the canonical core. Signal + drivers in scratch `_disco_work/`
(reproducible from the arc doc).

### arc_1001

**Volatility-contraction breakout long** (chat 1000–1999). Full record:
[`arcs/arc_1001_contraction_breakout_long.md`](arcs/arc_1001_contraction_breakout_long.md). No council
(cheap-killed at triage; no worthwhile-ceiling fork to evaluate).

**Idea + why.** Arc 0 + arc 1000 both bet on PRICE DIRECTION and failed (direction ≈ coin-flip on H4
majors). The ignored regularity: VOLATILITY clusters (autocorrelated). Idea: after a low-vol coil (ATR
percentile-in-trailing-100 < 0.33), enter long on a break above the prior 10-bar high — vol-clustering makes
expansion-timing predictable, and a squeeze-break triggers stops + breakout flow that generic breakouts
(which fade) lack. The coil is the differentiator.

**What happened.** Observation (honest +1R-before-SL capture, IS 2010–2020, unconditional 0.4877):
contraction-breakout **0.4880** (+0.0003) ≈ generic-breakout 0.4863 ≈ base. The coil adds ZERO directional
lift — the hypothesis is FALSIFIED at observation. (coil-only 0.4895, oversold-reversion 0.4844 — both dry.)
Formal cheap-kill: pool 3,186 IS trades (capture 0.4862, matches obs); 3-fold honest triage 2013 −9.41% /
2016 −1.15% / 2019 −5.43% — ALL negative, worst −9.41%, mean −5.33%, no lucky fold. → KILL at triage
(protocol §5d), before full WFO.

**Verdict: FAIL (cheap-kill).** The coil predicts vol EXPANSION but not DIRECTION; a long-only break is still
a directional coin-flip, inheriting the same sub-cost base as the prior two arcs.

**Threads / lessons.** (1) A volatility coil confers NO directional long edge on H4-major breakouts
(contraction-brk = generic-brk = unconditional ~0.487). (2) THIRD independent long family (pullback, XS
momentum, contraction-breakout) with the SAME ~coin-flip-base / sub-cost outcome — strong accumulating
evidence that the limiting factor for an H4-major LONG is the cost/SL-first hurdle against a ~0.49
directional base, NOT the entry construction. Direction-prediction entries (continuation, reversion,
momentum, breakout, squeeze) are systematically dry at this TF/RR. **Implication for future arcs:** try a
structurally different LEVER — payoff/RR asymmetry, portfolio/selection (the open XS-as-universe thread), a
different timeframe, or a non-directional construction — rather than another H4 directional-long entry
trigger. (Did NOT need the null baseline — an all-negative triage is decisive; the null is for distinguishing
a sub-cost edge that survives to a full WFO.)

**FLAGS (code not merged):** none. Signal + drivers scratch `_disco_work/` (reproducible from the arc doc).

### arc_1002

**D1 timeframe directional long (daily trend-following)** (chat 1000–1999). Full record:
[`arcs/arc_1002_d1_trend_following_long.md`](arcs/arc_1002_d1_trend_following_long.md). No council (cheap-kill).

**Idea + why.** 3 H4 directional longs failed identically (coin-flip base). Lever pivot = TIMEFRAME: daily
trends are classically more persistent. Test whether the coin-flip base is H4-specific or general.

**What happened.** D1 observation (honest +1R-before-SL LONG capture, IS 2010–2020): unconditional **0.4908**
(≈ H4 0.4877); uptrend +0.0000, momentum-breakout +0.0051, pullback −0.0013, oversold +0.0087 (0.4996) — NO
trigger lifts the D1 base above coin-flip, none cross-pair consistent. The directional base is TIMEFRAME-
INVARIANT. Apparatus check on the textbook case (D1 Donchian-20 breakout-in-uptrend, 831 trades, capture
0.4838): 3-fold triage 2013 −1.95% / 2016 −5.72% / 2019 −4.71% — all negative, mean −4.13% → cheap-kill.

**Verdict: FAIL (cheap-kill).** Changing TF does not escape the constraint.

**Threads / lessons.** (1) The FX-major directional-long coin-flip base is TIMEFRAME-INVARIANT (D1≈H4); even
textbook daily trend-following is sub-cost. (2) FOUR independent directional-long attempts now FAIL the same
way (pullback H4, XS-mom H4, contraction-brk H4, trend-following D1) — the approach is comprehensively dry
across entry construction AND timeframe. **STRONG STEER: stop testing directional-long entries on majors;
change a more fundamental lever** — instrument universe (less-efficient CROSSES; the backup has ~28 pairs),
portfolio/selection of decorrelated sub-cost signals (XS-as-universe thread), or a non-directional
construction. Another major-directional-long entry is very likely wasted compute.

**FLAGS (code not merged):** none. Signal + drivers scratch `_disco_work/`.

### arc_1003

**Cross trend-momentum long (instrument-universe lever)** (chat 1000–1999). Full record:
[`arcs/arc_1003_cross_trend_momentum_long.md`](arcs/arc_1003_cross_trend_momentum_long.md). No council (cheap-kill).

**Idea + why.** 4 major-directional-longs failed (coin-flip base). Lever = INSTRUMENT UNIVERSE: crosses
(JPY + commodity crosses) carry rate diffs / trend more than the hyper-efficient majors. Test if the cross
directional base clears coin-flip.

**What happened.** Cross observation (H4, honest +1R-before-SL LONG capture, IS 2010–2020): unconditional
cross base **0.4712** — WORSE than majors (0.4877); every cross < 0.50. Momentum-breakout lift +0.0270 (to
0.4982) — the run's LARGEST conditioning effect, but still coin-flip gross. Apparatus check (cross Donchian-20
breakout-in-uptrend, 4,078 trades, capture 0.4809, **mean final_r +0.1021 gross — crosses DO trend**): triage
2013 +1.43% / 2016 −8.15% / 2019 −9.75% — worst −9.75%, mean −5.49% → cheap-kill.

**Verdict: FAIL (cheap-kill).** Crosses trend (positive gross drift) but the WIDER cross spreads eat the thin
edge → net sub-cost.

**Threads / lessons.** (1) Crosses are NOT a better directional-long universe (base 0.4712 < majors 0.4877).
(2) **Binding constraint sharpened to EDGE < COST**: on crosses gross drift is POSITIVE (+0.10R/trade) yet
the wider spreads consume it; on majors gross edge ≈ 0. Both net sub-cost — the hurdle is finding an edge
LARGE enough to clear spread+commission+slippage+SL-first, not finding *a* directional edge. (3) FIFTH
directional-long FAIL across entry/TF/universe → **change the MECHANISM**: non-price-direction (calendar/flow
— turn-of-month rebalancing, NOT yet tested; arc 1000 covered hour/day-of-week but not day-of-month), or a
construction whose gross edge clears cost.

**FLAGS (code not merged):** none. Signal + drivers scratch `_disco_work/`.

### arc_1004

**Cross-trend exit/cost engineering (EXIT lever)** (chat 1000–1999). Full record:
[`arcs/arc_1004_cross_trend_exit_engineering.md`](arcs/arc_1004_cross_trend_exit_engineering.md). No council.

**Idea + why.** arc 1003 cross trend = +0.10R gross drift but sub-cost (EDGE<COST). Hypothesis: the
+1R-partial-runner CAPS the runner; a let-it-run exit captures more of the +drift per winner → higher
R-per-trade amortizes the per-trade cost. Test reasoned exit variants on the SAME cross trend signal.

**What happened.** Triage across exits (3 folds): baseline partial+runner worst −9.75%/mean −5.49%; wide-trail
let-it-run worst −9.75%/mean −4.07%; standard trail worst −11.45%/mean −5.16%; 3R target worst −11.71%/mean
−4.73%. Let-it-run/asymmetric IMPROVE the trending year (2013 +1.43%→+4.75–6.43%) but leave choppy folds
(2016/2019) deeply negative. No exit flips net-positive → cheap-kill.

**Verdict: FAIL (cheap-kill).** Exit engineering does NOT beat EDGE<COST.

**Threads / lessons.** (1) Exits can't rescue the cross trend — arc 0's "exits don't rescue a coin-flip entry"
extends to a POSITIVE-drift instrument: thin +drift + coin-flip split + per-trade spread = net-negative
regardless of harvest. (2) EDGE<COST is an ENTRY/COST problem, not an exit problem — the fix must raise
per-trade gross edge or cut per-trade cost (frequency), not redistribute the exit. (3) Cross-trend family now
CLOSED across entry + all exit structures. Next: change the MECHANISM — calendar-flow (turn-of-month) tested
with the CORRECT metric (mean forward DRIFT; +1R-before-SL is blind to small drifts) and correct long-only
USD-pair-structure handling.

**FLAGS (code not merged):** none. Driver scratch `_disco_work/arc1004_exits.py`.

### arc_3000

**Mean-reversion long on less-efficient / coupled crosses** (chat 3000–3999, FIRST continuous
arc of the 3000s range). Full record: [`arcs/arc_3000_reversion_long_crosses.md`](arcs/arc_3000_reversion_long_crosses.md).
No council (cheap-kill at triage; no reachable ceiling).

**Idea + why.** Fresh eyes; honest-era corpus = arcs 0 + 1000/1001/1002 (1003 landed on main mid-arc). Four
(then five) directional-long arcs share one signature: ~0.49 directional base, beats random, fails costs. Arc
1002's steer: change a fundamental lever — instrument universe (crosses), portfolio/selection, or a
non-directional construction. I attacked TWO untouched axes at once: (1) **instrument universe** — crosses are
less efficient than the hyper-arbitraged majors, and the tightly-coupled ones (EUR/GBP, EUR/CHF, AUD/NZD,
NZD/CAD…) behave like spreads between linked economies; (2) **mean-reversion** — the FIRST non-continuation
mechanism in the programme (all prior arcs bet on continuation/trend), and coupled crosses are where reversion
should be strongest. Long-only forced by the apparatus → "reversion" = buy the oversold/stretched-down side.

**What happened.** Observation #1 (honest +1R-before-SL LONG capture, all 28 pairs, IS 2010–2020): the base is
**instrument-invariant** — MAJOR 0.4859, COUPLED 0.4742, TREND_X 0.4719; nothing ≥ 0.50, crosses *worse* than
majors. z-decile buckets show NO monotone reversion lift as price stretches down on any group; the faint
high-capture cells sit at *positive* z (the already-dry continuation axis). Observation #2 (fail the BEST
reversion version: RSI/z/Bollinger/consec-down/big-drop × two barrier scales, on the 9 most-coupled crosses):
only **RSI<25** lifts non-trivially, and it peaks at **0.4907** (still < 0.50), thin (~30 fires/pair/yr), and
only 6/9 pairs lift. Characterize: pool 1,009 IS trades, capture **0.4936** (sanity-matches obs), mean final_r
**−0.14R**, 80% hard_sl. Cheap kills: pool floor PASS (1,009); oracle ceiling SKIPPED (capture<0.50 ⇒ no
reachable upside; a high oracle ceiling there is the Arc-0 hindsight trap — ritual not rigor); 3-fold honest
triage (A1, SL=2·ATR, 1% reset-floor, exposure 1/pair 2/ccy, `sl_partial_close_1r_runner_trail` = reversion's
best banking exit, FundedNext costs ON, SL-first; OOS 2013/2016/2019): **−20.22% / −5.17% / −12.89% → worst
−20.22%, mean −12.76%, 3/3 negative → KILL at triage.**

**Verdict: FAIL (cheap-kill).** Mean-reversion long on coupled crosses is not deployable. Reason reached
twice: the directional-long base is instrument-invariant (crosses ≤ majors, all sub-0.50), and the reversion
tilt — best version, best instruments — never crosses 0.50 (real but sub-cost). Buying oversold dips does NOT
catch the positive gross drift crosses have (mean final_r −0.14R), so reversion is gross-coin-flip *and*
cost-bled.

**Convergence with arc 1003 (landed mid-arc).** The 1000s chat independently hit crosses from the
*trend-momentum* angle and found the identical cross base **0.4712** < majors — two chats, two cross signals,
one conclusion: the instrument-universe lever is **closed** for a directional long. Arc 1003 sharpened the
constraint to **EDGE < COST** (crosses trend gross-positive +0.10R/trade but wider spreads eat it); my
reversion case is the complement (gross-coin-flip *and* cost-bled). Together arcs 1003 + 3000 close BOTH
directional mechanism families (continuation AND reversion) on crosses.

**Threads / lessons.** (1) The FX directional-long coin-flip base is **INSTRUMENT-INVARIANT** (28-pair scan;
crosses worse, not better) — the "less-efficient crosses" steer is closed, the way arc 1002 closed the
timeframe steer. Independently corroborated by arc 1003. (2) **Mean-reversion confers no deployable long
edge** — first non-continuation mechanism tested, lands sub-cost too (RSI<25 best, ~0.491 < 0.50). (3) **SIX
directional-long arcs now fail identically across BOTH mechanism families, two timeframes, the full 28-pair
universe, two chats** → overwhelming evidence the constraint is structural (gross edge too small × cost ×
SL-first), not the entry. (4) The excursion-banking partial/runner exit does not rescue a sub-0.50 entry
(re-confirmed). **Surviving steers (both away from trade-level price-direction; either is a fresh arc, not a
rescue):** (a) non-price-direction / calendar-flow (turn-of-month — arc 1003's steer, not yet tested), (b)
portfolio/selection of decorrelated sub-cost edges (mine; subsumes arc 0/1000's portfolio thread).
**Coordination:** arc 1003 named the calendar/flow steer, so the 3000s range should prefer
portfolio/selection to avoid collision.

**FLAGS (code not merged):** none requiring the canonical core. Signal + drivers scratch `_disco3_work/`
(reproducible from the arc doc). No reusable experiment tool needed (null baseline not required — an
all-negative triage is decisive).
