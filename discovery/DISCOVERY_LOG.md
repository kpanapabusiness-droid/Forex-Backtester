# DISCOVERY_LOG

This is the machine-scannable index of every discovery arc. The **Tier-1** table below is one row per
arc with fixed fields; the operator's check-in is a single scan of the `passed` column for `Y`. Discovery
chats **APPEND only** — they never edit or compress this file (compression is operator-run, out-of-band;
see [`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md) §6). The log is READ at arc step (a) and appended
at arc step (i).

## Tier 1 — Arc Ledger (strict schema)

| arc_id | chat | timestamp | hypothesis | IS_all_folds_pos | OOS_all_folds_pos | worst_fold_ROI_IS | worst_fold_ROI_OOS | worst_DD | n_trades | VERDICT | passed | disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | trial | 2026-06-04 | Pullback-in-uptrend long (close>SMA50 & pierce prior-5-bar-low); best ver = excursion-banking exit | N | N | -14.2% | -14.3% | 17.1% | 4985 | FAIL | N | KILL |
| 1000 | 1000s | 2026-06-04 | Cross-sectional momentum long (top-quintile 24-bar ret, rising; partial/runner exit) | N | N | -21.20% | -15.36% | 24.50% | 7757 | FAIL | N | KILL |
| 1001 | 1000s | 2026-06-04 | Volatility-contraction breakout long (low-ATR coil + 10-bar-high break) — cheap-kill at triage | N | n/e | -9.41% | n/e | 12.30% | 3186 | FAIL (triage) | N | KILL |
| 1002 | 1000s | 2026-06-04 | D1 daily trend-following long (Donchian-20 breakout in uptrend) — TF lever; cheap-kill at triage | N | n/e | -5.72% | n/e | 6.61% | 831 | FAIL (triage) | N | KILL |
| 1003 | 1000s | 2026-06-04 | Cross trend-momentum long (Donchian-20 breakout in uptrend, 8 trending crosses) — universe lever; cheap-kill | N | n/e | -9.75% | n/e | 12.39% | 4078 | FAIL (triage) | N | KILL |
| 1004 | 1000s | 2026-06-04 | Cross-trend exit/cost engineering (let-it-run / 3R vs partial-runner) — EXIT lever; cheap-kill | N | n/e | -9.75% | n/e | n/e | 4078 | FAIL (triage) | N | KILL |
| 1005 | 1000s | 2026-06-04 | Turn-of-month USD-long calendar drift (USDCHF/USDXXX, 6-bar time exit) — non-directional mechanism; cheap-kill | N | n/e | -2.78% | n/e | 2.96% | 128 | FAIL (triage) | N | KILL |
| 1006 | 1000s | 2026-06-04 | Weekend gap-down-fill long, JPY crosses (gap<-0.5ATR, 24-bar time exit) — generative-council idea; REAL but fold-fragile (FIRST mean-positive IS edge +0.69%, beats random) | N | N | -6.79% | -4.13% | 8.57% | 396 | FAIL | N | PORTFOLIO |
| 1007 | 1000s | 2026-06-04 | Gap-fill exit-AT-target (TP at gap origin) — best-version test of arc 1006; edge is OVERSHOOT not fill, capping it is WORSE (IS mean -1.21%, loses to null) | N | N | -6.76% | -3.93% | 6.95% | 396 | FAIL | N | KILL |
| 2000 | 2000s | 2026-06-04 | Trend-following long via full-size convexity harvest (Donchian breakout + full-size trailing) — fat tail is generic not trend-selected; cheap-kill at triage | N | n/e | -13.97% | n/e | 15.8% | 1617 | FAIL (triage) | N | KILL |
| 3000 | 3000s | 2026-06-04 | Mean-reversion long on coupled crosses (RSI<25 oversold) — instrument-universe + reversion lever; cheap-kill | N | n/e | -20.22% | n/e | 22.54% | 1009 | FAIL (triage) | N | KILL |
| 3001 | 3000s | 2026-06-04 | Drift-lens scan (mean fwd drift, the metric +1R-before-SL is blind to) across 28 pairs × 6 conds — directional-long death is METRIC-ROBUST; best cell post-up-spike trending-cross net −10.78% | N | n/e | -28.07% | n/e | 32.38% | 1375 | FAIL (triage) | N | KILL |
| 3002 | 3000s | 2026-06-04 | Volume-confirmation long (vol-spike on trending crosses) — the last untouched data column; full IS 7/10 neg | N | n/e | -16.25% | n/e | 20.57% | 839 | FAIL (IS WFO) | N | KILL |
| 3003 | 3000s | 2026-06-04 | Regime detection for momentum (Kaufman Efficiency-Ratio): is the trending regime detectable in advance? REGIME INVERSION (strong trends revert); best-case band catastrophic | N | n/e | -63.11% | n/e | 66.51% | 9250 | FAIL (IS WFO) | N | KILL |
| 2001 | 2000s | 2026-06-04 | Weekend-gap-fill long, MAJORS (buy big weekly-open down-gap, reversion to prior close) — CONVERGES w/ arc 1006 (JPY crosses); REAL edge but uncapturable: majors mean-NEG (vs 1006 crosses mean-POS), honest i+1 entry lands in adverse continuation (MAE −1.1R), thin (~13-17/fold); HEAVY diagnosis-council KILL, OOS preserved pristine | N | n/e (OOS preserved) | -4.57% | n/e | n/e | 176 | FAIL (IS+council) | N | KILL |
| 2002 | 2000s | 2026-06-05 | H1 entry-resolution test of the weekend gap-fill (enter ~6h closer to the gap open) — does finer resolution capture the fill the H4 i+1 entry missed? NO: H1 capture 0.450 ≡ H4 0.45-0.47, MAE −1.2R ≡ H4; the down-gap's adverse continuation is TF-INVARIANT. Confirmatory cheap-kill; vindicates arc-2001 council (uncapturable, not an H4 artifact) | N | n/e | -2.02% | n/e | n/e | 151 | FAIL (triage) | N | KILL |
| 2003 | 2000s | 2026-06-05 | Cross-pair relative-value divergence catch-up (long the under-performing leg when the synthetic cross is extreme) — LIGHT generative-council idea; cheap-kill at obs: the laggard CONTINUES lagging (z<−2 fwd drift −0.05 to −0.11 ATR; per-group z<−1 fwd10 −0.20 EURvGBP / −0.15 AUDvNZD), NO reversion — relative perf is momentum/coin-flip (re-confirms 1000/3001). Council's strongest call = CONCEDE: long-only constraint is the binding blocker (FLAG-1) | N | n/e | n/e | n/e | n/e | n/e | FAIL (obs) | N | KILL |
| 3004 | 3000s | 2026-06-05 | Stop-confound test (council-driven): is the universal 2*ATR SL the wall? Pure time-exit on best +drift entry — stop removed lifts mean ~4-5pp but STILL not all-folds-pos → ESCALATION (apparatus structurally incapable; needs 2nd leg/shorts/cost regime) | N | n/e | -20.43% | n/e | n/e | 799 | FAIL → ESCALATION | N | KILL |
| 3005 | 3000s | 2026-06-05 | Cross-rate triangulation residual (synthetic vs quoted cross) — last non-directional in-apparatus thread; residual sub-spread, conv-corr ~0 at H4 | N | n/e | n/e | n/e | n/e | n/e | FAIL (obs cheap-kill) | N | KILL |
| 1008 | 1000s | 2026-06-05 | Gotobi-day Tokyo-morning USDJPY long (5/10/15/20/25+EOM, fix-window) — documented JP corp USD-settlement microstructure flow; FALSIFIED at obs: gotobi premium ≤+0.25bp gross (best window, frac+ 0.501 coin-flip) vs ~2bp cost, absent at H4 AND H1; only +era (2010-14) is generic Abenomics drift not gotobi-specific | N | n/e | n/e | n/e | n/e | n/e | FAIL (obs cheap-kill) | N | KILL |

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

### arc_2000

**Trend-following long via full-size convexity harvest** (chat 2000–2999, first continuous arc). Full
record: [`arcs/arc_2000_trend_convexity_harvest.md`](arcs/arc_2000_trend_convexity_harvest.md). No
council (cheap-kill; falsified at observation).

**Idea + why (fresh insight, NOT "arc 1002 again").** The four prior arcs all *screened* on +1R-before-SL
*capture* — a WIN-RATE statistic, structurally blind to a low-win-rate / fat-tailed (convex) payoff (the
time-series-momentum profile, edge in the right tail not the hit rate). And every engine triage used the
50%-partial exit, which CAPS the right tail at half size. So the prior failures might be a measurement
artifact. Untested question: does a trend entry have a fat harvestable right tail, and does a FULL-SIZE
tail-preserving trailing exit (`sl_plus_trailing_atr`: −1R floor, uncapped upside, 1R-from-peak trail)
bank enough of it to clear costs? Entry = Donchian breakout long (canonical TSMOM/trend entry).

**What happened.** OBSERVATION (canonical pool MFE distribution, H4 majors, IS 2010-2020, hold 360,
SL=2·ATR) — built a periodic (time-random) long base and Donchian breakouts (N=20/55/120 ±SMA200). The
**fat tail is real but the trend entry does NOT create it**: periodic base already P(mfe≥5R)=.147,
P(mfe≥8R)=.077, meanMFE 2.46R; Donchian-120 lifts only to .165/.088/2.69 — trivial. The right tail is a
generic property of being long a vol-clustering FX major at any time, not trend-selected. cap+1R≈.49
everywhere too → win-rate lens and convexity lens AGREE: trend entry adds ~no separable edge on either.
Hypothesis falsified at observation. Cheap-kill triage (3 IS folds 2013/2016/2019, honest engine, costs
ON) on the untested full-size trails: `sl_plus_trailing_atr` worst −13.97% / mean −9.50% / 3-of-3 neg
(WORSE than the 50%-partial's −7.14% — the per-winner 1R give-back + full-size whipsaw losses dominate);
`sl_plus_trailing_swing` mean −0.04% but 2/3 neg with a +20.81% thin (n=23) fold and a −19.86% / DD-20%
blowup (regime-luck, the FX-majors-only trend signature); periodic null + trailing_atr −11.00% (donchian
−9.50% barely beats it). All N on all-folds-positive.

**Verdict: FAIL (cheap-kill).** Trend-following-long-via-convexity-harvest is not deployable on H4 majors.

**Convergence with arc 1004 (landed on main mid-arc).** The 1000s chat independently engineered exits
(let-it-run / 3R / wide-trail) on its CROSS trend-momentum signal and found no exit flips net-positive —
"EDGE<COST is an ENTRY/COST problem, not an exit problem." My arc is the MAJORS + full-size-trailing
complement and reaches the identical verdict from the convexity angle. Two chats, two universes, two exit
families: **payoff/exit engineering cannot rescue a sub-cost trend.** And my MFE observation explains WHY —
the harvestable tail is generic (a random long has it too), so there is no trend-specific convexity for any
exit to harvest.

**Threads / lessons.** (1) The fat right tail in FX-major longs is generic, not trend-selected — both the
win-rate lens (+1R≈.49) and the convexity lens (MFE tail) say the trend entry adds nothing; the prior
arcs' win-rate screen was NOT hiding a convexity edge. (2) No full-size trailing harvest rescues a
~coin-flip long (atr trail worse than partial; swing trail is regime-luck/blowup, not all-folds edge) —
converges with arc 1004 on crosses. (3) With arcs 1003/1004/3000, the count is now SEVEN+ directional-long
FAILs sharing the EDGE<COST signature across entry construction, timeframe (H4/D1), the full 28-pair
universe (majors+crosses), BOTH mechanism families (continuation+reversion), AND now payoff/exit structure
— the constraint is structural. (4) Surviving steers (coordinated): calendar-flow/non-price-direction
(1000s), portfolio/selection of decorrelated sub-cost edges (3000s). For the 2000s range, the open lane is
a construction that **raises per-trade gross edge or cuts per-trade cost/frequency** enough to clear the
EDGE<COST hurdle (not another entry/exit re-cut), OR a genuinely different instrument-structure (e.g.
relative-value/spread between coupled pairs) — to be decided fresh at arc 2001's observation.

**Tooling:** built + registered `discovery/tools/trend_entry_signals.py` (`DonchianBreakoutLongSignal`,
`PeriodicLongSignal`) — reusable trend-entry + random-base experiment tools (mask + ATR geometry only;
scoring stays canonical). TOOL_REGISTRY BUILT updated.

**FLAGS (code not merged):** none requiring the canonical core. Drivers in scratch `_disco2000_work/`
(reproducible from the arc doc).

### arc_1005

**Turn-of-month USD-long (calendar-flow mechanism)** (chat 1000–1999). Full record:
[`arcs/arc_1005_turn_of_month_usd_long.md`](arcs/arc_1005_turn_of_month_usd_long.md). No council (cheap-kill).

**Idea + why.** 6 prior families fail on EDGE<COST (entry/cost problem, not exit). Steer: change the
MECHANISM → non-price-direction CALENDAR FLOW (month-end rebalancing). Tested with the CORRECT metric (mean
forward DRIFT in ATR; +1R-before-SL is blind to small drifts), PER PAIR (month-end USD flow pushes XXXUSD vs
USDXXX oppositely).

**What happened.** D1 turn-of-month (last 2 + first 3 trading days), mean forward-5d drift: a REAL USD-strength
signal — EURUSD −0.18 ATR, GBPUSD −0.26 (18% years pos), USDCHF **+0.177** (73% years pos), concentrated in
EUR/GBP/CHF. Long-only-exploitable via USDCHF. Built a signal-class TIME-EXIT predicate (A1Config.time_exit_bars
is NOT wired into the Order — FLAG); long at 3rd-to-last trading day, 6-bar time exit, SL=2ATR. Triage: USDCHF
2013 +0.68% / 2016 −2.78% / 2019 −0.27% (worst −2.78%, mean −0.79%, tiny DDs 1–3%, mean final_r ≈ break-even
gross); USDXXX basket worst −3.55%, mean −1.61%. Both sub-cost → cheap-kill. (USDCHF = 1-of-8 cherry-pick.)

**Verdict: FAIL (cheap-kill).** The drift is REAL but too small (~0.18 ATR/5d) to clear FundedNext cost +
the ~31% 2ATR stop-out. EDGE<COST.

**Threads / lessons.** (1) A real turn-of-month USD-strength drift exists (USDCHF +0.177 ATR/5d) but is
SUB-COST — the FIRST non-directional mechanism, and EDGE<COST holds for it too → the cost hurdle is
**mechanism-general**. (2) SIX families across all levers (entry/TF/universe/exit/mechanism) now say the
realizable gross edge of simple long-only FX signals (~0.1–0.2 ATR or coin-flip capture) is below the
FundedNext cost hurdle. A deployable long needs a MUCH larger per-trade gross edge or a different cost regime.
(3) month-of-year seasonality = 11-sample noise. (4) Built reusable `make_time_exit_predicate` (BUILT) +
`build_null_signal_evaluation` (arc 1000) — calendar/hold and soundness tooling now in place.

**FLAGS (code not merged):** `A1Config.time_exit_bars` defined but NOT wired into the Order by A1 (worked
around with the signal-class time-exit predicate; config-level time exit silently no-ops — human-gated fix).

### arc_3001

**Drift-lens scan — is the directional-long death a capture-metric artifact?** (chat 3000–3999). Full record:
[`arcs/arc_3001_drift_lens_scan.md`](arcs/arc_3001_drift_lens_scan.md). No council (cheap-kill).

**Idea + why.** A *methodological* probe, not a new family. Six directional-long arcs (0,1000–1003,3000) all
judged entries with **+1R-before-SL capture**, which arc 1004 flagged as **blind to small persistent drift**
(arc 1003 saw +0.10R gross under sub-0.50 capture). Before declaring the directional-long space dead, re-scan
it with the **correct mean forward-DRIFT lens**. Distinct from the 1000s chat: they take the calendar/time
drift axis (month-end); I take price/volatility/structure. (Reconsidered my own arc-3000 "portfolio/selection"
suggestion and rejected it as **premature** — you can't diversify net-negative components positive; need a
net-positive component first, and none exists.)

**What happened.** `fwd_drift_12 = (mid_close[s+12] − entry@s+1 ask)/(2·ATR[s])`, R units, gross, 28 pairs, IS
2010–2020. **NO instrument × condition clears the ~0.05–0.10R cost hurdle.** By group (uncond / up-spike /
dn-spike / uptrend / mom_hi / low-vol): MAJOR −0.019/+0.007/−0.072/−0.031/−0.032/−0.004; COUPLED
−0.083/−0.101/−0.138/−0.074/−0.101/−0.131; TREND_X −0.012/**+0.023**/−0.014/−0.012/−0.005/−0.021. Best cell
anywhere = TREND_X post-up-spike **+0.023R gross** (< cost). Confirmatory triage on that best cell
(post-up-spike continuation, 12 trending crosses; pool n=1375, cap 0.4982, gross mean final_r +0.088R): net
**2013 +16.74% / 2016 −28.07% / 2019 −21.01% → worst −28.07%, mean −10.78%, 2/3 neg → KILL.**

**Verdict: FAIL (metric-robust closure).** The directional-long failure is NOT a capture-metric artifact — the
drift lens agrees with the capture lens. No cost-clearing forward drift exists in price/vol/structure on any
instrument; the lone positive cell is sub-cost and regime-fragile (the arc-1003/1004 momentum-on-crosses
signature). Directional price-structure long is now closed under BOTH metrics.

**Threads / lessons.** (1) **Directional-long death is METRIC-ROBUST** (capture AND drift; 28×6 scan, best
gross drift +0.023R < cost) — closes the arc-1004 "blind-to-drift" loophole. (2) **Coupled crosses drift
NEGATIVE for longs** (−0.08R/12bars; worse oversold/low-vol) — independent re-confirm of arc 3000 from the
drift angle. (3) Faint trending-cross momentum is regime-fragile + net-negative — re-confirms 1003/1004. (4)
**Portfolio/selection is premature/empty** until a net-positive component exists (none does). **Surviving steer
(singular, non-price-structure):** flow/calendar (1000s chat's axis) or a genuinely novel construction — arc
3002 needs a *generatively different* idea (candidate for a LIGHT generative council at the idea-fork, §5b),
since the 3000s side has now closed the price-structure directional space.

**FLAGS (code not merged):** none. Scan + signal + drivers scratch `_disco3_work/`.

### arc_3002

**Volume-confirmation long — the last untouched data column** (chat 3000–3999). Full record:
[`arcs/arc_3002_volume_confirmation_long.md`](arcs/arc_3002_volume_confirmation_long.md). No council
(no reachable ceiling once full IS unmasked the lucky triage).

**Idea + why.** Every arc conditioned on price/vol/structure/timeframe/calendar; **none used `volume`** — yet
the panel carries a tick-count `volume` column (activity proxy). Volume-confirmation (price moving on high
volume = informed = continuation) is the last untouched data axis. Re-used arc 3001's forward-drift lens with
volume conditionings, 28 pairs, IS.

**What happened.** Group-level volume-conditioned drift (gross R): NO cell clears the ~0.05–0.10R cost hurdle;
best group cell = TREND_X vol-spike(>2.5×) **+0.0475R** (thin n=710). Per-pair flags were scattered single-pair
(GBPJPY lovol_pull +0.092 etc.) = regime-luck, not built. Triaged the best cross-pair cell (vol-spike long, 12
trending crosses; pool n=839, cap 0.478, **gross mean final_r +0.0035** ≈ nil): 3-fold (2013/2016/2019) =
**+11.43 / +11.97 / −16.25 → mean +2.38%, 2/3 pos → NOT deeply negative → proceeded** (first non-negative
triage in 3000s). **Full IS WFO (all 10 folds) unmasked it: 2011−13.6, 2012−0.8, 2013+11.4, 2014−11.5,
2015−3.6, 2016+12.0, 2017−9.9, 2018−12.9, 2019−16.3, 2020+6.1 → worst −16.25%, mean −3.91%, 7/10 NEGATIVE →
FAIL.** The triage was lucky (sampled 2013 & 2016, the two most momentum-friendly years).

**Verdict: FAIL.** Volume-confirmation does not rescue the directional base. **Diagnosis (unifying): volume
predicts move MAGNITUDE, not DIRECTION** — exactly arc 1001's volatility-contraction finding re-derived on the
volume axis. Both volume & volatility say a *bigger* move is coming, not *which way*; a long-only bet on them is
a higher-variance directional coin-flip → regime-dependent, net sub-cost. Direction (≈0.49) is the binding
constraint regardless of the magnitude predictor.

**Threads / lessons.** (1) **Volume = magnitude, not direction** (unifies arc 1001); the last untouched data
column is now closed. (2) **METHODOLOGICAL — the 3-fold triage {2013,2016,2019} over-samples momentum-friendly
years** (2013/2016 are positive for almost every momentum-ish signal — cf. arc 1000 +11.98% 2013, arc 3001
+16.74% 2013); a regime-dependent signal can show a *positive* triage mean yet be 7/10 neg on full IS. **A
non-negative 3-fold triage is necessary but not sufficient — confirm with full IS before any
diagnosis/council investment** (or include a chop year 2014/2017/2018 in the triage). (3) Single-pair drift
flags = regime-luck (GBPJPY +0.092R), not edge — keep the cross-pair-robustness discipline. **Surviving steer:**
price/volume/structure/timeframe directional space is now systematically closed from the 3000s side and
calendar is weakening (1005) → **arc 3003 is the right point for a LIGHT generative council (§5b)** — a genuine
idea-fork — or a meta-pivot (is this apparatus capable of any edge, and what must change?).

**FLAGS (code not merged):** none requiring the canonical core. Methodological note only: the chat-convention
triage year-set {2013,2016,2019} is momentum-biased (lesson 2); canonical `build_v3_folds` is unaffected.
Scan + signal + drivers scratch `_disco3_work/`.

### arc_3003

**Regime detection for momentum — is the trending regime detectable in advance?** (chat 3000–3999). Full
record: [`arcs/arc_3003_regime_detection_momentum.md`](arcs/arc_3003_regime_detection_momentum.md). No council.

**Idea + why.** The central unsolved question after 12 arcs: momentum drift is real but regime-dependent (good
trending years 2013/2016/2020, bad chop — arcs 1003/3001/3002). If the trending regime were detectable IN
ADVANCE, trade momentum only then. Tested with Kaufman Efficiency Ratio (causal trending-ness ∈[0,1]) — the
most direct "is this trending now" measure, not yet tried (arcs 1000/1001 tried dispersion/vol-level regimes).

**What happened.** Bucketed forward 12-bar drift by ER60 on trending crosses (momentum entries above SMA50):
chop→trend drift = −0.010 / −0.001 / +0.011 / +0.019 / **−0.061** (top ER bucket). **REGIME INVERSION** — the
strongest-trending regime has the MOST NEGATIVE forward drift (strong trends REVERT, don't persist); the naive
"trade momentum when trending" is actively wrong; the mid-ER band is mildly positive but sub-cost. Fail-the-
best-version: gated momentum to the mid-ER sweet spot [0.09,0.23] (best band). Pool n=9250, capture 0.478,
**gross mean final_r −0.0547** (the +0.019R *raw* drift evaporates to NEGATIVE SL-honest expectancy). Full IS
WFO (skipped the lucky 3-fold triage per arc 3002's lesson): 2011−63, 2012+11, 2013+32, 2014−61, 2015−19,
2016−31, 2017−4, 2018−30, 2019−45, 2020−43 → **worst −63.11%, mean −25.30%, 8/10 neg, DDs to 66% →
catastrophic FAIL.**

**Verdict: FAIL.** The trending regime is NOT detectable in advance as a cost-clearing momentum filter; the
strongest-trend regime inverts (reverts). Closes the regime-detection lever — the central remaining hope for
rescuing the faint momentum drift.

**Threads / lessons.** (1) **Trending regime NOT detectable for momentum** (ER inverts: strong trends revert
−0.061R). **Regime-conditioning has now failed across THREE measures — dispersion (1000), vol-level (1001),
trending-ness/ER (3003)** → no exploitable regime structure for a long-only FX directional bet; extremes
mean-revert, the middle is random. (2) **METHODOLOGICAL: raw forward drift OVERSTATES SL-honest expectancy**
(+0.019R raw → −0.055R SL-honest; take-the-loss kills slow drifts). The drift lens (arc 3001) is an OPTIMISTIC
cheap pre-filter; a sub-+0.05R drift cell is ~guaranteed SL-honest-negative. Only the engine verdict counts.
(3) Wide regime band fires too often (9250 trades) → frequency amplifies a negative edge (−25%/66%DD).
**Surviving:** the 3000s side has now closed directional entries (both metrics), instrument universe, volume,
and regime detection; combined with 1000s/2000s (timeframe, exits, calendar, convexity) the
price/volume/structure/regime directional space is exhausted. **Arc 3004 → LIGHT generative council (§5b):** I
am genuinely at a stuck-point/idea-fork; convene the council for generative perspectives / to pressure-test a
meta-pivot (is this apparatus capable of any edge, and what structural change is required?) before another
lone guess. Portfolio/selection still premature (no net-positive component).

**FLAGS (code not merged):** none. Scan + signal + drivers scratch `_disco3_work/`.

### arc_1006

**Weekend gap-down-fill long, JPY crosses (generative-council idea)** (chat 1000–1999). Full record:
[`arcs/arc_1006_weekend_gap_fill_long.md`](arcs/arc_1006_weekend_gap_fill_long.md). Council transcript (LIGHT
generative, §5b): [`results/arc_1006_weekend_gap_fill/council_transcript.md`](results/arc_1006_weekend_gap_fill/council_transcript.md).

**Idea + why.** Directional + calendar closed fleet-wide (arc 3001 flagged a §5b generative council). Convened
it (5 lenses); synthesized the WEEKEND GAP-FILL (untouched by arcs 0–3001; price-only, long-only): big weekend
gap-downs tend to FILL (weekend positioning/liquidity overhang reverts Monday). Rejected SL-geometry (1/W cancels
the sign), triangular divergence (dead at H4), deferred the Devil's null-confirmation (premature).

**What happened.** Observation: big gap-downs (<−0.5 ATR) fill (+0.20 ATR pooled fwd); **JPY crosses strongest**
(EURJPY +0.48 ATR, capture 0.54 — biggest gross edge of the run); moderate gap-downs CONTINUE down. Best version:
long a JPY cross on a >0.5-ATR weekend gap-down, 24-bar time exit, SL=2ATR. Pool 396 trades, **mean final_r
+0.1127 gross**. Triage 2013 −2.06% / 2016 +3.20% / 2019 +7.45% (2/3 POSITIVE, mean +2.86%, tiny DDs) → PROCEED.
Full WFO: IS mean **+0.69% (FIRST mean-positive IS of the run)**, beats random null (−0.60%), DDs 2–9% — BUT
all-folds-positive NO (5/10 IS neg worst −6.79%; OOS 4/6 neg worst −4.13%). One reasoned refinement (uptrend
filter close>SMA50): negatives 5/10→4/10 but still not all-folds-positive AND over-thinned (min 5 trades/fold).

**Verdict: FAIL** (not all-folds-positive) — but **categorically the most promising result of the run**: a REAL,
mean-positive, beats-random, low-DD edge that is merely FOLD-FRAGILE (JPY-cross weekend gaps are tail-event-
timing-dependent), not a coin-flip.

**Threads / lessons.** (1) Weekend gap-FILL is REAL on JPY crosses — first net-positive long edge; fold-fragile,
large-gap-only (moderate gap-downs continue down). (2) **RE-OPENS the portfolio thread**: arc 3001 noted you
can't diversify net-NEGATIVE components positive — but this is the FIRST net-POSITIVE (fold-fragile) component;
a decorrelated combination of ≥2 such edges could plausibly reach all-folds-positive (diversification cuts
fold-variance, preserves positive mean). **HIGH-VALUE fleet steer: hunt for a 2nd net-positive edge to combine.**
(3) Generative council earned its keep (steered off the exhausted rut to the first promising lead) and queued
untested ideas: spread-tier gating, vol/cost-ratio conditioning, Asia→London timing, Devil's null-confirmation.

**FLAGS (code not merged):** none. Reused BUILT `make_time_exit_predicate`; drivers scratch `_disco_work/`.

### arc_2001

**Weekend-gap-fill long, MAJORS** (chat 2000s). Full record:
[`arcs/arc_2001_weekend_gap_fill_long.md`](arcs/arc_2001_weekend_gap_fill_long.md). HEAVY diagnosis council:
[`results/arc_2001_weekend_gap_fill_long/council_transcript.md`](results/arc_2001_weekend_gap_fill_long/council_transcript.md).

**Idea + why.** EDGE<COST exhausts price-DIRECTION; the price-only corpus leaves price-STRUCTURE / time-
STRUCTURE. Untouched axis: weekend/weekly-open GAPs — discrete, sometimes-large moves with a documented
partial-fill tendency, attacking EDGE<COST from the move-size side. Observe first (drift metric).

**What happened.** OBSERVATION (H4 majors, IS 2010-2020): a clean, monotone, SYMMETRIC gap-fill — DOWN
>1·ATR weekly-open gaps drift +0.45 ATR/~1day (frac+ 0.59); UP >1·ATR gaps drift −0.57 ATR (frac+ 0.36).
The ONLY conditioning variable in the programme to push the directional hit-rate clearly >0.50 with sizable
drift. Long-only tradeable side = buy big down-gaps. Characterize (threshold×SL grid): the down-gap's
ADVERSE continuation is large (median MAE −1.1 to −1.3R, P(MAE≤−0.5R)=0.77-0.89) — the honest i+1 entry
(after the gap bar) lands INSIDE the continuation, so honest +1R-before-SL capture only 0.45-0.47 and gross
meanFinalR is positive but small (+0.07..+0.18R). 3-fold triage looked near-break-even (favorable-fold
LUCK), but the FULL 10-fold IS WFO is mean −1 to −2.5%, 6-7/10 folds negative for every threshold/SL/exit.

**Diagnosis + HEAVY council.** Diagnosis: real edge, but the i+1 entry sits in the adverse continuation and
the small surviving edge is cost-eaten. Council (5 lenses + 3 reviewers + chairman) → **KILL**: (1) the
delayed/confirmation-entry refinement is a contamination trap (invented+tuned on IS, validatable only by
spending pristine OOS); (2) the sample can't support the all-folds-positive judge even if real (~6-8
trades/OOS-year filtered → coin-flip); (3) the exciting 0.59/monotone drift is measured from the
UNTRADEABLE gap-bar open (hindsight framing); the tradeable i+1 edge is the small one costs eat. "The
mechanism survives; the trade does not." CC committed (no override). **OOS deliberately NOT touched** (no
IS-clearing system + preserve the holdout).

**Verdict: FAIL** (real edge, uncapturable) — the 9th EDGE<COST result, but the first whose binding
constraint is CAPTURABILITY/frequency/long-only-access, not absence of edge.

**Convergence with arc 1006 (chat 1000s, landed mid-arc).** 1000s independently found weekend gap-down-fill
(generative council) on JPY CROSSES — the run's MOST promising result (FIRST mean-POSITIVE IS edge +0.69%,
beats random) but fold-fragile FAIL. **Two chats, two universes, converged: weekend gap-fill is the realest
edge found.** CROSS-FINDING: mean-POSITIVE on JPY crosses, mean-NEGATIVE on majors → **JPY crosses are the
better gap universe; majors too efficient.** My distinct adds: (1) majors are the weaker gap universe (don't
pursue gap-fill there); (2) **FLAG-1 the long-only apparatus blocks the STRONGER short side** (UP-gaps:
−0.57 ATR drift, 0.64 accuracy — the better edge, structurally untradeable; short support = human-gated
code, NOT self-merged); (3) the OOS-pristine + n-too-thin-for-the-judge discipline.

**Threads / lessons.** (1) Weekend gap-fill = the first real, monotone, mechanistically-clean edge (weekend-
illiquidity repricing toward prior close), fails on CAPTURABILITY not edge. (2) A 3-fold triage can be
favorable-fold LUCK — run the full IS WFO before believing it (the same 2013-regime-luck that burned
1000/1003). (3) OOS-pristine discipline: when the only rescue is an IS-tuned refinement, FAIL-on-IS +
preserve-OOS, don't burn the holdout (HEAVY council). (4) **HIGH-VALUE open thread (shared w/ 1006's steer):
gap-fill on JPY crosses is net-POSITIVE-but-fold-fragile; a decorrelated COMBINATION of ≥2 net-positive
gap-type edges may reach all-folds-positive — but the single most promising gap signal (the UP-gap short) is
blocked by long-only.** The 2000s lane next should AVOID the majors gap-fill (done) and the 1000s
gap-portfolio thread (theirs); a lower-timeframe gap test (more events) or the FLAG-2 upstream diagnostic
(is H4+FundedNext generically hostile to fill/reversion?) are open.

**Tooling:** built + registered `discovery/tools/gap_signals.py` (`WeekendGapFillLongSignal`). TOOL_REGISTRY
BUILT updated.

**FLAGS (code not merged):** **FLAG-1** long-only apparatus blocks the stronger UP-gap short side (canonical-
core change, operator-gated, NOT merged); **FLAG-2** future-arc seed: is H4+FundedNext cost structure
generically hostile to fill/mean-reversion entries? Drivers scratch `_disco2000_work/`.

### arc_1007

**Gap-fill best-version test: exit AT the fill target** (chat 1000–1999). Full record:
[`arcs/arc_1007_gap_fill_exit_at_target.md`](arcs/arc_1007_gap_fill_exit_at_target.md). No council.

**Idea + why.** Develop the run's only net-positive lead (arc 1006 gap-fill, 24-bar exit was arbitrary).
Mechanism-aligned hypothesis: exit AT the gap origin (prior-week close) to take the reversion cleanly. Built
`make_price_target_exit_predicate` (BUILT). One reasoned version, measured IS then OOS.

**What happened.** EXIT-AT-TARGET IS WORSE: IS mean +0.69% (24-bar) → **−1.21%** (exit-at-target), now LOSES
to the random null (−0.60%); not all-folds-positive (6/10 IS neg, OOS 4/6 neg). **Diagnosis:** the gap-fill
edge is the OVERSHOOT — the JPY-cross snapback runs PAST the gap origin and continues; capping at the origin
cut the big winners while keeping full −1R losers → asymmetry flipped negative. The arbitrary 24-bar "let it
run" exit was capturing the overshoot.

**Verdict: FAIL.** Arc 1006's 24-bar version remains the best version of the gap-fill; it is NOT
all-folds-positive in any tested exit.

**Threads / lessons.** (1) The JPY-cross weekend gap-down edge is OVERSHOOT/continuation, NOT fill-to-origin
(re-frames the mechanism: big gap-down → snapback past the prior close). (2) Confirms arc 1004 ON THE POSITIVE
LEAD: exit engineering can't fix fold-fragility (it's tail-event-timing-driven, not exit-driven). (3) The BEST
version of a lead can be the naive one — the "mechanism-aligned" refinement was a worse hypothesis (mechanism
mis-stated). (4) gap-fill stays the portfolio-component candidate (arc 1006 24-bar version); needs a 2nd
net-positive edge. Queued/untested: spread-tier gating, vol/cost conditioning, Asia→London, Devil's
null-confirmation (the decisive closure test as leads narrow).

**FLAGS (code not merged):** none. Built + registered `make_price_target_exit_predicate` (works; just not the
right exit for an overshoot edge). Drivers scratch `_disco_work/`.

### arc_2002

**H1 entry-resolution test of the weekend gap-fill** (chat 2000s). Full record:
[`arcs/arc_2002_h1_gap_resolution.md`](arcs/arc_2002_h1_gap_resolution.md). No council (confirmatory
cheap-kill); reused BUILT `WeekendGapFillLongSignal` on H1.

**Idea + why.** The arc-2001 HEAVY council ruled the gap-fill "uncapturable at H4 RESOLUTION" — the honest
i+1 entry fills 8h after the gap open, deep in the adverse continuation. Entry resolution is the one
UNBLOCKED, non-curve-fit, non-colliding lever (distinct from 1000s' cross-gap-portfolio): on **H1** the i+1
entry is ~2h after the open (~6h earlier), so it should capture the fill the H4 entry misses. Mechanistic
change (finer bars), not a tuned parameter — so the council's contamination/thinness objection doesn't block
it. Is "at H4 resolution" load-bearing, or uncapturable at any resolution?

**What happened.** H1 majors, IS 2010–2020 (H1 loads ~1s/pair from m1 cache). Characterize: H1 capture
**0.450–0.454 ≡ H4 0.45–0.47**; meanMAE **−1.2 to −1.3R ≡ H4**; stop 0.76–0.80. Entering 6h closer changes
NOTHING — because the down-gap continues adversely for hours-to-days, so shaving 6h off a multi-hour
continuation doesn't escape it. Triage (3 folds, comparable thr2.0 SL2.0 partial): +2.43/−2.02/−1.04, mean
−0.21%, 2/3 neg — the same near-break-even-to-sub-cost as H4. → confirmatory cheap-kill.

**Verdict: FAIL (confirmatory).** Entry resolution is NOT the binding constraint. The gap-fill's adverse
continuation is **TIMEFRAME-INVARIANT** (H4 ≡ H1); the long-only fill entry is structurally late at any bar
size. Empirically vindicates the arc-2001 council ("the mechanism survives; the trade does not") — and
complements arc 1007 (1000s: the gap edge is OVERSHOOT not fill, capping at the target is worse) and arc
3001 (intraday down-spikes CONTINUE not revert). The long-side gap-fill is now closed across resolutions
AND exit structures.

**Threads / lessons.** (1) The weekend gap-fill's adverse continuation is TF-INVARIANT (H4≡H1) — entry
resolution was never the lever. (2) Convergent closure of the long-side gap-fill from 4 angles across 2
chats: universe (1006 crosses>majors), exit-at-target (1007 overshoot-not-fill), entry-resolution (2002
TF-invariant), and intraday-analog (3001 spikes continue). (3) The only live gap paths remain the two named
in arc 2001 — neither a long-only price-signal: the cross-universe net-positive PORTFOLIO (1000s' active
thread) and the blocked UP-gap SHORT (FLAG-1, stronger side, human-gated code). (4) **The price-only
long-only H4/H1/D1 space is now comprehensively mapped across 15 arcs / 3 chats with no capturable
EDGE>COST edge found.** Next 2000s arc should weigh a LIGHT generative council (§5b, flagged by arc 3001)
for a genuinely non-price-structure construction, vs documenting the structural blockers as the standing
result.

**FLAGS (code not merged):** none new (carries FLAG-1 long-only-blocks-short-side, FLAG-2 H4-cost-vs-
reversion from arc 2001). No new tool (reused `WeekendGapFillLongSignal(primary_tf="H1")`). Driver scratch
`_disco2000_work/`.

### arc_2003

**Cross-pair relative-value divergence (laggard catch-up long)** (chat 2000s). Full record:
[`arcs/arc_2003_crosspair_divergence.md`](arcs/arc_2003_crosspair_divergence.md). LIGHT generative council
(§5b) summarized inline in the arc doc (no separate transcript — light/generative).

**Idea + why.** 15 arcs closed the obvious price-structure space → genuine dry-log fork → LIGHT generative
council (3 lenses). Two of three converged on **cross-pair relative-value divergence**: long the
under-performing leg of a correlated XXX/USD pair when the synthetic cross is at an extreme (relative-value,
NOT single-pair direction — distinct from arc 3000's absolute RSI-oversold-on-the-cross). The Devil lens'
strongest self-judged call was **CONCEDE** (the binding blocker is the long-only constraint; the gap-fill's
strong side is short). CC synthesized: test the convergent divergence idea; defer the M1 spread-spike idea
(sub-pip reversion vs ~3-pip cost → EDGE<COST).

**What happened.** Drift observation (D1, ex-ante, correlated groups EURUSD↔GBPUSD, AUDUSD↔NZDUSD): the
laggard does NOT catch up. At z<−2 (A under-performed) fwd drift of A = −0.109/−0.046/−0.007 ATR (5/10/20
bars), frac+ ~0.49–0.53; per-group z<−1 fwd10 −0.202 EURvGBP, −0.150 AUDvNZD (laggard keeps lagging). The
catch-up hypothesis is FALSIFIED — relative performance of correlated majors is weak momentum/coin-flip, not
mean-reverting. → cheap-kill at observation.

**Verdict: FAIL (cheap-kill).** No relative-value reversion in FX majors (3rd confirm with 1000/3001).

**Threads / lessons.** (1) No relative-value catch-up in correlated majors — the laggard continues lagging
(pairwise-divergence angle confirms the cross-sectional 1000 + drift 3001 findings). (2) **The generative
council INDEPENDENTLY reached arc 2001's FLAG-1 via its CONCEDE lens**: across 16 arcs/3 chats the one real
edge (weekend gap-fill) is mechanism+direction+magnitude-confirmed and its STRONGER leg is the UP-gap SHORT,
which long-only structurally forbids — **the binding blocker is the constraint, not the search space.** (3) A
generative council on a heavily-mapped space mostly re-surfaces the known answer-space; its value here was
the independent CONCEDE corroboration + cheap in/out on the divergence idea. (4) Low-odds untested seed: M1
bid-ask spread-spike mean-reversion (microstructure liquidity dislocation) — almost certainly EDGE<COST.

**FLAG-1 (REINFORced — most actionable):** add short support (operator/human-gated canonical-core change,
NOT self-merged) to unlock the gap-fill's dominant UP-gap short leg — the single highest-leverage operator
decision the programme has surfaced. **FLAGS (code not merged):** none new. Driver scratch `_disco2000_work/`.

### arc_3004

**The stop confound (council-driven) → clean ESCALATION** (chat 3000–3999). Full record:
[`arcs/arc_3004_stop_confound_and_escalation.md`](arcs/arc_3004_stop_confound_and_escalation.md). Council
transcript: [`results/arc_3004_stop_confound/council_transcript.md`](results/arc_3004_stop_confound/council_transcript.md).
**Operator-facing finding:** [`ESCALATION_apparatus_capability.md`](ESCALATION_apparatus_capability.md).

**Idea + why.** Genuine stuck-point after 3000–3003 closed the price/volume/structure/regime directional space.
Per §5b/§7 convened the LIGHT generative council (`/llm-council-discovery`). The council surfaced a confound I
(and ALL 13 prior arcs) had missed: **every arc used a 2·ATR SL-first stop.** The take-the-loss tax converts
positive gross drift → negative honest expectancy (+0.019R raw → −0.055R, arc 3003), so "directional space is
closed" was confounded with "a tight stop truncates a slow drift." Chairman (commit): investigate-X-first —
re-score the strongest +gross-drift entry with the stop REMOVED (pure time-exit) before escalating; read TWO
numbers (stop effect, cost effect); stop-removed+all-folds-positive → stop was the wall, else → escalate clean.

**What happened.** Clean A/B on the post-up-spike trending-cross entry (arc 3001 pool, gross +0.088R), full IS
WFO, FundedNext costs ON, scored only by MultiPairBacktester. Stop removed via the BUILT
`make_time_exit_predicate` (arc 1005 — reused, not rebuilt; geometry only) + wide 6·ATR sizing-only SL.
BASELINE (2·ATR partial/runner): mean −4.52%, 6/10 neg. STOP-REMOVED time-exit: N=6 mean −2.13%/8neg; N=12
−3.46%/8neg; N=24 −2.77%/7neg; N=48 **+0.62%/5neg, worst −20.4%**. **Removing the stop lifts mean ~4–5pp (the
take-the-loss tax is REAL) but NO horizon is all-folds-positive** — the residual ≈ cost and is regime-dependent
(chop years stay negative regardless of stop). The chairman's "stop-removed but still sub-cost" branch.

**Verdict: FAIL — and the directional space is now CLEANLY closed.** The stop was a contributing ~4–5pp drag,
NOT the binding wall. → **ESCALATION raised** (operator decision, NOT a halt; loop continues): the fixed
apparatus (long-only, single-instrument, per-trade SL/TP, price+volume, FundedNext costs) cannot express a
deployable FX edge — it can only bet DIRECTION, which liquid FX prices to a coin-flip; durable FX edges are
relative-value/carry needing a structural unlock. Ranked unlocks: (1) second simultaneous leg (market-neutral /
cost-mutualising — the only lever not requiring beating 0.50 per-trade), (2) shorting, (3) a real tighter-cost
execution regime (NB must match an obtainable broker or it re-introduces Arc-10 gate-mismatch).

**Threads / lessons.** (1) **The 2·ATR SL was a real ~4–5pp drag but NOT the wall** — removing it (pure
time-exit) lifts mean −4.52%→~0% yet still 5–8/10 folds negative; the deeper wall is gross-drift≈cost +
regime-dependence. Retires "the stop was the wall" with a clean unconfounded test. (2) **Directional space is
closed structurally, not by tuning** → escalation. (3) **TOOLING DISCIPLINE: check TOOL_REGISTRY BUILT FIRST
before writing** — I started to rebuild the time-exit predicate; the Write failed (it existed, arc 1005); I
called the BUILT tool instead. Cross-chat tools compound (reused arc 1005's time-exit). (4) The council earns
its keep at a real stuck-point — it found a confound 4 arcs missed and its measure-then-decide prevented
escalating on confounded data (Arc-10 lesson in research-conclusion space). **Remaining thin in-apparatus
thread:** cross-rate triangulation lag (Mechanism lens) — likely M1/sub-cost, a future cheap-kill candidate.

**FLAGS (code not merged):** **`A1Config.time_exit_bars` defined but UNWIRED** (never consumed by A1; a time
exit must be an `ExitPredicate`). Caused a silent no-op in an early version of this arc's test. Independently
flagged by arc 1005; re-flagged. Fix human-gated (wire it, or remove the dead param). Scratch drivers
`_disco3_work/`.

### arc_3005

**Cross-rate triangulation residual — closing the last in-apparatus thread** (chat 3000–3999). Full record:
[`arcs/arc_3005_triangulation_residual.md`](arcs/arc_3005_triangulation_residual.md). No council (cheap-kill at
observation).

**Idea + why.** The arc-3004 council named ONE remaining concrete in-apparatus idea that is genuinely
*non-directional*: cross-rate triangulation lag (synthetic cross from its two USD legs vs the quoted cross).
Low conviction (Mechanism lens: "likely M1, dislocation may be < the cross spread") but closing it completes the
escalation's coverage. Tested at H4: residual = (quoted_mid − synthetic)/synthetic, bps, contemporaneous closes.

**What happened.** Across 6 triples (EURGBP, EURJPY, GBPJPY, AUDJPY, EURAUD, AUDCAD): residual median ≈ 0
(±0.05 bps); |residual| exceeds the cross spread only 1.3–6.1% of bars (within spread-noise); **forward-6-bar
convergence correlation ≈ 0.00–0.02** (no predictive power). Triangular arb pins quoted=synthetic at H4 closes;
any lag is sub-second/tick — out of apparatus scope and below the cost floor. Nothing to trade.

**Verdict: FAIL (cheap-kill at observation).** No H4 triangulation dislocation. The last concrete
non-directional in-apparatus thread is closed.

**Threads / lessons.** (1) No H4 triangular-arb residual (median ≈0, within spread, conv-corr ≈0). (2) **The
3000s chat's in-apparatus space is now comprehensively exhausted** — directional (both metrics; all
instruments/timeframes/regimes/exits/stop-geometry), volume, calendar, convexity, and triangulation all closed.
The operative finding is the **arc-3004 escalation**: a deployable FX edge needs a structural unlock (second
leg / shorts / real tighter-cost regime) = the operator's decision. No grounded in-apparatus arc remains;
further long-only single-instrument directional variants would grind a cleanly-closed space. **3000s chat has
reached its in-apparatus terminus; escalation pending operator.**

**FLAGS (code not merged):** none. Driver scratch `_disco3_work/observe6_triangulation.py`.

### arc_1008

**Gotobi-day Tokyo-morning USDJPY long (microstructure settlement flow)** (chat 1000–1999). Full record:
[`arcs/arc_1008_gotobi_usdjpy_tokyo_fix.md`](arcs/arc_1008_gotobi_usdjpy_tokyo_fix.md). No council (falsified
at observation; no worthwhile-ceiling fork).

**Idea + why.** Fresh eyes, honest-era only. 16 arcs closed shallow directional prediction; the ONE net-positive
lead (weekend gap-fill, arc 1006, PORTFOLIO) is a discrete-FLOW-event reversion, not continuous prediction. My
pre-shorts lane (dispatch) = hunt a **decorrelated** 2nd net-positive long-only component; refining the gap-fill
gives a *correlated* one (already mapped 4 ways). The most-documented, untested, decorrelated long-only FLOW is
the **gotobi effect**: on days divisible by 5 (+ month-end), Japanese corporates settle invoices → banks buy
USD/JPY → documented upward USDJPY drift into the 9:55 JST (=00:55 UTC) fixing. Long-only-tradeable (long
USDJPY, a major), event-timed, decorrelated from weekend gaps. Attacks frontier item 3 (execution/
microstructure, available NOW pre-shorts). Documented to have decayed post-2015 → **observe before believing.**

**What happened.** The panel's EET-00:00 H4 bar (starts 22 UTC winter / 21 UTC summer, ends 01/02 UTC) fully
contains the Tokyo run-up + the 00:55 UTC fix. OBSERVATION (gross mid bp, gotobi tagged 5/10/15/20/25 + last
trading day, prior-trading-day shift; EURUSD placebo): gotobi Tokyo-morning USDJPY return **−0.06 bp** vs
non-gotobi **+0.13 bp** — gotobi is WORSE (frac+ 0.490). By era: 2010–2014 gotobi +2.00bp ≈ non +1.98bp (NOT
gotobi-specific — generic Abenomics Tokyo-morning USDJPY drift, identical on non-gotobi days, gone by 2015);
2015–2020 −0.98 vs −0.88; 2021+ −0.97 vs −0.49 (gotobi worse). The gotobi *premium* ≈ −0.2 bp gross vs ~2 bp
FundedNext round-trip. H1 fix-window check (rule out a run-into-fix the H4 bar nets against the post-fix hour):
largest gotobi premium anywhere **+0.25 bp** (fix-hour-only, frac+ 0.501 = coin flip), negative in the other
four windows — confirms the H4 finding at finer resolution (arc 2002 discipline). **No edge at any resolution.**

**Verdict: FAIL (cheap-kill at observation) → KILL.** The documented gotobi mechanism produces no capturable
H4/H1 USDJPY bias in 2010–2026 — fully arbitraged (one of the most-published FX seasonalities) or never of a
surviving magnitude. Premium ≤ +0.25 bp gross (coin-flip) ≪ ~2 bp cost; the honest engine (costs + SL-first)
can only make a no-edge gross signal worse (arc 3003) → engine compute not spent. No pool/council/null needed.

**Threads / lessons.** (1) **Gotobi is absent/arbitraged in 2010–2026** (gotobi ≤ non-gotobi every era; the
only +era is generic regime drift). A real *because* (the flow genuinely existed) does NOT imply a still-
capturable edge — the famous seasonality dissolves on honest data. (2) **A documented flow is not a free pass
past observation** — arc 1005 (turn-of-month) found a real-but-sub-cost flow, arc 1008 a famous-but-~zero one;
calendar/microstructure-flow magnitude (not existence) is what's measured against the cost floor. The
microstructure-settlement-flow frontier item is now tested and dry for gotobi. (3) **Mean is additive across
sub-bars** — the cheap H1 run-into-fix vs post-fix split is the right way to rule out a netted-away intraday
run (quantifies arc 2002's resolution discipline). (4) **Surviving pre-shorts long-only lanes** narrow further:
calendar/flow now weakly mapped (TOM sub-cost 1005, gotobi absent 1008); remaining untested = multi-timeframe
STRUCTURE as a setup (not a regime filter), and the standing escalation note — after 17 arcs the only net-
positive long-only edge is the weekend gap-fill (PORTFOLIO); no 2nd decorrelated net-positive long-only
component has been found, reinforcing arc 3004's escalation + the 2001/2003 FLAG-1 shorts/second-leg unlock as
the highest-leverage operator move.

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (gotobi-date tagging is a
one-off calendar helper kept in scratch, like arc 3005's triangulation observer). Drivers scratch
`_disco_work/arc1008_observe.py`, `arc1008_h1_fixwindow.py` (reproducible from the arc doc).
