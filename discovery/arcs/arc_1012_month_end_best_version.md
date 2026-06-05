# Arc 1012 — Month-End Reversion: best-version completion (§5f) + survivor-ceiling test

> **Arc id:** 1012 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL the survivor judge → disposition PORTFOLIO (re-affirmed,
> ceiling confirmed).** The §5f best-version test that arc 1011 left informal is now
> complete: the full registered exit/SL menu (24 configs) yields **0 all-folds-positive**,
> and the 3 negative folds are an irreducible **2014–2016 USD-bull regime block** that
> neither of two pre-registered, mechanism-reasoned refinements (quarter-end-only;
> trend filter) can remove. The month-end reversion **cannot be lifted to a solo
> survivor**; it stays a valid PORTFOLIO component. OOS deliberately **NOT touched**.
> **Lever tested:** EXIT/SL (§5f nested WFO) + regime diagnosis on a NON-coin-flip entry.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first take-the-loss) via
canonical `ArcFoldRunner` → `build_v3_folds` IS folds + `judge_all_folds_positive`. No
council (not a stuck-point fork, not a survivor — a protocol-mandated best-version
completion + diagnosis on an existing PORTFOLIO component, cf. arc 1009 for the gap-fill).
Reused BUILT `MonthEndReversionLongSignal` + `make_time_exit_predicate` (geometry/timing
only; scoring canonical). No new tool. Additive record:
[`portfolio-candidates/arc_1011_month_end_reversion_long/best_version_audit_arc1012.md`](../portfolio-candidates/arc_1011_month_end_reversion_long/best_version_audit_arc1012.md).

## (a) Log read — FRESH EYES (honest-era only)

Pulled main (in sync). Honest-era corpus = 23 arcs / 3 chats. **Comprehensively closed:**
shallow single-condition directional prediction (momentum / breakout / reversion / trend,
long OR short), H1/H4/D1, majors + crosses, both capture & drift lenses, every exit/SL,
stop-removed (3004/2004); regime conditioning dead 3 ways; volume = magnitude; gotobi
(1008), round-numbers (1010), triangulation (3005), breakout-retest (3006) all absent;
relative-value catch-up falsified (2003). The arc-3004 **escalation** stands (solo deployable
needs shorts/second-leg — pre-shorts here). **The ONLY winning family = discrete
liquidity/flow-EVENT over-extension → reversion**, with two PORTFOLIO components: the weekend
gap-fill (1006, thoroughly worked across 6 arcs → firmly fold-fragile PORTFOLIO) and the
month-end reversion (1011, **3 days old, under-explored**).

**My pick.** Before hunting a 3rd thin flow component, complete the best-version test on the
under-explored asset — which §5f *mandates*: the month-end entry is NON-coin-flip (beats a
fair null +0.56pp, mechanism-controlled +0.249 ATR excess vs random-day), so its best HONEST
exit MUST be tested before any FAIL. Arc 1011 sampled exits informally (sl_only 2/3/5-bar) and
never diagnosed its 3 negative folds. This is the one asset with genuine **survivor** upside
(all-folds-positive → PASS), not just another PORTFOLIO. Decisive either way: lift it, or
confirm PORTFOLIO is its ceiling and move to the 3rd-component hunt. STOP absent.

## (b) Reproduction (registered tools)
`MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)` → D1, 7 USD majors, canonical
engine, FundedNext costs. **Baseline reproduced byte-for-byte:** sl_only / time-exit 2-bar /
sl 2.0 → `+0.40 +0.29 +0.96 −0.23 −1.14 −0.51 +0.34 +0.90 +1.16 +0.15`, mean **+0.232%**,
7/10 (= arc 1011). fold_id→OOS year: {2:2011,3:2012,4:2013,**5:2014,6:2015,7:2016**,8:2017,
9:2018,10:2019,11:2020} → the 3 negatives are folds 5/6/7 = **2014/2015/2016**.

## (c) §5f nested exit/SL sweep — 0 of 24 all-folds-positive
Full registered menu (`sl_only`, `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_plus_trailing_atr`,
`sl_plus_trailing_swing`, `sl_partial_close_1r_runner_trail`) × SL ∈ {1.5, 2.0, 2.5}, with
time-exit caps {2,3,5}. **Anti-fishing (§5f):** a config qualifies only if the SAME config is
all-folds-positive — no per-fold cherry-pick. **Result: 0/24.** Best fold count 7/10
(sl_only/te2/sl2.0; sl_only/te3/sl1.5). Higher-mean configs (sl_only/te3/sl2.0 +0.505%) are
WORSE on folds (6/10). Longer holds (te5) and all TP/trail policies **dilute** (mean ≤0,
≤6/10) — the reversion is a tight ~2–3 day event; holding past it gives the move back and
exposes to continuation. Re-confirms the corpus-wide "exits don't rescue fold-fragility"
(arcs 0/1004/1007/2004). Because no config cleared IS, **OOS was never touched** (holdout
preserved, §4 + arc-2001 discipline).

## (d) Negative-fold diagnosis — 2014–2016 USD-bull regime block
The 3 negatives are a **contiguous** 2014/2015/2016 block = the strong-USD-bull regime (EUR
1.39→1.05, SNB unpeg Jan-2015, oil crash) — a regime signature, not random. Two
**pre-registered, mechanism-reasoned** refinements (no tuning to the bad folds):
- **(A) Quarter-end only** (*because* quarter-end rebalancing is larger/cleaner) → **FALSIFIED:**
  4/10, mean +0.058%, min-trades 0 (thins catastrophically). Bigger-flow hypothesis wrong.
- **(B) Trend filter close>SMA100** (*because* an informed trend overwhelms the mechanical
  flow) → **FALSIFIED:** 4/10 (SMA50 3/10, SMA200 3/10). Requiring "not a downtrend" HURTS.

The **inverse** (close<SMA100, fire only in a downtrend) is marginally better (8/10, +0.302%)
— a coherent "rebalancing buys the already-weak/underweight leg" story — but it is **post-hoc**
(filter-vs-inverse selected after the fact = fishing) and **still leaves 2015 & 2016 negative**.
The SNB-unpeg / USD-peak folds survive every honest cut. Not claimed; logged as a low-priority
fresh-arc thread only.

## (e) Verdict + threads
**FAIL the survivor judge → PORTFOLIO (re-affirmed, survivor-ceiling closed).** The month-end
reversion's best HONEST version is not all-folds-positive; its 2014–2016 regime drag is
irreducible by exit/SL or pre-registered filter. It remains a valid PORTFOLIO component
(mean-positive, decorrelated corr +0.12 vs gap-fill, threshold-robust) — it simply cannot
stand alone. This is the month-end analogue of arc 1009 (gap-fill audit): the second of the
two PORTFOLIO components is now best-version-completed and firmly fold-fragile.

**Threads / lessons.** (1) **Both PORTFOLIO components are now best-version-closed as
non-survivors** — gap-fill (1006/1007/1009/2002/2004) and month-end (1011/1012). Neither exits
nor reasoned filters lift either to all-folds-positive; both are regime/tail-timing-fragile.
The PORTFOLIO route (combine ≥2 decorrelated components, gate the *combined* book) and the
arc-3004 escalation (shorts/second-leg for a solo) remain the only live paths. (2) **§5f
completed on the month-end entry** — the exit menu is exhausted (0/24); a future arc need not
re-sweep exits on this signal. (3) **A negative-fold block that is contiguous-in-time is a
REGIME drag, not exit-fixable** — diagnose the regime, but do NOT let "which filter flips
those folds" drive the choice (that is the Arc-10/§5f fishing trap; here both reasoned
refinements FAILED and the post-hoc inverse still fell short, so the verdict rests on no
fished number). (4) **Thread (fresh-arc, pre-register before believing):** month-end reversion
may be stronger for below-trend/oversold currencies ("rebalancing buys the underweight"); test
with its own ex-ante population + fair null, not as a post-hoc contrast. (5) **Next in this
range:** a 3rd decorrelated discrete-flow-event component (arc 1011's named hunt), since both
existing components' solo ceilings are now closed and a 2-way combination is gap-fill-variance
dominated (1011: 6/10).

**FLAGS (code not merged):** none requiring the canonical core. No new BUILT tool (reused
`MonthEndReversionLongSignal` + `make_time_exit_predicate`; the quarter-end / SMA-filter masks
are one-off conditioning helpers kept in scratch, like prior arcs' one-offs). Carries the
standing `A1Config.time_exit_bars`-unwired flag (arcs 1005/3004; worked around via the BUILT
`make_time_exit_predicate`). Drivers scratch `_disco_work/arc1012_*.py` (reproducible from this
doc).
