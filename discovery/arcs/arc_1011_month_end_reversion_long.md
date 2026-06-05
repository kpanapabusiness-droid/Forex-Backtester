# Arc 1011 — Month-End Reversion Long (mechanical rebalancing-flow over-extension)

> **Arc id:** 1011 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL the sole judge (not all-folds-positive) → disposition PORTFOLIO.** A big DOWN
> move into month-end REVERSES (post-fix rebalancing over-extension); the long capture is **mean-positive
> net of FundedNext costs** (IS mean **+0.23%** sl_only/2-bar, +0.50% 3-bar), **beats a fair same-exit null
> by +0.56pp**, is **threshold-robust (0.75–1.5 all positive), broad-based (7/10 folds), and not
> single-pair (leave-one-out all positive)** — but **NOT all-folds-positive (7/10)** → not a survivor.
> The **2nd net-positive, decorrelated (corr +0.12 vs the gap-fill), long-only component** — the find the
> pre-shorts lane has hunted since arc 1006.
> **Lever tested:** MECHANISM — a discrete CALENDAR-FLOW reversion (the winning weekend-gap template,
> applied to a structurally different event/universe), frontier item 3, available pre-shorts.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first take-the-loss) via the canonical
`ArcFoldRunner` → `build_v3_folds` IS + the discovery judge. No council (council is mandatory only for a PASS
survivor → `passed/`; this is PORTFOLIO). OOS deliberately **NOT touched** (not all-folds-positive on IS →
preserve the holdout, §4 + the arc-2001 discipline). Reused BUILT `make_time_exit_predicate`,
`build_null_signal_evaluation`, `observe_long_capture`; built + registered `MonthEndReversionLongSignal`.

## (a) Log read — FRESH EYES (honest-era only)

Pulled main (incl. concurrent arc 3006 = multi-TF breakout-retest long KILL, which closes the "multi-TF
structure" sub-lane). Honest-era corpus = 22 arcs / 3 chats. **Closed comprehensively:** shallow
single-condition directional prediction (momentum / breakout / mean-reversion / trend, long OR short),
H1/H4/D1, majors + crosses, both lenses, every exit/SL, stop-removed (3004/2004); regime conditioning dead 3
ways; volume = magnitude; gotobi absent (1008); round-numbers absent (1010, my prior arc); triangulation ≈ 0
(3005); relative-value catch-up falsified (2003); breakout-retest dead (3006). **The one positive lead:**
weekend gap-down-fill long on JPY crosses (arc 1006; audited 1009) → **PORTFOLIO** — a *discrete
liquidity/flow-EVENT reversion*, the only mechanism family that has produced net-positive.

**Critical prior (arcs 3000/3001/3003):** generic short-term reversion is DEAD — a big down move on a random
day CONTINUES down. So any reversion claim must prove its *timing* is load-bearing vs generic reversion.

**My lane (dispatch, pre-shorts):** a 2nd net-positive **decorrelated** long-only component for the PORTFOLIO
route. The winning template is a discrete flow-event reversion (weekend gap). The natural decorrelated sibling
is a DIFFERENT discrete flow event on DIFFERENT instruments. STOP absent.

## (b) Idea + observation — month-end mechanical-rebalancing reversion (documented *because*)

**Because:** month-end portfolio rebalancing (the WMR 4pm London fix on the last business day) generates
large, INELASTIC, mechanical FX flows. A currency sold into the fix over-extends and **reverses** once the
flow completes — the move is mechanical, not informed. This is distinct from arc 1005, which measured
turn-of-month *drift over the whole window* (real but sub-cost); here the object is the discrete
**spike-into-month-end → reversion** — the weekend-gap-fill template (an over-extension that reverts) applied
to a calendar-flow event. Long-tradeable side: a big DOWN move into month-end (a currency sold into the fix) →
buy the reversion UP. Decorrelated from the weekend gap (different event, calendar, and universe).

**Observation (D1, 7 USD majors, IS 2010–2020;** `_disco_work/arc1011_observe_monthend_reversion.py`**).**
For each month-end (last trading day), the move INTO it (last 2 days, ATR) vs the forward 2-day reversion:

| move into month-end (ATR) | n | fwd2 (ATR) | frac_pos |
|---|---|---|---|
| ≤ −1.0 | 125 | **+0.186** | **0.616** |
| −1.0 to −0.5 | 136 | +0.010 | 0.500 |
| ... | | | |

The reversion concentrates in the **extreme** down moves (≤ −1 ATR): fwd2 **+0.186 ATR, 61.6% positive**
(fwd3 +0.185, holds) — comparable magnitude to the weekend-gap winner. Per-pair big-down (≤−0.5): AUDUSD
+0.42, NZDUSD +0.47, GBPUSD/USDCHF strong; EURUSD/USDJPY weak-negative (5/7 positive).

## (c) CRITICAL CONTROL — is month-end load-bearing, or just generic reversion (which is dead)?

`_disco_work/arc1011_control_monthend_vs_random.py` — the SAME big 2-day down move (≤ −1 ATR), month-end vs
RANDOM non-month-end days:

| ≤ −1 ATR down move | n | fwd2 (ATR) | frac_pos |
|---|---|---|---|
| **MONTH-END** | 125 | **+0.186** | 0.616 |
| **RANDOM DAY** | 2783 | **−0.063** | 0.480 |

**Month-end EXCESS = +0.249 ATR.** Generic big-down CONTINUES down (−0.063 — the dead generic-reversion
finding of arcs 3000/3001, re-confirmed across all 7 pairs); only the MONTH-END big-down reverses. **The
timing IS the mechanism** — this is a calendar-flow effect, not generic reversion. Per-pair: the random-day
control is negative/flat for all 7 pairs; the month-end reversion is positive in 5/7 (AUD/GBP/NZD/CHF strong:
+0.32 to +0.47).

## (d)–(g) Canonical engine WFO + robustness

Pool (`build_arc_pool`, D1, sl 2·ATR, thr 1.0, into 2): **n=121 (floor PASS)**, gross mean_final_r **+0.0635**.
Full IS WFO (skipped the lucky 3-fold triage per arc 3002), scored by `MultiPairBacktester`, FundedNext costs
ON (`_disco_work/arc1011_wfo.py`):

| exit | folds_pos | mean | worst | all-folds-pos |
|---|---|---|---|---|
| sl_only, time-exit 2-bar | **7/10** | **+0.23%** | −1.14% | N |
| sl_only, time-exit 3-bar | 6/10 | +0.50% | −0.79% | N |
| sl_only, 5-bar / partial-runner / tp_2r | 3–4/10 | ≈ 0 | | N |
| **FAIR NULL** (random entry, matched fire-rate, sl_only 2-bar) | 4/10 | **−0.33%** | −1.54% | N |

**Real beats the fair same-exit null by +0.56pp** (the month-end-timing-specific excess, the arc-1009
apples-to-apples discipline — independent confirmation of the +0.249 ATR gross control). NOT all-folds-positive
→ **OOS NOT touched.**

**Robustness (`_disco_work/arc1011_robustness.py`, the arc-1009 audit discipline):**
- **Per-fold (sl_only 2-bar):** `+0.40 +0.29 +0.96 −0.23 −1.14 −0.51 +0.34 +0.90 +1.16 +0.15` — positives
  broad-based across 7 folds, not one-fold luck.
- **Threshold-robust:** thr 0.75/1.0/1.25/1.5 → mean +0.18/+0.23/+0.27/+0.19%, 6–7/10 positive. *More* robust
  than the gap-fill (which lived only at 0.5 ATR).
- **Leave-one-pair-out:** all 7 drops stay positive (mean +0.15 to +0.28%, 6–7/10) — NOT single-pair-driven.

## (h)/(i) Verdict, decorrelation, threads

**Verdict: FAIL the sole judge (not all-folds-positive) → PORTFOLIO** (§11): mean-positive net of costs, beats
a fair null, mechanism-controlled, robust across threshold/hold/pair — but fold-fragile (7/10). A CANDIDATE
component, NOT deployable solo, never a survivor.

**Decorrelation vs the existing candidate (arc 1006 gap-fill):** Pearson on the 10 IS fold-ROI vectors =
**+0.117** (near-zero); structurally they cannot co-fire (weekly-open vs month-end-last-trading-day) and trade
disjoint universes (JPY crosses vs USD majors). A naive equal-add 2-way combination is still 6/10 positive
(the gap-fill's ±8% fold swings dominate my ±1%) → **a real combined-book all-folds-positive WFO (co-simulated,
risk-weighted, likely needing a 3rd component) is the gated next step** (§11), NOT claimed here.

**Threads / lessons.** (1) **Month-end mechanical-rebalancing reversion is a REAL, mechanism-controlled,
decorrelated net-positive long-only edge** — the 2nd PORTFOLIO component, and the first found by *applying the
winning discrete-flow-event template to a new event/universe* (deliberate, not luck). (2) **The random-day
control is the decisive test for any reversion claim** (generic reversion is dead — 3000/3001): month-end
+0.249 ATR EXCESS over generic = the timing is causal. Re-usable discipline: a reversion edge must beat its
own same-magnitude random-day control, not just the base. (3) **More threshold-robust than the gap-fill** (0.75–1.5
vs the gap-fill's single 0.5) — a sturdier component on that axis, though thinner in absolute mean (+0.23% vs
+0.69%). (4) **The portfolio thread is now ACTIVE** (2 decorrelated net-positive components, corr +0.12); the
combination is its own gated arc and likely wants ≥3 components or risk-weighting to tame the gap-fill's fold
variance. **Surviving pre-shorts lane:** a 3rd decorrelated discrete-flow-event component; the arc-3004
escalation (shorts/second-leg) still dominates for a *solo* deployable.

**FLAGS (code not merged):** none requiring the canonical core. Built + registered
`discovery/tools/month_end_signals.py :: MonthEndReversionLongSignal` (EXPERIMENT tool — mask + ATR geometry
only; scoring canonical). Carries the standing `A1Config.time_exit_bars`-unwired flag (arcs 1005/3004; worked
around via the BUILT `make_time_exit_predicate`). Drivers scratch `_disco_work/arc1011_*.py` (reproducible from
this doc).
