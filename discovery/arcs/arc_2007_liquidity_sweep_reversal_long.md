# arc_2007 — Multi-factor liquidity-sweep reversal long (DEEP directional)

**Chat:** 2000s | **Date:** 2026-06-05 | **Disposition:** KILL
**Verdict:** FAIL the sole judge (all-folds-positive) AND loses to the fair null → KILL.
**OOS:** deliberately NOT touched (IS never cleared; holdout preserved).

This is the 2000s range's **DEEP-arc** entry — a multi-factor, structurally-grounded directional
long developed freely on IS, the open frontier the shallow corpus left untested. It is the first arc
in the programme to take the "can DEPTH clear the bar nothing has cleared?" question and run it to an
airtight honest-engine falsification on a genuinely multi-condition structural entry.

---

## 1. The *because* (order-flow / structural, not a chart shape)

A DOWN move that pierces a well-defined prior swing low — running resting **sell-stops** and
triggering breakout-shorts — and then **immediately reclaims** that low within the same bar is the
classic **liquidity-grab / Wyckoff-spring**: a large buyer used the stop-cascade liquidity to fill a
long, so the path should revert UP. The mechanism is *forced/triggered flow* (stops are mechanical,
not informed), which is the same family that produced the programme's only net-positive edges (weekend
gap-fill 1006, month-end reversion 1011 — both forced-flow over-extension → reversion).

**Why MULTI-factor (the deep claim).** Each single condition is a corpus-proven coin-flip:
arc-0 (pierce-prior-5-low pullback) FAIL, arc-1010 (round-number spring) sub-0.50, arc-3000 (oversold
reversion) dead, arc-3006 (breakout-retest) dead. The §5(a) carve-out is that a genuinely novel
*conjunction* with a documented *because* still earns a fresh test. The hypothesis: the **intersection**
— (sweep below prior L-bar low) × (immediate reclaim) × (fast 3-bar drop in, the forced-speed proxy) ×
(higher-TF uptrend context) — isolates the genuine forced-sweep-in-uptrend cell that no single
condition reveals. Developed FREELY on IS (2010–2020), 7 USD majors, H4.

**Conditions considered and why (incl. rejected).**
- *Sweep of a prior swing low* — the liquidity pool; the event the grab needs. KEPT (core).
- *Immediate reclaim (close back above the low)* — distinguishes a grab (rejection) from a clean
  breakdown. KEPT.
- *Speed / forced-ness* — tested two proxies: **3-bar drop in** (`drop3 < −1 ATR`, KEPT) and **bar-range
  climax** (`range > k·ATR`, **REJECTED** — see §2, it is a falling knife).
- *Higher-TF uptrend (close > SMA200)* — the grab should be a dip-buy WITH the bigger flow. KEPT
  (it was the lift in the best cell), though ambiguous on the drift lens.
- *Reclaim strength (close near bar high)* — **REJECTED**: it lowered capture to 0.45 (you'd be buying
  *after* the bounce already ran; the entry is next-bar open).
- *Pierce depth band (shallow vs deep)* — explored; shallow added no lift once `drop3` already excluded
  the slow grind, so left disabled.

---

## 2. Observation (IS 2010–2020, H4 majors, honest `observe_long_capture`: +1R-before-SL capture + fwd drift)

Base capture **0.4860**, drift −0.0474 (the known H4-major coin-flip). Nested conjunction (n reported
at every cut; floor 50):

| cell | n | capture | fwd-drift (ATR) | fpos |
|---|---|---|---|---|
| sweep | 13,550 | 0.4905 | −0.031 | 0.496 |
| sweep & reclaim (spring) | 6,725 | 0.4928 | −0.013 | 0.497 |
| sweep & reclaim & uptrend | 1,922 | 0.5073 | **−0.124** | 0.497 |
| sweep & reclaim & uptrend & **climax** (rng>1.5·ATR) | 558 | 0.5036 | **−0.280** | 0.495 |
| sweep & reclaim & uptrend & climax (rng>2·ATR) | 277 | 0.5271 | **−0.333** | 0.495 |
| **sweep & reclaim & uptrend & fast3** (L20) | 771 | 0.5123 | −0.007 | 0.524 |
| **sweep & reclaim & uptrend & fast3** (L30) | 484 | **0.5207** | **+0.034** | 0.552 |

**Two findings that shaped the best version:**
1. **The climax (forced, big-range) sweep is a FALLING KNIFE** — capture ticks up but forward drift goes
   strongly NEGATIVE, *worse the bigger the climax* (−0.124 → −0.280 → −0.333). My *because*'s strongest
   prediction (violent forced sweep = exhaustion = reversion) is **falsified**: a climactic sweep below a
   swing low in an uptrend continues DOWN. (This is the strong **SHORT** leg — blocked by long-only,
   FLAG-1 again.)
2. The **only** non-knife long cell is the *fast-but-not-climactic* drop that reclaims in an uptrend
   (cap ~0.51–0.52, drift ~0 to +0.03, fpos 0.52–0.55) — marginally above coin-flip, **sub-cost** drift.

Round-2 refinement (depth bands × reclaim-strength × trend × drift-horizon 6/12/24) confirmed: adding
"strong reclaim" *drops* capture to 0.45; the only >0.55 cells were tiny-n noise (n=4–57, sign-unstable
across drift horizons). The robust best long version = **sweep & reclaim & close>SMA200 & drop3<−1·ATR**,
L∈{20,30}. Marginally non-coin-flip → §5f mandates the honest-engine best-version test before any FAIL
(no observation cheap-kill for a non-coin-flip entry).

---

## 3. Honest WFO (the sole judge — `MultiPairBacktester` via `ArcFoldRunner`, FundedNext costs, SL-first)

Full 10-fold IS WFO (`build_v3_folds`), registered exit menu × SL × short time-caps (§5f nested
exit/SL; anti-fishing = a SINGLE config must be all-folds-positive). Pool floor PASS (508–763
trades/config). **Every config mean-NEGATIVE; none all-folds-positive:**

**PRIMARY L20 d3=1.0 sma200** (1,150 IS fires)

| exit | n | folds + | worst | mean |
|---|---|---|---|---|
| sl_only sl2.0 te6 | 570 | 1/10 | −5.61% | −1.76% |
| sl_only sl2.0 te12 | 538 | 3/10 | −7.83% | −2.39% |
| sl_only sl2.0 te24 | 508 | 5/10 | −8.88% | −1.96% |
| sl_only sl1.5 te12 | 572 | 1/10 | −9.25% | −5.26% |
| sl_plus_tp_2r sl2.0 | 509 | 3/10 | −8.33% | −1.98% |
| partial_runner sl2.0 | 763 | 3/10 | −8.67% | −2.51% |
| trailing_atr sl2.0 | 511 | 3/10 | −10.27% | −2.91% |

**L30 d3=1.0 sma200** (739 fires): milder but still all mean-negative (−0.28% to −1.79%), 3–4/10 folds
positive, none all-folds-positive.

**The decisive test — fair same-conditions NULL** (random entry, matched fire-rate, identical exit/SL/
universe, seed 42), PRIMARY config:

| exit | REAL mean | NULL mean |
|---|---|---|
| sl_only sl2.0 te6 | **−1.76%** | −0.33% |
| sl_only sl2.0 te12 | **−2.39%** | −0.66% |
| sl_only sl2.0 te24 | **−1.96%** | −0.59% |

**The real multi-factor entry LOSES to random entry** at every horizon (consistently ~1.4–1.7pp worse).
It is not merely sub-cost — it is **worse than no signal**.

---

## 4. Diagnosis — why it loses to random (the mechanism)

The +1R-before-SL capture lift (0.486 → 0.52) is a **thin-tail artifact**: the sweep-reclaim produces a
quick +1R bounce often enough to tick capture above 0.50, but the forward drift is already ~0/sub-cost,
and the entry **buys into recent weakness** (a fast down move). Buying weakness sits in large adverse
excursion → take-the-loss (SL-first) converts a disproportionate share into −1R losers — *more* than a
random entry suffers, which has no adverse-excursion bias. So under the honest engine the small capture
edge is more than offset by the fat adverse tail, and the entry underperforms random. This is the exact
**arc-2001/2004 reversion stop-tax mechanism** (gap-fill MAE −1.1R; buy-into-weakness = worst case for a
2·ATR stop), now reproduced on a *structural chart* setup.

The deeper reason it fails where month-end (1011) succeeded: 1011 has a **forced-flow control**
(month-end vs random-day +0.249 ATR *excess* — the timing is causally load-bearing). The sweep-reclaim
*looks* like a grab but is **not tied to any measurable forced participant** — it is a chart pattern, and
chart-pattern reversion on liquid H4 majors is arbitrated to a coin-flip (round-numbers 1010,
breakout-retest 3006). **Edge comes from forced FLOW, not from structure that resembles forced flow.**

---

## 5. Verdict

**FAIL → KILL** (§11: an entry that loses to the fair null is net-negative and below the null baseline —
not PORTFOLIO, which requires mean-positive net of cost). The DEEP multi-factor conjunction does **not**
rescue directional prediction: it reproduces the corpus coin-flip on the capture lens and, under honest
scoring, an entry *worse than random* because it buys adverse-excursion weakness with no offsetting
forced flow. OOS preserved (IS never cleared; holdout pristine).

This is the clean, unconfounded **directional falsification the programme had never run on a genuinely
multi-factor structural entry** — the dispatch's stated value-if-it-breaks. Depth was developed freely
(2 swing-lookbacks, 7 exits, 2 SLs, 3 time-caps, depth/reclaim/trend/horizon observation sweep) and the
best honest version still loses to random.

---

## 6. Threads / lessons

1. **Multi-factor depth does NOT escape the directional coin-flip** — the first genuinely multi-condition
   structural long, developed freely toward all-folds-positive, lands at cap 0.52 (capture lens) and
   *below the fair null* (honest engine). Reinforces the arc-3004 escalation from a new, deeper angle:
   the wall is direction itself, not the shallowness of prior cuts.
2. **A climactic (forced, big-range) sweep below a swing low is a FALLING KNIFE** (drift −0.33 ATR, worse
   the bigger) — it CONTINUES down. The intuition "violent = exhaustion = reversion" is wrong for liquid
   H4 majors; the violent leg is the **short** edge (FLAG-1). A future shorts-enabled arc should test the
   climax-sweep SHORT, where direction is genuinely non-coin-flip on the observation.
3. **Buying structural weakness is worse than random under take-the-loss** unless a forced-flow control
   proves the timing is causal (the arc-1011 random-day-control discipline). A chart pattern that
   *resembles* a grab is not a grab; the edge needs a measurable forced participant, which sweep-reclaim
   lacks. (Re-confirms the corpus theme: the two wins are forced-flow events, every chart-structure
   reversion — 1010/3006/2007 — is arbitraged.)
4. **Capture-lens lift with ~0 drift is a thin-tail artifact that the honest engine reverses for a
   buy-weakness entry** — a marginally-above-0.50 capture is NOT evidence of edge when the entry sits in
   adverse excursion; only the fair null + engine settle it (here it lost both). Re-usable screen: for a
   reversion/buy-weakness entry, distrust a capture-only lift; require beats-fair-null on the engine.
5. **Surviving lanes unchanged:** the two PORTFOLIO components (1006 gap-fill, 1011 month-end) need a 3rd
   regime-orthogonal component positive in 2015/16/18/20 (arc-2006 spec) — and the directional-long menu
   that would win those risk-off years is comprehensively dead, so the shorts/second-leg unlock (FLAG-1)
   stays the highest-leverage operator move. The climax-sweep SHORT (lesson 2) is a concrete, well-
   motivated first target the moment shorts are enabled.

---

## 7. Tooling

Built + registered **`SweepReclaimReversalLongSignal`** (`discovery/tools/sweep_signals.py`) — EXPERIMENT
`SignalModule` (mask + ATR geometry only; never realizes P&L). Parameterizes swing-lookback, fast-3-bar
threshold, SMA trend filter, reclaim/depth options. Reusable for any sweep/spring structural test (e.g.
the shorts-enabled climax-sweep short). Reused BUILT `observe_long_capture`, `make_time_exit_predicate`,
`build_null_signal_evaluation`. Canonical measurement throughout (`build_arc_pool`, `ArcFoldRunner` →
`MultiPairBacktester`, `build_v3_folds`, `judge_all_folds_positive`).

## 8. FLAGS (code not merged)

None new requiring the canonical core. Carries **FLAG-1** (long-only blocks the stronger leg — here the
climax-sweep SHORT, lesson 2; operator/human-gated) + the standing `A1Config.time_exit_bars`-unwired flag
(arcs 1005/3004; worked around via the BUILT `make_time_exit_predicate`).

**Reproduction:** `PYTHONPATH=. py _disco2000_work/arc2007_observe.py` (round-1 obs),
`arc2007_observe2.py` (round-2 refinement), `arc2007_wfo.py` (honest WFO + null). Frame = H4 5ers_eet
from `C:\Users\panap\histdata_backup`, 7 USD majors, IS 2010–2020.
