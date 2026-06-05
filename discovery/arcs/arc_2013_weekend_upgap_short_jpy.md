# arc 2013 — Weekend UP-gap weekend SHORT, JPY crosses (the gap-fill's "stronger leg", finally engine-tested) — KILL

- **Chat / range:** 2000s (2000–2999)
- **Arc id:** 2013
- **Date:** 2026-06-05
- **Disposition:** **KILL** — **CONVERGES with the independent 1000s arc 1016** (same idea, same KILL).
  The honest engine shows a thin mean-positive (+0.745% IS best exit) but it is **2018/2019-thin-regime-luck**
  (without those 2 folds the other 8 average **−0.44%**), the entry observation is **coin-flip-to-adverse**
  (independently reproduced by 1016: JPY cap 0.448 / drift −0.093), and against the **fair weekly-open null**
  the lift is marginal (+0.27 ATR). Not a robust component → KILL (NOT PORTFOLIO).
- **Council:** none (convergent independent reproduction + own fold-decomposition are decisive; arc-3010/1016
  regime-luck discipline).
- **Tooling:** BUILT `WeekendUpGapShortSignal` (kept — valid reusable short signal); made BUILT
  `build_null_signal_evaluation` direction-aware (additive, longs byte-identical). Reused
  `observe_long_capture(direction="short")`. **No canonical-core change** (short path merged, PR #273).
  Drivers: `_disco2000_work/arc2013_observe_upgap_short.py`, `arc2013_robustness.py`,
  `arc2013_engine_wfo.py`, `arc2013_null.py`.

---

## Idea + why

arc 2001 found the weekly-open gap-fill is monotone, **symmetric** on H4 (UP >1ATR gaps drift −0.57, frac
DOWN 0.64 — flagged the "stronger leg", FLAG-1). arc 1006 found the DOWN-gap fill LONG is mean-positive on
**JPY crosses** (the corpus's one PORTFOLIO long). Shorts merged (PR #273). The up-gap SHORT on JPY crosses
— the direction-mirror of 1006's strong leg, and 1015's named candidate 2018-leg — was the highest-acc
still-untested short. (Concurrently the 1000s chat tested the SAME idea as arc 1016; we converge.)

## What happened

### Observation (direction-aware `observe_long_capture`, H4)
- **JPY crosses, up-gap≥1.0:** honest i+1 capture 0.518, drift +0.08 (median +0.146) — but **≥0.5: drift
  −0.004 / median −0.115 (negative typical)**, and **≥1.5 INVERTS to −0.22** (non-monotone — thin-tail tell).
  Leave-one-pair-out: drop AUDJPY → **−0.107** (single-pair dependence). Only a single fragile threshold band.
- **Majors:** drift −0.355 (continues up) → dead (1006 cross>major asymmetry mirrored).
- **Independent 1000s arc 1016** found the SAME entry coin-flip-to-adverse (JPY cap **0.448** / drift
  **−0.093**, per-pair 1/5) — they KILLED at observation. My +0.08/≥1.0 is the fragile edge of their result,
  not a contradiction: the up-gap reversion is spent by the i+1 entry (backward-confirming, arc 1014 mode),
  and the **JPY-basket up-drift** (arc 1009) is a short headwind.

### Honest engine IS WFO (§5f exit menu) — the FIRST end-to-end short engine run
This DID validate the merged short path end-to-end: the short pool builds sign-correctly (mean final_r
+0.137 at thr 1.0), WFO runs clean, FundedNext costs net symmetrically — **no FLAG, no canonical change**.
- thr ≥0.5: all exits net-negative → dead.
- thr ≥1.0: mean-positive under the overshoot-exit family (`sl_plus_trailing_atr` +0.745%, tp_3r +0.510%,
  partial +0.324%, tp_2r +0.015%); non-overshoot exits negative. **NOT all-folds-positive** (5/10 neg).
- **The positive mean is REGIME-LUCK, not an edge** — per-fold (trailing_atr): 2011 +1.91(n5), 2012
  +2.21(n14), 2013 −2.24, 2014 −3.14, 2015 −0.57(n2), 2016 −3.18, 2017 +1.97, **2018 +5.08(n8)**, **2019
  +5.92(n10)**, 2020 −0.51(n1). **Excluding 2018+2019, the other 8 folds average −0.44%** (net-negative).
  The entire IS positivity is 2 thin lucky years (n=8, n=10) — exactly the "2018-positivity is uncapturable
  regime-luck within a yearly coin-flip" the 1000s arc 1016 / arc 3010 diagnosed. And it does **not** even
  satisfy 1015's spec (2015 = −0.57, negative).
- **Fair-null caveat:** the engine null (`build_null_signal_evaluation`) is random-ANY-bar (−0.99%), so the
  +1.7% "lift" partly reflects up-gaps firing at price extremes vs any bar. The FAIRER weekly-open null
  (obs R3) gives only +0.27 ATR — a much smaller, regime-luck-consistent margin.

## Verdict: KILL

Mean-positive on the engine ONLY via 2 thin regime-luck folds (2018/2019); net-negative across the other 8;
entry coin-flip-to-adverse (independently reproduced by 1016); fragile (≥0.5 median-neg, ≥1.5 inverts,
single-pair dependent); fails the 2015 half of the 4th-leg spec. A real-but-regime-luck signal is KILL, not
PORTFOLIO (§11 — you cannot diversify thin regime-luck into a robust book; arcs 3000/3001 spirit).

## Threads / lessons

1. **The up-gap weekend SHORT is dead — convergent KILL across two independent chats** (2000s arc 2013
   engine + 1000s arc 1016 observation). The honest i+1 short is coin-flip-to-adverse; arc-2001's −0.57 ATR
   "stronger leg" was the **untradeable gap-bar-OPEN (hindsight)** — by i+1 the reversion is spent
   (backward-confirming, arc 1014) and the JPY-basket up-drift taxes the short. The gap-fill's tradeable
   edge is the DOWN-gap LONG only (1006); its "stronger" up leg has no honest short.
2. **The Arc-10 defense worked (again)** — independent reproduction (1016) + my own fold-decomposition
   caught a thin-regime-luck "positive" before it was recorded as a PORTFOLIO component. The engine's
   mean-positive was real but driven entirely by 2 thin years; trusting it would have been the
   single-engine-trust trap (§11). **Lesson: for a thin signal, decompose the mean by fold/year before any
   PORTFOLIO claim — a +mean carried by 1–2 thin folds is regime-luck, not a component.**
3. **First end-to-end SHORT engine run is GREEN** — the merged PR #273 short path (Step-1 pool +
   architecture + `MultiPairBacktester` + cost netting) builds sign-correctly and nets costs symmetrically
   on a real arc. The short engine path is now exercised, not just observation-verified (closes arc 2011's
   "never exercised end-to-end" note). Reusable: `WeekendUpGapShortSignal` (BUILT), direction-aware null.
4. **The 2018-positive 4th leg remains unfound in ANY short construction** — structure (1014/2009/2011/3011),
   trend (3010), continuation-long (2012), flow (1016/2013), vol-state short (3012): all KILL. The portfolio
   route's 2018 wall stands; arc-3004 escalation reinforced.

## FLAGS (code not merged)
None requiring the canonical core. Two EXPERIMENT-tool additions (discovery/tools/, flow freely §9): BUILT
`WeekendUpGapShortSignal`; `build_null_signal_evaluation` made direction-aware (additive, longs
byte-identical). Both registered in `TOOL_REGISTRY.md`.
