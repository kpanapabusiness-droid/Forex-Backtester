# arc_2030 — fbr × level-test-count: is a *repeatedly-tested* level a cleaner fbr? (collinear with shadow → KILL)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (refinement redundant; fbr unchanged) · **Disposition:** KILL

## Seed (step a/b — from arc 2029's mechanism finding)
Arc 2029 established fbr's load-bearing liquidity feature: **significance-by-survival** — a rolling-K swing low that is *still intact* has survived K bars of testing, which is what concentrates resting stops (calendar-anchored prior-day/week lows, mechanically refreshed, are a coin-flip once fbr's survived-swing subset is removed). Direct sharpening: a swing low that was **repeatedly TESTED** (approached within a tight band, without breaking) before the sweep should hold even denser resting stops than one that merely survived passively → a cleaner, possibly fold-lifting fbr. *because:* multiply-tested support is where stops genuinely pile up. **Prediction:** honest +1R capture should rise with prior touch-count, and a high-touch subset should lift fbr's binding folds (2016/2018/2019, the strong-USD years).

## What I did
1. **Observation (gross capture/drift, characterization).** On the fbr swept-swing-low population (H4 4 majors, IS 2010–2020, K=40), counted prior **touches** of the swept level = bars in the prior 40 whose `low_bid` ∈ `[level, level+0.25·ATR]` (approached from above, didn't break). Scored via canonical `observe_long_capture`, grouped by touch-count.
2. **Honest engine (the verdict instrument).** Built the committed signal `FailedBreakdownReclaimTouchFilteredLongSignal` (fbr + `min_touches≥2`); ran raw fbr vs touch≥2 over the IS v3 folds via `ArcFoldRunner` → `MultiPairBacktester`, net of FundedNext costs, fbr's canonical exit (`sl_partial_close_1r_runner_trail`, NOT exit-fished). **OOS (2021+) NOT touched — holdout frozen.**

## What happened
**Observation (shadow 1.0) looked like a clear win** — capture by touch bucket: 0–1 → **0.480** (sub-coin-flip, drift −0.54) | 2–3 → **0.614** (+0.61) | 4–6 → 0.567. A `touches≥2` filter cut the weak ~43% (0–1) tail and lifted fbr capture 0.547→0.598, drift +0.086→+0.555, per-pair balanced (all 4 majors ≥0.548). But the per-year check already showed it raises the **mean**, not the gate: 2016/2018/2019 kept negative forward drift.

**Honest engine, CANONICAL config (shadow 1.25 — the integrity anchor), net of costs:**

| | mean ROI | worst fold | folds+ | AFP | n |
|---|---|---|---|---|---|
| raw fbr (sh1.25) | +0.007% | −0.030% | 8/10 | False | 190 |
| **fbr touch≥2 (sh1.25)** | **+0.007%** | −0.016% | **8/10** | False | **107** |

**Identical mean ROI, identical 8/10 folds-positive, the same binding folds still negative — and HALF the trades.** The modest worst-fold improvement (−0.030→−0.016) lands on 4-trade folds (noise). The refinement adds nothing to the gate; it only thins the pool → *worse* fold resolution (the arc-2017 problem).

## Why the gross win evaporated (the mechanism)
**Touch-count and shadow-depth are collinear "level-significance" proxies.** Both measure the same latent thing — "is this a significant, stop-laden level." At shadow **1.0** (a *weak* fbr) the touch filter helps because shadow 1.0 admits low-significance trades for it to remove. At the **canonical shadow ≥1.25**, the deep rejection wick has *already* removed those same weak trades, so the touch filter is redundant: same mean, same AFP, just fewer trades. This is the O1 collinearity warning (`DISCOVERY_DIRECTION`: forced-flow-density / spread-z / trigger-shallowing are "the same variable in three costumes — build it once") and a direct confirmation of **arc 1025** (fbr trigger depth and edge strength are coupled — fbr does not thicken). It also *confirms* arc 2029's survival-significance mechanism while showing canonical fbr's `shadow≥1.25` is already its sufficient operationalization.

## Arc-10 discipline note (why this arc matters beyond the KILL)
The gross capture observation (0.547→0.598, drift +0.086→+0.555) read as an unambiguous improvement to the crown-jewel component. **Verdicting on it — or even on my first engine run at a non-canonical shadow 1.0 (which showed 5/10→7/10) — would have falsely "improved" fbr.** Re-anchoring the comparison to the *canonical* fbr config (shadow 1.25) is what exposed the redundancy. Gross capture is not a gate; the honest engine at the component's real config is. Textbook internal-consistency-≠-correctness catch.

## Verdict + what this closes
**KILL the refinement** (NOT the family — fbr stays the recorded PORTFOLIO component, unchanged). Touch-count adds nothing to canonical fbr. Closes the level-significance-conditioning axis for fbr: survival-length (2029), trigger depth (1025/2014), and touch-frequency (this arc) are one collinear quality dimension that fbr's `shadow≥1.25` already harvests; you cannot lift fbr's strong-USD binding folds by sharpening *entry quality* (confirms arc 2019 — the binding folds are a regime property, not an entry-noise property). Components UNCHANGED (all 4 PORTFOLIO). Deployable-system count = 0.

**Threads.** None on the fbr entry-quality axis (collinear, harvested). The book's binding constraint stays the strong-USD risk-off folds — a regime/governance problem (operator path-A), not an entry-refinement one.

**Tooling:** BUILT `FailedBreakdownReclaimTouchFilteredLongSignal` (`discovery/tools/failed_breakdown_touch_filtered.py`) — committed + registered (TOOL_REGISTRY), reusable for touch-conditioning a base where shadow is NOT already filtering. **FLAGS:** none. Driver: `_arc2030_work/` (scratch, not committed).
