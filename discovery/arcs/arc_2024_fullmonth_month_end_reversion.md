# arc_2024 — Full-month-conditioned, dual-direction month-end reversion (option-B thick-standalone test)

**Chat:** 2000s | **Date:** 2026-06-05 | **Disposition:** KILL (obs cheap-kill, §5d) | **OOS:** untouched

## (a) Log read — what's been tried, what's dead, what's open (fresh eyes, honest era)

Read the full Tier-1 ledger + the 2000s/1000s/3000s recent Tier-2 + LESSONS + TOOL_REGISTRY (pulled main
first; STOP absent). State of the corpus:

- **The 4-component PORTFOLIO book** (gap-fill 1006 JPY-cross H4 + me_long 1011 USD-major D1 + fbr 1013
  USD-major H4 + me_short 1019 USD-major D1) is the corpus's deepest result: mean-positive (risk-parity
  +0.589%, P(mean<0)=0.004, arc 2019), ~3 independent bets (ENB 3.32), temporally robust (arc 2021), but
  **NOT all-folds-positive**.
- **The leg-hunt / portfolio route is PROVEN closed** three independent ways: arc 2019 (no diversification
  deficit, no tail to hedge → a 5th reversion leg can't make AFP), arc 2022 (the weighting dilemma — even a
  perfectly-targeted, maximally-decorrelated 5th leg fails honest weighting), and arc 3021 (portfolio-math
  proof: at realistic residual corr ρ≈0.1 P(AFP) plateaus ~0.33 and NEVER reaches 0.9 at any N — a shared
  dollar/risk-regime factor floors book variance → path-B densification cannot satisfy the per-year gate).
  ⇒ the deployability lever is the operator's **path-A gate-governance call**, not more leg-hunting.
- **fbr's −2018 is mechanism-intrinsic / entry-unconditionable**, confirmed 5 ways (2014 regime-gate / 2017
  per-fold CI / 2020 M1-microstructure / 1025 depth / 3013 level / 3020 breadth). And the **EXIT axis is
  effectively closed too** — arc 1013 swept all 6 registry exits and 2018 is negative under EVERY one, and
  arc 2014 found 2018 fbr trades are ~all −1R (18/19; 6/7 pairs 100% −1R) → no favorable excursion to exit
  into. So fbr solo-PASS is closed on BOTH entry and exit axes.
- **Closed elsewhere:** relative-value / market-neutral (2010/2018, doubled-cost-vs-coin-flip even with
  month-end timing); structural shorts have no tradeable mirror (1014/2009/2011/3011); triangulation
  first-moment (3005/1027/2023); session structure (1026/3016); shock-continuation epoch-died OOS (3019);
  fbr does not port off USD majors (1022/3018); option-B (a thick fold-resolving standalone) closed on 3
  prior constructions — fbr-thicken (1025), cross-sectional me (2018), thick-by-design session reversal
  (1026).
- **Unified theory the corpus has converged on:** real FX edges are intrinsically-rare forced-flow
  reversions → thin → the per-calendar-year AFP gate is structurally unsatisfiable for them, and they share
  a risk-off factor that floors any book's per-fold variance.

**Open thread I attacked:** arc-2017 **option B** — "a component THICK enough that per-year folds RESOLVE
and clear zero" is the one autonomous route to a standalone deployable (the portfolio route being closed).
The `me` family (1011/1019) is the ONLY mechanism that demonstrably carries the binding 2018 fold (me_long
+0.90, me_short +0.86). No arc has built the **unified dual-direction**, **full-month-conditioned**, thick
`me` and gated it standalone — arc 2018's option-B closure was the CROSS-SECTIONAL (2-leg, USD-neutral,
doubled-cost) version, a different construction.

## (b) Idea + because (observation-grounded)

`me` (1011/1019) fires on a `into_bars=2` move (the last 2 D1 bars) into month-end. The documented economic
driver of the WMR 4pm-fix month-end rebalancing flow is **equity-hedge rebalancing scaled to the MONTH's
currency appreciation** — i.e. the **full-month (~20 D1 bar) return**, not the last 2 days. Hypotheses:

- **Q1.** The full-month return is the economically-correct trigger → into_bars=20 should give a CLEANER
  (higher-capture / higher-excess) reversion than into_bars=2.
- **Q2/option-B.** A **unified dual-direction** fade (at month-end: long if the currency fell over the
  window, short if it rose) is ~2× thicker than each one-sided `me` leg → could RESOLVE folds (option B),
  while inheriting the +2015/+2018 property from the same forced flow → a shot at a thick standalone.
- **Q3.** Or — per the unified theory (arc 1025/2018) — thickening a forced-flow reversion just DILUTES it
  to coin-flip.

## (c)/(d) Characterize + cheap-kill (obs, §5d — gross capture/drift, NOT a gate)

Driver: `discovery/_disco2_work/arc_2024_fullmonth_me_obs.py`. Canonical month-end geometry
(`_month_end_into_move`, the locked `me` tool) + the direction-aware `observe_long_capture` harness
(take-the-loss +1R-before-SL capture + forward 10-bar drift in ATR). 7 USD majors, D1, IS 2010-2020. For
each `into_bars ∈ {2,5,10,20}` × `thr ∈ {1.0, 0.5}`: unified fade (long fires `into≤−thr`, short fires
`into≥+thr`, capture/drift direction-correct) vs a **random-day same-move control** (same |into| & sign on
NON-month-end bars) → month-end EXCESS.

```
 thr  into     n     cap  excess_cap  excess_drift   yr+    y2015    y2018
 1.0     2   240  0.5292      0.0414        0.2387     8   0.4072   0.0449
 1.0     5   420  0.5548      0.0592        0.3222     8  -0.0933  -0.0289
 1.0    10   588  0.5272      0.0324        0.2641     7  -0.1513  -0.0843
 1.0    20   681  0.5492      0.0577        0.2851     9   0.1504  -0.1699
 0.5     2   519  0.5145      0.0189        0.1600     7   0.1665   0.0825
 0.5     5   653  0.5360      0.0341        0.2493     8  -0.1629   0.0210
 0.5    10   741  0.5088      0.0110        0.1308     7  -0.2240  -0.2845
 0.5    20   783  0.5377      0.0455        0.2457     8  -0.0457  -0.2879
```

## (e) Diagnose — what the data says (question everything)

1. **Month-end timing is LOAD-BEARING at every window** — month-end EXCESS (drift vs the random-day
   same-move control, n=5–18k) is POSITIVE everywhere (+0.13 to +0.32 ATR). The dual-direction unified
   construction reproduces arc-1011's core control: the SAME move on a random day barely drifts (+0.01 to
   +0.04), the month-end one reverts. The mechanism is real and direction-symmetric. (Confirmatory.)

2. **Q1 REFUTED — the full-month trigger is NOT cleaner.** Capture is non-monotone in window length,
   peaking near into_bars=5 (0.5548) with into_bars=2/20 both ~0.53-0.55. The economically-"correct"
   full-month return does not beat the corpus's 2-day trigger; the `into_bars=2` `me` config is vindicated
   as already ~optimal.

3. **Q2 DECISIVE — no cell is option-B, and the reason is a clean window×2018 mechanism.** No cell is
   all-folds-positive (best 9/11, into20/thr1 — and that cell is 2018-NEGATIVE). The binding fold tells the
   story: **2018 drift decreases MONOTONICALLY with window length, robustly under BOTH thresholds:**
   - thr 1.0: into2 **+0.045** → into5 −0.029 → into10 −0.084 → into20 **−0.170**
   - thr 0.5: into2 **+0.083** → into5 +0.021 → into10 −0.285 → into20 −0.288

   Mechanism: 2018 was a strong-USD **trend** year. A 2-day move into month-end that reverts isolates the
   mechanical fix over-extension. A 20-day move IS the trend → fading it at month-end = fading a real trend
   → loses precisely in 2018 (the same "in a strong-USD regime the move IS the trend, it continues" failure
   that makes me_long 2015-negative, now appearing for the full-month window in 2018). The short (2-bar)
   window is what SEPARATES the fix-flow over-extension from the trend.

4. **Q3 — thickening does NOT collapse to coin-flip here** (unlike fbr, arc 1025): cap stays 0.51-0.55 from
   n=240 to n=783. But thickness buys NOTHING toward option-B: fold-resolution stays 7-9/11, never AFP, and
   the only cells keeping +2018 are the THIN 2-day cells (i.e. the existing `me`, already engine-tested).

## (f)/(g) Verdict — KILL (obs cheap-kill, §5d; no engine run owed)

The refinement (full-month trigger / unified-thick fade) is **falsified at observation**:
- It does NOT beat the existing `into_bars=2` `me` config (Q1 refuted).
- The both-blocker (+2015 AND +2018) property survives ONLY at the thin 2-day cell — which IS me_long+
  me_short, already through the engine + exit sweep as PORTFOLIO (1011/1019; 2018 there is marginal/sub-cost:
  gross drift +0.045 ATR « the ~0.05-0.10R cost hurdle).
- Longer windows are STRICTLY WORSE on the binding 2018 fold.

So there is no NEW best-version entry to put on the honest engine (§5f doesn't bite — the only non-coin-flip
cell is the already-validated `me`; the arc's hypothesis is what's falsified). This mirrors the arc 2014 /
2020 discipline (entry-refinement falsified at obs → §5d cheap-kill, no engine/null/council). OOS untouched.

## (i) What this closes + new lesson

- **Closes the "full-month trigger improves `me`" refinement** — `into_bars=2` is ~optimal; the full-month
  return is not a cleaner trigger and is strictly worse on 2018.
- **Closes single-leg option-B for `me` via the trigger-window lever** — complements arc 2018 (cross-
  sectional me) and arc 1025 (fbr-depth): three independent levers (cross-section / fbr-depth / me-window)
  all fail to deliver a thick fold-resolving standalone. Option B is now closed on a 4th construction.
- **NEW lesson:** the month-end reversion is a **SHORT-horizon (2-5 bar) fix-flow over-extension, NOT a
  monthly-appreciation rebalancing effect.** Lengthening the conditioning window contaminates the signal
  with trend and specifically destroys the strong-USD 2018 fold (2018 drift monotone-decreasing in window
  length under both thresholds). The intuitive "edge scales with the monthly move" economic model is the
  WRONG model for the tradeable signal — generalizes arc 2017's thinness finding (the edge lives in the
  acute, short-horizon over-extension, exactly where the pool is thin).
- Reinforces the corpus's terminal state: the operative deployability lever remains the operator's path-A
  gate-governance call (arc 2016/2017/2019/3021); the autonomous standalone-PASS route is closed on the one
  +2018-carrying mechanism's last untried trigger axis. Components UNCHANGED (me legs stay PORTFOLIO).

**Tooling:** no new tool (reused canonical `_month_end_into_move` + BUILT `observe_long_capture`). No
TOOL_REGISTRY append. **FLAGS:** none (no canonical-core change).
