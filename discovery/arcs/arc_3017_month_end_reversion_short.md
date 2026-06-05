# Arc 3017 — Month-End Reversion SHORT (the unharvested symmetric mirror of arc 1011)

> **Arc id:** 3017 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL the sole judge (not all-folds-positive) → disposition KILL.** The first short in
> the corpus to clear **>0.50 capture (0.5508)** with a **PASSING structure control** (+0.0996 ATR month-end
> excess; generic big-UP *continues* up) — a GENUINELY REAL mechanism (the direction-symmetric mirror of the
> validated 1011 long) that leans the RIGHT WAY for the 2018 leg (2015 obs drift **+0.437**, 2018 capture
> **0.818**). But on the honest engine it is **sub-cost / noise-floor**: not all-folds-positive (best 7/10),
> mean **+0.007%**, beats the fair same-side null by only **+0.012pp** (the noise floor, ~30× below 1006), and
> **un-scalable** (winning trades cluster on simultaneous USDXXX month-end up-extensions in strong-USD years →
> the 2-per-USD exposure cap erodes them, arc-1017 mode). Right SIGN, real mechanism — fails on **MAGNITUDE**.
> **Lever tested:** the unharvested SHORT side of the one demonstrably-2018-positive mechanism (1011), the
> precise 2015 & 2018-positive 4th-portfolio-leg spec (arc 1015/2008/3009), shorts now open (PR #273).

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first take-the-loss) via the canonical
`ArcFoldRunner` → `build_v3_folds` IS + the discovery judge. No council (council is mandatory only for a PASS
survivor; this is a cheap-engine KILL). **OOS deliberately NOT touched** (not all-folds-positive on IS →
preserve the holdout, §4). Reused BUILT `observe_long_capture` (dir-aware), `make_time_exit_predicate`,
`build_null_signal_evaluation` (dir-aware); built + registered `MonthEndReversionShortSignal`.

## (a) Log read — FRESH EYES (honest-era only)

Pulled main. Honest-era corpus = 40+ arcs / 3 chats, exhaustively mapped. **Three net-positive long-only
PORTFOLIO components** exist: gap-fill 1006 (JPY-cross H4), month-end 1011 (USD-major D1), failed-breakdown
reclaim 1013 (USD-major H4, strongest, 9/10). The deployable route (arcs 1015/2008/3009, triple-independent):
the 3-way book is cut to **2/10 neg {2015, 2018}, worst −0.77%** but is **provably blocked** (0/5151 convex
weightings all-folds-positive): **2015 positive ONLY in fbr, 2018 positive ONLY in me** → no convex weighting
passes both. The precise need: a **4th component positive in BOTH 2015 & 2018** (strong-USD/risk-off).

**Closed comprehensively:** shallow single-condition directional prediction (momentum/breakout/mean-reversion/
trend, long OR short), H1/H4/D1/W1, majors + crosses, both lenses, every exit/SL, stop-removed. The
**2018-leg hunt is the live frontier and has died on ~10 routes:** structure shorts (1014/2009/2011/3011 — all
capture <0.50 or failed structure control), trend short (3010 inverts), up-gap flow short (1016/2013 —
regime-luck), vol short (3012 — 2018 negative), relative-value (2010 — doubled-cost coin-flip), deep
continuation long (2012), carry-unwind flow short (1017 — real but un-scalable cascade), weekly trend (3014),
end-of-week reversion (3015 — no forced rebalancing), intraday session (3016 — sub-cost). **Every prior short
failed on SIGN/capture or structure control.**

**My lane:** the one mechanism nobody had mirrored. `me` (1011) is the **only demonstrably 2018-positive
mechanical-flow reversion**, but only its LONG side is harvested.

## (b) Idea + observation — month-end reversion SHORT (documented *because*)

**Because:** month-end WMR/index rebalancing is INELASTIC, **direction-symmetric** mechanical flow — a big
move INTO the fix over-extends and reverts, *either way*. Arc 1011 harvested only the LONG side (buy big DOWN
moves → revert UP), which is **NEGATIVE in the strong-USD block 2014/15/16** (arc 1012) because in USD-bull
years EURUSD-type down-moves *continue down*. The unharvested SHORT side (sell big UP moves into month-end →
revert DOWN) fires on **USDXXX over-extensions in exactly those strong-USD years** → a candidate **2015 &
2018-positive** 4th leg, sharing the *validated* 1011 mechanism (not a fishing construction). This is NOT
closed shallow-directional ground — it is a specific validated mechanical-flow mechanism's mirror, shorts now
open (PR #273), aimed at the exact binding fold.

**Observation (D1, 7 USD majors, IS 2010–2020, dir-aware `observe_long_capture(direction="short")`;**
`_disco_work/arc3017_observe_monthend_short.py`**).** Restrict = last-trading-day-of-month AND big UP move
(into ≥ +1.0 ATR):

| group | n | capture (short +1R-before-SL) | fwd2 drift (ATR) | median | frac_pos |
|---|---|---|---|---|---|
| **MONTH-END big-UP SHORT (signal)** | 118 | **0.5508** | **+0.0591** | +0.0354 | 0.525 |
| **RANDOM-DAY big-UP SHORT (control)** | 2674 | 0.4963 | −0.0405 | −0.0303 | 0.483 |

**Month-end EXCESS = +0.0996 ATR.** Capture clears **0.50** (the first short to do so — every dead short was
≤0.50: 1014 .489 / 2011·3011 .473 / 1016 .448), median is positive (passes the thin-tail test that killed
2011/3011/3012/1018), and the control PASSES (generic big-up *continues* up −0.0405 = bad for a short; only
the month-end big-up reverts). **The timing is load-bearing — the mechanism is real and direction-symmetric.**

**Acceptance test (the whole point):** **2015 drift +0.437 / median +0.326 / cap 0.545** (strongly positive,
exactly where the long bleeds); **2018 cap 0.818 (9/11 hit +1R before SL) / mean +0.293 / median −0.002**
(capture-strong). **First construction in the corpus where BOTH 2015 & 2018 read positive.**

## (c) Robustness — the yellow flag (`_disco_work/arc3017_robustness_obs.py`)

- **Threshold sweep:** capture robust >0.50 across 0.75–1.5 (.540–.575); drift median goes negative at thr≥1.25
  (thinning). Capture robust, drift concentrated in the 0.75–1.0 band.
- **Leave-one-pair-out:** drift mean stays positive in all 7 drops; capture >0.50 in 6/7 (drop-GBPUSD = 0.5053).
- **DROP TOP-2 pairs (GBPUSD+USDJPY): capture COLLAPSES to 0.4605, drift −0.027, median −0.033** — the
  arc-2011 pair-mix tell fires. The gross edge is concentrated in 2 pairs.

Mixed: capture (the SL-honest-engine-relevant metric) is robust; drift is pair-concentrated/fragile. Per §5f
(non-coin-flip entry + passing control), this is NOT a clean cheap-kill → the engine must decide.

## (d)–(g) Honest-engine WFO (§5f, the decisive test) — `_disco_work/arc3017_wfo.py`

Pool (`build_arc_pool`, D1, sl 2·ATR, thr 1.0, into 2): **n=116 (floor PASS)**, gross mean_final_r **+0.1713**
(notably HIGHER than the 1011 long's +0.0635). Full IS WFO, `MultiPairBacktester`, FundedNext ON, registered
exit menu × time-exit:

| exit | folds_pos | mean | worst | AFP | 2015 | 2018 |
|---|---|---|---|---|---|---|
| sl_only | 5/10 | −0.005% | −0.031% | N | +0.025% | +0.010% |
| sl_plus_tp_3r | 6/10 | +0.006% | −0.031% | N | +0.031% | +0.011% |
| sl_plus_trailing_atr | 6/10 | +0.005% | −0.014% | N | −0.001% | −0.000% |
| **sl_partial_close_1r_runner_trail** | **7/10** | **+0.007%** | −0.009% | N | +0.004% | +0.009% |
| sl_only + 2/3/5-bar time-exit | 5–6/10 | ≈0.000–0.002% | | N | + | + |
| **FAIR SAME-SIDE NULL** (partial_runner) | 5/10 | **−0.005%** | −0.036% | N | +0.004% | +0.019% |

**Decisive numbers:** (1) **0 exits all-folds-positive** (best 7/10). (2) Mean **+0.007%** — microscopic, ~30×
below 1011's long (+0.23%). (3) **Beats the fair same-side null by only +0.012pp** — the noise floor (~30×
below 1006's +0.36pp, below even 1017's +0.044pp). (4) 2015 passes (+0.004–0.031%), 2018 marginal (≈0 to
+0.011%) — but at magnitudes ~30× smaller than the 3-way book's 2015/2018 deficits (~−0.77%).

**Why a higher-gross-R short collapses to net ≈0 — un-scalability (arc-1017 mode;**
`_disco_work/` simultaneity check**):** 122 fires on 74 month-end dates; **10 dates carry ≥3 simultaneous
USDXXX fires, concentrated in strong-USD years (2011:6, 2012:7, 2014:6, 2018:7).** The 2-per-USD exposure cap
(USD is in all 7 pairs) drops the 3rd+ simultaneous fire → it erodes precisely the **clustered winning short
trades in strong-USD years**, the same correlated-cascade un-scalability that killed 1017. The uncapped pool
shows +0.171R and 2015/2018-positive drift; the capped, costed engine washes it to the noise floor.

## (h)/(i) Verdict, disposition, threads

**Verdict: FAIL the sole judge (not all-folds-positive) → KILL** (§11). It IS marginally mean-positive
(+0.007%, best exit) but beats the fair null by only **+0.012pp = noise floor** → not a *real* net-positive
edge (§11: beating null is necessary-but-NOT-sufficient; the 1017 precedent KILLs real-but-sub-cost +
noise-floor + un-scalable, even when "genuinely 2018-positive"). The per-fold magnitudes are 30× too small and
too noise-indistinguishable to rescue the book's 2015/2018 deficits; forcing a risk-parity combine would *game*
the artificially-tiny fold-vol (fitting to noise). NOT PORTFOLIO. OOS preserved.

**NEW lesson (the valuable one): the 2018-leg hunt has now found a candidate with the RIGHT SIGN and a REAL,
control-passing mechanism — and it STILL fails, on MAGNITUDE, not sign.** Every prior short died on
sign/capture (<0.50) or a failed structure control (1014/2009/2011/3010/1016/3012); the month-end short is the
**first to clear >0.50 capture with a passing control AND lean 2015/2018-positive** — confirming the month-end
mechanism is *genuinely direction-symmetric* (strengthening 1011's mechanism story). But the strong-USD-year
mechanical-reversion edge is **inherently tiny** (the month-end flow is a small effect) AND **un-scalable** (its
winning fires cluster on USD pairs in strong-USD years → exposure-capped, arc-1017). So a correctly-signed,
mechanism-real 4th leg fails because the *available magnitude* in the strong-USD-year reversion well is below
cost — not because no such mechanism exists. The route's 2018 wall is now shown to be a **magnitude/scale**
wall, not only a sign wall.

**Threads.** (1) The month-end mechanism is symmetric and real on BOTH sides; only the LONG side (on
EUR/commodity pairs, un-capped, larger) was ever big enough to be PORTFOLIO (1011), and even that is not
deployable. (2) Any 2015/2018-positive leg that fires on USD pairs in strong-USD years will hit the same
exposure-cap un-scalability (1017 + 3017) — a structural ceiling on USD-clustered strong-USD-year edges.
(3) Closes the "mirror the proven 2018-positive mechanism's short side" sub-route. 11th dead route to the
2018 leg; the strongest-signed of them. Components UNCHANGED (1006/1011/1013 still PORTFOLIO).

**FLAGS (code not merged):** none requiring the canonical core. Built + registered
`discovery/tools/month_end_signals.py :: MonthEndReversionShortSignal` (EXPERIMENT tool — mask + ATR geometry
+ `Direction.SHORT` only; scoring canonical). Drivers scratch `_disco_work/arc3017_*.py` (reproducible from
this doc).
