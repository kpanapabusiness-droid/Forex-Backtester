# Arc 3017 — Month-End Reversion SHORT (INDEPENDENT REPRODUCTION of arc 1019; Arc-10 defense)

> **Arc id:** 3017 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **PORTFOLIO — independent reproduction CONFIRMS arc 1019** (chat 1000s, which ran the
> same idea concurrently). My initial KILL was an **error** (retracted below): a measurement scale-mismatch,
> not a signal disagreement. The signal-level findings reproduce EXACTLY (pool n=116, gross mean_final_r
> +0.1713, honest short capture **0.5508 >0.50** — first corpus short to clear it, month-end structure-control
> excess **+0.0996 ATR**, 2015 & 2018 the right sign); and in the **linear (low-risk) regime the fold-sign
> pattern matches 1019** (partial-runner 7/10, 2015 +, 2018 +, beats null). The single net-positive month-end
> short component lives at [`../portfolio-candidates/arc_1019_month_end_reversion_short/`](../portfolio-candidates/arc_1019_month_end_reversion_short/)
> (1019's folder — NOT duplicated here). **This arc's lasting contribution is the Arc-10 defense it forced: a
> measurement-convention trap (`A1Config.risk_pct` units + daily-DD-cap nonlinearity) that flipped an honest
> verdict, now FLAGGED.**
> **Lever tested:** the unharvested SHORT side of the one demonstrably-2018-positive mechanism (1011), the
> precise 2015 & 2018-positive 4th-portfolio-leg spec (1015/2008/3009), shorts open (PR #273).

## (a) Log read — FRESH EYES (honest-era only)

Pulled main. The route is one regime-orthogonal component from deployable: the 3-way book (gap-fill 1006 +
month-end 1011 + failed-breakdown 1013) is provably blocked by {2015, 2018} — 2015 positive ONLY via fbr, 2018
positive ONLY via me — so the 4th leg must be positive in BOTH. `me` (1011) is the ONE demonstrably
2018-positive mechanism, but only its LONG side was harvested. ~12 prior routes to the 2018 leg are dead
(structure/trend/flow/vol shorts, relative-value, deep-continuation, carry-unwind, weekly, end-of-week,
intraday).

## (b)–(c) Idea, observation, structure control — REPRODUCED arc 1019 exactly

**Because:** month-end WMR/index rebalancing is INELASTIC, **direction-symmetric** mechanical flow. The LONG
side (1011) bleeds the strong-USD block 2014/15/16 (EURUSD-type down-moves continue); the unharvested SHORT
side (sell big UP moves into month-end → revert DOWN) fires on USDXXX over-extensions in those years → a
candidate 2015 & 2018-positive leg, sharing the *validated* 1011 mechanism. (I formed this independently before
pulling 1019 — genuine concurrent convergence.)

**Observation (D1, 7 USD majors, IS 2010–2020;** `_disco_work/arc3017_observe_monthend_short.py`**):**

| group | n | capture (short +1R-before-SL) | fwd2 drift (ATR) | median | frac_pos |
|---|---|---|---|---|---|
| **MONTH-END big-UP SHORT (signal)** | 118 | **0.5508** | +0.0591 | +0.0354 | 0.525 |
| **RANDOM-DAY big-UP SHORT (control)** | 2674 | 0.4963 | −0.0405 | −0.0303 | 0.483 |

**Month-end EXCESS = +0.0996 ATR** (1019 got +0.089 — matches). Capture clears 0.50 (the first short to do
so), median positive (passes the thin-tail test), control passes (generic big-up continues up). **2015 drift
+0.437/cap 0.545; 2018 cap 0.818.** All matching 1019. The mechanism is real and direction-symmetric.

**Robustness yellow flag (`_disco_work/arc3017_robustness_obs.py`):** capture robust >0.50 across thresholds &
6/7 leave-one-outs; but **drop-top-2-pairs (GBPUSD+USDJPY) → capture 0.4605/drift −0.027** (pair-concentrated;
1019 independently found 2015 leans on GBPUSD). The gross edge is concentrated — a real durability caveat.

## (d)–(g) Honest-engine WFO + the measurement discrepancy that forced an Arc-10 defense

Pool (`build_arc_pool`, D1, sl 2·ATR, thr 1.0, into 2): **n=116, gross mean_final_r +0.1713** — byte-identical
to 1019. But my first WFO pass (`_disco_work/arc3017_wfo.py`) read the engine ROI as **~100× too small** vs
1019 and the whole corpus, and I wrongly concluded KILL. Diagnosis (`A1Config.risk_pct` test):

| config | A1Config.risk_pct | 1011-LONG mean/fold | notes |
|---|---|---|---|
| my first run | 0.005 (the `ArcPoolConfig` fraction) | **+0.0023%** | 100× below 1011's reported **+0.23%** |
| corpus scale | 0.5 (PERCENT units) | +0.176% | matches corpus magnitude |

**THE TRAP (FLAG, Arc-10 class):** `ArcPoolConfig.risk_pct` is a **FRACTION** (0.005 = 0.5%) but
`A1Config.risk_pct` is in **PERCENT** (0.5 = 0.5%); its default is 0.005 (= 0.005%). Re-using the pool's 0.005
for the A1Config compresses every per-fold ROI by 100×. **My KILL was a scale-mismatch error:** I compared the
100×-compressed real-vs-null margin (+0.012pp) against the corpus's uncompressed benchmarks (1006 +0.36pp) and
called it "noise floor." **The scale-INVARIANT judgments agreed with 1019 the whole time:** at my (linear,
low-risk) scale the partial-runner was **7/10, 2015 +0.004%, 2018 +0.009%, real > null** — the SAME fold-sign
pattern 1019 reports (partial-runner 7/10, 2015 +, 2018 +, beats null +0.80pp). Reading the *pattern* not the
absolute pp would have caught my error.

**A SECOND, deeper FLAG (the genuinely worrying one):** the per-fold ROI is **not just scaled but reshaped by
`risk_pct`** through the daily-DD cap. At risk 0.5 the partial-runner fold-SIGNS flip (3/10; 2015 −1.07%, 2018
−1.08% — the high-vol strong-USD folds blow through the 5% daily cap and get truncated), the OPPOSITE of the
low-risk 7/10/2015+/2018+. So **the all-folds-positive / 2015-2018-sign verdict for this thin (n≈116, ~10/yr),
USD-concurrency-clustered short is risk_pct-convention-DEPENDENT.** The signal edge is robustly real; the
*engine disposition* is only as firm as the pinned risk convention.

## (h)/(i) Verdict, retraction, threads

**Verdict: PORTFOLIO (confirming arc 1019), KILL RETRACTED.** Independent reproduction confirms the signal
(pool/capture/structure-control/2015-18 sign all match) and the linear-regime fold-sign pattern. The component
is recorded under 1019's `portfolio-candidates/` folder; this arc does not duplicate it. OOS preserved
(not all-folds-positive on IS → §4).

**FLAGS (code human-gated, NOT patched — protocol §9):**
1. **`A1Config.risk_pct` is PERCENT (0.5 = 0.5%) while `ArcPoolConfig.risk_pct` is a FRACTION (0.005 = 0.5%);
   A1Config's default 0.005 = 0.005% silently 100×-compresses ROI.** The registry's standard-entry-point
   example omits `risk_pct` from `A1Config` (uses the default) → anyone copying it gets 100×-low magnitudes.
   **Recommendation:** pin the canonical discovery `A1Config.risk_pct` (0.5) in `TOOL_REGISTRY.md`'s entry
   point, or unify the two configs' units. Until pinned, **judge fold-SIGN/real-vs-null in the linear
   (low-risk) regime; never trust an absolute pp-margin across configs.**
2. **The daily-DD cap makes per-fold ROI (hence all-folds-positive and the 2015/2018 sign) nonlinear in
   risk_pct** for thin, concurrency-clustered books. The gated 4-way combination (arc 1020) must report its
   verdict's risk_pct-sensitivity, not a single-risk number — else it repeats the Arc-10 single-config trust.

**Lessons.** (1) **Independent reproduction works exactly as the protocol intends:** two chats, same idea, a
100× engine discrepancy → traced to a measurement convention, not the signal. The signal is confirmed; a real
apparatus trap is surfaced. (2) **Scale-invariant metrics (fold-sign pattern, real-vs-null) are the
trustworthy cross-config judges; absolute ROI magnitude is config-fragile.** (3) The month-end short is the
corpus's first 2018-positive PORTFOLIO short, but its engine disposition rests on the risk convention — flag,
don't celebrate (Arc-10). Components UNCHANGED (1006/1011/1013 + now the 1019 month-end short).

Built/confirmed `MonthEndReversionShortSignal` (registry credits arc 1019 + 3017). Drivers scratch
`_disco_work/arc3017_*.py` (reproducible from this doc).
