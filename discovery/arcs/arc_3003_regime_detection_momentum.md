# Arc 3003 — Regime Detection for Momentum (is the trending regime detectable in advance?)

> **Arc id:** 3003 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (full IS WFO, 8/10 negative, mean −25.30%)** — the trending regime is NOT a
> forward-predictive, cost-clearing filter. Efficiency-Ratio regime does not predict forward momentum drift;
> the **strongest-trend regime INVERTS** (strong trends revert, −0.061R), and the best-case mid-ER band is
> catastrophic after SL-honest costs.
> **Idea-family:** the central unsolved question after 12 arcs. Momentum drift is real but **regime-dependent**
> (good in trending years 2013/2016/2020, bad in chop — arcs 1003/3001/3002). If the trending regime were
> detectable IN ADVANCE, you could trade momentum only then and escape the chop. Test with a causal
> trending-ness measure (Kaufman Efficiency Ratio / variance-ratio) — not yet tried.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Engine/measurement **called, never
re-rolled**.

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main`. Honest-era corpus = 11 arcs across 3 chats, all FAIL. Consolidated: directional
price-structure long is closed under both metrics (capture + drift), all instruments, timeframes, exits, the
volume axis (my arc 3002), and calendar is weakening (1005). The one recurring *positive* thread is that a
faint momentum drift exists but is **regime-dependent** — positive only in trending years, negative in chop
(arc 3001 post-up-spike +16.74% 2013 / −28% 2016; arc 3002 vol-spike +12% 2013/2016, 7/10 neg overall). The
binding question this leaves: **is that profitable (trending) regime detectable in advance?** Arcs 0/1000
showed entry-time observables don't separate good/bad *trades* (AUC≈0.50); arc 1000 showed a *dispersion*
regime *inverts*; arc 1001 showed *vol-level* regime doesn't help. But the most direct measure of "is this
pair trending right now" — a **variance-ratio / Efficiency-Ratio** — has NOT been tested as a forward regime
predictor. That is this arc.

## (b) Observation → idea (Efficiency-Ratio regime buckets)

Kaufman **Efficiency Ratio** `ER_W(s) = |close[s] − close[s−W]| / Σ|Δclose|` over trailing W (causal, ∈[0,1];
1 = clean trend, 0 = chop). Bucket forward 12-bar drift (arc 3001's lens, gross R) by ER, on the 12 trending
crosses (where momentum is strongest) and majors. Cost hurdle ~+0.05–0.10R.

**TREND_X, momentum entries (above SMA50), by ER60 quintile:**

| ER60 band | 0.00–0.04 | 0.04–0.09 | 0.09–0.15 | 0.15–0.23 | **0.23–0.62** |
|---|---|---|---|---|---|
| fwd drift | −0.010 | −0.001 | +0.011 | +0.019 | **−0.061** |

**REGIME INVERSION (falsification).** The strongest-trending regime (top ER bucket) has the **most negative**
forward drift (−0.061R) — strong recent trends **revert**, they do not persist. The mid-ER band is mildly
positive (+0.011 to +0.019R) but **below cost**. On MAJORS the pattern inverts the other way (choppiest low-ER
bucket mildly best, +0.023R, also sub-cost). **No ER band on any group clears cost, and the naive hypothesis
"trade momentum when trending" is actively wrong** (the trending regime is where momentum drift is worst).
This is arc 1000's dispersion inversion, re-derived on the trending-ness axis: there is no exploitable
trend-persistence regime.

**Idea (fail the BEST version):** even so, test the best-case band — momentum-long (up & above SMA50) gated to
the mid-ER sweet spot 0.09 ≤ ER60 ≤ 0.23 (avoiding both chop and exhausting trends), on trending crosses. If
even the best band fails the honest engine, the regime lever is definitively dead.

## (c)+(d)+(g) Characterize + validate (full IS WFO — learned arc 3002's lesson, skipped the lucky triage)

`build_arc_pool`, 12 trending crosses, IS 2010–2020, SL=2·ATR, hold 120. `pool_sha256 9c5d87000e916d96…`.
**9,250 IS trades** (high-frequency band), capture 0.4783, **gross mean final_r −0.0547** — *negative*. Note:
the band's +0.019R *raw forward* drift (observation) **evaporates to −0.055R SL-honest** pool expectancy.
Pool floor PASS. Ran the **full IS WFO directly** (arc 3002 taught me the 3-fold triage over-samples
momentum-friendly years):

| OOS year | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|---|---|---|---|
| ROI | −63.11 | +11.13 | +31.64 | −60.77 | −19.10 | −30.59 | −4.46 | −29.61 | −45.02 | −43.15 |

**worst −63.11%, mean −25.30%, median −30.10%, 8/10 NEGATIVE → catastrophic FAIL.** Massive DDs (up to 66%).
The best-case regime band is not just sub-cost — it bleeds severely (high-frequency × negative SL-honest
expectancy × compounding).

## (e)/(h) Diagnose / council / survivor — NOT REACHED

The observation already falsified the hypothesis (regime inversion) and the best-case full IS is catastrophic
— no worthwhile/reachable ceiling, so the heavy diagnosis council (§5e) is not triggered, and there is no
survivor. OOS not measured (IS fails decisively).

## Final verdict — FAIL

The trending regime is **not detectable in advance** as a cost-clearing momentum filter. Efficiency-Ratio /
variance-ratio regime does not predict forward momentum drift; the strongest-trend regime *inverts* (trends
revert), and the best-case band is catastrophically net-negative under SL-honest costs. This closes the
regime-detection lever — the central remaining hope for rescuing the faint momentum drift.

## Lessons (candidate for LESSONS.md compression)

1. **The trending regime is NOT detectable in advance for momentum** (Efficiency-Ratio / variance-ratio). The
   strongest-trend regime has the *most negative* forward drift (−0.061R) — strong trends REVERT. This is
   arc 1000's dispersion inversion generalized: **regime-conditioning has now failed across THREE independent
   regime measures — dispersion (1000), vol-level (1001), and trending-ness/ER (3003).** There is no
   exploitable regime structure for a long-only FX directional bet. Strong recent trends mean-revert; the
   middle is random.
2. **METHODOLOGICAL: raw forward drift OVERSTATES SL-honest expectancy.** The mid-ER band showed +0.019R raw
   forward drift but −0.0547R SL-honest pool expectancy — take-the-loss (the −1R stop hit before the slow
   drift accrues) destroys small positive drifts. The drift lens (arc 3001) is a useful *cheap pre-filter* but
   it is OPTIMISTIC vs the SL-honest engine; only the engine's verdict counts (good thing it is the gate). A
   sub-+0.05R drift cell is essentially guaranteed to be SL-honest-negative.
3. **A wide regime band fires too often** — 9,250 trades, high-frequency × negative per-trade expectancy ×
   compounding = −25% mean / 66% DD. Frequency amplifies a negative edge.

## Threads / what didn't help

- **Closed (this arc):** regime detection (Efficiency-Ratio / variance-ratio) for momentum; "trade momentum
  when trending" (inverts); the mid-ER best-case band (catastrophic). Regime-conditioning is now closed across
  all three regime measures tried.
- **The 3000s side has now systematically closed:** directional entries (continuation + reversion, both
  metrics), instrument universe, the volume data axis, and regime detection. Combined with the 1000s/2000s
  chats (timeframe, exits, calendar, convexity), **the price/volume/structure/regime directional space is
  comprehensively exhausted.**
- **Arc 3004 → LIGHT generative council (§5b).** I am now genuinely at a stuck-point / idea-fork: the obvious
  levers are closed. Per protocol §5b I will convene the discovery council for *generative* perspectives
  before another lone guess — or to pressure-test a meta-pivot (is this apparatus — long-only,
  single-instrument, these costs — capable of expressing any edge, and what structural change would be
  required?). This is the designed use of the council as the operator-judgment stand-in at a real fork.
- **Premature, not closed:** portfolio/selection (still no net-positive component).

## Flags (code NOT merged — human-gated, per protocol §9)

None requiring the canonical core. Scan + signal + drivers scratch `_disco3_work/`.

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Pairs:** 12 trending crosses (EURJPY GBPJPY AUDJPY NZDJPY
  CADJPY CHFJPY EURAUD EURNZD EURCAD GBPAUD GBPNZD GBPCAD). Observation also scanned the 7 USD majors.
- **Drivers (scratch):** `observe5_regime.py` (b, ER buckets), `arc3003_signal.py`
  (`RegimeBandMomentumLong(er_win=60, er_lo=0.09, er_hi=0.23, sma=50, refractory=6)`), `arc3003_triage.py`
  (full IS WFO). `pool_sha256 9c5d87000e916d96…`. Run `PYTHONPATH=. py _disco3_work/<script>.py`.
- **Engine:** `MultiPairBacktester` via `A1Architecture` + `ArcFoldRunner`; FundedNext costs at
  `build_fold_stats_from_run`; IS folds `build_v3_folds`; judge `judge_all_folds_positive`.
